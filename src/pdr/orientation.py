"""Device attitude, orientation estimation, and magnetic disturbance detection (Task 21)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logging import setup_logger
from src.pdr.config import PDRConfig
from src.pdr.coordinate_frames import (
    quaternion_from_android_rotation_vector,
    quaternion_to_euler_angles,
    quaternion_to_rotation_matrix,
)

logger = setup_logger("PDROrientation")


@dataclass
class OrientationResult:
    """Output DataFrame and quality metrics of the orientation estimation stage."""

    df: pd.DataFrame
    primary_source: str
    mean_confidence: float
    magnetic_disturbance_count: int


class OrientationEstimator:
    """Estimates 3D smartphone attitude (roll, pitch, yaw) and detects magnetic anomalies."""

    def __init__(self, config: Optional[PDRConfig] = None):
        self.config = config or PDRConfig()

    def process(
        self,
        rot_df: Optional[pd.DataFrame] = None,
        accel_df: Optional[pd.DataFrame] = None,
        gyro_df: Optional[pd.DataFrame] = None,
        mag_df: Optional[pd.DataFrame] = None
    ) -> OrientationResult:
        """Estimates continuous attitude from available sensor combinations."""
        # Method 1: Android Rotation Vector (highest fidelity Sensor Hub fusion)
        if rot_df is not None and not rot_df.empty:
            return self._process_from_rotation_vector(rot_df, mag_df)

        # Method 2: Accelerometer + Magnetometer tilt-compensated e-Compass
        if accel_df is not None and not accel_df.empty and mag_df is not None and not mag_df.empty:
            return self._process_from_accel_mag(accel_df, mag_df, gyro_df)

        # Method 3: Gyroscope integration fallback
        if gyro_df is not None and not gyro_df.empty:
            return self._process_from_gyro_integration(gyro_df)

        # No orientation source available
        logger.warning("No orientation sensor streams available; returning identity attitude.")
        return OrientationResult(
            df=pd.DataFrame(),
            primary_source="NONE",
            mean_confidence=0.0,
            magnetic_disturbance_count=0
        )

    def _process_from_rotation_vector(
        self,
        rot_df: pd.DataFrame,
        mag_df: Optional[pd.DataFrame]
    ) -> OrientationResult:
        """Processes attitude directly from Android Rotation Vector quaternion stream."""
        logger.info("Estimating orientation from Android Rotation Vector stream.")
        df = rot_df.sort_values("normalized_epoch_ms").reset_index(drop=True).copy()

        rolls, pitches, yaws = [], [], []
        confidences = []

        for i in range(len(df)):
            rx, ry, rz = float(df["x"].iloc[i]), float(df["y"].iloc[i]), float(df["z"].iloc[i])
            rw = float(df["w"].iloc[i]) if "w" in df and not pd.isna(df["w"].iloc[i]) else None
            q = quaternion_from_android_rotation_vector(rx, ry, rz, rw)
            roll, pitch, yaw = quaternion_to_euler_angles(q)
            rolls.append(roll)
            pitches.append(pitch)
            yaws.append(yaw)
            confidences.append(0.95)

        df["roll_rad"] = rolls
        df["pitch_rad"] = pitches
        df["yaw_rad"] = yaws
        df["roll_deg"] = np.degrees(rolls)
        df["pitch_deg"] = np.degrees(pitches)
        df["yaw_deg"] = np.degrees(yaws)
        df["heading_source"] = "ROTATION_VECTOR"
        df["orientation_confidence"] = confidences
        df["magnetic_disturbance"] = False

        # Check magnetic disturbance if mag stream exists
        mag_disturbances = 0
        if mag_df is not None and not mag_df.empty:
            mag_disturbances = self._count_magnetic_disturbances(mag_df)

        return OrientationResult(
            df=df,
            primary_source="ROTATION_VECTOR",
            mean_confidence=float(np.mean(confidences)),
            magnetic_disturbance_count=mag_disturbances
        )

    def _process_from_accel_mag(
        self,
        accel_df: pd.DataFrame,
        mag_df: pd.DataFrame,
        gyro_df: Optional[pd.DataFrame]
    ) -> OrientationResult:
        """Method 2: Tilt-compensated magnetic heading from Accelerometer + Magnetometer."""
        logger.info("Estimating orientation from Accelerometer + Magnetometer e-Compass.")
        a = accel_df.sort_values("normalized_epoch_ms").copy()
        m = mag_df.sort_values("normalized_epoch_ms").copy()

        # Merge on closest timestamps
        a_times = a["normalized_epoch_ms"].values
        m_times = m["normalized_epoch_ms"].values
        closest_m_idx = np.searchsorted(m_times, a_times)
        closest_m_idx = np.clip(closest_m_idx, 0, len(m_times) - 1)

        rolls, pitches, yaws = [], [], []
        disturbances = []
        confidences = []

        b0 = self.config.mag_nominal_field_uT
        b_thresh = self.config.mag_disturbance_threshold_uT

        for i in range(len(a)):
            ax, ay, az = float(a["x"].iloc[i]), float(a["y"].iloc[i]), float(a["z"].iloc[i])
            m_row = m.iloc[closest_m_idx[i]]
            mx, my, mz = float(m_row["x"]), float(m_row["y"]), float(m_row["z"])

            # Pitch and roll from gravity vector
            roll = math.atan2(ay, az)
            pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))

            # Tilt-compensation for magnetometer
            # Project magnetometer onto horizontal plane
            b_xh = mx * math.cos(pitch) + mz * math.sin(pitch)
            b_yh = mx * math.sin(roll) * math.sin(pitch) + my * math.cos(roll) - mz * math.sin(roll) * math.cos(pitch)
            yaw = math.atan2(-b_yh, b_xh)

            # Check magnetic anomaly
            mag_norm = math.sqrt(mx**2 + my**2 + mz**2)
            is_disturbed = abs(mag_norm - b0) > b_thresh
            disturbances.append(is_disturbed)
            confidences.append(0.4 if is_disturbed else 0.75)

            rolls.append(roll)
            pitches.append(pitch)
            yaws.append(yaw)

        res_df = a.copy()
        res_df["roll_rad"] = rolls
        res_df["pitch_rad"] = pitches
        res_df["yaw_rad"] = yaws
        res_df["roll_deg"] = np.degrees(rolls)
        res_df["pitch_deg"] = np.degrees(pitches)
        res_df["yaw_deg"] = np.degrees(yaws)
        res_df["heading_source"] = "ACCEL_MAG_TILT"
        res_df["orientation_confidence"] = confidences
        res_df["magnetic_disturbance"] = disturbances

        return OrientationResult(
            df=res_df,
            primary_source="ACCEL_MAG_TILT",
            mean_confidence=float(np.mean(confidences)),
            magnetic_disturbance_count=sum(disturbances)
        )

    def _process_from_gyro_integration(self, gyro_df: pd.DataFrame) -> OrientationResult:
        """Method 3: Relative heading via trapezoidal integration of Gyroscope z-axis."""
        logger.info("Estimating orientation from Gyroscope angular rate integration.")
        g = gyro_df.sort_values("normalized_epoch_ms").reset_index(drop=True).copy()
        n = len(g)

        t_s = g["normalized_epoch_ms"].values / 1000.0
        wz = g["z"].values  # rad/s

        yaw = np.zeros(n, dtype=np.float64)
        yaw[0] = self.config.initial_heading_rad
        confidences = np.zeros(n, dtype=np.float64)
        confidences[0] = 0.8

        drift_rate = self.config.gyro_drift_rate_rad_per_s

        for i in range(1, n):
            dt = t_s[i] - t_s[i - 1]
            if 0 < dt < (self.config.sensor_gap_threshold_ms / 1000.0):
                # Trapezoidal integration
                yaw[i] = yaw[i - 1] + 0.5 * (wz[i] + wz[i - 1]) * dt
            else:
                # Discontinuity/gap
                yaw[i] = yaw[i - 1]

            # Confidence decays as sqrt of integration duration
            elapsed = t_s[i] - t_s[0]
            confidences[i] = max(0.2, 0.8 - math.sqrt(max(0.0, elapsed)) * drift_rate)

        g["roll_rad"] = 0.0
        g["pitch_rad"] = 0.0
        g["yaw_rad"] = yaw
        g["roll_deg"] = 0.0
        g["pitch_deg"] = 0.0
        g["yaw_deg"] = np.degrees(yaw)
        g["heading_source"] = "GYRO_INTEGRATED"
        g["orientation_confidence"] = confidences
        g["magnetic_disturbance"] = False

        return OrientationResult(
            df=g,
            primary_source="GYRO_INTEGRATED",
            mean_confidence=float(np.mean(confidences)),
            magnetic_disturbance_count=0
        )

    def _count_magnetic_disturbances(self, mag_df: pd.DataFrame) -> int:
        """Calculates total samples with anomalous magnetic field norm."""
        norms = np.sqrt(mag_df["x"]**2 + mag_df["y"]**2 + mag_df["z"]**2)
        devs = np.abs(norms - self.config.mag_nominal_field_uT)
        return int((devs > self.config.mag_disturbance_threshold_uT).sum())
