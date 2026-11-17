"""Gravity separation and vertical/horizontal acceleration decomposition (Task 21)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logging import setup_logger
from src.pdr.config import PDRConfig
from src.pdr.coordinate_frames import (
    project_to_horizontal_plane,
    quaternion_from_android_rotation_vector,
    quaternion_to_rotation_matrix,
    rotate_body_to_world,
    rotate_world_to_body,
)
from src.pdr.filtering import exponential_smoothing_online

logger = setup_logger("PDRGravity")


@dataclass
class GravityCompResult:
    """Output data of the gravity separation stage."""

    df: pd.DataFrame
    method_used: str
    mean_compensation_quality: float
    rotation_vector_coverage_ratio: float


class GravityCompensator:
    """Separates Earth's gravitational acceleration from user dynamic movement acceleration."""

    def __init__(self, config: Optional[PDRConfig] = None):
        self.config = config or PDRConfig()

    def process(
        self,
        accel_df: pd.DataFrame,
        rot_df: Optional[pd.DataFrame] = None
    ) -> GravityCompResult:
        """Decomposes acceleration into gravity and dynamic linear acceleration.

        Args:
            accel_df: Cleaned accelerometer records (x, y, z, normalized_epoch_ms).
            rot_df: Optional cleaned rotation vector records.

        Returns:
            GravityCompResult containing DataFrame with gravity, linear, and vertical acceleration.
        """
        if accel_df.empty:
            logger.warning("Empty accelerometer DataFrame provided to GravityCompensator.")
            return GravityCompResult(
                df=pd.DataFrame(),
                method_used="NONE",
                mean_compensation_quality=0.0,
                rotation_vector_coverage_ratio=0.0
            )

        df = accel_df.sort_values("normalized_epoch_ms").reset_index(drop=True).copy()
        n = len(df)
        g0 = self.config.nominal_gravity_mps2

        has_rot = (
            rot_df is not None and not rot_df.empty and
            len(rot_df) >= max(3, int(n * 0.2)) and
            self.config.gravity_method != "low_pass"
        )

        if has_rot:
            return self._process_with_rotation_vector(df, rot_df, g0)
        else:
            return self._process_with_low_pass(df, g0)

    def _process_with_rotation_vector(
        self,
        accel_df: pd.DataFrame,
        rot_df: pd.DataFrame,
        g0: float
    ) -> GravityCompResult:
        """Method A: Uses Android rotation vector orientation to remove gravity in navigation frame."""
        logger.info("Gravity separation using Method A (Android Rotation Vector).")
        rot = rot_df.sort_values("normalized_epoch_ms").copy()

        # Build nearest-neighbor / merge on timestamp for attitude alignment
        accel_times = accel_df["normalized_epoch_ms"].values
        rot_times = rot["normalized_epoch_ms"].values

        # Find closest rotation vector for each acceleration sample
        rot_indices = np.searchsorted(rot_times, accel_times)
        rot_indices = np.clip(rot_indices, 0, len(rot_times) - 1)

        # For samples where right neighbor is closer than left
        left_indices = np.maximum(0, rot_indices - 1)
        dist_right = np.abs(rot_times[rot_indices] - accel_times)
        dist_left = np.abs(rot_times[left_indices] - accel_times)
        best_indices = np.where(dist_left < dist_right, left_indices, rot_indices)

        # Tolerance check: if closest rotation vector is > 200 ms away, fall back to low pass for that sample
        time_diffs = np.abs(rot_times[best_indices] - accel_times)

        gravity_x = np.zeros(len(accel_df), dtype=np.float64)
        gravity_y = np.zeros(len(accel_df), dtype=np.float64)
        gravity_z = np.zeros(len(accel_df), dtype=np.float64)
        vertical_accel = np.zeros(len(accel_df), dtype=np.float64)
        quality = np.zeros(len(accel_df), dtype=np.float64)

        g_world = np.array([0.0, 0.0, -g0], dtype=np.float64)

        for i in range(len(accel_df)):
            if time_diffs[i] <= 300:  # Valid attitude match
                r_row = rot.iloc[best_indices[i]]
                rx, ry, rz = float(r_row["x"]), float(r_row["y"]), float(r_row["z"])
                rw = float(r_row["w"]) if "w" in r_row and not pd.isna(r_row["w"]) else None
                q = quaternion_from_android_rotation_vector(rx, ry, rz, rw)
                R = quaternion_to_rotation_matrix(q)

                # Gravity in body frame: R^T * g_world
                g_body = rotate_world_to_body(g_world, R)
                gravity_x[i] = g_body[0]
                gravity_y[i] = g_body[1]
                gravity_z[i] = g_body[2]

                # Accelerometer reading in body frame
                a_body = np.array([accel_df["x"].iloc[i], accel_df["y"].iloc[i], accel_df["z"].iloc[i]], dtype=np.float64)
                # Rotate total acceleration to world frame: a_world = R * a_body
                a_world = rotate_body_to_world(a_body, R)
                # Vertical acceleration (upward): a_world[2] - (-g0) = a_world[2] + g0
                vertical_accel[i] = a_world[2] + g0
                quality[i] = 1.0
            else:
                # Stale attitude fallback
                quality[i] = 0.4
                gravity_z[i] = g0  # Approximate vertical

        accel_df["gravity_x"] = gravity_x
        accel_df["gravity_y"] = gravity_y
        accel_df["gravity_z"] = gravity_z
        accel_df["linear_accel_x"] = accel_df["x"] - gravity_x
        accel_df["linear_accel_y"] = accel_df["y"] - gravity_y
        accel_df["linear_accel_z"] = accel_df["z"] - gravity_z
        accel_df["vertical_accel"] = vertical_accel
        accel_df["horizontal_accel_mag"] = np.sqrt(
            accel_df["linear_accel_x"]**2 + accel_df["linear_accel_y"]**2
        )
        accel_df["compensation_quality"] = quality
        accel_df["gravity_method"] = "ROTATION_VECTOR"

        valid_rot_ratio = float((time_diffs <= 300).mean())
        return GravityCompResult(
            df=accel_df,
            method_used="ROTATION_VECTOR",
            mean_compensation_quality=float(quality.mean()),
            rotation_vector_coverage_ratio=valid_rot_ratio
        )

    def _process_with_low_pass(self, accel_df: pd.DataFrame, g0: float) -> GravityCompResult:
        """Method B: Uses low-pass filtering on accelerometer axes when rotation vector is absent."""
        logger.info("Gravity separation using Method B (Low-Pass Accelerometer Filtering).")
        cutoff = self.config.gravity_low_pass_cutoff_hz
        gap_thresh = self.config.sensor_gap_threshold_ms

        t_ms = accel_df["normalized_epoch_ms"].values
        ax = accel_df["x"].values
        ay = accel_df["y"].values
        az = accel_df["z"].values

        gx = exponential_smoothing_online(ax, t_ms, cutoff_hz=cutoff, gap_threshold_ms=gap_thresh)
        gy = exponential_smoothing_online(ay, t_ms, cutoff_hz=cutoff, gap_threshold_ms=gap_thresh)
        gz = exponential_smoothing_online(az, t_ms, cutoff_hz=cutoff, gap_threshold_ms=gap_thresh)

        # Normalize gravity magnitude to nominal g0
        g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        safe_g_mag = np.where(g_mag > 1e-3, g_mag, g0)
        gx_norm = gx * (g0 / safe_g_mag)
        gy_norm = gy * (g0 / safe_g_mag)
        gz_norm = gz * (g0 / safe_g_mag)

        lin_x = ax - gx_norm
        lin_y = ay - gy_norm
        lin_z = az - gz_norm

        # Vertical acceleration: component along upward unit vector (-g / |g|)
        unit_gx = gx_norm / g0
        unit_gy = gy_norm / g0
        unit_gz = gz_norm / g0

        # a_vert = a_total . (-unit_g) - g0
        vert_accel = -(ax * unit_gx + ay * unit_gy + az * unit_gz) - g0

        # Horizontal linear acceleration
        horiz_accel_mag = np.sqrt(np.maximum(0.0, (lin_x**2 + lin_y**2 + lin_z**2) - vert_accel**2))

        accel_df["gravity_x"] = gx_norm
        accel_df["gravity_y"] = gy_norm
        accel_df["gravity_z"] = gz_norm
        accel_df["linear_accel_x"] = lin_x
        accel_df["linear_accel_y"] = lin_y
        accel_df["linear_accel_z"] = lin_z
        accel_df["vertical_accel"] = vert_accel
        accel_df["horizontal_accel_mag"] = horiz_accel_mag
        accel_df["compensation_quality"] = 0.65
        accel_df["gravity_method"] = "LOW_PASS_ACCELEROMETER"

        return GravityCompResult(
            df=accel_df,
            method_used="LOW_PASS_ACCELEROMETER",
            mean_compensation_quality=0.65,
            rotation_vector_coverage_ratio=0.0
        )
