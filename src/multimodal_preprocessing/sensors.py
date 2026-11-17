"""Tri-axial IMU and magnetometer sensor cleaning, gap analysis, and feature derivation (Task 20)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logging import setup_logger
from src.multimodal_preprocessing.config import MultimodalPreprocessingConfig

logger = setup_logger("MultimodalSensorPreprocessing")

SENSOR_UNITS = {
    "ACCELEROMETER": "m/s^2",
    "GYROSCOPE": "rad/s",
    "MAGNETIC_FIELD": "uT (microteslas)",
    "ROTATION_VECTOR": "unitless quaternion component",
}


@dataclass
class SensorCleaningStats:
    """Summary metrics of sensor data cleaning and sampling continuity."""

    records_input: int = 0
    records_retained: int = 0
    records_removed_nan_inf: int = 0
    records_removed_invalid_time: int = 0
    records_removed_duplicates: int = 0
    records_removed_unsupported_type: int = 0
    total_gaps_detected: int = 0
    gaps_by_sensor: Dict[str, int] = field(default_factory=dict)
    sampling_freq_by_sensor: Dict[str, float] = field(default_factory=dict)


class SensorPreprocessor:
    """Cleans multi-sensor streams, calculates kinematics, and preserves discontinuities without interpolation."""

    def __init__(self, config: Optional[MultimodalPreprocessingConfig] = None):
        self.config = config or MultimodalPreprocessingConfig()

    def process(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, SensorCleaningStats]:
        """Cleans sensor readings, detects sampling gaps, and extracts rolling kinematic features.

        Returns:
            Tuple of:
            - cleaned_df: Cleaned sensor stream with magnitude and gap flags.
            - features_df: Per-sensor aggregate statistical profile.
            - stats: Cleaning statistics and gap counts.
        """
        stats = SensorCleaningStats(records_input=len(raw_df))
        if raw_df.empty:
            logger.info("Sensor raw dataframe is empty; returning empty results.")
            empty_cleaned = pd.DataFrame(columns=[
                "observation_id", "session_id", "original_timestamp_ns", "time_source",
                "normalized_epoch_ms", "normalized_iso", "sync_method", "uncertainty_ms",
                "sensor_type", "x", "y", "z", "magnitude", "accuracy", "sampling_rate",
                "delta_t_ms", "is_gap", "is_mock"
            ])
            empty_features = pd.DataFrame(columns=[
                "sensor_type", "units", "record_count", "mean_sampling_freq_hz",
                "gap_count", "mean_magnitude", "std_magnitude", "mean_x", "mean_y", "mean_z"
            ])
            return empty_cleaned, empty_features, stats

        df = raw_df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        # 1. Check required sensor columns
        req_cols = ["normalized_epoch_ms", "sensor_type", "x", "y", "z"]
        for c in req_cols:
            if c not in df.columns:
                logger.error("Missing critical sensor column: %s", c)
                return pd.DataFrame(), pd.DataFrame(), stats

        # 2. Filter unsupported sensor types
        df["sensor_type"] = df["sensor_type"].astype(str).str.strip().str.upper()
        supported_mask = df["sensor_type"].isin(self.config.supported_sensor_types)
        stats.records_removed_unsupported_type = int((~supported_mask).sum())
        df = df[supported_mask].copy()

        # 3. Filter invalid timestamps
        df["normalized_epoch_ms"] = pd.to_numeric(df["normalized_epoch_ms"], errors="coerce")
        time_mask = df["normalized_epoch_ms"].notna() & (df["normalized_epoch_ms"] > 0)
        stats.records_removed_invalid_time = int((~time_mask).sum())
        df = df[time_mask].copy()
        df["normalized_epoch_ms"] = df["normalized_epoch_ms"].astype("int64")

        # 4. Clean numerical x, y, z: drop NaN, Inf, non-numeric
        for axis in ["x", "y", "z"]:
            df[axis] = pd.to_numeric(df[axis], errors="coerce")

        num_mask = (
            df["x"].notna() & ~np.isinf(df["x"]) &
            df["y"].notna() & ~np.isinf(df["y"]) &
            df["z"].notna() & ~np.isinf(df["z"])
        )
        stats.records_removed_nan_inf = int((~num_mask).sum())
        df = df[num_mask].copy()

        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), stats

        # 5. Process per-sensor stream: deduplicate timestamps, sort, detect gaps, compute derivatives
        processed_dfs: List[pd.DataFrame] = []
        feature_rows: List[Dict[str, Any]] = []

        for sensor_type, group in df.groupby("sensor_type"):
            g = group.sort_values("normalized_epoch_ms").copy()

            # Deduplicate exact timestamps per sensor type
            before_len = len(g)
            g = g.drop_duplicates(subset=["normalized_epoch_ms"], keep="first")
            stats.records_removed_duplicates += (before_len - len(g))

            # Recalculate or verify vector magnitude
            g["magnitude"] = np.sqrt(g["x"]**2 + g["y"]**2 + g["z"]**2)

            # Compute delta_t in milliseconds
            delta_t = g["normalized_epoch_ms"].diff().fillna(0.0)
            g["delta_t_ms"] = delta_t

            # Detect large sampling gaps (no silent interpolation!)
            is_gap = (delta_t > self.config.sensor_gap_threshold_ms) & (delta_t != g["normalized_epoch_ms"].iloc[0])
            g["is_gap"] = is_gap
            gap_count = int(is_gap.sum())
            stats.gaps_by_sensor[sensor_type] = gap_count
            stats.total_gaps_detected += gap_count

            # Calculate instantaneous sampling rate
            valid_dt = delta_t[delta_t > 0]
            if len(valid_dt) > 0:
                mean_dt = float(valid_dt.median())
                mean_freq = float(1000.0 / mean_dt) if mean_dt > 0 else 0.0
            else:
                mean_freq = 0.0
            stats.sampling_freq_by_sensor[sensor_type] = mean_freq

            # Calculate rolling statistics per segment bounded by gaps
            # Group by gap occurrences so rolling statistics do not bleed across gaps
            gap_segments = is_gap.cumsum()
            roll_win = self.config.sensor_rolling_window_samples

            g["magnitude_rolling_mean"] = g.groupby(gap_segments)["magnitude"].transform(
                lambda s: s.rolling(roll_win, min_periods=1).mean()
            )
            g["magnitude_rolling_std"] = g.groupby(gap_segments)["magnitude"].transform(
                lambda s: s.rolling(roll_win, min_periods=1).std().fillna(0.0)
            )

            # Jerk / signal derivative (finite difference)
            dt_safe = g["delta_t_ms"].replace(0.0, np.nan)
            g["jerk_x"] = (g["x"].diff() / dt_safe).fillna(0.0)
            g["jerk_y"] = (g["y"].diff() / dt_safe).fillna(0.0)
            g["jerk_z"] = (g["z"].diff() / dt_safe).fillna(0.0)
            # Nullify jerk at gap points
            g.loc[g["is_gap"], ["jerk_x", "jerk_y", "jerk_z"]] = 0.0

            processed_dfs.append(g)

            # Aggregate profile features
            feat = {
                "sensor_type": sensor_type,
                "units": SENSOR_UNITS.get(sensor_type, "unknown"),
                "record_count": int(len(g)),
                "mean_sampling_freq_hz": round(mean_freq, 2),
                "gap_count": gap_count,
                "mean_magnitude": float(g["magnitude"].mean()),
                "std_magnitude": float(g["magnitude"].std()) if len(g) > 1 else 0.0,
                "min_magnitude": float(g["magnitude"].min()),
                "max_magnitude": float(g["magnitude"].max()),
                "mean_x": float(g["x"].mean()),
                "mean_y": float(g["y"].mean()),
                "mean_z": float(g["z"].mean()),
                "std_x": float(g["x"].std()) if len(g) > 1 else 0.0,
                "std_y": float(g["y"].std()) if len(g) > 1 else 0.0,
                "std_z": float(g["z"].std()) if len(g) > 1 else 0.0,
            }
            feature_rows.append(feat)

        cleaned_df = pd.concat(processed_dfs, ignore_index=True) if processed_dfs else pd.DataFrame()
        features_df = pd.DataFrame(feature_rows)
        stats.records_retained = len(cleaned_df)

        logger.info(
            "Sensor Preprocessing: %d input -> %d retained across %d sensor types. Total gaps detected: %d.",
            stats.records_input, stats.records_retained, len(feature_rows), stats.total_gaps_detected
        )
        return cleaned_df, features_df, stats
