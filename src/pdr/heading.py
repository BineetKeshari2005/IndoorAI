"""Pedestrian heading extraction, unwrapping, and heading uncertainty propagation (Task 21)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logging import setup_logger
from src.pdr.config import PDRConfig

logger = setup_logger("PDRHeading")


@dataclass
class StepHeading:
    """Estimated pedestrian heading angle and uncertainty for an individual step event."""

    heading_rad: float
    heading_deg: float
    heading_uncertainty_rad: float
    heading_source: str
    is_reliable: bool


class HeadingEstimator:
    """Extracts continuous pedestrian heading and synchronizes with discrete step events."""

    def __init__(self, config: Optional[PDRConfig] = None):
        self.config = config or PDRConfig()

    def get_heading_at_timestamp(
        self,
        timestamp_ms: int,
        orientation_df: pd.DataFrame,
        session_start_ms: int
    ) -> StepHeading:
        """Retrieves and computes heading and uncertainty at a specific step timestamp."""
        if orientation_df.empty or "yaw_rad" not in orientation_df.columns:
            # Fallback when no orientation is available
            return StepHeading(
                heading_rad=self.config.initial_heading_rad,
                heading_deg=float(np.degrees(self.config.initial_heading_rad)),
                heading_uncertainty_rad=float(math.radians(45.0)),
                heading_source="DEFAULT_FALLBACK",
                is_reliable=False
            )

        t_vals = orientation_df["normalized_epoch_ms"].values
        idx = int(np.searchsorted(t_vals, timestamp_ms))
        idx = min(len(t_vals) - 1, max(0, idx))

        row = orientation_df.iloc[idx]
        raw_yaw = float(row["yaw_rad"])
        source = str(row.get("heading_source", "UNKNOWN"))
        is_disturbed = bool(row.get("magnetic_disturbance", False))
        conf = float(row.get("orientation_confidence", 0.5))

        # Calculate time-dependent heading uncertainty
        elapsed_s = max(0.0, (timestamp_ms - session_start_ms) / 1000.0)

        if source == "ROTATION_VECTOR":
            # High-fidelity sensor hub fusion: low bounded uncertainty (~3 deg)
            sigma_theta = math.radians(3.5)
            is_reliable = True
        elif source == "GYRO_INTEGRATED":
            # Gyroscope integration drifts as sqrt(time)
            drift_rate = self.config.gyro_drift_rate_rad_per_s
            sigma_theta = math.sqrt(math.radians(5.0)**2 + (drift_rate**2) * elapsed_s)
            is_reliable = conf > 0.4
        elif source == "ACCEL_MAG_TILT":
            # Magnetometer heading affected by local metallic disturbances
            sigma_theta = math.radians(25.0) if is_disturbed else math.radians(8.0)
            is_reliable = not is_disturbed
        else:
            sigma_theta = math.radians(30.0)
            is_reliable = False

        # Apply coordinate convention
        # Default Cartesian: heading_rad is angle counterclockwise from +X
        heading_rad = raw_yaw

        return StepHeading(
            heading_rad=heading_rad,
            heading_deg=float(np.degrees(heading_rad)),
            heading_uncertainty_rad=sigma_theta,
            heading_source=source,
            is_reliable=is_reliable
        )
