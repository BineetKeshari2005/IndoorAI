"""Configuration dataclass and algorithm parameter definitions for Pedestrian Dead Reckoning (Task 21)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.config import find_project_root


@dataclass
class PDRConfig:
    """Configurable parameters for gravity compensation, filtering, step detection, step length, heading, and trajectory."""

    # Versioning
    version: str = "1.0.0"

    # Directory Paths
    base_output_dir: Path = field(
        default_factory=lambda: find_project_root() / "data" / "processed" / "multimodal"
    )

    # Gravity Compensation Parameters
    gravity_method: str = "auto"  # "auto", "rotation_vector", "low_pass"
    gravity_low_pass_cutoff_hz: float = 0.5
    nominal_gravity_mps2: float = 9.80665

    # Signal Filtering Parameters
    accel_low_pass_cutoff_hz: float = 4.0
    accel_band_pass_low_hz: float = 0.5
    accel_band_pass_high_hz: float = 3.5
    sensor_gap_threshold_ms: float = 500.0

    # Step Detection Parameters
    step_detection_signal: str = "vertical"  # "vertical", "magnitude"
    min_step_interval_s: float = 0.33        # Max ~3 steps per second
    max_step_interval_s: float = 2.0         # Min ~30 steps per minute
    min_peak_prominence_mps2: float = 0.4
    peak_threshold_mps2: float = 0.8

    # Step Length Estimation Parameters (Weinberg heuristic: L = K * (amax - amin)^0.25)
    step_length_model: str = "weinberg"      # "weinberg", "fixed"
    weinberg_k: float = 0.42
    min_step_length_m: float = 0.30
    max_step_length_m: float = 1.20
    fixed_step_length_m: float = 0.70

    # Orientation and Heading Parameters
    heading_source: str = "auto"             # "auto", "rotation_vector", "gyroscope", "magnetometer"
    initial_heading_rad: float = 0.0
    gyro_drift_rate_rad_per_s: float = 0.005  # Approximate random walk ~0.3 deg/s
    mag_nominal_field_uT: float = 45.0
    mag_disturbance_threshold_uT: float = 15.0
    coordinate_convention: str = "cartesian"  # "cartesian" (dx = L cos theta, dy = L sin theta)

    # Trajectory Integration Parameters
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0
    initial_pos_uncertainty_m: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serializes configuration to JSON-compatible dictionary."""
        d = asdict(self)
        d["base_output_dir"] = str(self.base_output_dir)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PDRConfig:
        """Constructs PDRConfig from dictionary with type safety."""
        init_kwargs = {}
        for k, v in data.items():
            if k == "base_output_dir":
                init_kwargs[k] = Path(v)
            elif hasattr(cls, k):
                init_kwargs[k] = v
        return cls(**init_kwargs)
