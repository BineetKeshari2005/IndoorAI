"""Configuration dataclass and parameter constants for multimodal data preprocessing (Task 20)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.core.config import find_project_root


@dataclass
class MultimodalPreprocessingConfig:
    """Configurable parameters for cleaning, feature engineering, window alignment, and segmentation."""

    # Versioning
    version: str = "1.0.0"

    # Directory Paths
    base_output_dir: Path = field(
        default_factory=lambda: find_project_root() / "data" / "processed" / "multimodal"
    )

    # Wi-Fi Preprocessing Parameters
    min_rssi_dbm: float = -105.0
    max_rssi_dbm: float = -20.0
    min_frequency_mhz: float = 2400.0
    max_frequency_mhz: float = 7125.0
    deduplicate_wifi_ap: str = "keep_strongest"  # "keep_strongest", "keep_first", "reject"
    require_fresh_scans_only: bool = False

    # Sensor Preprocessing Parameters
    sensor_gap_threshold_ms: float = 500.0
    sensor_rolling_window_samples: int = 5
    supported_sensor_types: Set[str] = field(
        default_factory=lambda: {
            "ACCELEROMETER",
            "GYROSCOPE",
            "MAGNETIC_FIELD",
            "ROTATION_VECTOR",
        }
    )

    # Temporal Alignment Parameters
    temporal_window_size_ms: float = 500.0
    temporal_window_stride_ms: float = 500.0  # Non-overlapping by default
    flag_ambiguous_timing_boundary: bool = True

    # Photo Preprocessing Parameters
    filter_unsanitized_photos: bool = True
    require_image_file_exists: bool = True

    # Annotation Parameters
    supported_annotation_types: Set[str] = field(
        default_factory=lambda: {
            "CORRIDOR",
            "DOOR",
            "ROOM_ENTRANCE",
            "STAIRCASE",
            "ELEVATOR",
            "LANDMARK",
            "REFERENCE_POINT",
            "MANUAL_LOCATION_CORRECTION",
            "CUSTOM",
        }
    )

    # Session Segmentation Parameters
    segmentation_strategy: str = "fixed_duration"  # "fixed_duration", "observation_count", "annotation_boundaries", "temporal_gaps"
    segmentation_duration_seconds: float = 60.0
    segmentation_observation_count: int = 100
    segmentation_gap_threshold_seconds: float = 5.0

    # Output Format & Options
    output_format: str = "parquet"
    write_manifest: bool = True
    write_report: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to JSON-serializable dictionary."""
        d = asdict(self)
        d["base_output_dir"] = str(self.base_output_dir)
        d["supported_sensor_types"] = sorted(list(self.supported_sensor_types))
        d["supported_annotation_types"] = sorted(list(self.supported_annotation_types))
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MultimodalPreprocessingConfig:
        """Constructs configuration from dictionary with type safety."""
        init_kwargs = {}
        for k, v in data.items():
            if k == "base_output_dir":
                init_kwargs[k] = Path(v)
            elif k in ("supported_sensor_types", "supported_annotation_types"):
                init_kwargs[k] = set(v)
            elif hasattr(cls, k):
                init_kwargs[k] = v
        return cls(**init_kwargs)
