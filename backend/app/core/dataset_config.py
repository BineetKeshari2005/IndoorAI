"""Configuration settings and security thresholds for multimodal dataset ingestion."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set

from src.core.config import find_project_root


@dataclass
class DatasetConfig:
    """Security thresholds, path layouts, and validation parameters for survey dataset ingestion."""

    # Directory Paths
    base_dir: Path = field(default_factory=lambda: find_project_root() / "data" / "survey_datasets")
    raw_zips_dir_name: str = "raw_zips"
    extracted_dir_name: str = "extracted"
    registry_file_name: str = "registry.json"

    # Security Thresholds
    max_upload_size_bytes: int = 500 * 1024 * 1024  # 500 MB
    max_extracted_size_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GB
    max_archive_entries: int = 10_000
    max_compression_ratio: float = 100.0  # Max uncompressed:compressed ratio per entry

    # Allowed File Extensions & Structure
    allowed_extensions: Set[str] = field(
        default_factory=lambda: {".json", ".csv", ".jpg", ".jpeg", ".png"}
    )
    required_archive_files: Set[str] = field(
        default_factory=lambda: {
            "session.json",
            "wifi_observations.csv",
            "imu_observations.csv",
            "photo_metadata.csv",
            "annotations.csv",
            "manifest.json",
        }
    )
    images_subfolder: str = "images"

    # Export Schema Versions
    supported_export_versions: Set[str] = field(default_factory=lambda: {"1.0.0"})

    @property
    def raw_zips_path(self) -> Path:
        """Absolute path to the raw uploaded ZIP preservation directory."""
        return self.base_dir / self.raw_zips_dir_name

    @property
    def extracted_path(self) -> Path:
        """Absolute path to extracted dataset workspaces."""
        return self.base_dir / self.extracted_dir_name

    @property
    def registry_path(self) -> Path:
        """Absolute path to the central JSON dataset registry."""
        return self.base_dir / self.registry_file_name

    def ensure_directories(self) -> None:
        """Ensure all required dataset directories exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.raw_zips_path.mkdir(parents=True, exist_ok=True)
        self.extracted_path.mkdir(parents=True, exist_ok=True)


# Default global instance
default_dataset_config = DatasetConfig()
