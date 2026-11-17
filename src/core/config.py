"""Centralized Configuration Management for IndoorAI.

Loads settings from YAML files, .env files, and environment variables.
Avoids hardcoded paths throughout the codebase.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_project_root() -> Path:
    """Traverse upwards to find the IndoorAI project root directory."""
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "pyproject.toml").exists() or (current / "configs").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class PathSettings:
    """Standard filesystem paths used across the platform."""
    root_dir: Path = field(default_factory=find_project_root)
    data_dir: Path = field(default_factory=lambda: find_project_root() / "data")
    raw_data_dir: Path = field(default_factory=lambda: find_project_root() / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: find_project_root() / "data" / "processed")
    external_data_dir: Path = field(default_factory=lambda: find_project_root() / "data" / "external")
    configs_dir: Path = field(default_factory=lambda: find_project_root() / "configs")
    experiments_dir: Path = field(default_factory=lambda: find_project_root() / "experiments")
    notebooks_dir: Path = field(default_factory=lambda: find_project_root() / "notebooks")


@dataclass
class AppSettings:
    """Platform runtime settings."""
    app_name: str = "IndoorAI"
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "t")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    paths: PathSettings = field(default_factory=PathSettings)
    
    # Wi-Fi Fingerprint Standards
    missing_rssi_value: float = -104.0  # IEEE 802.11 threshold representation
    rssi_min: float = -104.0
    rssi_max: float = 0.0

    # Sensor Processing
    imu_sample_rate_hz: int = 50

    # Privacy and Security Constraints
    strip_image_exif: bool = True
    enable_face_blurring: bool = True
    require_device_authorization: bool = True


_settings_instance: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Return a singleton configuration instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AppSettings()
    return _settings_instance

