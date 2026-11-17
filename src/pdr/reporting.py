"""Reporting, manifest generation, and cryptographic checksum logging for Pedestrian Dead Reckoning (Task 21)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def calculate_sha256(file_path: Path) -> str:
    """Computes hexadecimal SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


@dataclass
class PDRReport:
    """Comprehensive diagnostic and kinematic performance report for a PDR run."""

    dataset_id: str
    session_id: str
    status: str  # "COMPLETED", "WARNINGS", "FAILED"
    movement_summary: Dict[str, Any] = field(default_factory=dict)
    gravity_summary: Dict[str, Any] = field(default_factory=dict)
    orientation_summary: Dict[str, Any] = field(default_factory=dict)
    step_summary: Dict[str, Any] = field(default_factory=dict)
    output_summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes report to dictionary."""
        return asdict(self)


@dataclass
class PDRManifest:
    """Immutable audit manifest of PDR inputs, outputs, checksums, and parameters."""

    source_dataset_id: str
    session_id: str
    pdr_version: str
    config: Dict[str, Any]
    creation_timestamp_utc: str
    input_file_checksums: Dict[str, str] = field(default_factory=dict)
    output_file_checksums: Dict[str, str] = field(default_factory=dict)
    number_of_detected_steps: int = 0
    trajectory_point_count: int = 0
    processing_warnings: List[str] = field(default_factory=list)
    processing_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes manifest to dictionary."""
        return asdict(self)
