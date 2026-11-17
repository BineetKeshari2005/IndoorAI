"""Reporting, manifest generation, and cryptographic checksum auditing for multimodal preprocessing (Task 20)."""

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
class PreprocessingReport:
    """Comprehensive diagnostic and quality report for a preprocessing run."""

    dataset_id: str
    session_id: str
    session_name: str
    status: str  # "COMPLETED", "WARNINGS", "FAILED"
    input_summary: Dict[str, int] = field(default_factory=dict)
    cleaning_summary: Dict[str, Any] = field(default_factory=dict)
    feature_summary: Dict[str, Any] = field(default_factory=dict)
    alignment_summary: Dict[str, Any] = field(default_factory=dict)
    output_summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes report to dictionary."""
        return asdict(self)


@dataclass
class PreprocessingManifest:
    """Immutable audit manifest of inputs, outputs, checksums, and parameters."""

    source_dataset_id: str
    session_id: str
    preprocessing_version: str
    config: Dict[str, Any]
    creation_timestamp_utc: str
    input_file_checksums: Dict[str, str] = field(default_factory=dict)
    output_file_checksums: Dict[str, str] = field(default_factory=dict)
    input_record_counts: Dict[str, int] = field(default_factory=dict)
    output_record_counts: Dict[str, int] = field(default_factory=dict)
    available_modalities: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes manifest to dictionary."""
        return asdict(self)
