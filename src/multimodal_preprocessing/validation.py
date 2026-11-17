"""Pre-flight validation for multimodal survey datasets before preprocessing (Task 20)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.core.logging import setup_logger

logger = setup_logger("MultimodalPreValidation")

REQUIRED_METADATA_FILES = {"session.json", "manifest.json"}
EXPECTED_MODALITY_FILES = {
    "wifi": "wifi_observations.csv",
    "sensors": "imu_observations.csv",
    "photos": "photo_metadata.csv",
    "annotations": "annotations.csv",
}


@dataclass
class PreValidationResult:
    """Outcome of pre-preprocessing validation."""

    is_valid: bool
    session_id: str
    session_name: str
    available_modalities: Set[str] = field(default_factory=set)
    record_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class MultimodalPreValidator:
    """Validates structural readiness of an extracted survey dataset for preprocessing."""

    def __init__(self, extracted_dir: Path):
        self.extracted_dir = Path(extracted_dir)

    def validate(self) -> PreValidationResult:
        """Executes pre-flight checks on files, headers, and metadata consistency."""
        errors: List[str] = []
        warnings: List[str] = []
        record_counts: Dict[str, int] = {}
        available_modalities: Set[str] = set()

        if not self.extracted_dir.exists() or not self.extracted_dir.is_dir():
            errors.append(f"Extracted dataset directory not found: {self.extracted_dir}")
            return PreValidationResult(
                is_valid=False,
                session_id="UNKNOWN",
                session_name="Nonexistent Directory",
                errors=errors
            )

        # 1. Check required metadata files
        for req in REQUIRED_METADATA_FILES:
            target = self.extracted_dir / req
            if not target.is_file():
                errors.append(f"Missing required metadata file: '{req}'")

        if errors:
            return PreValidationResult(
                is_valid=False,
                session_id="UNKNOWN",
                session_name="Missing Metadata",
                errors=errors
            )

        # 2. Parse session.json
        session_id = "UNKNOWN"
        session_name = "Survey Session"
        try:
            with open(self.extracted_dir / "session.json", "r", encoding="utf-8") as f:
                session_json = json.load(f)
                session_id = session_json.get("sessionId", "UNKNOWN")
                session_name = session_json.get("sessionName", "Survey Session")
        except Exception as e:
            errors.append(f"Corrupted or unreadable session.json: {e}")

        # 3. Parse manifest.json
        try:
            with open(self.extracted_dir / "manifest.json", "r", encoding="utf-8") as f:
                manifest_json = json.load(f)
                m_session_id = manifest_json.get("sessionId")
                if m_session_id and m_session_id != session_id:
                    warnings.append(
                        f"Session ID discrepancy: session.json='{session_id}' vs manifest.json='{m_session_id}'"
                    )
        except Exception as e:
            errors.append(f"Corrupted or unreadable manifest.json: {e}")

        # 4. Check presence and row counts for each modality CSV
        for mod, filename in EXPECTED_MODALITY_FILES.items():
            csv_path = self.extracted_dir / filename
            if not csv_path.exists():
                warnings.append(f"Modality file '{filename}' is absent (valid single-modality survey).")
                record_counts[mod] = 0
                continue

            try:
                with open(csv_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header is None:
                        record_counts[mod] = 0
                        warnings.append(f"Modality file '{filename}' is completely empty.")
                        continue

                    # Count rows
                    count = sum(1 for _ in reader)
                    record_counts[mod] = count
                    if count > 0:
                        available_modalities.add(mod)
            except Exception as e:
                errors.append(f"Failed to read CSV '{filename}': {e}")

        is_valid = len(errors) == 0
        return PreValidationResult(
            is_valid=is_valid,
            session_id=session_id,
            session_name=session_name,
            available_modalities=available_modalities,
            record_counts=record_counts,
            warnings=warnings,
            errors=errors
        )
