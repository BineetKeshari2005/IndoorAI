"""Multimodal dataset ingestion orchestrator coordinating extraction, validation, and registry."""

from __future__ import annotations

import io
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.app.core.dataset_config import DatasetConfig, default_dataset_config
from backend.app.schemas.dataset import (
    DataSourceType,
    DatasetQualityReport,
    ValidationStatus,
)
from backend.app.services.dataset_extraction import DatasetExtractor, SecurityValidationError
from backend.app.services.dataset_registry import DatasetRegistry
from backend.app.services.dataset_validation import DatasetValidator
from src.core.logging import setup_logger

logger = setup_logger("DatasetIngestionService")


class DatasetIngestionService:
    """End-to-end ingestion orchestrator for survey datasets exported by Android client."""

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        extractor: Optional[DatasetExtractor] = None,
        validator: Optional[DatasetValidator] = None,
        registry: Optional[DatasetRegistry] = None
    ):
        self.config = config or default_dataset_config
        self.config.ensure_directories()
        self.extractor = extractor or DatasetExtractor(self.config)
        self.validator = validator or DatasetValidator(self.config)
        self.registry = registry or DatasetRegistry(self.config)

    def ingest_dataset_zip(
        self,
        zip_input: Path | io.BytesIO,
        original_filename: str = "survey_export.zip",
        reported_size: Optional[int] = None,
        validate_only: bool = False
    ) -> Tuple[str, DatasetQualityReport, Dict[str, Any]]:
        """Ingests, validates, and catalog-registers an uploaded survey ZIP.

        Args:
            zip_input: Path to local zip or in-memory BytesIO stream.
            original_filename: Original name of uploaded archive.
            reported_size: Optional size in bytes for security checks.
            validate_only: If True, validates without permanently saving to registry.

        Returns:
            Tuple of (dataset_id, quality_report, registry_entry_or_summary).
        """
        dataset_id = str(uuid.uuid4())
        extracted_workspace = self.config.extracted_path / dataset_id
        raw_zip_dest = self.config.raw_zips_path / f"{dataset_id}.zip"

        try:
            # 1. Preserve raw ZIP file first if persisting
            if not validate_only:
                if isinstance(zip_input, Path):
                    shutil.copy2(zip_input, raw_zip_dest)
                else:
                    zip_input.seek(0)
                    with open(raw_zip_dest, "wb") as f:
                        shutil.copyfileobj(zip_input, f)
                    zip_input.seek(0)

            # 2. Extract safely into dedicated dataset workspace
            source_for_extraction = raw_zip_dest if not validate_only else zip_input
            self.extractor.extract_dataset(
                source_zip=source_for_extraction,
                dest_dir=extracted_workspace,
                reported_size=reported_size
            )

            # 3. Validate dataset schema, cryptographic checksums, and quality
            quality_report = self.validator.validate_dataset(extracted_workspace, dataset_id)

            # 4. Save machine-readable quality report inside extracted directory
            report_file = extracted_workspace / "quality_report.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(quality_report.model_dump(), f, indent=2, default=str)

            # If validate_only mode, cleanup and return report early
            if validate_only:
                shutil.rmtree(extracted_workspace, ignore_errors=True)
                return dataset_id, quality_report, {"dataset_id": dataset_id, "validate_only": True}

            # 5. Extract metadata from session.json for registry
            session_file = extracted_workspace / "session.json"
            session_data: Dict[str, Any] = {}
            if session_file.exists():
                try:
                    with open(session_file, "r", encoding="utf-8") as f:
                        session_data = json.load(f)
                except Exception as e:
                    logger.warning("Could not read session.json for registry metadata: %s", e)

            # Determine source classification
            real_cnt = session_data.get("realObservationCount", 0)
            mock_cnt = session_data.get("mockObservationCount", 0)
            if real_cnt > 0 and mock_cnt > 0:
                data_source_type = DataSourceType.HYBRID
            elif mock_cnt > 0:
                data_source_type = DataSourceType.MOCK
            else:
                data_source_type = DataSourceType.REAL

            wifi_q = quality_report.modalities.wifi
            imu_q = quality_report.modalities.sensors
            photo_q = quality_report.modalities.photos
            ann_q = quality_report.modalities.annotations

            wifi_count = wifi_q.total_scans if wifi_q else 0
            imu_count = imu_q.total_records if imu_q else 0
            photo_count = photo_q.total_photo_records if photo_q else 0
            ann_count = ann_q.total_annotations if ann_q else 0

            modality_avail = {
                "wifi": wifi_count > 0,
                "sensors": imu_count > 0,
                "photos": photo_count > 0,
                "annotations": ann_count > 0,
            }

            obs_counts = {
                "wifi": wifi_count,
                "imu": imu_count,
                "photos": photo_count,
                "annotations": ann_count,
                "total": wifi_count + imu_count + photo_count + ann_count,
            }

            now_iso = datetime.now(timezone.utc).isoformat()
            file_size_bytes = raw_zip_dest.stat().st_size if raw_zip_dest.exists() else 0

            # 6. Build registry entry
            registry_entry = {
                "dataset_id": dataset_id,
                "session_id": session_data.get("sessionId", quality_report.session_id),
                "session_name": session_data.get("sessionName", quality_report.session_name),
                "ingested_at_iso": now_iso,
                "validation_status": quality_report.validation_status.value,
                "data_source_type": data_source_type.value,
                "building_id": session_data.get("buildingId") or None,
                "floor_id": session_data.get("floorId") or None,
                "room_id": session_data.get("roomId") or None,
                "created_at_ms": session_data.get("createdAtMs"),
                "start_time_ms": session_data.get("startTimeMs"),
                "end_time_ms": session_data.get("endTimeMs"),
                "session_status": session_data.get("status", "COMPLETED"),
                "duration_seconds": session_data.get("durationSeconds", 0.0),
                "clock_anchor": session_data.get("clockAnchor"),
                "is_interrupted": session_data.get("isInterrupted", False),
                "interruption_note": session_data.get("interruptionNote"),
                "modality_availability": modality_avail,
                "observation_counts": obs_counts,
                "file_size_bytes": file_size_bytes,
                "warnings_count": len(quality_report.warnings),
                "errors_count": len(quality_report.errors),
                "quality_report": quality_report.model_dump(),
            }

            self.registry.register_dataset(registry_entry)
            return dataset_id, quality_report, registry_entry

        except Exception as e:
            logger.error("Failed to ingest dataset %s: %s", dataset_id, e)
            # Cleanup on failure
            if extracted_workspace.exists():
                shutil.rmtree(extracted_workspace, ignore_errors=True)
            if raw_zip_dest.exists():
                raw_zip_dest.unlink(missing_ok=True)
            raise

    def revalidate_dataset(self, dataset_id: str) -> Optional[DatasetQualityReport]:
        """Re-runs validation and quality assessment on an already ingested dataset."""
        entry = self.registry.get_dataset(dataset_id)
        if not entry:
            return None

        extracted_workspace = self.config.extracted_path / dataset_id
        if not extracted_workspace.exists():
            return None

        report = self.validator.validate_dataset(extracted_workspace, dataset_id)
        report_file = extracted_workspace / "quality_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)

        self.registry.update_quality_report(dataset_id, report)
        return report

    def get_quality_report(self, dataset_id: str) -> Optional[DatasetQualityReport]:
        """Retrieves the machine-readable quality report for a registered dataset."""
        extracted_workspace = self.config.extracted_path / dataset_id
        report_file = extracted_workspace / "quality_report.json"
        if report_file.exists():
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return DatasetQualityReport.model_validate(data)
            except Exception as e:
                logger.error("Failed to parse quality_report.json for %s: %s", dataset_id, e)

        entry = self.registry.get_dataset(dataset_id)
        if entry and "quality_report" in entry:
            return DatasetQualityReport.model_validate(entry["quality_report"])

        return None
