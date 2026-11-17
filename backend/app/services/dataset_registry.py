"""Thread-safe local dataset registry managing cataloged multimodal survey datasets."""

from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.core.dataset_config import DatasetConfig, default_dataset_config
from backend.app.schemas.dataset import DatasetQualityReport, ValidationStatus
from src.core.logging import setup_logger

logger = setup_logger("DatasetRegistry")


class DatasetRegistry:
    """Manages index, metadata lookup, and lifecycle states of ingested survey datasets."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or default_dataset_config
        self.config.ensure_directories()
        self._lock = threading.Lock()
        self._datasets: Dict[str, Dict[str, Any]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Loads dataset index from disk if present."""
        registry_file = self.config.registry_path
        if registry_file.exists():
            try:
                with open(registry_file, "r", encoding="utf-8") as f:
                    self._datasets = json.load(f)
                logger.info("Loaded %d registered datasets from %s", len(self._datasets), registry_file)
            except Exception as e:
                logger.error("Failed to load registry file %s: %e", registry_file, e)
                self._datasets = {}
        else:
            self._datasets = {}

    def _save_registry(self) -> None:
        """Atomically persists dataset index to disk."""
        registry_file = self.config.registry_path
        tmp_file = registry_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(self._datasets, f, indent=2, default=str)
            os.replace(tmp_file, registry_file)
        except Exception as e:
            logger.error("Failed to atomically save registry: %s", e)
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)
            raise

    def register_dataset(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Registers or updates a dataset in the catalog."""
        dataset_id = entry["dataset_id"]
        with self._lock:
            self._datasets[dataset_id] = entry
            self._save_registry()
            logger.info("Registered dataset %s (session %s)", dataset_id, entry.get("session_id"))
            return entry

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata for a specific dataset ID."""
        with self._lock:
            return self._datasets.get(dataset_id)

    def list_datasets(
        self,
        status: Optional[str] = None,
        source_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Lists all registered datasets with optional filtering."""
        with self._lock:
            results = list(self._datasets.values())

        if status:
            target_status = status.upper()
            results = [d for d in results if d.get("validation_status", "").upper() == target_status]

        if source_type:
            target_type = source_type.upper()
            results = [d for d in results if d.get("data_source_type", "").upper() == target_type]

        # Sort descending by ingestion timestamp
        return sorted(results, key=lambda d: d.get("ingested_at_iso", ""), reverse=True)

    def delete_dataset(self, dataset_id: str) -> bool:
        """Deletes dataset entry and safely purges all associated files."""
        with self._lock:
            entry = self._datasets.get(dataset_id)
            if not entry:
                return False

            # Delete extracted directory
            extracted_dir = self.config.extracted_path / dataset_id
            if extracted_dir.exists():
                shutil.rmtree(extracted_dir, ignore_errors=True)

            # Delete preserved raw ZIP
            raw_zip = self.config.raw_zips_path / f"{dataset_id}.zip"
            if raw_zip.exists():
                raw_zip.unlink(missing_ok=True)

            del self._datasets[dataset_id]
            self._save_registry()
            logger.info("Deleted dataset %s and cleaned up storage.", dataset_id)
            return True

    def update_quality_report(self, dataset_id: str, report: DatasetQualityReport) -> Optional[Dict[str, Any]]:
        """Updates the recorded quality report and validation status for a registered dataset."""
        with self._lock:
            entry = self._datasets.get(dataset_id)
            if not entry:
                return None

            entry["validation_status"] = report.validation_status.value
            entry["warnings_count"] = len(report.warnings)
            entry["errors_count"] = len(report.errors)
            entry["quality_report"] = report.model_dump()
            self._save_registry()
            return entry
