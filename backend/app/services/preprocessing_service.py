"""Service layer managing multimodal preprocessing execution and artifact cataloging (Task 20)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.app.core.dataset_config import DatasetConfig, default_dataset_config
from backend.app.schemas.preprocessing import PreprocessRequest, ProcessedFileInfo
from backend.app.services.dataset_registry import DatasetRegistry
from src.core.logging import setup_logger
from src.multimodal_preprocessing.config import MultimodalPreprocessingConfig
from src.multimodal_preprocessing.pipeline import MultimodalPreprocessingPipeline
from src.multimodal_preprocessing.reporting import PreprocessingManifest, PreprocessingReport

logger = setup_logger("PreprocessingService")


class PreprocessingService:
    """Manages preprocessing execution, artifact storage, and status inspection for survey datasets."""

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        registry: Optional[DatasetRegistry] = None
    ):
        self.config = config or default_dataset_config
        self.registry = registry or DatasetRegistry(self.config)
        self.output_base_dir = self.config.base_dir.parent / "processed" / "multimodal"
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_dataset(
        self,
        dataset_id: str,
        params: Optional[PreprocessRequest] = None
    ) -> Tuple[PreprocessingReport, PreprocessingManifest]:
        """Runs the preprocessing pipeline on an extracted dataset workspace."""
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_id}' not found in registry.")

        extracted_dir = self.config.extracted_path / dataset_id
        if not extracted_dir.is_dir():
            raise FileNotFoundError(f"Extracted directory for '{dataset_id}' does not exist.")

        # Build pipeline configuration
        pipeline_config = MultimodalPreprocessingConfig(
            base_output_dir=self.output_base_dir
        )
        if params is not None:
            pipeline_config.min_rssi_dbm = params.min_rssi_dbm
            pipeline_config.max_rssi_dbm = params.max_rssi_dbm
            pipeline_config.deduplicate_wifi_ap = params.deduplicate_wifi_ap
            pipeline_config.sensor_gap_threshold_ms = params.sensor_gap_threshold_ms
            pipeline_config.temporal_window_size_ms = params.temporal_window_size_ms
            pipeline_config.segmentation_strategy = params.segmentation_strategy
            pipeline_config.segmentation_duration_seconds = params.segmentation_duration_seconds
            pipeline_config.filter_unsanitized_photos = params.filter_unsanitized_photos

        pipeline = MultimodalPreprocessingPipeline(pipeline_config)
        report, manifest = pipeline.run(extracted_dir, dataset_id)
        return report, manifest

    def get_status(self, dataset_id: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Retrieves latest preprocessing status, report, and manifest for a dataset."""
        dataset_output_dir = self.output_base_dir / dataset_id
        report_file = dataset_output_dir / "preprocessing_report.json"
        manifest_file = dataset_output_dir / "preprocessing_manifest.json"

        if not report_file.is_file():
            return False, None, None

        report_data = None
        manifest_data = None
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        except Exception as e:
            logger.error("Failed to read report for %s: %s", dataset_id, e)

        if manifest_file.is_file():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception as e:
                logger.error("Failed to read manifest for %s: %s", dataset_id, e)

        return True, report_data, manifest_data

    def list_processed_files(self, dataset_id: str) -> List[ProcessedFileInfo]:
        """Lists all generated Parquet and JSON files for a dataset with sizes and checksums."""
        dataset_output_dir = self.output_base_dir / dataset_id
        manifest_file = dataset_output_dir / "preprocessing_manifest.json"

        if not manifest_file.is_file():
            return []

        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as e:
            logger.error("Could not load manifest for %s: %s", dataset_id, e)
            return []

        out_checksums = manifest_data.get("output_file_checksums", {})
        out_counts = manifest_data.get("output_record_counts", {})

        files: List[ProcessedFileInfo] = []
        for rel_path, sha in out_checksums.items():
            full_path = dataset_output_dir / rel_path
            size = full_path.stat().st_size if full_path.is_file() else 0
            count = out_counts.get(rel_path, 0)
            files.append(
                ProcessedFileInfo(
                    filename=full_path.name,
                    relative_path=rel_path,
                    record_count=count,
                    file_size_bytes=size,
                    sha256=sha
                )
            )

        return sorted(files, key=lambda x: x.relative_path)
