"""End-to-end multimodal data preprocessing pipeline orchestrator (Task 20)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.core.logging import setup_logger
from src.multimodal_preprocessing.annotations import AnnotationProcessor
from src.multimodal_preprocessing.config import MultimodalPreprocessingConfig
from src.multimodal_preprocessing.photos import PhotoMetadataProcessor
from src.multimodal_preprocessing.reporting import (
    PreprocessingManifest,
    PreprocessingReport,
    calculate_sha256,
)
from src.multimodal_preprocessing.segmentation import SessionSegmenter
from src.multimodal_preprocessing.sensors import SensorPreprocessor
from src.multimodal_preprocessing.timestamps import TimestampAligner
from src.multimodal_preprocessing.validation import MultimodalPreValidator
from src.multimodal_preprocessing.wifi import WifiPreprocessor

logger = setup_logger("MultimodalPreprocessingPipeline")


class MultimodalPreprocessingPipeline:
    """Orchestrates ingestion validation, modality cleaning, feature extraction, temporal alignment, and manifest generation."""

    def __init__(self, config: Optional[MultimodalPreprocessingConfig] = None):
        self.config = config or MultimodalPreprocessingConfig()

    def run(self, extracted_dir: Path, dataset_id: str) -> Tuple[PreprocessingReport, PreprocessingManifest]:
        """Executes the complete preprocessing pipeline on an extracted dataset workspace.

        Args:
            extracted_dir: Directory containing extracted survey files.
            dataset_id: Identifier of the source dataset.

        Returns:
            Tuple of (PreprocessingReport, PreprocessingManifest).
        """
        start_time = time.perf_counter()
        extracted_dir = Path(extracted_dir).resolve()
        output_dir = self.config.base_output_dir / dataset_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Pre-validation
        validator = MultimodalPreValidator(extracted_dir)
        pre_val = validator.validate()

        warnings = list(pre_val.warnings)
        errors = list(pre_val.errors)

        if not pre_val.is_valid:
            logger.error("Pre-validation failed for dataset %s: %s", dataset_id, errors)
            report = PreprocessingReport(
                dataset_id=dataset_id,
                session_id=pre_val.session_id,
                session_name=pre_val.session_name,
                status="FAILED",
                input_summary=pre_val.record_counts,
                warnings=warnings,
                errors=errors,
                processing_time_ms=round((time.perf_counter() - start_time) * 1000, 2)
            )
            manifest = PreprocessingManifest(
                source_dataset_id=dataset_id,
                session_id=pre_val.session_id,
                preprocessing_version=self.config.version,
                config=self.config.to_dict(),
                creation_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                warnings=warnings,
                errors=errors
            )
            return report, manifest

        # 2. Compute input file checksums
        input_checksums: Dict[str, str] = {}
        for item in extracted_dir.glob("*"):
            if item.is_file():
                try:
                    input_checksums[item.name] = calculate_sha256(item)
                except Exception as e:
                    warnings.append(f"Could not compute checksum for input {item.name}: {e}")

        # 3. Wi-Fi Preprocessing
        wifi_file = extracted_dir / "wifi_observations.csv"
        wifi_cleaned_df = pd.DataFrame()
        wifi_features_df = pd.DataFrame()
        wifi_stats = None
        bssid_stats = {}
        if wifi_file.is_file() and wifi_file.stat().st_size > 0:
            try:
                raw_wifi = pd.read_csv(wifi_file)
                wifi_proc = WifiPreprocessor(self.config)
                wifi_cleaned_df, wifi_features_df, wifi_stats, bssid_stats = wifi_proc.process(raw_wifi)
            except Exception as e:
                errors.append(f"Wi-Fi preprocessing failed: {e}")
                logger.error("Wi-Fi preprocessing error: %s", e)

        # 4. Sensor Preprocessing
        imu_file = extracted_dir / "imu_observations.csv"
        sensor_cleaned_df = pd.DataFrame()
        sensor_features_df = pd.DataFrame()
        sensor_stats = None
        if imu_file.is_file() and imu_file.stat().st_size > 0:
            try:
                raw_imu = pd.read_csv(imu_file)
                sensor_proc = SensorPreprocessor(self.config)
                sensor_cleaned_df, sensor_features_df, sensor_stats = sensor_proc.process(raw_imu)
            except Exception as e:
                errors.append(f"Sensor preprocessing failed: {e}")
                logger.error("Sensor preprocessing error: %s", e)

        # 5. Photo Preprocessing
        photo_file = extracted_dir / "photo_metadata.csv"
        photos_approved_df = pd.DataFrame()
        photos_excluded_df = pd.DataFrame()
        photo_stats = None
        images_dir = extracted_dir / "images"
        if photo_file.is_file() and photo_file.stat().st_size > 0:
            try:
                raw_photo = pd.read_csv(photo_file)
                photo_proc = PhotoMetadataProcessor(self.config, images_dir=images_dir)
                photos_approved_df, photos_excluded_df, photo_stats = photo_proc.process(raw_photo)
            except Exception as e:
                errors.append(f"Photo preprocessing failed: {e}")
                logger.error("Photo preprocessing error: %s", e)

        # 6. Annotation Preprocessing
        ann_file = extracted_dir / "annotations.csv"
        annotations_df = pd.DataFrame()
        ann_stats = None
        if ann_file.is_file() and ann_file.stat().st_size > 0:
            try:
                raw_ann = pd.read_csv(ann_file)
                ann_proc = AnnotationProcessor(self.config)
                annotations_df, ann_stats = ann_proc.process(raw_ann)
            except Exception as e:
                errors.append(f"Annotation preprocessing failed: {e}")
                logger.error("Annotation preprocessing error: %s", e)

        # 7. Temporal Window Alignment
        aligner = TimestampAligner(self.config)
        windows_df, alignment_stats = aligner.align(
            wifi_df=wifi_cleaned_df,
            sensors_df=sensor_cleaned_df,
            photos_df=photos_approved_df,
            annotations_df=annotations_df
        )

        # 8. Session Segmentation
        segmenter = SessionSegmenter(self.config)
        segments = segmenter.segment(
            session_id=pre_val.session_id,
            wifi_df=wifi_cleaned_df,
            sensors_df=sensor_cleaned_df,
            photos_df=photos_approved_df,
            annotations_df=annotations_df
        )

        # 9. Write outputs to Parquet & JSON files
        output_files_info: Dict[str, Any] = {}
        output_checksums: Dict[str, str] = {}
        output_record_counts: Dict[str, int] = {}

        def save_parquet(df: pd.DataFrame, rel_path: str):
            if df is not None and not df.empty:
                full_path = output_dir / rel_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(full_path, index=False, engine="pyarrow")
                sha = calculate_sha256(full_path)
                output_checksums[rel_path] = sha
                output_record_counts[rel_path] = len(df)
                output_files_info[rel_path] = {
                    "records": len(df),
                    "file_size_bytes": full_path.stat().st_size,
                    "sha256": sha,
                }

        save_parquet(wifi_cleaned_df, "wifi/wifi_cleaned.parquet")
        save_parquet(wifi_features_df, "wifi/wifi_features.parquet")
        save_parquet(sensor_cleaned_df, "sensors/sensor_cleaned.parquet")
        save_parquet(sensor_features_df, "sensors/sensor_features.parquet")
        save_parquet(photos_approved_df, "photos/photo_metadata.parquet")
        save_parquet(annotations_df, "annotations/annotations_cleaned.parquet")
        save_parquet(windows_df, "aligned/temporal_windows.parquet")

        # Save segments JSON
        segments_path = output_dir / "segments" / "segments.json"
        segments_path.parent.mkdir(parents=True, exist_ok=True)
        segments_data = [asdict(s) for s in segments]
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(segments_data, f, indent=2)
        sha_seg = calculate_sha256(segments_path)
        rel_seg = "segments/segments.json"
        output_checksums[rel_seg] = sha_seg
        output_record_counts[rel_seg] = len(segments)
        output_files_info[rel_seg] = {
            "records": len(segments),
            "file_size_bytes": segments_path.stat().st_size,
            "sha256": sha_seg,
        }

        # 10. Build Report & Manifest
        cleaning_summary = {
            "wifi": asdict(wifi_stats) if wifi_stats else {},
            "sensors": asdict(sensor_stats) if sensor_stats else {},
            "photos": asdict(photo_stats) if photo_stats else {},
            "annotations": asdict(ann_stats) if ann_stats else {},
        }

        feature_summary = {
            "wifi_sweeps": len(wifi_features_df),
            "unique_bssids": len(bssid_stats),
            "sensor_streams": len(sensor_features_df),
            "temporal_windows": len(windows_df),
            "session_segments": len(segments),
        }

        status = "FAILED" if errors else ("WARNINGS" if warnings else "COMPLETED")
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        report = PreprocessingReport(
            dataset_id=dataset_id,
            session_id=pre_val.session_id,
            session_name=pre_val.session_name,
            status=status,
            input_summary=pre_val.record_counts,
            cleaning_summary=cleaning_summary,
            feature_summary=feature_summary,
            alignment_summary=asdict(alignment_stats),
            output_summary=output_files_info,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms
        )

        manifest = PreprocessingManifest(
            source_dataset_id=dataset_id,
            session_id=pre_val.session_id,
            preprocessing_version=self.config.version,
            config=self.config.to_dict(),
            creation_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            input_file_checksums=input_checksums,
            output_file_checksums=output_checksums,
            input_record_counts=pre_val.record_counts,
            output_record_counts=output_record_counts,
            available_modalities=sorted(list(pre_val.available_modalities)),
            warnings=warnings,
            errors=errors
        )

        # Write report and manifest to output directory
        report_path = output_dir / "preprocessing_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        manifest_path = output_dir / "preprocessing_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        logger.info(
            "Multimodal Preprocessing Pipeline for %s finished in %0.2f ms with status %s.",
            dataset_id, elapsed_ms, status
        )
        return report, manifest
