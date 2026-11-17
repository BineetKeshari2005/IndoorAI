"""Comprehensive unit and integration test suite for Multimodal Preprocessing & Alignment (Task 20)."""

import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.core.dataset_config import DatasetConfig
from backend.app.main import create_app
from backend.app.services.dataset_extraction import DatasetExtractor
from backend.app.services.dataset_ingestion import DatasetIngestionService
from backend.app.services.dataset_registry import DatasetRegistry
from backend.app.services.dataset_validation import DatasetValidator
from backend.app.services.preprocessing_service import PreprocessingService
from src.multimodal_preprocessing.annotations import AnnotationProcessor
from src.multimodal_preprocessing.config import MultimodalPreprocessingConfig
from src.multimodal_preprocessing.photos import PhotoMetadataProcessor
from src.multimodal_preprocessing.pipeline import MultimodalPreprocessingPipeline
from src.multimodal_preprocessing.reporting import calculate_sha256
from src.multimodal_preprocessing.segmentation import SessionSegmenter
from src.multimodal_preprocessing.sensors import SensorPreprocessor
from src.multimodal_preprocessing.timestamps import TimestampAligner
from src.multimodal_preprocessing.wifi import WifiPreprocessor
from tests.test_dataset_ingestion import build_synthetic_survey_zip, create_test_image_bytes


class TestMultimodalPreprocessingUnits(unittest.TestCase):
    """Unit tests for individual modality preprocessors, alignment, and segmentation."""

    def setUp(self):
        self.config = MultimodalPreprocessingConfig()

    def test_wifi_cleaning_and_deduplication(self):
        """Verifies Wi-Fi preprocessor cleans invalid RSSIs, missing BSSIDs, and deduplicates within sweep."""
        raw_wifi = pd.DataFrame([
            # Valid record 1
            {
                "observation_id": "WIFI-001", "session_id": "SES-1", "normalized_epoch_ms": 1000,
                "normalized_iso": "2026-11-17T15:00:01.000Z", "time_source": "BOOTTIME_NANOS",
                "sync_method": "CLOCK_ANCHOR", "uncertainty_ms": 10.0, "is_fresh": True, "is_mock": False,
                "bssid": "00:11:22:33:44:55", "ssid": "Campus-Net", "rssi_dbm": -65.0, "frequency_mhz": 2412.0,
                "channel": 1, "channel_band": "2.4GHz", "building": "B1", "floor": "F1", "room": "R101"
            },
            # Duplicate BSSID in same sweep with weaker RSSI (-75 vs -65)
            {
                "observation_id": "WIFI-002", "session_id": "SES-1", "normalized_epoch_ms": 1050,
                "normalized_iso": "2026-11-17T15:00:01.050Z", "time_source": "BOOTTIME_NANOS",
                "sync_method": "CLOCK_ANCHOR", "uncertainty_ms": 10.0, "is_fresh": True, "is_mock": False,
                "bssid": "00:11:22:33:44:55", "ssid": "Campus-Net", "rssi_dbm": -75.0, "frequency_mhz": 2412.0,
                "channel": 1, "channel_band": "2.4GHz", "building": "B1", "floor": "F1", "room": "R101"
            },
            # Invalid RSSI (0.0 dBm is above -20 dBm threshold)
            {
                "observation_id": "WIFI-003", "session_id": "SES-1", "normalized_epoch_ms": 1080,
                "bssid": "aa:bb:cc:dd:ee:01", "rssi_dbm": 5.0, "frequency_mhz": 2412.0
            },
            # Missing / invalid BSSID
            {
                "observation_id": "WIFI-004", "session_id": "SES-1", "normalized_epoch_ms": 1100,
                "bssid": "not-a-mac", "rssi_dbm": -70.0, "frequency_mhz": 2412.0
            },
            # Valid second AP in same sweep
            {
                "observation_id": "WIFI-005", "session_id": "SES-1", "normalized_epoch_ms": 1020,
                "bssid": "00:11:22:33:44:66", "ssid": "Guest-Net", "rssi_dbm": -80.0, "frequency_mhz": 5180.0,
                "channel": 36, "channel_band": "5GHz"
            }
        ])

        proc = WifiPreprocessor(self.config)
        cleaned_df, features_df, stats, bssid_stats = proc.process(raw_wifi)

        # Missing BSSID removed
        self.assertEqual(stats.records_removed_missing_bssid, 1)
        # Invalid RSSI (5.0 dBm) removed
        self.assertEqual(stats.records_removed_invalid_rssi, 1)
        # Duplicate BSSID deduplicated (kept strongest -65 dBm)
        self.assertEqual(stats.records_removed_duplicates, 1)
        self.assertEqual(len(cleaned_df), 2)
        self.assertIn("00:11:22:33:44:55", cleaned_df["bssid"].values)
        kept_rssi = cleaned_df[cleaned_df["bssid"] == "00:11:22:33:44:55"]["rssi_dbm"].iloc[0]
        self.assertEqual(kept_rssi, -65.0)

        # Feature sweep validation
        self.assertEqual(len(features_df), 1)
        feat = features_df.iloc[0]
        self.assertEqual(feat["num_visible_aps"], 2)
        self.assertEqual(feat["strongest_bssid"], "00:11:22:33:44:55")
        self.assertEqual(feat["strongest_rssi"], -65.0)

    def test_sensor_cleaning_and_gap_detection(self):
        """Verifies Sensor preprocessor filters NaN/Inf, calculates magnitude, and identifies gaps without interpolation."""
        raw_sensor = pd.DataFrame([
            # Valid record 1
            {"observation_id": "S1", "sensor_type": "ACCELEROMETER", "normalized_epoch_ms": 1000, "x": 0.0, "y": 9.8, "z": 0.0},
            # Valid record 2 (50ms later)
            {"observation_id": "S2", "sensor_type": "ACCELEROMETER", "normalized_epoch_ms": 1050, "x": 0.1, "y": 9.7, "z": 0.1},
            # NaN value record
            {"observation_id": "S3", "sensor_type": "ACCELEROMETER", "normalized_epoch_ms": 1100, "x": np.nan, "y": 9.8, "z": 0.0},
            # Inf value record
            {"observation_id": "S4", "sensor_type": "ACCELEROMETER", "normalized_epoch_ms": 1150, "x": 0.0, "y": np.inf, "z": 0.0},
            # Duplicate timestamp (deduplicated)
            {"observation_id": "S5", "sensor_type": "ACCELEROMETER", "normalized_epoch_ms": 1050, "x": 0.2, "y": 9.6, "z": 0.2},
            # Large sampling gap (> 500 ms threshold): jump from 1050 to 2000 ms (950 ms gap)
            {"observation_id": "S6", "sensor_type": "ACCELEROMETER", "normalized_epoch_ms": 2000, "x": 0.0, "y": 9.81, "z": 0.0},
        ])

        proc = SensorPreprocessor(self.config)
        cleaned_df, features_df, stats = proc.process(raw_sensor)

        # Verify removal counts
        self.assertEqual(stats.records_removed_nan_inf, 2)
        self.assertEqual(stats.records_removed_duplicates, 1)
        self.assertEqual(len(cleaned_df), 3)

        # Verify gap detection without interpolation
        self.assertEqual(stats.total_gaps_detected, 1)
        gap_rows = cleaned_df[cleaned_df["is_gap"]]
        self.assertEqual(len(gap_rows), 1)
        self.assertEqual(gap_rows.iloc[0]["normalized_epoch_ms"], 2000)
        self.assertEqual(gap_rows.iloc[0]["delta_t_ms"], 950.0)

        # Magnitude verification: sqrt(0^2 + 9.8^2 + 0^2) == 9.8
        self.assertAlmostEqual(cleaned_df.iloc[0]["magnitude"], 9.8, places=2)

    def test_temporal_alignment_and_ambiguous_timing(self):
        """Verifies temporal window alignment and boundary uncertainty flagging."""
        wifi_df = pd.DataFrame([
            {"observation_id": "W-1", "normalized_epoch_ms": 1000, "uncertainty_ms": 50.0},
            {"observation_id": "W-2", "normalized_epoch_ms": 1600, "uncertainty_ms": 20.0},
        ])
        # Sensor record near boundary with large uncertainty: at t=1480ms with unc=50ms,
        # interval is [1430, 1530] ms which crosses the window boundary at 1500 ms
        sensor_df = pd.DataFrame([
            {"observation_id": "S-1", "normalized_epoch_ms": 1100, "uncertainty_ms": 10.0},
            {"observation_id": "S-2", "normalized_epoch_ms": 1480, "uncertainty_ms": 50.0},
            {"observation_id": "S-3", "normalized_epoch_ms": 1700, "uncertainty_ms": 10.0},
        ])

        self.config.temporal_window_size_ms = 500.0
        self.config.temporal_window_stride_ms = 500.0
        aligner = TimestampAligner(self.config)
        windows_df, stats = aligner.align(wifi_df=wifi_df, sensors_df=sensor_df)

        self.assertGreaterEqual(len(windows_df), 2)
        # Window 1: [1000, 1500)
        win1 = windows_df.iloc[0]
        self.assertEqual(win1["window_start_ms"], 1000)
        self.assertEqual(win1["window_end_ms"], 1500)
        self.assertTrue(win1["has_wifi"])
        self.assertTrue(win1["has_sensors"])
        self.assertTrue(win1["is_multimodal_concurrent"])
        self.assertTrue(win1["is_timing_ambiguous"])
        self.assertGreaterEqual(win1["ambiguous_obs_count"], 1)

    def test_photo_sanitization_gating(self):
        """Verifies unsanitized photos and non-existent image files are gated."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            images_dir = Path(tmp_dir) / "images"
            images_dir.mkdir()
            # Create real files
            (images_dir / "photo_valid.jpg").write_bytes(create_test_image_bytes())
            (images_dir / "photo_unsanitized.jpg").write_bytes(create_test_image_bytes())

            raw_photos = pd.DataFrame([
                # Valid sanitized photo with existing file
                {
                    "photo_id": "P1", "normalized_epoch_ms": 1000, "filename": "photo_valid.jpg",
                    "is_exif_sanitized": True, "landmark_type_code": "DOOR"
                },
                # Unsanitized photo (rejected)
                {
                    "photo_id": "P2", "normalized_epoch_ms": 1200, "filename": "photo_unsanitized.jpg",
                    "is_exif_sanitized": False, "landmark_type_code": "DOOR"
                },
                # Sanitized but missing on disk (rejected)
                {
                    "photo_id": "P3", "normalized_epoch_ms": 1400, "filename": "photo_missing.jpg",
                    "is_exif_sanitized": True, "landmark_type_code": "DOOR"
                }
            ])

            proc = PhotoMetadataProcessor(self.config, images_dir=images_dir)
            approved_df, excluded_df, stats = proc.process(raw_photos)

            self.assertEqual(len(approved_df), 1)
            self.assertEqual(approved_df.iloc[0]["photo_id"], "P1")
            self.assertEqual(approved_df.iloc[0]["visual_audit_status"], "PENDING_AUDIT")
            self.assertEqual(stats.records_excluded_unsanitized, 1)
            self.assertEqual(stats.records_excluded_missing_file, 1)

    def test_annotation_canonicalization(self):
        """Verifies manual annotations are canonicalized and surveyor provenance is assigned."""
        raw_ann = pd.DataFrame([
            {"annotation_id": "A1", "normalized_epoch_ms": 1000, "type_name": "door entrance", "description": "Front Door"},
            {"annotation_id": "A2", "normalized_epoch_ms": 1500, "type_code": "STAIRCASE", "description": "Floor 2 Stairs"},
            {"annotation_id": "A3", "normalized_epoch_ms": 2000, "type_code": "random_note", "description": "Water fountain"},
        ])

        proc = AnnotationProcessor(self.config)
        cleaned_df, stats = proc.process(raw_ann)

        self.assertEqual(len(cleaned_df), 3)
        self.assertEqual(cleaned_df.iloc[0]["type_code"], "DOOR")
        self.assertEqual(cleaned_df.iloc[1]["type_code"], "STAIRCASE")
        self.assertEqual(cleaned_df.iloc[2]["type_code"], "CUSTOM")
        self.assertTrue((cleaned_df["label_provenance"] == "SURVEYOR_MANUAL").all())

    def test_session_segmentation_strategies(self):
        """Verifies session segmentation under fixed duration and annotation boundaries."""
        sensor_df = pd.DataFrame([
            {"observation_id": f"S-{i}", "normalized_epoch_ms": 1000 + i * 10000} for i in range(10)
        ])  # 10 samples from 1000 to 91000 ms (90 seconds)
        ann_df = pd.DataFrame([
            {"observation_id": "A-1", "normalized_epoch_ms": 35000, "type_code": "DOOR"}
        ])

        # 1. Fixed duration (30 seconds per segment)
        self.config.segmentation_strategy = "fixed_duration"
        self.config.segmentation_duration_seconds = 30.0
        segmenter = SessionSegmenter(self.config)
        segments = segmenter.segment("SES-TEST", sensors_df=sensor_df, annotations_df=ann_df)

        self.assertGreaterEqual(len(segments), 3)
        self.assertEqual(segments[0].segment_index, 1)
        self.assertEqual(segments[0].duration_ms, 30000)

        # 2. Annotation boundaries
        self.config.segmentation_strategy = "annotation_boundaries"
        segmenter_ann = SessionSegmenter(self.config)
        ann_segments = segmenter_ann.segment("SES-TEST", sensors_df=sensor_df, annotations_df=ann_df)
        self.assertEqual(len(ann_segments), 2)
        self.assertEqual(ann_segments[0].end_timestamp_ms, 35000)

    def test_gap_based_segmentation(self):
        """Verifies session segmentation splits when temporal gaps exceed threshold."""
        sensor_df = pd.DataFrame([
            {"observation_id": "S1", "normalized_epoch_ms": 1000},
            {"observation_id": "S2", "normalized_epoch_ms": 2000},
            # Gap of 10 seconds (exceeds 5.0 s threshold)
            {"observation_id": "S3", "normalized_epoch_ms": 12000},
            {"observation_id": "S4", "normalized_epoch_ms": 13000},
        ])
        self.config.segmentation_strategy = "temporal_gaps"
        self.config.segmentation_gap_threshold_seconds = 5.0
        segmenter = SessionSegmenter(self.config)
        segments = segmenter.segment("SES-GAP", sensors_df=sensor_df)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].observation_counts["sensors"], 2)
        self.assertEqual(segments[1].observation_counts["sensors"], 2)

    def test_wifi_frequency_bands_and_stale_filter(self):
        """Verifies Wi-Fi preprocessor rejects invalid frequencies and stale scans when configured."""
        raw_wifi = pd.DataFrame([
            # 2.4 GHz valid fresh
            {"observation_id": "W1", "normalized_epoch_ms": 1000, "bssid": "00:11:22:33:44:01", "rssi_dbm": -50.0, "frequency_mhz": 2412.0, "is_fresh": True},
            # Stale scan (should be filtered out)
            {"observation_id": "W2", "normalized_epoch_ms": 1000, "bssid": "00:11:22:33:44:02", "rssi_dbm": -55.0, "frequency_mhz": 2412.0, "is_fresh": False},
            # Invalid frequency (e.g. 900 MHz cellular / out of band)
            {"observation_id": "W3", "normalized_epoch_ms": 1000, "bssid": "00:11:22:33:44:03", "rssi_dbm": -60.0, "frequency_mhz": 900.0, "is_fresh": True},
        ])
        self.config.require_fresh_scans_only = True
        proc = WifiPreprocessor(self.config)
        cleaned_df, _, stats, _ = proc.process(raw_wifi)
        self.assertEqual(len(cleaned_df), 1)
        self.assertEqual(cleaned_df.iloc[0]["bssid"], "00:11:22:33:44:01")
        self.assertEqual(stats.records_removed_stale, 1)
        self.assertEqual(stats.records_removed_invalid_freq, 1)

    def test_sensor_kinematics_and_rolling_stats(self):
        """Verifies vector magnitude, jerk derivative, and rolling window calculation."""
        raw_sensor = pd.DataFrame([
            {"observation_id": f"S{i}", "sensor_type": "ACCELEROMETER", "normalized_epoch_ms": 1000 + i * 100, "x": float(i), "y": 0.0, "z": 0.0}
            for i in range(10)
        ])
        proc = SensorPreprocessor(self.config)
        cleaned_df, features_df, stats = proc.process(raw_sensor)

        self.assertEqual(len(cleaned_df), 10)
        self.assertIn("magnitude_rolling_mean", cleaned_df.columns)
        self.assertIn("jerk_x", cleaned_df.columns)
        # For x = [0, 1, 2, ...], jerk_x should be delta_x / delta_t = 1 / 100 = 0.01
        self.assertAlmostEqual(cleaned_df["jerk_x"].iloc[2], 0.01, places=3)


class TestMultimodalPreprocessingPipelineAndAPI(unittest.TestCase):
    """Integration tests for the complete pipeline, reproducibility, and FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="test_pipeline_")
        cls.config = DatasetConfig(base_dir=Path(cls.temp_dir) / "survey_datasets")
        cls.config.ensure_directories()
        cls.extractor = DatasetExtractor(cls.config)
        cls.validator = DatasetValidator(cls.config)
        cls.registry = DatasetRegistry(cls.config)
        cls.ingestion_service = DatasetIngestionService(
            config=cls.config,
            extractor=cls.extractor,
            validator=cls.validator,
            registry=cls.registry
        )

        cls.app = create_app()
        from backend.app.api.routes.datasets import get_ingestion_service, get_preprocessing_service
        cls.app.dependency_overrides[get_ingestion_service] = lambda: cls.ingestion_service
        cls.prep_service = PreprocessingService(config=cls.config, registry=cls.registry)
        cls.app.dependency_overrides[get_preprocessing_service] = lambda: cls.prep_service
        cls.test_client = TestClient(cls.app)

        # Ingest a valid survey ZIP
        zip_bytes = build_synthetic_survey_zip(
            session_id="SES-PIPE-TEST",
            session_name="Pipeline Survey Test",
            wifi_sweeps=3,
            imu_records=30,
            photo_records=2,
            annotation_records=2
        )
        cls.dataset_id, cls.report, cls.entry = cls.ingestion_service.ingest_dataset_zip(
            io.BytesIO(zip_bytes),
            original_filename="pipeline_survey.zip"
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_end_to_end_pipeline_execution(self):
        """Verifies full pipeline execution produces all required Parquet files, report, and manifest."""
        extracted_dir = self.config.extracted_path / self.dataset_id
        pipeline_cfg = MultimodalPreprocessingConfig(
            base_output_dir=self.config.base_dir.parent / "processed" / "multimodal"
        )
        pipeline = MultimodalPreprocessingPipeline(pipeline_cfg)
        report, manifest = pipeline.run(extracted_dir, self.dataset_id)

        self.assertIn(report.status, ["COMPLETED", "WARNINGS"])
        self.assertEqual(report.session_id, "SES-PIPE-TEST")

        # Verify Parquet outputs exist
        out_dir = pipeline_cfg.base_output_dir / self.dataset_id
        self.assertTrue((out_dir / "wifi" / "wifi_cleaned.parquet").is_file())
        self.assertTrue((out_dir / "wifi" / "wifi_features.parquet").is_file())
        self.assertTrue((out_dir / "sensors" / "sensor_cleaned.parquet").is_file())
        self.assertTrue((out_dir / "sensors" / "sensor_features.parquet").is_file())
        self.assertTrue((out_dir / "photos" / "photo_metadata.parquet").is_file())
        self.assertTrue((out_dir / "annotations" / "annotations_cleaned.parquet").is_file())
        self.assertTrue((out_dir / "aligned" / "temporal_windows.parquet").is_file())
        self.assertTrue((out_dir / "segments" / "segments.json").is_file())
        self.assertTrue((out_dir / "preprocessing_report.json").is_file())
        self.assertTrue((out_dir / "preprocessing_manifest.json").is_file())

        # Check Parquet reading
        wifi_cleaned = pd.read_parquet(out_dir / "wifi" / "wifi_cleaned.parquet")
        self.assertGreater(len(wifi_cleaned), 0)
        self.assertIn("bssid", wifi_cleaned.columns)

        windows = pd.read_parquet(out_dir / "aligned" / "temporal_windows.parquet")
        self.assertGreater(len(windows), 0)
        self.assertIn("window_id", windows.columns)

    def test_reproducibility(self):
        """Verifies running the pipeline twice with identical inputs and config produces exact identical checksums."""
        extracted_dir = self.config.extracted_path / self.dataset_id
        pipeline_cfg = MultimodalPreprocessingConfig(
            base_output_dir=self.config.base_dir.parent / "processed" / "multimodal"
        )
        pipeline = MultimodalPreprocessingPipeline(pipeline_cfg)

        _, manifest1 = pipeline.run(extracted_dir, self.dataset_id)
        _, manifest2 = pipeline.run(extracted_dir, self.dataset_id)

        # Compare output checksums
        self.assertEqual(
            manifest1.output_file_checksums["wifi/wifi_cleaned.parquet"],
            manifest2.output_file_checksums["wifi/wifi_cleaned.parquet"]
        )
        self.assertEqual(
            manifest1.output_file_checksums["aligned/temporal_windows.parquet"],
            manifest2.output_file_checksums["aligned/temporal_windows.parquet"]
        )

    def test_single_modality_dataset(self):
        """Verifies pipeline succeeds on a valid single-modality dataset without generating fake data."""
        zip_bytes = build_synthetic_survey_zip(
            session_id="SES-WIFI-ONLY",
            session_name="Wi-Fi Only Survey",
            wifi_sweeps=4,
            imu_records=0,
            photo_records=0,
            annotation_records=0,
            empty_modalities=False
        )
        ds_id, _, _ = self.ingestion_service.ingest_dataset_zip(
            io.BytesIO(zip_bytes),
            original_filename="wifi_only.zip"
        )
        extracted_dir = self.config.extracted_path / ds_id
        pipeline_cfg = MultimodalPreprocessingConfig(
            base_output_dir=self.config.base_dir.parent / "processed" / "multimodal"
        )
        pipeline = MultimodalPreprocessingPipeline(pipeline_cfg)
        report, manifest = pipeline.run(extracted_dir, ds_id)

        self.assertIn(report.status, ["COMPLETED", "WARNINGS"])
        # Sensors should have 0 input and 0 output
        self.assertEqual(report.input_summary.get("sensors", 0), 0)

    def test_fastapi_preprocessing_endpoints(self):
        """Verifies POST /preprocess, GET /preprocessing, and GET /preprocessing/files."""
        # 1. Trigger preprocessing via API
        req_payload = {
            "min_rssi_dbm": -100.0,
            "max_rssi_dbm": -25.0,
            "temporal_window_size_ms": 250.0,
            "segmentation_duration_seconds": 45.0
        }
        res_post = self.test_client.post(
            f"/api/datasets/{self.dataset_id}/preprocess",
            json=req_payload
        )
        self.assertEqual(res_post.status_code, 200)
        data_post = res_post.json()
        self.assertEqual(data_post["dataset_id"], self.dataset_id)
        self.assertIn("report", data_post)

        # 2. Query status
        res_status = self.test_client.get(f"/api/datasets/{self.dataset_id}/preprocessing")
        self.assertEqual(res_status.status_code, 200)
        data_status = res_status.json()
        self.assertTrue(data_status["is_preprocessed"])
        self.assertIsNotNone(data_status["report"])
        self.assertIsNotNone(data_status["manifest"])

        # 3. Query files catalog
        res_files = self.test_client.get(f"/api/datasets/{self.dataset_id}/preprocessing/files")
        self.assertEqual(res_files.status_code, 200)
        data_files = res_files.json()
        files = data_files["files"]
        self.assertGreaterEqual(len(files), 5)
        file_names = [f["filename"] for f in files]
        self.assertIn("wifi_cleaned.parquet", file_names)
        self.assertIn("temporal_windows.parquet", file_names)
        self.assertIn("segments.json", file_names)

    def test_corrupted_dataset_fails_gracefully(self):
        """Verifies pipeline records FAILED status cleanly when metadata files are missing."""
        with tempfile.TemporaryDirectory() as empty_dir:
            pipeline_cfg = MultimodalPreprocessingConfig(
                base_output_dir=self.config.base_dir.parent / "processed" / "multimodal"
            )
            pipeline = MultimodalPreprocessingPipeline(pipeline_cfg)
            report, manifest = pipeline.run(Path(empty_dir), "nonexistent-id")
            self.assertEqual(report.status, "FAILED")
            self.assertGreater(len(report.errors), 0)


if __name__ == "__main__":
    unittest.main()
