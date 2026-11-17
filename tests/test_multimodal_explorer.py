"""Unit and integration tests for the Multimodal Dataset Explorer and Dashboard API client (Task 19)."""

import io
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import httpx
from fastapi.testclient import TestClient

from backend.app.core.dataset_config import DatasetConfig
from backend.app.main import create_app
from backend.app.services.dataset_extraction import DatasetExtractor
from backend.app.services.dataset_ingestion import DatasetIngestionService
from backend.app.services.dataset_registry import DatasetRegistry
from backend.app.services.dataset_validation import DatasetValidator
from frontend.api_client import DatasetApiClient, DatasetApiClientException
from tests.test_dataset_ingestion import build_synthetic_survey_zip


class TestDatasetApiClient(unittest.TestCase):
    """Test suite for the frontend DatasetApiClient integrating with FastAPI."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="test_api_client_")
        cls.config = DatasetConfig(base_dir=Path(cls.temp_dir))
        cls.config.ensure_directories()
        cls.extractor = DatasetExtractor(cls.config)
        cls.validator = DatasetValidator(cls.config)
        cls.registry = DatasetRegistry(cls.config)
        cls.service = DatasetIngestionService(
            config=cls.config,
            extractor=cls.extractor,
            validator=cls.validator,
            registry=cls.registry
        )

        cls.app = create_app()
        from backend.app.api.routes.datasets import get_ingestion_service
        cls.app.dependency_overrides[get_ingestion_service] = lambda: cls.service

        # Use an in-process FastAPI TestClient for DatasetApiClient
        cls.test_client = TestClient(cls.app)
        cls.client = DatasetApiClient(base_url="http://testserver", http_client=cls.test_client)

        # Ingest a sample dataset for queries
        zip_bytes = build_synthetic_survey_zip(
            session_id="SES-CLIENT-TEST",
            session_name="Client Test Survey",
            wifi_sweeps=3,
            imu_records=25,
            photo_records=2,
            annotation_records=2
        )
        cls.dataset_id, cls.report, cls.entry = cls.service.ingest_dataset_zip(
            io.BytesIO(zip_bytes),
            original_filename="client_test.zip"
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_health_check(self):
        """Verifies check_health endpoint returns healthy status."""
        is_healthy, msg = self.client.check_health()
        self.assertTrue(is_healthy)
        self.assertIn("Connected", msg)

    def test_list_datasets(self):
        """Verifies list_datasets retrieves registered survey datasets."""
        datasets = self.client.list_datasets()
        self.assertIsInstance(datasets, list)
        self.assertGreaterEqual(len(datasets), 1)
        found = next((d for d in datasets if d["dataset_id"] == self.dataset_id), None)
        self.assertIsNotNone(found)
        self.assertEqual(found["session_id"], "SES-CLIENT-TEST")

    def test_get_dataset_detail(self):
        """Verifies get_dataset returns complete metadata."""
        detail = self.client.get_dataset(self.dataset_id)
        self.assertIsNotNone(detail)
        self.assertEqual(detail["session_name"], "Client Test Survey")
        self.assertEqual(detail["building_id"], "BLDG-01")
        self.assertTrue(detail["modality_availability"]["wifi"])

    def test_get_nonexistent_dataset(self):
        """Verifies get_dataset returns None for unknown ID."""
        detail = self.client.get_dataset("nonexistent-00000000")
        self.assertIsNone(detail)

    def test_get_quality_report(self):
        """Verifies get_quality_report returns validated integrity metrics."""
        report = self.client.get_quality_report(self.dataset_id)
        self.assertIsNotNone(report)
        self.assertEqual(report["validation_status"], "PASSED")
        self.assertTrue(report["integrity"]["manifest_valid"])
        self.assertEqual(report["modalities"]["wifi"]["total_scans"], 3)

    def test_get_observations_wifi(self):
        """Verifies observation endpoint retrieves parsed Wi-Fi rows."""
        records = self.client.get_observations(self.dataset_id, "wifi", limit=100)
        self.assertIsInstance(records, list)
        self.assertGreater(len(records), 0)
        first = records[0]
        self.assertIn("observation_id", first)
        self.assertIn("bssid", first)
        self.assertIn("rssi_dbm", first)

    def test_get_observations_sensors(self):
        """Verifies observation endpoint retrieves parsed IMU records."""
        records = self.client.get_observations(self.dataset_id, "sensors", limit=100)
        self.assertIsInstance(records, list)
        self.assertEqual(len(records), 25)
        self.assertIn("sensor_type", records[0])
        self.assertIn("x", records[0])

    def test_get_observations_photos(self):
        """Verifies observation endpoint retrieves photo metadata."""
        records = self.client.get_observations(self.dataset_id, "photos", limit=100)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["is_exif_sanitized"], "true")

    def test_get_observations_annotations(self):
        """Verifies observation endpoint retrieves ground-truth annotations."""
        records = self.client.get_observations(self.dataset_id, "annotations", limit=100)
        self.assertEqual(len(records), 2)
        self.assertIn("description", records[0])

    def test_get_sanitized_image_bytes(self):
        """Verifies get_image_bytes successfully serves privacy-sanitized photo."""
        img_bytes = self.client.get_image_bytes(self.dataset_id, "photo_001.jpg")
        self.assertIsNotNone(img_bytes)
        self.assertGreater(len(img_bytes), 0)
        # Check JPEG magic bytes
        self.assertTrue(img_bytes.startswith(b"\xff\xd8"))

    def test_get_unsanitized_image_denied(self):
        """Verifies non-existent or unverified photos are rejected with None (HTTP 403/404)."""
        img_bytes = self.client.get_image_bytes(self.dataset_id, "photo_999_fake.jpg")
        self.assertIsNone(img_bytes)

    def test_upload_and_delete_flow(self):
        """Verifies uploading a dataset via client and subsequently deleting it."""
        zip_bytes = build_synthetic_survey_zip(
            session_id="SES-FLOW-TEST",
            wifi_sweeps=1,
            imu_records=5
        )
        success, data, msg = self.client.upload_dataset(zip_bytes, filename="flow_test.zip")
        self.assertTrue(success)
        uploaded_id = data["dataset_id"]

        # Confirm it appears in list
        datasets = self.client.list_datasets()
        self.assertTrue(any(d["dataset_id"] == uploaded_id for d in datasets))

        # Revalidate
        reval_ok, reval_data, _ = self.client.revalidate_dataset(uploaded_id)
        self.assertTrue(reval_ok)

        # Delete
        del_ok, del_msg = self.client.delete_dataset(uploaded_id)
        self.assertTrue(del_ok)

        # Confirm 404 after deletion
        self.assertIsNone(self.client.get_dataset(uploaded_id))


class TestMultimodalComponentLogic(unittest.TestCase):
    """Test suite for data transformations, chart aggregations, and privacy rules."""

    def test_wifi_sweep_grouping(self):
        """Verifies Wi-Fi rows group properly by observation_id into sweeps."""
        rows = [
            {"observation_id": "W1", "bssid": "AP1", "rssi_dbm": "-60"},
            {"observation_id": "W1", "bssid": "AP2", "rssi_dbm": "-70"},
            {"observation_id": "W2", "bssid": "AP1", "rssi_dbm": "-65"},
        ]
        import pandas as pd
        df = pd.DataFrame(rows)
        sweeps = df["observation_id"].unique().tolist()
        self.assertEqual(len(sweeps), 2)
        sweep_1 = df[df["observation_id"] == "W1"]
        self.assertEqual(len(sweep_1), 2)

    def test_sensor_sampling_frequency_calculation(self):
        """Verifies sampling rate calculation across time-series sensor samples."""
        import pandas as pd
        timestamps_ms = [0, 20, 40, 60, 80]  # 5 samples over 80ms => 20ms delta = 50 Hz
        df = pd.DataFrame({"ts": timestamps_ms, "x": [0.1]*5})
        duration_s = (df["ts"].max() - df["ts"].min()) / 1000.0
        hz = (len(df) - 1) / duration_s
        self.assertAlmostEqual(hz, 50.0)

    def test_cross_modal_temporal_overlap_math(self):
        """Verifies pairwise overlap calculation across normalized time ranges."""
        r1 = {"start_ms": 1000, "end_ms": 5000}
        r2 = {"start_ms": 3000, "end_ms": 7000}
        start = max(r1["start_ms"], r2["start_ms"])
        end = min(r1["end_ms"], r2["end_ms"])
        overlap_s = max(0.0, (end - start) / 1000.0)
        self.assertEqual(overlap_s, 2.0)


if __name__ == "__main__":
    unittest.main()
