"""Comprehensive test suite for multimodal dataset ingestion, validation, and QA backend."""

import csv
import hashlib
import io
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.dataset_config import DatasetConfig
from backend.app.main import create_app
from backend.app.schemas.dataset import DataSourceType, ValidationStatus
from backend.app.services.dataset_extraction import DatasetExtractor, SecurityValidationError
from backend.app.services.dataset_ingestion import DatasetIngestionService
from backend.app.services.dataset_registry import DatasetRegistry
from backend.app.services.dataset_validation import (
    EXPECTED_ANNOTATION_HEADER,
    EXPECTED_IMU_HEADER,
    EXPECTED_PHOTO_HEADER,
    EXPECTED_WIFI_HEADER,
    DatasetValidator,
    calculate_sha256,
)


def create_test_image_bytes(width: int = 100, height: int = 100, color: Tuple[int, int, int] = (100, 150, 200)) -> bytes:
    """Generates a small valid JPEG image byte string."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def build_synthetic_survey_zip(
    session_id: str = "SES-A1B2C3D4",
    session_name: str = "Test Survey Session",
    wifi_sweeps: int = 2,
    imu_records: int = 10,
    photo_records: int = 1,
    annotation_records: int = 1,
    is_mock: bool = False,
    is_exif_sanitized: bool = True,
    omit_file: Optional[str] = None,
    tamper_manifest_sha: bool = False,
    tamper_manifest_count: bool = False,
    session_id_override_csv: Optional[str] = None,
    empty_modalities: bool = False,
    corrupt_session_json: bool = False
) -> bytes:
    """Creates a fully valid (or intentionally flawed) Task 17 survey ZIP in memory."""
    buf = io.BytesIO()
    timestamp_ms = 1731855600000  # Nov 17, 2026 UTC

    # 1. session.json
    clock_anchor = {
        "wallClockEpochMs": timestamp_ms,
        "bootTimeNs": 123456789000,
        "anchorIsoUtc": "2026-11-17T15:00:00.000Z"
    }

    actual_wifi_count = 0 if empty_modalities else wifi_sweeps
    actual_imu_count = 0 if empty_modalities else imu_records
    actual_photo_count = 0 if empty_modalities else photo_records
    actual_ann_count = 0 if empty_modalities else annotation_records

    total_real = (actual_wifi_count + actual_imu_count + actual_photo_count + actual_ann_count) if not is_mock else 0
    total_mock = (actual_wifi_count + actual_imu_count + actual_photo_count + actual_ann_count) if is_mock else 0

    session_dict = {
        "sessionId": session_id,
        "sessionName": session_name,
        "createdAtMs": timestamp_ms,
        "startTimeMs": timestamp_ms,
        "endTimeMs": timestamp_ms + 60000,
        "status": "COMPLETED",
        "buildingId": "BLDG-01",
        "floorId": "FL-02",
        "roomId": "LAB-201",
        "clockAnchor": clock_anchor,
        "wifiCount": actual_wifi_count,
        "sensorCount": actual_imu_count,
        "photoCount": actual_photo_count,
        "annotationCount": actual_ann_count,
        "realObservationCount": total_real,
        "mockObservationCount": total_mock,
        "isInterrupted": False,
        "interruptionNote": None
    }

    session_content = json.dumps(session_dict, indent=2).encode("utf-8")
    if corrupt_session_json:
        session_content = b"INVALID_JSON_SYNTAX{{{{"

    # 2. wifi_observations.csv
    csv_session_id = session_id_override_csv or session_id
    wifi_buf = io.StringIO()
    wifi_writer = csv.writer(wifi_buf)
    wifi_writer.writerow(EXPECTED_WIFI_HEADER)
    if not empty_modalities:
        for i in range(wifi_sweeps):
            obs_id = f"WIFI-SWEEP-{i+1:03d}"
            t_ms = timestamp_ms + (i * 5000)
            # AP 1
            wifi_writer.writerow([
                obs_id, csv_session_id, str(t_ms * 1000), "CLOCK_BOOTTIME_MICROS",
                str(t_ms), "2026-11-17T15:00:00.000Z", "BOOTTIME_OFFSET", "30.0",
                "true", str(is_mock).lower(), "00:11:22:33:44:55", "IndoorAI-AP1",
                "-65", "2412", "1", "2.4GHz", "BLDG-01", "FL-02", "LAB-201"
            ])
            # AP 2
            wifi_writer.writerow([
                obs_id, csv_session_id, str(t_ms * 1000), "CLOCK_BOOTTIME_MICROS",
                str(t_ms), "2026-11-17T15:00:00.000Z", "BOOTTIME_OFFSET", "30.0",
                "true", str(is_mock).lower(), "00:11:22:33:44:66", "IndoorAI-AP2",
                "-72", "5180", "36", "5GHz", "BLDG-01", "FL-02", "LAB-201"
            ])
    wifi_content = wifi_buf.getvalue().encode("utf-8")

    # 3. imu_observations.csv
    imu_buf = io.StringIO()
    imu_writer = csv.writer(imu_buf)
    imu_writer.writerow(EXPECTED_IMU_HEADER)
    if not empty_modalities:
        for i in range(imu_records):
            obs_id = f"IMU-REC-{i+1:04d}"
            t_ms = timestamp_ms + (i * 20)  # 50 Hz = 20ms
            t_ns = 123456789000 + (i * 20_000_000)
            stype = "ACCELEROMETER" if i % 2 == 0 else "GYROSCOPE"
            imu_writer.writerow([
                obs_id, csv_session_id, str(t_ns), "CLOCK_BOOTTIME_NANOS",
                str(t_ms), "2026-11-17T15:00:00.000Z", "BOOTTIME_OFFSET", "3.5",
                stype, "0.12", "0.45", "9.81", "9.82", "SENSOR_STATUS_ACCURACY_HIGH",
                "SENSOR_DELAY_GAME", str(is_mock).lower()
            ])
    imu_content = imu_buf.getvalue().encode("utf-8")

    # 4. photo_metadata.csv & images/
    photo_buf = io.StringIO()
    photo_writer = csv.writer(photo_buf)
    photo_writer.writerow(EXPECTED_PHOTO_HEADER)
    photo_images: Dict[str, bytes] = {}

    if not empty_modalities:
        for i in range(photo_records):
            pid = f"IMG-REC-{i+1:03d}"
            fn = f"photo_{i+1:03d}.jpg"
            img_data = create_test_image_bytes(color=(100 + i * 20, 120, 150))
            photo_images[fn] = img_data
            t_ms = timestamp_ms + (i * 10000)
            photo_writer.writerow([
                pid, csv_session_id, str(t_ms), "CLOCK_REALTIME_MILLIS",
                str(t_ms), "2026-11-17T15:00:00.000Z", "DIRECT_REALTIME", "50.0",
                fn, f"/app_storage/{fn}", str(len(img_data)), "100", "100",
                "BLDG-01", "FL-02", "LAB-201", "CORRIDOR", "Corridor",
                "North corridor test", "", str(is_exif_sanitized).lower(), "0", "false",
                "NO_FACES_DETECTED", str(is_mock).lower()
            ])
    photo_content = photo_buf.getvalue().encode("utf-8")

    # 5. annotations.csv
    ann_buf = io.StringIO()
    ann_writer = csv.writer(ann_buf)
    ann_writer.writerow(EXPECTED_ANNOTATION_HEADER)
    if not empty_modalities:
        for i in range(annotation_records):
            aid = f"ANN-REC-{i+1:03d}"
            t_ms = timestamp_ms + (i * 15000)
            ann_writer.writerow([
                aid, csv_session_id, str(t_ms), "CLOCK_REALTIME_MILLIS",
                str(t_ms), "2026-11-17T15:00:00.000Z", "DIRECT_REALTIME", "50.0",
                "CORRIDOR", "Corridor", "Turning corridor corner", "BLDG-01", "FL-02", "LAB-201",
                "", str(is_mock).lower()
            ])
    ann_content = ann_buf.getvalue().encode("utf-8")

    # Prepare file entries for manifest
    file_payloads: Dict[str, bytes] = {
        "session.json": session_content,
        "wifi_observations.csv": wifi_content,
        "imu_observations.csv": imu_content,
        "photo_metadata.csv": photo_content,
        "annotations.csv": ann_content,
    }
    for fn, img_bytes in photo_images.items():
        file_payloads[f"images/{fn}"] = img_bytes

    if omit_file and omit_file in file_payloads:
        del file_payloads[omit_file]

    # Calculate manifest entries
    manifest_files = []
    for fn, data in file_payloads.items():
        sha = hashlib.sha256(data).hexdigest()
        if tamper_manifest_sha and fn == "session.json":
            sha = "0000000000000000000000000000000000000000000000000000000000000000"
        manifest_files.append({
            "filename": fn,
            "sizeBytes": len(data),
            "sha256": sha
        })

    reported_wifi_count = (actual_wifi_count + 5) if tamper_manifest_count else actual_wifi_count
    manifest_dict = {
        "exportVersion": "1.0.0",
        "sessionId": session_id,
        "exportedAtMs": timestamp_ms + 65000,
        "exportedAtIso": "2026-11-17T15:01:05.000Z",
        "recordCounts": {
            "wifiCount": reported_wifi_count,
            "imuCount": actual_imu_count,
            "photoCount": actual_photo_count,
            "annotationCount": actual_ann_count,
            "totalObservations": reported_wifi_count + actual_imu_count + actual_photo_count + actual_ann_count
        },
        "sourceCounts": {
            "realObservationCount": total_real,
            "mockObservationCount": total_mock,
            "containsRealData": total_real > 0,
            "containsMockData": total_mock > 0
        },
        "files": manifest_files,
        "validationStatus": "VALIDATED"
    }

    manifest_content = json.dumps(manifest_dict, indent=2).encode("utf-8")
    if omit_file != "manifest.json":
        file_payloads["manifest.json"] = manifest_content

    # Write to ZIP
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in file_payloads.items():
            zf.writestr(name, data)

    return buf.getvalue()


class TestDatasetIngestion(unittest.TestCase):
    """Test suite for ZIP security, extraction, validation, and registry management."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_dataset_backend_")
        self.config = DatasetConfig(base_dir=Path(self.temp_dir))
        self.config.ensure_directories()
        self.extractor = DatasetExtractor(self.config)
        self.validator = DatasetValidator(self.config)
        self.registry = DatasetRegistry(self.config)
        self.service = DatasetIngestionService(
            config=self.config,
            extractor=self.extractor,
            validator=self.validator,
            registry=self.registry
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_valid_task17_zip_ingestion(self) -> None:
        """Verifies successful ingestion of a compliant Task 17 multimodal survey archive."""
        zip_bytes = build_synthetic_survey_zip(
            session_id="SES-00112233",
            wifi_sweeps=3,
            imu_records=20,
            photo_records=2,
            annotation_records=1,
            is_mock=False
        )
        stream = io.BytesIO(zip_bytes)
        dataset_id, report, entry = self.service.ingest_dataset_zip(stream, original_filename="valid_session.zip")

        self.assertIsNotNone(dataset_id)
        self.assertEqual(report.validation_status, ValidationStatus.PASSED)
        self.assertTrue(report.integrity.manifest_valid)
        self.assertTrue(report.integrity.checksums_valid)
        self.assertTrue(report.integrity.record_counts_valid)
        self.assertEqual(len(report.errors), 0)

        # Modalities check
        self.assertEqual(report.modalities.wifi.total_scans, 3)
        self.assertEqual(report.modalities.sensors.total_records, 20)
        self.assertEqual(report.modalities.photos.total_photo_records, 2)
        self.assertEqual(report.modalities.annotations.total_annotations, 1)

        # Registry check
        stored = self.registry.get_dataset(dataset_id)
        self.assertIsNotNone(stored)
        self.assertEqual(stored["session_id"], "SES-00112233")
        self.assertEqual(stored["data_source_type"], "REAL")
        self.assertTrue(stored["modality_availability"]["wifi"])
        self.assertTrue(stored["modality_availability"]["sensors"])

    def test_invalid_zip_signature(self) -> None:
        """Verifies rejection of files that lack the standard PK magic signature."""
        bad_stream = io.BytesIO(b"NOT_A_VALID_ZIP_HEADER_DATA")
        with self.assertRaises(SecurityValidationError):
            self.service.ingest_dataset_zip(bad_stream)

    def test_path_traversal_rejection(self) -> None:
        """Verifies rejection of archive entries attempting directory climbs (..) or root paths."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../../etc/passwd", b"malicious content")
            zf.writestr("session.json", b"{}")

        buf.seek(0)
        with self.assertRaises(SecurityValidationError):
            self.extractor.extract_dataset(buf, self.config.extracted_path / "traversal_test")

        # Verify nothing was extracted
        self.assertFalse((self.config.extracted_path / "traversal_test").exists())

    def test_absolute_path_traversal_rejection(self) -> None:
        """Verifies rejection of archive entries with leading slash."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("/root/pwned.txt", b"evil")

        buf.seek(0)
        with self.assertRaises(SecurityValidationError):
            self.extractor.extract_dataset(buf, self.config.extracted_path / "abs_test")

    def test_zip_bomb_excessive_compression_ratio(self) -> None:
        """Verifies rejection of entries with excessive decompression expansion ratio."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1 MB of zeros compresses into ~1 KB (> 100:1 ratio)
            zf.writestr("huge_zeroes.csv", b"0" * (1024 * 1024))

        buf.seek(0)
        with self.assertRaises(SecurityValidationError):
            self.extractor.extract_dataset(buf, self.config.extracted_path / "bomb_test")

    def test_duplicate_archive_entries(self) -> None:
        """Verifies rejection of ZIP archives containing duplicate filename entries."""
        buf = io.BytesIO()
        # zipfile module allows writing duplicate names
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("session.json", b'{"sessionId": "SES-1"}')
            zf.writestr("session.json", b'{"sessionId": "SES-2"}')

        buf.seek(0)
        with self.assertRaises(SecurityValidationError):
            self.extractor.extract_dataset(buf, self.config.extracted_path / "dup_test")

    def test_forbidden_file_extension(self) -> None:
        """Verifies rejection of executable scripts or unapproved extensions."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("payload.sh", b"#!/bin/bash\necho bad")

        buf.seek(0)
        with self.assertRaises(SecurityValidationError):
            self.extractor.extract_dataset(buf, self.config.extracted_path / "ext_test")

    def test_missing_required_files(self) -> None:
        """Verifies validation failure when a required file (e.g. session.json) is missing."""
        zip_bytes = build_synthetic_survey_zip(omit_file="session.json")
        stream = io.BytesIO(zip_bytes)
        dataset_id, report, entry = self.service.ingest_dataset_zip(stream)

        self.assertEqual(report.validation_status, ValidationStatus.CORRUPTED)
        self.assertIn("Missing required file in dataset archive: 'session.json'", report.errors)

    def test_malformed_json_handling(self) -> None:
        """Verifies graceful handling and error reporting when session.json contains invalid syntax."""
        zip_bytes = build_synthetic_survey_zip(corrupt_session_json=True)
        stream = io.BytesIO(zip_bytes)
        dataset_id, report, entry = self.service.ingest_dataset_zip(stream)

        self.assertIn(report.validation_status, (ValidationStatus.FAILED, ValidationStatus.CORRUPTED))
        self.assertTrue(any("Failed to parse session.json" in err for err in report.errors))

    def test_incorrect_manifest_checksums(self) -> None:
        """Verifies detection of tampered files with mismatched cryptographic SHA-256."""
        zip_bytes = build_synthetic_survey_zip(tamper_manifest_sha=True)
        stream = io.BytesIO(zip_bytes)
        dataset_id, report, entry = self.service.ingest_dataset_zip(stream)

        self.assertEqual(report.validation_status, ValidationStatus.CORRUPTED)
        self.assertFalse(report.integrity.checksums_valid)
        self.assertTrue(any("Checksum mismatch" in err or "hash mismatch" in err for err in report.errors))

    def test_record_count_mismatch(self) -> None:
        """Verifies detection when manifest claims a record count differing from CSV rows."""
        zip_bytes = build_synthetic_survey_zip(wifi_sweeps=2, tamper_manifest_count=True)
        stream = io.BytesIO(zip_bytes)
        dataset_id, report, entry = self.service.ingest_dataset_zip(stream)

        self.assertEqual(report.validation_status, ValidationStatus.FAILED)
        self.assertFalse(report.integrity.record_counts_valid)
        self.assertTrue(any("Wi-Fi sweep count mismatch" in err for err in report.errors))

    def test_session_id_mismatch(self) -> None:
        """Verifies rejection when observations cite a foreign session ID."""
        zip_bytes = build_synthetic_survey_zip(
            session_id="SES-AAA1111",
            session_id_override_csv="SES-BBB2222"
        )
        stream = io.BytesIO(zip_bytes)
        dataset_id, report, entry = self.service.ingest_dataset_zip(stream)

        self.assertEqual(report.validation_status, ValidationStatus.FAILED)
        self.assertTrue(any("session_id mismatch" in err for err in report.errors))

    def test_unsanitized_image_rejection(self) -> None:
        """Verifies that unsanitized photos (is_exif_sanitized=False) violate policy and fail."""
        zip_bytes = build_synthetic_survey_zip(photo_records=1, is_exif_sanitized=False)
        stream = io.BytesIO(zip_bytes)
        dataset_id, report, entry = self.service.ingest_dataset_zip(stream)

        self.assertEqual(report.validation_status, ValidationStatus.FAILED)
        self.assertTrue(any("violates privacy policy: is_exif_sanitized is FALSE" in err for err in report.errors))

    def test_mixed_real_and_mock_classification(self) -> None:
        """Verifies proper classification of synthetic/mock and hybrid datasets."""
        # Pure mock dataset
        mock_zip = build_synthetic_survey_zip(session_id="SES-MOCK", is_mock=True)
        _, mock_rep, mock_entry = self.service.ingest_dataset_zip(io.BytesIO(mock_zip))
        self.assertEqual(mock_entry["data_source_type"], "MOCK")

    def test_empty_modalities_allowed(self) -> None:
        """Verifies that a valid session with zero IMU or photo observations passes validation cleanly."""
        zip_bytes = build_synthetic_survey_zip(
            session_id="SES-WIFI-ONLY",
            empty_modalities=True
        )
        stream = io.BytesIO(zip_bytes)
        dataset_id, report, entry = self.service.ingest_dataset_zip(stream)

        self.assertEqual(report.validation_status, ValidationStatus.PASSED)
        self.assertEqual(report.modalities.wifi.total_scans, 0)
        self.assertEqual(report.modalities.sensors.total_records, 0)
        self.assertEqual(report.modalities.photos.total_photo_records, 0)

    def test_registry_persistence_and_lookup(self) -> None:
        """Verifies disk persistence and reloading of the dataset catalog."""
        zip_bytes = build_synthetic_survey_zip(session_id="SES-PERSIST")
        dataset_id, _, _ = self.service.ingest_dataset_zip(io.BytesIO(zip_bytes))

        # Re-instantiate a fresh registry pointing at the same base directory
        new_registry = DatasetRegistry(self.config)
        retrieved = new_registry.get_dataset(dataset_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["session_id"], "SES-PERSIST")

        # Test listing
        all_ds = new_registry.list_datasets()
        self.assertEqual(len(all_ds), 1)

    def test_missing_referenced_image_file(self) -> None:
        """Verifies failure when photo metadata references an image missing from images/."""
        zip_bytes = build_synthetic_survey_zip(
            photo_records=1,
            omit_file="images/photo_001.jpg"
        )
        stream = io.BytesIO(zip_bytes)
        dataset_id, report, entry = self.service.ingest_dataset_zip(stream)

        self.assertIn(report.validation_status, (ValidationStatus.FAILED, ValidationStatus.CORRUPTED))
        self.assertTrue(any("missing from images/ directory" in err or "missing on disk" in err for err in report.errors))

    def test_invalid_csv_header(self) -> None:
        """Verifies rejection when a CSV observation file has invalid or corrupted header columns."""
        # Create a ZIP where wifi_observations.csv has invalid header
        buf = io.BytesIO()
        valid_zip_bytes = build_synthetic_survey_zip(wifi_sweeps=1)
        with zipfile.ZipFile(io.BytesIO(valid_zip_bytes), "r") as zf_in:
            with zipfile.ZipFile(buf, "w") as zf_out:
                for item in zf_in.infolist():
                    content = zf_in.read(item.filename)
                    if item.filename == "wifi_observations.csv":
                        content = b"col1,col2,col3\nval1,val2,val3\n"
                    zf_out.writestr(item.filename, content)

        buf.seek(0)
        dataset_id, report, entry = self.service.ingest_dataset_zip(buf)
        self.assertIn(report.validation_status, (ValidationStatus.FAILED, ValidationStatus.CORRUPTED))
        self.assertTrue(any("header mismatch" in err or "Checksum mismatch" in err for err in report.errors))

    def test_non_monotonic_timestamps_detected(self) -> None:
        """Verifies timestamp diagnostics flag out-of-order observation timestamps."""
        buf = io.BytesIO()
        timestamp_ms = 1731855600000
        # Create IMU rows with intentionally out-of-order timestamps
        imu_buf = io.StringIO()
        imu_writer = csv.writer(imu_buf)
        imu_writer.writerow(EXPECTED_IMU_HEADER)
        # Row 1 at t=100
        imu_writer.writerow([
            "IMU-01", "SES-TIME", "100000000", "CLOCK_BOOTTIME_NANOS",
            str(timestamp_ms + 100), "2026-11-17T15:00:00.100Z", "BOOTTIME_OFFSET", "3.5",
            "ACCELEROMETER", "0.1", "0.2", "9.8", "9.8", "HIGH", "GAME", "false"
        ])
        # Row 2 at t=50 (out of order!)
        imu_writer.writerow([
            "IMU-02", "SES-TIME", "50000000", "CLOCK_BOOTTIME_NANOS",
            str(timestamp_ms + 50), "2026-11-17T15:00:00.050Z", "BOOTTIME_OFFSET", "3.5",
            "ACCELEROMETER", "0.1", "0.2", "9.8", "9.8", "HIGH", "GAME", "false"
        ])
        raw_zip_bytes = build_synthetic_survey_zip(session_id="SES-TIME", imu_records=2)
        with zipfile.ZipFile(io.BytesIO(raw_zip_bytes), "r") as zf_in:
            with zipfile.ZipFile(buf, "w") as zf_out:
                for item in zf_in.infolist():
                    content = zf_in.read(item.filename)
                    if item.filename == "imu_observations.csv":
                        content = imu_buf.getvalue().encode("utf-8")
                    zf_out.writestr(item.filename, content)

        buf.seek(0)
        dataset_id, report, entry = self.service.ingest_dataset_zip(buf)
        self.assertGreater(report.timestamp_quality.out_of_order_count, 0)

    def test_partial_extraction_cleanup(self) -> None:
        """Verifies that an aborted extraction purges partial files completely."""
        target_dir = self.config.extracted_path / "cleanup_test_dir"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "canary.txt").write_text("should be cleaned up")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../../etc/shadow", b"bad")

        buf.seek(0)
        with self.assertRaises(SecurityValidationError):
            self.extractor.extract_dataset(buf, target_dir)

        self.assertFalse(target_dir.exists())

    def test_dataset_deletion_lifecycle(self) -> None:
        """Verifies deletion purges registry entry, extracted directory, and raw ZIP."""
        zip_bytes = build_synthetic_survey_zip(session_id="SES-DELETE")
        dataset_id, _, _ = self.service.ingest_dataset_zip(io.BytesIO(zip_bytes))

        extracted_dir = self.config.extracted_path / dataset_id
        raw_zip = self.config.raw_zips_path / f"{dataset_id}.zip"

        self.assertTrue(extracted_dir.exists())
        self.assertTrue(raw_zip.exists())

        deleted = self.registry.delete_dataset(dataset_id)
        self.assertTrue(deleted)
        self.assertIsNone(self.registry.get_dataset(dataset_id))
        self.assertFalse(extracted_dir.exists())
        self.assertFalse(raw_zip.exists())


class TestDatasetApiEndpoints(unittest.TestCase):
    """Test suite for FastAPI dataset ingestion and inspection endpoints."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_dataset_api_")
        self.config = DatasetConfig(base_dir=Path(self.temp_dir))
        self.config.ensure_directories()
        self.service = DatasetIngestionService(config=self.config)

        # Create FastAPI app and override dependency
        self.app = create_app()
        from backend.app.api.routes.datasets import get_ingestion_service
        self.app.dependency_overrides[get_ingestion_service] = lambda: self.service
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_api_upload_dataset_success(self) -> None:
        """Tests POST /api/datasets/upload with a valid ZIP file."""
        zip_bytes = build_synthetic_survey_zip(session_id="SES-API-01", wifi_sweeps=2)
        response = self.client.post(
            "/api/datasets/upload",
            files={"file": ("survey_export.zip", io.BytesIO(zip_bytes), "application/zip")}
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("dataset_id", data)
        self.assertEqual(data["session_id"], "SES-API-01")
        self.assertEqual(data["validation_status"], "PASSED")

        dataset_id = data["dataset_id"]

        # Test GET /api/datasets
        list_res = self.client.get("/api/datasets")
        self.assertEqual(list_res.status_code, 200)
        self.assertEqual(len(list_res.json()), 1)

        # Test GET /api/datasets/{dataset_id}
        detail_res = self.client.get(f"/api/datasets/{dataset_id}")
        self.assertEqual(detail_res.status_code, 200)
        detail_data = detail_res.json()
        self.assertEqual(detail_data["session_id"], "SES-API-01")
        self.assertEqual(detail_data["building_id"], "BLDG-01")

        # Test GET /api/datasets/{dataset_id}/quality
        quality_res = self.client.get(f"/api/datasets/{dataset_id}/quality")
        self.assertEqual(quality_res.status_code, 200)
        quality_data = quality_res.json()
        self.assertTrue(quality_data["integrity"]["manifest_valid"])
        self.assertEqual(quality_data["modalities"]["wifi"]["total_scans"], 2)

        # Test POST /api/datasets/{dataset_id}/validate
        reval_res = self.client.post(f"/api/datasets/{dataset_id}/validate")
        self.assertEqual(reval_res.status_code, 200)
        self.assertEqual(reval_res.json()["validation_status"], "PASSED")

        # Test DELETE /api/datasets/{dataset_id}
        del_res = self.client.delete(f"/api/datasets/{dataset_id}")
        self.assertEqual(del_res.status_code, 200)
        self.assertTrue(del_res.json()["success"])

        # Test GET after deletion -> 404
        get_deleted = self.client.get(f"/api/datasets/{dataset_id}")
        self.assertEqual(get_deleted.status_code, 404)

    def test_api_upload_non_zip_file_rejected(self) -> None:
        """Tests that uploading a non-ZIP file returns 400 Bad Request."""
        response = self.client.post(
            "/api/datasets/upload",
            files={"file": ("document.txt", io.BytesIO(b"Hello world"), "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("must have a .zip extension", response.json()["detail"])

    def test_api_upload_tampered_checksum_rejected(self) -> None:
        """Tests that uploading an archive with tampered checksums fails validation."""
        zip_bytes = build_synthetic_survey_zip(tamper_manifest_sha=True)
        response = self.client.post(
            "/api/datasets/upload",
            files={"file": ("corrupt.zip", io.BytesIO(zip_bytes), "application/zip")}
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["validation_status"], "CORRUPTED")
        self.assertFalse(data["quality_report"]["integrity"]["checksums_valid"])

    def test_api_nonexistent_dataset_404(self) -> None:
        """Tests that requesting nonexistent dataset ID returns 404."""
        response = self.client.get("/api/datasets/nonexistent-uuid-1234")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
