"""Unit and API integration test suite for Task 25: Computer Vision-Based Indoor Landmark Detection."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.dataset_config import DatasetConfig
from backend.app.services.dataset_registry import DatasetRegistry
from backend.app.main import create_app
from src.computer_vision import (
    BoundingBox,
    GroundTruthLandmark,
    ImageLoadingError,
    ImagePreprocessor,
    ImagePrivacyValidator,
    LandmarkCategory,
    LandmarkClassifier,
    LandmarkDetection,
    LandmarkDetector,
    LandmarkRecord,
    LandmarkRegistry,
    OcrTextRecord,
    SafeImageLoader,
    SignboardOcrEngine,
    VerificationStatus,
    VisionEvaluationReport,
    VisionEvaluator,
    VisionPipeline,
    VisualFeatureExtractor,
    VisualLandmarkMatcher,
    VisualMatchStatus,
)


class TestBoundingBoxGeometry(unittest.TestCase):
    """Tests for normalized BoundingBox coordinates, clamping, and IoU."""

    def test_box_clamping_and_dimensions(self):
        bb = BoundingBox(x_min=-0.1, y_min=0.2, x_max=1.2, y_max=0.8)
        self.assertEqual(bb.x_min, 0.0)
        self.assertEqual(bb.x_max, 1.0)
        self.assertEqual(bb.y_min, 0.2)
        self.assertEqual(bb.y_max, 0.8)
        self.assertAlmostEqual(bb.width, 1.0)
        self.assertAlmostEqual(bb.height, 0.6)
        self.assertAlmostEqual(bb.area, 0.6)

    def test_inverted_coordinates_swap(self):
        bb = BoundingBox(x_min=0.8, y_min=0.9, x_max=0.2, y_max=0.1)
        self.assertEqual(bb.x_min, 0.2)
        self.assertEqual(bb.x_max, 0.8)
        self.assertEqual(bb.y_min, 0.1)
        self.assertEqual(bb.y_max, 0.9)

    def test_pixel_conversions(self):
        bb = BoundingBox(0.25, 0.50, 0.75, 1.00)
        px_min, py_min, px_max, py_max = bb.to_pixel_coords(800, 600)
        self.assertEqual((px_min, py_min, px_max, py_max), (200, 300, 600, 600))

        reconstructed = BoundingBox.from_pixel_coords(px_min, py_min, px_max, py_max, 800, 600)
        self.assertAlmostEqual(reconstructed.x_min, bb.x_min)
        self.assertAlmostEqual(reconstructed.y_min, bb.y_min)
        self.assertAlmostEqual(reconstructed.x_max, bb.x_max)
        self.assertAlmostEqual(reconstructed.y_max, bb.y_max)

    def test_iou_calculation(self):
        # Disjoint boxes
        b1 = BoundingBox(0.0, 0.0, 0.4, 0.4)
        b2 = BoundingBox(0.6, 0.6, 1.0, 1.0)
        self.assertEqual(b1.iou(b2), 0.0)

        # Identical boxes
        self.assertAlmostEqual(b1.iou(b1), 1.0)

        # 50% overlap box
        b3 = BoundingBox(0.2, 0.0, 0.6, 0.4)
        # Inter = 0.2 * 0.4 = 0.08, Union = 0.16 + 0.16 - 0.08 = 0.24, IoU = 0.08 / 0.24 = 1/3
        self.assertAlmostEqual(b1.iou(b3), 1.0 / 3.0, places=4)


class TestImageValidationAndSafeLoader(unittest.TestCase):
    """Tests for privacy validation, EXIF sanitization check, and safe image loading."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        # Create a valid test image
        self.valid_img_path = self.temp_dir / "valid_test.jpg"
        img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        img.save(self.valid_img_path, format="JPEG")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_privacy_rejection_when_not_sanitized(self):
        meta_unsanitized = {"is_exif_sanitized": "false", "face_detection_status": "NONE"}
        val_res = ImagePrivacyValidator.validate_photo_file(
            image_path=self.valid_img_path,
            photo_id="photo_01",
            metadata_dict=meta_unsanitized,
            enforce_sanitization=True,
        )
        self.assertFalse(val_res.is_valid)
        self.assertIn("Privacy check failed", val_res.errors[0])

        with self.assertRaises(ImageLoadingError):
            SafeImageLoader.load_rgb(
                image_path=self.valid_img_path,
                photo_id="photo_01",
                enforce_sanitization=True,
                metadata_dict=meta_unsanitized,
            )

    def test_privacy_acceptance_when_sanitized(self):
        meta_sanitized = {"is_exif_sanitized": "true", "face_detection_status": "PROCESSED"}
        val_res = ImagePrivacyValidator.validate_photo_file(
            image_path=self.valid_img_path,
            photo_id="photo_01",
            metadata_dict=meta_sanitized,
            enforce_sanitization=True,
        )
        self.assertTrue(val_res.is_valid)
        self.assertTrue(val_res.is_exif_sanitized)

        rgb_arr, res = SafeImageLoader.load_rgb(
            image_path=self.valid_img_path,
            photo_id="photo_01",
            enforce_sanitization=True,
            metadata_dict=meta_sanitized,
        )
        self.assertEqual(rgb_arr.shape, (480, 640, 3))
        self.assertTrue(res.is_valid)

    def test_rejection_of_corrupt_file(self):
        corrupt_path = self.temp_dir / "corrupt.jpg"
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_REAL_JPEG_IMAGE_CONTENT_HERE")

        meta = {"is_exif_sanitized": "true"}
        val_res = ImagePrivacyValidator.validate_photo_file(
            image_path=corrupt_path,
            photo_id="corrupt_01",
            metadata_dict=meta,
            enforce_sanitization=True,
        )
        self.assertFalse(val_res.is_valid)

    def test_rejection_of_disallowed_extension(self):
        bad_ext_path = self.temp_dir / "exploit.exe"
        with open(bad_ext_path, "wb") as f:
            f.write(b"\x00" * 1024)

        val_res = ImagePrivacyValidator.validate_photo_file(
            image_path=bad_ext_path,
            photo_id="bad_01",
            metadata_dict={"is_exif_sanitized": "true"},
        )
        self.assertFalse(val_res.is_valid)
        self.assertTrue(any("Disallowed image extension" in e for e in val_res.errors))


class TestLandmarkDetectorAndOcr(unittest.TestCase):
    """Tests for indoor landmark detection heuristics and OCR integration."""

    def setUp(self):
        # Create a synthetic image with an artificial door rectangle and room sign
        self.img = np.full((600, 800, 3), 220, dtype=np.uint8)

        # Draw a door rectangle (vertical aspect ratio ~ 2.2)
        cv2.rectangle(self.img, (200, 100), (400, 550), (60, 40, 20), -1)
        cv2.rectangle(self.img, (200, 100), (400, 550), (20, 10, 0), 4)

        # Draw a signboard beside the door
        cv2.rectangle(self.img, (450, 200), (620, 280), (40, 40, 40), -1)
        cv2.putText(self.img, "ROOM 101", (460, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def test_detector_finds_door_and_sign(self):
        detector = LandmarkDetector(confidence_threshold=0.35, enable_ocr=True)
        dets = detector.detect(
            self.img,
            photo_id="test_photo_01",
            building_id="BLDG-1",
            floor_id=2,
        )
        self.assertGreaterEqual(len(dets), 1)

        categories = [d.category for d in dets]
        # At least one architectural element (DOOR or SIGN) must be identified
        self.assertTrue(any(c in (LandmarkCategory.DOOR, LandmarkCategory.ROOM_NUMBER_SIGN, LandmarkCategory.EXIT_SIGN) for c in categories))

        # Check metadata preservation without hallucinating
        for d in dets:
            self.assertEqual(d.building_id, "BLDG-1")
            self.assertEqual(d.floor_id, 2)
            self.assertIsNotNone(d.bounding_box)
            self.assertGreater(d.confidence, 0.0)

    def test_empty_or_featureless_image_returns_empty_list(self):
        blank_img = np.full((400, 400, 3), 128, dtype=np.uint8)
        detector = LandmarkDetector(confidence_threshold=0.50)
        dets = detector.detect(blank_img)
        self.assertEqual(len(dets), 0)

    def test_ocr_unverified_status(self):
        sign_crop = self.img[200:280, 450:620]
        ocr_engine = SignboardOcrEngine(use_easyocr=False)
        record = ocr_engine.recognize_text(sign_crop, source_image_id="img_01")
        if record:
            # Must remain UNVERIFIED until manual confirmation
            self.assertFalse(record.is_manually_verified)


class TestLandmarkRegistry(unittest.TestCase):
    """Tests for persistent landmark registry lifecycle and manual verification."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.registry = LandmarkRegistry(base_storage_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_detection_as_unverified(self):
        det = LandmarkDetection(
            category=LandmarkCategory.DOOR,
            bounding_box=BoundingBox(0.2, 0.1, 0.4, 0.8),
            confidence=0.85,
            photo_id="photo_101.jpg",
            building_id="BLDG-TEST",
            floor_id=3,
            reference_location_id="LOC-301",
        )
        rec = self.registry.register_detection(det, name="East Entrance Door")
        self.assertEqual(rec.verification_status, VerificationStatus.UNVERIFIED)
        self.assertEqual(rec.building_id, "BLDG-TEST")
        self.assertEqual(rec.floor_id, 3)
        self.assertIn("photo_101.jpg", rec.source_image_ids)

        # Test querying
        retrieved = self.registry.get_landmark(rec.landmark_id, "BLDG-TEST")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "East Entrance Door")

    def test_manual_verification_of_landmark(self):
        det = LandmarkDetection(
            category=LandmarkCategory.ROOM_NUMBER_SIGN,
            bounding_box=BoundingBox(0.4, 0.2, 0.6, 0.3),
            confidence=0.92,
            building_id="BLDG-TEST",
            floor_id=1,
        )
        rec = self.registry.register_detection(det)
        self.assertEqual(rec.verification_status, VerificationStatus.UNVERIFIED)

        # Manually confirm landmark
        verified = self.registry.verify_landmark(
            landmark_id=rec.landmark_id,
            building_id="BLDG-TEST",
            verified_name="Room 101 Lab Plaque",
            coordinates_m=(25.4, 18.2),
            status=VerificationStatus.MANUALLY_VERIFIED,
        )
        self.assertIsNotNone(verified)
        self.assertEqual(verified.verification_status, VerificationStatus.MANUALLY_VERIFIED)
        self.assertEqual(verified.verified_coordinates_m, (25.4, 18.2))
        self.assertEqual(verified.name, "Room 101 Lab Plaque")

    def test_disk_persistence_and_reload(self):
        det = LandmarkDetection(
            category=LandmarkCategory.STAIRCASE,
            bounding_box=BoundingBox(0.1, 0.1, 0.9, 0.9),
            confidence=0.88,
            building_id="BLDG-PERSIST",
            floor_id=1,
        )
        rec = self.registry.register_detection(det, name="Central Stairwell")

        # Create fresh registry instance pointing to same storage
        new_registry = LandmarkRegistry(base_storage_dir=self.temp_dir)
        loaded = new_registry.get_landmark(rec.landmark_id, "BLDG-PERSIST")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "Central Stairwell")
        self.assertEqual(loaded.category, LandmarkCategory.STAIRCASE)


class TestVisualLandmarkMatching(unittest.TestCase):
    """Tests for visual descriptor matching and NO_MATCH behavior."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.registry = LandmarkRegistry(base_storage_dir=self.temp_dir)
        self.matcher = VisualLandmarkMatcher(registry=self.registry)
        self.extractor = VisualFeatureExtractor()

        # Create two distinct test patterns
        self.pattern_a = np.zeros((200, 200, 3), dtype=np.uint8)
        self.pattern_a[:, :] = (0, 0, 255)  # Pure Red
        cv2.circle(self.pattern_a, (100, 100), 40, (255, 255, 255), -1)

        self.pattern_b = np.zeros((200, 200, 3), dtype=np.uint8)
        self.pattern_b[:, :] = (255, 0, 0)  # Pure Blue
        cv2.rectangle(self.pattern_b, (50, 50), (150, 150), (0, 255, 0), -1)

        # Register pattern A as a landmark
        det_a = LandmarkDetection(
            category=LandmarkCategory.DISTINCTIVE_STRUCTURE,
            bounding_box=BoundingBox(0, 0, 1, 1),
            confidence=0.95,
            photo_id="photo_a.jpg",
            building_id="BLDG-1",
            floor_id=1,
            descriptor_vector=self.extractor.extract_global_embedding(self.pattern_a),
        )
        self.lm_a = self.registry.register_detection(det_a, name="Red Wall Target")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_strong_match_for_similar_image(self):
        # Query identical to registered landmark A
        report = self.matcher.match_query_image(
            query_bgr=self.pattern_a,
            query_image_id="query_similar",
            building_id="BLDG-1",
            floor_id=1,
        )
        self.assertEqual(report.match_status, VisualMatchStatus.STRONG_MATCH)
        self.assertEqual(report.best_candidate_id, self.lm_a.landmark_id)
        self.assertGreater(report.similarity_score, 0.80)

    def test_no_match_for_dissimilar_image(self):
        # Query pattern B (Blue/Green) is completely dissimilar to pattern A (Red/White)
        report = self.matcher.match_query_image(
            query_bgr=self.pattern_b,
            query_image_id="query_dissimilar",
            building_id="BLDG-1",
            floor_id=1,
        )
        self.assertEqual(report.match_status, VisualMatchStatus.NO_MATCH)


class TestVisionEvaluationFramework(unittest.TestCase):
    """Tests for multi-class detection evaluation, confusion matrix, and OCR error metrics."""

    def test_synthetic_vision_benchmark_evaluation(self):
        evaluator = VisionEvaluator(iou_threshold=0.50)
        preds, gt, matches, gt_matches = evaluator.generate_controlled_synthetic_vision_fixture()

        report = evaluator.evaluate_detections(
            predictions_by_photo=preds,
            ground_truth_by_photo=gt,
            is_synthetic_fixture=True,
            matching_reports=matches,
            ground_truth_matches=gt_matches,
        )

        self.assertTrue(report.is_synthetic_fixture)
        self.assertEqual(report.total_images_evaluated, 3)
        self.assertEqual(report.total_ground_truth_boxes, 4)
        self.assertEqual(report.total_predictions, 5)

        # 4 true positives, 1 false positive -> precision = 4 / 5 = 0.80, recall = 4 / 4 = 1.00
        self.assertAlmostEqual(report.overall_precision, 0.80, places=2)
        self.assertAlmostEqual(report.overall_recall, 1.00, places=2)
        self.assertGreater(report.overall_f1, 0.85)

        # OCR evaluation check
        self.assertIsNotNone(report.ocr_exact_match_rate)
        self.assertEqual(report.ocr_exact_match_rate, 1.0)  # Both signs matched exactly

        # Visual matching check
        self.assertIsNotNone(report.matching_precision)
        self.assertEqual(report.matching_precision, 1.0)


class TestComputerVisionAPIEndpoints(unittest.TestCase):
    """Integration test suite for FastAPI /datasets/{id}/vision endpoints."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = DatasetConfig(base_dir=self.temp_dir)
        self.registry = DatasetRegistry(self.config)

        from backend.app.services.dataset_ingestion import DatasetIngestionService
        from backend.app.services.computer_vision_service import ComputerVisionService
        from backend.app.api.routes import datasets

        self.ingestion_service = DatasetIngestionService(self.config, self.registry)
        self.vision_service = ComputerVisionService(self.config, self.registry)

        # Dependency injection
        datasets._ingestion_service = self.ingestion_service
        datasets._vision_service = self.vision_service

        app = create_app()
        self.client = TestClient(app)

        # Create a mock extracted survey dataset with 1 valid photo
        self.dataset_id = "test-vision-ds"
        extracted_dir = self.config.extracted_path / self.dataset_id
        extracted_dir.mkdir(parents=True, exist_ok=True)
        images_dir = extracted_dir / "photos"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Generate a test JPEG
        photo_path = images_dir / "door_01.jpg"
        img = Image.new("RGB", (640, 480), color=(180, 180, 180))
        img.save(photo_path, format="JPEG")

        # Write photo_metadata.csv confirming is_exif_sanitized=true
        csv_path = extracted_dir / "photo_metadata.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("photo_id,filename,is_exif_sanitized,building,floor,room,normalized_epoch_ms\n")
            f.write("door_01,door_01.jpg,true,BLDG-VISION,2,ROOM-201,1700000000000\n")

        # Register dataset in mock registry
        self.registry.register_dataset({
            "dataset_id": self.dataset_id,
            "session_id": "SES-VISION-01",
            "session_name": "Vision Survey Session",
            "raw_filename": "survey.zip",
            "extracted_path": str(extracted_dir),
            "entry_count": 1,
            "status": "VALID",
            "validation_report": {},
        })

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_vision_analyze_endpoint(self):
        res = self.client.post(
            f"/api/datasets/{self.dataset_id}/vision/analyze",
            json={"confidence_threshold": 0.30, "enable_ocr": True, "auto_register": True},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["dataset_id"], self.dataset_id)
        self.assertEqual(data["total_photos_scanned"], 1)
        self.assertEqual(data["photos_processed_valid"], 1)
        self.assertEqual(data["photos_blocked_privacy"], 0)

    def test_vision_detections_endpoint(self):
        # Run analyze first
        self.client.post(f"/api/datasets/{self.dataset_id}/vision/analyze", json={})
        res = self.client.get(f"/api/datasets/{self.dataset_id}/vision/detections")
        self.assertEqual(res.status_code, 200)
        self.assertIsInstance(res.json(), list)

    def test_vision_landmarks_and_verify_endpoint(self):
        # Auto-register landmarks
        self.client.post(f"/api/datasets/{self.dataset_id}/vision/analyze", json={"auto_register": True})

        # List landmarks
        res_list = self.client.get(f"/api/datasets/{self.dataset_id}/vision/landmarks")
        self.assertEqual(res_list.status_code, 200)
        landmarks = res_list.json()

        if landmarks:
            target_id = landmarks[0]["landmark_id"]
            # Verify landmark
            verify_payload = {
                "landmark_id": target_id,
                "building_id": landmarks[0]["building_id"],
                "status": "MANUALLY_VERIFIED",
                "verified_name": "Verified Door 201",
                "coordinates_m": [15.5, 30.2],
            }
            res_verify = self.client.post(
                f"/api/datasets/{self.dataset_id}/vision/landmarks/verify",
                json=verify_payload,
            )
            self.assertEqual(res_verify.status_code, 200)
            verified_data = res_verify.json()
            self.assertEqual(verified_data["verification_status"], "MANUALLY_VERIFIED")
            self.assertEqual(verified_data["name"], "Verified Door 201")
            self.assertEqual(verified_data["verified_coordinates_m"], [15.5, 30.2])

    def test_vision_matching_endpoint(self):
        # Register a landmark first
        self.client.post(f"/api/datasets/{self.dataset_id}/vision/analyze", json={"auto_register": True})
        match_payload = {
            "photo_filename": "door_01.jpg",
            "building_id": "BLDG-VISION",
            "floor_id": 2,
        }
        res_match = self.client.post(
            f"/api/datasets/{self.dataset_id}/vision/match",
            json=match_payload,
        )
        self.assertEqual(res_match.status_code, 200)
        match_data = res_match.json()
        self.assertIn("match_status", match_data)
        self.assertIn(match_data["match_status"], ["STRONG_MATCH", "WEAK_MATCH", "NO_MATCH"])

    def test_vision_evaluate_endpoint(self):
        res = self.client.post(
            f"/api/datasets/{self.dataset_id}/vision/evaluate",
            json={"is_synthetic": True, "iou_threshold": 0.50},
        )
        self.assertEqual(res.status_code, 200)
        eval_data = res.json()
        self.assertTrue(eval_data["is_synthetic_fixture"])
        self.assertGreaterEqual(eval_data["overall_precision"], 0.0)
        self.assertGreaterEqual(eval_data["overall_recall"], 0.0)
