"""Comprehensive unit and API integration test suite for PDR Calibration, Ground-Truth Validation & Drift Analysis (Task 22)."""

import io
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.core.dataset_config import DatasetConfig
from backend.app.main import create_app
from backend.app.services.dataset_extraction import DatasetExtractor
from backend.app.services.dataset_ingestion import DatasetIngestionService
from backend.app.services.dataset_registry import DatasetRegistry
from backend.app.services.dataset_validation import DatasetValidator
from backend.app.services.pdr_calibration_service import PDRCalibrationService
from backend.app.services.pdr_service import PDRService
from backend.app.services.preprocessing_service import PreprocessingService
from src.pdr.calibration.evaluator import PDREvaluator
from src.pdr.calibration.heading_evaluator import HeadingEvaluator, circular_difference_deg
from src.pdr.calibration.models import (
    CalibrationProfile,
    CalibrationStatus,
    GroundTruthMeasurement,
    PhonePlacement,
    ProfileType,
    ReferenceWaypoint,
    TrialType,
    WalkingCondition,
    WalkingTrial,
)
from src.pdr.calibration.step_evaluator import StepDetectionEvaluator
from src.pdr.calibration.step_length_calibrator import WeinbergCalibrator
from src.pdr.calibration.storage import CalibrationStorage
from src.pdr.calibration.trajectory_evaluator import TrajectoryDriftEvaluator, point_to_segment_distance
from src.pdr.calibration.uncertainty_evaluator import (
    CHI2_2DOF_95,
    UncertaintyAuditor,
    compute_confidence_ellipse_2d,
    is_point_inside_ellipse,
)
from src.pdr.uncertainty import PDRUncertaintyPropagator
from tests.test_pdr import build_synthetic_walking_survey_zip


class TestGroundTruthTrialModels(unittest.TestCase):
    """Verifies trial data modeling, serializability, and validation invariants."""

    def test_trial_creation_and_roundtrip_dict(self):
        """Verifies WalkingTrial creation, JSON serialization, and deserialization."""
        gt = GroundTruthMeasurement(
            measured_distance_m=25.5,
            measurement_method="LASER_RANGEFINDER",
            measurement_uncertainty_m=0.03,
            measured_start_heading_deg=90.0,
            measured_end_heading_deg=90.0,
            ground_truth_step_count=36,
            num_turns=0,
            waypoints=[
                ReferenceWaypoint("WP-0", 0.0, 0.0, 1000),
                ReferenceWaypoint("WP-1", 25.5, 0.0, 20000)
            ]
        )
        trial = WalkingTrial(
            trial_id="TRIAL-TEST-001",
            dataset_id="DATASET-01",
            session_id="SES-01",
            participant_id="PARTICIPANT-ANON",
            device_model="Pixel 7 Pro",
            phone_placement=PhonePlacement.HAND_HELD,
            walking_condition=WalkingCondition.NORMAL_PACE,
            trial_type=TrialType.STRAIGHT_LINE,
            start_timestamp_ms=1000,
            end_timestamp_ms=20000,
            ground_truth=gt,
            is_synthetic=False
        )

        d = trial.to_dict()
        self.assertEqual(d["trial_id"], "TRIAL-TEST-001")
        self.assertEqual(d["phone_placement"], "HAND_HELD")
        self.assertFalse(d["is_synthetic"])

        reloaded = WalkingTrial.from_dict(d)
        self.assertEqual(reloaded.trial_id, trial.trial_id)
        self.assertEqual(reloaded.ground_truth.measured_distance_m, 25.5)
        self.assertEqual(len(reloaded.ground_truth.waypoints), 2)


class TestWeinbergStepLengthCalibration(unittest.TestCase):
    """Verifies optimization of Weinberg stride constant K from ground-truth trials."""

    def setUp(self):
        self.calibrator = WeinbergCalibrator(min_k=0.20, max_k=0.80, default_k=0.42)

    def test_weinberg_parameter_estimation(self):
        """Generates synthetic walking trials with known K=0.48 and verifies calibrator recovers K accurately."""
        target_k = 0.48
        # Simulate 15 steps with mean bounce range 2.5 m/s^2
        np.random.seed(42)
        accel_ranges = np.random.uniform(1.8, 3.2, size=15)
        # Ground truth distance calculated with target_k
        true_dist = float(np.sum(np.clip(target_k * (accel_ranges ** 0.25), 0.30, 1.20)))

        gt = GroundTruthMeasurement(
            measured_distance_m=true_dist,
            measurement_method="LASER_RANGEFINDER",
            measurement_uncertainty_m=0.02
        )
        trial = WalkingTrial(
            trial_id="TRIAL-CAL-1",
            dataset_id="DS-1",
            session_id="SES-1",
            participant_id="P1",
            device_model="Pixel",
            phone_placement=PhonePlacement.HAND_HELD,
            walking_condition=WalkingCondition.NORMAL_PACE,
            trial_type=TrialType.STRAIGHT_LINE,
            start_timestamp_ms=0,
            end_timestamp_ms=10000,
            ground_truth=gt
        )
        steps_df = pd.DataFrame({"accel_range_mps2": accel_ranges})

        profile = self.calibrator.calibrate(
            training_trials=[(trial, steps_df)],
            profile_name="Optimized Profile",
            profile_type=ProfileType.PARTICIPANT_SPECIFIC,
            participant_id="P1"
        )

        self.assertEqual(profile.calibration_status, CalibrationStatus.CALIBRATED)
        # Fitted K should closely match the generative K=0.48
        self.assertAlmostEqual(profile.weinberg_k, target_k, places=2)
        # Training MAE should be nearly zero (< 0.05m)
        self.assertLess(profile.training_metrics["mae_m"], 0.05)

    def test_insufficient_calibration_data(self):
        """Verifies fallback to default K=0.42 and INSUFFICIENT_DATA status when no valid trials provided."""
        profile = self.calibrator.calibrate(training_trials=[])
        self.assertEqual(profile.calibration_status, CalibrationStatus.INSUFFICIENT_DATA)
        self.assertEqual(profile.weinberg_k, 0.42)

    def test_held_out_validation_separation(self):
        """Verifies that held-out trials are evaluated independently without data leakage."""
        target_k = 0.45
        accel_ranges_train = np.array([2.0, 2.5, 3.0, 2.2, 2.8])
        dist_train = float(np.sum(np.clip(target_k * (accel_ranges_train ** 0.25), 0.30, 1.20)))

        accel_ranges_val = np.array([2.1, 2.4, 2.9, 2.3])
        dist_val = float(np.sum(np.clip(target_k * (accel_ranges_val ** 0.25), 0.30, 1.20)))

        trial_train = WalkingTrial(
            trial_id="TR-TRAIN", dataset_id="D1", session_id="S1", participant_id="P1",
            device_model="Dev", phone_placement=PhonePlacement.POCKET, walking_condition=WalkingCondition.NORMAL_PACE,
            trial_type=TrialType.STRAIGHT_LINE, start_timestamp_ms=0, end_timestamp_ms=5000,
            ground_truth=GroundTruthMeasurement(dist_train, "LASER", 0.05)
        )
        trial_val = WalkingTrial(
            trial_id="TR-VAL", dataset_id="D1", session_id="S2", participant_id="P1",
            device_model="Dev", phone_placement=PhonePlacement.POCKET, walking_condition=WalkingCondition.NORMAL_PACE,
            trial_type=TrialType.STRAIGHT_LINE, start_timestamp_ms=6000, end_timestamp_ms=10000,
            ground_truth=GroundTruthMeasurement(dist_val, "LASER", 0.05)
        )

        profile = self.calibrator.calibrate(
            training_trials=[(trial_train, pd.DataFrame({"accel_range_mps2": accel_ranges_train}))],
            held_out_trials=[(trial_val, pd.DataFrame({"accel_range_mps2": accel_ranges_val}))],
            profile_name="Train-Val Profile"
        )

        self.assertEqual(profile.calibration_status, CalibrationStatus.VALIDATED)
        self.assertIn("mae_m", profile.validation_metrics)
        self.assertEqual(profile.validation_metrics["num_held_out_trials"], 1)


class TestStepDetectionAndHeadingEvaluation(unittest.TestCase):
    """Verifies classification metrics for step events and circular statistics for direction."""

    def test_step_precision_recall_f1(self):
        """Verifies True Positive, False Positive, and False Negative calculation with timestamps."""
        evaluator = StepDetectionEvaluator(timing_tolerance_ms=200.0)

        # Ground truth step timestamps: [1000, 1600, 2200, 2800] (4 steps)
        gt_times = [1000, 1600, 2200, 2800]

        # Detected timestamps: [1020 (TP), 1590 (TP), 2180 (TP), 2500 (FP - extra), 3500 (FP - extra)]
        # True step 2800 is missed (FN)
        det_times = [1020, 1590, 2180, 2500, 3500]
        det_df = pd.DataFrame({"timestamp_ms": det_times})

        metrics = evaluator.evaluate(det_df, ground_truth_timestamps_ms=gt_times)

        self.assertEqual(metrics.total_detected_steps, 5)
        self.assertEqual(metrics.ground_truth_steps, 4)
        self.assertEqual(metrics.true_positives, 3)
        self.assertEqual(metrics.false_positives, 2)
        self.assertEqual(metrics.false_negatives, 1)

        # Precision = 3 / (3 + 2) = 0.60
        self.assertAlmostEqual(metrics.precision, 0.60, places=3)
        # Recall = 3 / (3 + 1) = 0.75
        self.assertAlmostEqual(metrics.recall, 0.75, places=3)
        # F1 = 2 * (0.6 * 0.75) / (0.6 + 0.75) = 0.6667
        self.assertAlmostEqual(metrics.f1_score, 0.6667, places=3)

    def test_circular_difference_wraparound(self):
        """Verifies angular difference handles 0/360 wraparound correctly."""
        # 1 deg vs 359 deg -> diff should be 2 deg (not 358 deg)
        diff1 = circular_difference_deg(1.0, 359.0)
        self.assertAlmostEqual(diff1, 2.0, places=4)

        # 359 deg vs 1 deg -> diff should be -2 deg
        diff2 = circular_difference_deg(359.0, 1.0)
        self.assertAlmostEqual(diff2, -2.0, places=4)

        # 180 deg vs -180 deg -> diff should be 0 deg
        diff3 = circular_difference_deg(180.0, -180.0)
        self.assertAlmostEqual(diff3, 0.0, places=4)

        # 45 deg vs 55 deg -> diff should be -10 deg
        diff4 = circular_difference_deg(45.0, 55.0)
        self.assertAlmostEqual(diff4, -10.0, places=4)

    def test_heading_metrics_and_drift_rate(self):
        """Verifies heading evaluation computes mean absolute error and drift rate per minute."""
        evaluator = HeadingEvaluator()
        # Simulated walk of 60 seconds (1 minute): heading slowly drifts from 90.0 deg to 93.0 deg
        t_ms = np.linspace(0, 60000, 31, dtype=int)
        headings = np.linspace(90.0, 93.0, 31)

        traj_df = pd.DataFrame({
            "timestamp_ms": t_ms,
            "heading_deg": headings,
            "heading_source": ["ROTATION_VECTOR"] * 31
        })

        metrics = evaluator.evaluate(traj_df, reference_start_heading_deg=90.0, reference_end_heading_deg=90.0)

        # Total drift = |93.0 - 90.0| = 3.0 deg over 1 minute -> drift rate ~ 3.0 deg/min
        self.assertAlmostEqual(metrics.drift_rate_deg_per_min, 3.0, delta=0.2)
        self.assertGreater(metrics.mean_absolute_error_deg, 0.0)


class TestTrajectoryAndUncertaintyEvaluation(unittest.TestCase):
    """Verifies distance error, endpoint error, cross-track error, and Chi-square confidence ellipse."""

    def test_distance_and_endpoint_error(self):
        """Verifies dead-reckoning drift per meter and relative endpoint error."""
        evaluator = TrajectoryDriftEvaluator()

        # Straight-line walk: ground truth is 20.0m along Y axis (endpoint at (0, 20))
        # Estimated path terminates at (1.0, 19.0) with total distance 19.5m
        traj_df = pd.DataFrame({
            "timestamp_ms": [0, 10000, 20000],
            "rel_x_m": [0.0, 0.5, 1.0],
            "rel_y_m": [0.0, 9.5, 19.0],
            "cum_distance_m": [0.0, 9.8, 19.5]
        })

        metrics = evaluator.evaluate(
            trajectory_df=traj_df,
            ground_truth_distance_m=20.0,
            ground_truth_endpoint=(0.0, 20.0)
        )

        # Distance error = |19.5 - 20.0| = 0.5m
        self.assertAlmostEqual(metrics.absolute_distance_error_m, 0.5, delta=0.01)
        # Endpoint error = sqrt((1.0 - 0)^2 + (19.0 - 20.0)^2) = sqrt(2) ≈ 1.414m
        self.assertAlmostEqual(metrics.endpoint_error_m, math.sqrt(2.0), delta=0.01)
        # Drift per meter = 1.414 / 20.0 = 0.0707 m/m
        self.assertAlmostEqual(metrics.drift_per_meter, math.sqrt(2.0) / 20.0, delta=0.002)

    def test_point_to_segment_cross_track(self):
        """Verifies minimum perpendicular cross-track distance math."""
        # Horizontal segment from (0, 0) to (10, 0)
        # Point at (5, 3) -> distance is exactly 3.0
        d1 = point_to_segment_distance(5.0, 3.0, 0.0, 0.0, 10.0, 0.0)
        self.assertAlmostEqual(d1, 3.0, places=5)

        # Point past endpoint at (14, 0) -> distance to (10, 0) is 4.0
        d2 = point_to_segment_distance(14.0, 0.0, 0.0, 0.0, 10.0, 0.0)
        self.assertAlmostEqual(d2, 4.0, places=5)

    def test_2d_confidence_ellipse_chi_square(self):
        """Verifies 2D Gaussian confidence ellipse calculation using chi2(2, 0.95) = 5.99146."""
        # Diagonal covariance: var_x = 4.0, var_y = 1.0, cov_xy = 0.0
        cov = np.array([[4.0, 0.0], [0.0, 1.0]], dtype=float)

        ellipse = compute_confidence_ellipse_2d(cov, quantile=CHI2_2DOF_95)

        # lambda_1 = 4.0, lambda_2 = 1.0
        # semi_major = sqrt(5.99146 * 4.0) = 2.4477 * 2.0 = 4.8955 m
        expected_major = math.sqrt(CHI2_2DOF_95 * 4.0)
        # semi_minor = sqrt(5.99146 * 1.0) = 2.4477 m
        expected_minor = math.sqrt(CHI2_2DOF_95 * 1.0)

        self.assertAlmostEqual(ellipse.semi_major_axis_m, expected_major, delta=0.01)
        self.assertAlmostEqual(ellipse.semi_minor_axis_m, expected_minor, delta=0.01)
        self.assertAlmostEqual(ellipse.orientation_deg, 0.0, delta=0.1)

        # Check Mahalanobis containment:
        # Point at (2.0, 0) -> dx=2, dy=0 -> dx^2/4 + dy^2/1 = 1 <= 5.991 (INSIDE)
        self.assertTrue(is_point_inside_ellipse((2.0, 0.0), (0.0, 0.0), cov))
        # Point at (6.0, 0) -> dx=6 -> dx^2/4 = 9 > 5.991 (OUTSIDE)
        self.assertFalse(is_point_inside_ellipse((6.0, 0.0), (0.0, 0.0), cov))

    def test_3_state_covariance_propagation(self):
        """Verifies augmented 3-state state-transition covariance propagation F P F^T + G Q G^T."""
        propagator = PDRUncertaintyPropagator(initial_uncertainty_m=0.5, initial_heading_uncertainty_rad=0.05)

        # Step 1: L=0.8m, theta=0 rad
        cov1 = propagator.update_with_heading_drift(
            step_length_m=0.8,
            heading_rad=0.0,
            sigma_length_m=0.08,
            sigma_heading_drift_rad=0.03
        )
        # Step 2: L=0.8m, theta=0 rad
        cov2 = propagator.update_with_heading_drift(
            step_length_m=0.8,
            heading_rad=0.0,
            sigma_length_m=0.08,
            sigma_heading_drift_rad=0.03
        )

        # Semi-major axis should grow monotonically
        self.assertGreater(cov2.semi_major_axis_m, cov1.semi_major_axis_m)
        self.assertGreater(cov2.uncertainty_radius_m, cov1.uncertainty_radius_m)


class TestCalibrationStorageAndProfileResolution(unittest.TestCase):
    """Verifies profile persistence and hierarchical matching fallback logic."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_cal_storage_")
        self.storage = CalibrationStorage(base_dir=Path(self.temp_dir))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_profile_initialized(self):
        """Verifies default uncalibrated profile PROF-DEFAULT exists upon initialization."""
        prof = self.storage.get_profile("PROF-DEFAULT")
        self.assertIsNotNone(prof)
        self.assertEqual(prof.weinberg_k, 0.42)
        self.assertEqual(prof.calibration_status, CalibrationStatus.UNCALIBRATED)

    def test_hierarchical_profile_resolution(self):
        """Verifies specific profile matching falls back properly to placement or default."""
        # Register a placement-specific calibrated profile for POCKET
        pocket_prof = CalibrationProfile(
            profile_id="PROF-POCKET-01",
            profile_name="Pocket Profile",
            profile_type=ProfileType.PHONE_PLACEMENT_SPECIFIC,
            calibration_method="WEINBERG_BOUNDED_OPT",
            weinberg_k=0.46,
            phone_placement=PhonePlacement.POCKET,
            calibration_status=CalibrationStatus.CALIBRATED
        )
        self.storage.save_profile(pocket_prof)

        # Query for POCKET -> should resolve PROF-POCKET-01
        resolved_pocket = self.storage.resolve_best_profile(phone_placement=PhonePlacement.POCKET)
        self.assertEqual(resolved_pocket.profile_id, "PROF-POCKET-01")
        self.assertEqual(resolved_pocket.weinberg_k, 0.46)

        # Query for BAG (no bag profile exists) -> should fallback to PROF-DEFAULT
        resolved_bag = self.storage.resolve_best_profile(phone_placement=PhonePlacement.BAG)
        self.assertEqual(resolved_bag.profile_id, "PROF-DEFAULT")
        self.assertEqual(resolved_bag.weinberg_k, 0.42)


class TestFastAPICalibrationEndpoints(unittest.TestCase):
    """Integration test suite for FastAPI trial registration, calibration, and evaluation endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="test_fastapi_cal_")
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
        from backend.app.api.routes.datasets import (
            get_ingestion_service,
            get_pdr_calibration_service,
            get_pdr_service,
            get_preprocessing_service,
        )
        cls.app.dependency_overrides[get_ingestion_service] = lambda: cls.ingestion_service
        cls.prep_service = PreprocessingService(config=cls.config, registry=cls.registry)
        cls.app.dependency_overrides[get_preprocessing_service] = lambda: cls.prep_service
        cls.pdr_service = PDRService(config=cls.config, registry=cls.registry)
        cls.app.dependency_overrides[get_pdr_service] = lambda: cls.pdr_service

        cls.cal_storage = CalibrationStorage(base_dir=Path(cls.temp_dir) / "calibration" / "pdr")
        cls.cal_service = PDRCalibrationService(
            config=cls.config,
            registry=cls.registry,
            storage=cls.cal_storage,
            pdr_service=cls.pdr_service
        )
        cls.app.dependency_overrides[get_pdr_calibration_service] = lambda: cls.cal_service

        cls.test_client = TestClient(cls.app)

        # Ingest a valid survey ZIP with walking IMU observations
        zip_bytes = build_synthetic_walking_survey_zip(
            session_id="SES-CALIB-INTEG",
            session_name="PDR Calibration Survey",
            duration_s=6.0,
            fs_hz=50.0,
            cadence_hz=1.8
        )
        cls.dataset_id, cls.report, cls.entry = cls.ingestion_service.ingest_dataset_zip(
            io.BytesIO(zip_bytes),
            original_filename="pdr_cal_survey.zip"
        )

        # Preprocess and run PDR
        cls.prep_service.preprocess_dataset(cls.dataset_id)
        cls.pdr_service.run_pdr(cls.dataset_id)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_trial_registration_and_listing(self):
        """Verifies POST and GET /datasets/{id}/pdr/trials."""
        payload = {
            "session_id": "SES-CALIB-INTEG",
            "participant_id": "PART-01",
            "device_model": "Pixel 7 Pro",
            "phone_placement": "HAND_HELD",
            "walking_condition": "NORMAL_PACE",
            "trial_type": "STRAIGHT_LINE",
            "start_timestamp_ms": 1731855600000,
            "end_timestamp_ms": 1731855606000,
            "ground_truth": {
                "measured_distance_m": 6.8,
                "measurement_method": "LASER_RANGEFINDER",
                "measurement_uncertainty_m": 0.05,
                "ground_truth_step_count": 11,
                "num_turns": 0
            },
            "trial_notes": "Integration test trial",
            "is_synthetic": True
        }

        # 1. Register trial
        res_post = self.test_client.post(f"/api/datasets/{self.dataset_id}/pdr/trials", json=payload)
        self.assertEqual(res_post.status_code, 201)
        data_post = res_post.json()
        self.assertIn("trial_id", data_post)
        trial_id = data_post["trial_id"]

        # 2. List trials
        res_list = self.test_client.get(f"/api/datasets/{self.dataset_id}/pdr/trials")
        self.assertEqual(res_list.status_code, 200)
        data_list = res_list.json()
        self.assertGreaterEqual(data_list["total_trials"], 1)

        # 3. Get single trial detail
        res_get = self.test_client.get(f"/api/datasets/{self.dataset_id}/pdr/trials/{trial_id}")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.json()
        self.assertEqual(data_get["trial_id"], trial_id)
        self.assertEqual(data_get["ground_truth"]["measured_distance_m"], 6.8)

    def test_weinberg_calibration_and_evaluation_endpoints(self):
        """Verifies POST /pdr/calibrate, GET /pdr/calibration, POST /pdr/evaluate, and GET /pdr/evaluation."""
        # 1. Register two trials
        t1_payload = {
            "session_id": "SES-CALIB-INTEG",
            "participant_id": "PART-02",
            "device_model": "Pixel 7 Pro",
            "phone_placement": "POCKET",
            "walking_condition": "NORMAL_PACE",
            "trial_type": "STRAIGHT_LINE",
            "start_timestamp_ms": 1731855600000,
            "end_timestamp_ms": 1731855606000,
            "ground_truth": {
                "measured_distance_m": 6.7,
                "measurement_method": "LASER_RANGEFINDER",
                "measurement_uncertainty_m": 0.05,
                "ground_truth_step_count": 11
            },
            "is_synthetic": True
        }
        res_t1 = self.test_client.post(f"/api/datasets/{self.dataset_id}/pdr/trials", json=t1_payload)
        self.assertEqual(res_t1.status_code, 201)
        trial_1_id = res_t1.json()["trial_id"]

        # 2. Calibrate Weinberg stride
        cal_payload = {
            "training_trial_ids": [trial_1_id],
            "profile_name": "Pocket Calibrated",
            "profile_type": "PHONE_PLACEMENT_SPECIFIC",
            "phone_placement": "POCKET"
        }
        res_cal = self.test_client.post(f"/api/datasets/{self.dataset_id}/pdr/calibrate", json=cal_payload)
        self.assertEqual(res_cal.status_code, 200)
        data_cal = res_cal.json()
        self.assertIn("weinberg_k", data_cal)
        self.assertEqual(data_cal["calibration_status"], "CALIBRATED")
        profile_id = data_cal["profile_id"]

        # 3. List profiles
        res_profs = self.test_client.get(f"/api/datasets/{self.dataset_id}/pdr/calibration")
        self.assertEqual(res_profs.status_code, 200)
        self.assertGreaterEqual(res_profs.json()["total_profiles"], 2)  # Default + Pocket

        # 4. Evaluate against trial
        eval_payload = {
            "trial_ids": [trial_1_id],
            "calibration_profile_id": profile_id
        }
        res_eval = self.test_client.post(f"/api/datasets/{self.dataset_id}/pdr/evaluate", json=eval_payload)
        self.assertEqual(res_eval.status_code, 200)
        reports = res_eval.json()
        self.assertEqual(len(reports), 1)
        rep = reports[0]
        self.assertIn("step_metrics", rep)
        self.assertIn("trajectory_metrics", rep)
        self.assertIn("uncertainty_metrics", rep)
        self.assertGreater(rep["step_metrics"]["f1_score"], 0.8)

        # 5. Get evaluation history
        res_history = self.test_client.get(f"/api/datasets/{self.dataset_id}/pdr/evaluation")
        self.assertEqual(res_history.status_code, 200)
        self.assertGreaterEqual(len(res_history.json()), 1)


if __name__ == "__main__":
    unittest.main()
