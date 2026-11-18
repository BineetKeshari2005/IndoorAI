"""Comprehensive unit and integration test suite for Wi-Fi + PDR Sensor Fusion using EKF (Task 24)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from backend.app.core.dataset_config import DatasetConfig
from backend.app.main import create_app
from backend.app.services.dataset_registry import DatasetRegistry
from backend.app.services.sensor_fusion_service import SensorFusionService
from src.sensor_fusion.coordinates import (
    CoordinateFrameMetadata,
    Pose2D,
    compass_azimuth_to_local_heading_rad,
    local_heading_rad_to_compass_azimuth,
    wrap_angle_deg,
    wrap_angle_rad,
)
from src.sensor_fusion.ekf import IndoorEKF, enforce_psd_covariance
from src.sensor_fusion.evaluation import FusionComparisonReport, FusionEvaluator
from src.sensor_fusion.models import (
    CHI2_2DOF_95,
    ConfidenceEllipse2D,
    FusedState,
    FusionConfig,
    FusionStatus,
    PdrStepEvent,
    UncertaintyType,
    WifiMeasurementEvent,
)
from src.sensor_fusion.pipeline import FusionPipeline
from src.sensor_fusion.uncertainty import WifiUncertaintyEstimator


class TestCoordinateConventions(unittest.TestCase):
    """Test building-local coordinate conventions, angular wrapping, and frame validations."""

    def test_wrap_angle_rad(self):
        self.assertAlmostEqual(wrap_angle_rad(0.0), 0.0)
        self.assertAlmostEqual(wrap_angle_rad(math.pi), -math.pi)
        self.assertAlmostEqual(wrap_angle_rad(3.0 * math.pi), -math.pi)
        self.assertAlmostEqual(wrap_angle_rad(-3.0 * math.pi), -math.pi)
        self.assertAlmostEqual(wrap_angle_rad(2.0 * math.pi + 0.5), 0.5)

    def test_wrap_angle_deg(self):
        self.assertAlmostEqual(wrap_angle_deg(0.0), 0.0)
        self.assertAlmostEqual(wrap_angle_deg(180.0), -180.0)
        self.assertAlmostEqual(wrap_angle_deg(370.0), 10.0)
        self.assertAlmostEqual(wrap_angle_deg(-190.0), 170.0)

    def test_compass_azimuth_conversions(self):
        # Azimuth 0 deg (North) -> Local +Y (pi/2 rad)
        theta_north = compass_azimuth_to_local_heading_rad(0.0)
        self.assertAlmostEqual(theta_north, math.pi / 2.0, places=4)
        self.assertAlmostEqual(local_heading_rad_to_compass_azimuth(theta_north), 0.0)

        # Azimuth 90 deg (East) -> Local +X (0 rad)
        theta_east = compass_azimuth_to_local_heading_rad(90.0)
        self.assertAlmostEqual(theta_east, 0.0, places=4)
        self.assertAlmostEqual(local_heading_rad_to_compass_azimuth(theta_east), 90.0)

        # Azimuth 180 deg (South) -> Local -Y (-pi/2 rad)
        theta_south = compass_azimuth_to_local_heading_rad(180.0)
        self.assertAlmostEqual(theta_south, -math.pi / 2.0, places=4)
        self.assertAlmostEqual(local_heading_rad_to_compass_azimuth(theta_south), 180.0)

    def test_coordinate_frame_validation(self):
        frame_a = CoordinateFrameMetadata(building_id="BLDG-1", floor_id=2)
        frame_b = CoordinateFrameMetadata(building_id="BLDG-1", floor_id=2)
        is_compat, err = frame_a.validate_compatibility(frame_b)
        self.assertTrue(is_compat)
        self.assertIsNone(err)

        # Building mismatch
        frame_c = CoordinateFrameMetadata(building_id="BLDG-2", floor_id=2)
        is_compat_c, err_c = frame_a.validate_compatibility(frame_c)
        self.assertFalse(is_compat_c)
        self.assertIn("Building mismatch", err_c)

        # Floor mismatch
        frame_d = CoordinateFrameMetadata(building_id="BLDG-1", floor_id=3)
        is_compat_d, err_d = frame_a.validate_compatibility(frame_d)
        self.assertFalse(is_compat_d)
        self.assertIn("Floor mismatch", err_d)


class TestEKFEngine(unittest.TestCase):
    """Test Extended Kalman Filter state transitions, Jacobians, Joseph updates, and gating."""

    def setUp(self):
        self.config = FusionConfig(
            chi2_threshold=CHI2_2DOF_95,
            initial_pos_uncertainty_m=1.0,
            initial_heading_uncertainty_rad=0.1,
        )
        self.ekf = IndoorEKF(config=self.config)
        self.initial_pose = Pose2D(x_m=10.0, y_m=20.0, theta_rad=0.0, building_id="BLDG-1", floor_id=1)
        self.ekf.initialize(self.initial_pose, timestamp_ms=1000)

    def test_initialization(self):
        self.assertTrue(self.ekf.is_initialized)
        self.assertEqual(self.ekf.x[0], 10.0)
        self.assertEqual(self.ekf.x[1], 20.0)
        self.assertEqual(self.ekf.x[2], 0.0)
        self.assertAlmostEqual(self.ekf.P[0, 0], 1.0)
        self.assertAlmostEqual(self.ekf.P[1, 1], 1.0)
        self.assertAlmostEqual(self.ekf.P[2, 2], 0.01)

    def test_pdr_prediction_step(self):
        # Step of 1.0m East (theta = 0)
        step = PdrStepEvent(
            step_index=1,
            timestamp_epoch_ms=1600,
            step_length_m=1.0,
            heading_rad=0.0,
            heading_delta_rad=0.0,
            sigma_length_m=0.05,
            sigma_heading_rad=0.02,
        )
        state = self.ekf.predict_pdr_step(step)

        self.assertEqual(state.status, FusionStatus.PDR_ONLY)
        self.assertAlmostEqual(self.ekf.x[0], 11.0, places=3)
        self.assertAlmostEqual(self.ekf.x[1], 20.0, places=3)
        # Covariance must grow after prediction
        self.assertGreater(self.ekf.P[0, 0], 1.0)
        # Verify symmetry and positive definiteness
        np.testing.assert_allclose(self.ekf.P, self.ekf.P.T, atol=1e-8)
        eigvals = np.linalg.eigvalsh(self.ekf.P)
        self.assertTrue(np.all(eigvals > 0))

    def test_wifi_measurement_update_accepted(self):
        # Wi-Fi measurement close to true position (x=10.5, y=20.2)
        wifi = WifiMeasurementEvent(
            scan_index=1,
            timestamp_epoch_ms=2000,
            x_m=10.5,
            y_m=20.2,
            building_id="BLDG-1",
            floor_id=1,
            confidence_score=0.9,
            matched_ap_count=6,
        )
        R_cov = np.eye(2) * 4.0  # (2.0m standard deviation)^2
        prior_var_x = self.ekf.P[0, 0]

        state = self.ekf.update_wifi(wifi, R_cov)

        self.assertEqual(state.status, FusionStatus.WIFI_CORRECTED)
        self.assertIsNotNone(state.innovation)
        self.assertLess(state.mahalanobis_dist_sq, self.config.chi2_threshold)
        # Measurement update must reduce position uncertainty
        self.assertLess(self.ekf.P[0, 0], prior_var_x)
        # Joseph update preserves symmetry
        np.testing.assert_allclose(self.ekf.P, self.ekf.P.T, atol=1e-8)

    def test_statistical_outlier_rejection(self):
        # Wi-Fi measurement with extreme +40m teleport jump
        wifi_outlier = WifiMeasurementEvent(
            scan_index=2,
            timestamp_epoch_ms=3000,
            x_m=50.0,
            y_m=20.0,
            building_id="BLDG-1",
            floor_id=1,
            confidence_score=0.8,
            matched_ap_count=5,
        )
        R_cov = np.eye(2) * 4.0
        prior_x = float(self.ekf.x[0])
        prior_P = self.ekf.P.copy()

        state = self.ekf.update_wifi(wifi_outlier, R_cov)

        self.assertEqual(state.status, FusionStatus.WIFI_REJECTED)
        self.assertEqual(state.rejection_reason, "STATISTICAL_GATE_EXCEEDED")
        self.assertGreater(state.mahalanobis_dist_sq, self.config.chi2_threshold)
        # Filter state and covariance must be preserved unmodified
        self.assertEqual(self.ekf.x[0], prior_x)
        np.testing.assert_array_equal(self.ekf.P, prior_P)

    def test_floor_mismatch_rejection(self):
        # Wi-Fi observation reported on Floor 2 when filter is on Floor 1
        wifi_floor2 = WifiMeasurementEvent(
            scan_index=3,
            timestamp_epoch_ms=4000,
            x_m=10.0,
            y_m=20.0,
            building_id="BLDG-1",
            floor_id=2,
        )
        state = self.ekf.update_wifi(wifi_floor2, np.eye(2) * 4.0)
        self.assertEqual(state.status, FusionStatus.WIFI_REJECTED)
        self.assertEqual(state.rejection_reason, "BUILDING_OR_FLOOR_MISMATCH")


class TestUncertaintyAndQualityEstimator(unittest.TestCase):
    """Test Wi-Fi covariance hierarchy, candidate dispersion, and quality gating."""

    def setUp(self):
        self.estimator = WifiUncertaintyEstimator()

    def test_empirical_covariance_registration(self):
        cov_emp = np.array([[6.25, 0.5], [0.5, 4.0]])
        self.estimator.register_empirical_covariance("SITE-A", cov_emp)

        evt = WifiMeasurementEvent(1, 1000, 5.0, 5.0, "BLDG-1", 1)
        cov, unc_type = self.estimator.estimate_covariance(evt, site_id="SITE-A")
        self.assertEqual(unc_type, UncertaintyType.EMPIRICAL_SURVEY_RESIDUALS)
        self.assertAlmostEqual(cov[0, 0], 6.25)

    def test_candidate_dispersion_covariance(self):
        evt = WifiMeasurementEvent(1, 1000, 10.0, 10.0, "BLDG-1", 1)
        candidates = [
            {"x_m": 8.0, "y_m": 10.0, "score": 0.5},
            {"x_m": 12.0, "y_m": 10.0, "score": 0.5},
        ]
        cov, unc_type = self.estimator.estimate_covariance(evt, candidate_matches=candidates)
        self.assertEqual(unc_type, UncertaintyType.CANDIDATE_DISPERSION)
        # Dispersion in X should be significantly higher than base floor
        self.assertGreater(cov[0, 0], 2.0)

    def test_quality_validation_filters(self):
        # Cached scan rejected
        evt_cached = WifiMeasurementEvent(1, 1000, 0.0, 0.0, "B1", 1, is_fresh=False)
        valid, reason = self.estimator.validate_measurement_quality(evt_cached)
        self.assertFalse(valid)
        self.assertIn("CACHED_OR_STALE_SCAN", reason)

        # Insufficient AP overlap rejected
        evt_low_ap = WifiMeasurementEvent(2, 1000, 0.0, 0.0, "B1", 1, matched_ap_count=1)
        valid, reason = self.estimator.validate_measurement_quality(evt_low_ap)
        self.assertFalse(valid)
        self.assertIn("INSUFFICIENT_AP_OVERLAP", reason)

        # Low confidence score rejected
        evt_low_conf = WifiMeasurementEvent(3, 1000, 0.0, 0.0, "B1", 1, matched_ap_count=5, confidence_score=0.1)
        valid, reason = self.estimator.validate_measurement_quality(evt_low_conf)
        self.assertFalse(valid)
        self.assertIn("LOW_CONFIDENCE", reason)


class TestFusionPipelineAndChronology(unittest.TestCase):
    """Test asynchronous event interleaving and pipeline execution."""

    def test_chronological_interleaving(self):
        pdr_steps = [
            PdrStepEvent(step_index=1, timestamp_epoch_ms=1000, step_length_m=0.8, heading_rad=0.0),
            PdrStepEvent(step_index=2, timestamp_epoch_ms=2000, step_length_m=0.8, heading_rad=0.0),
            PdrStepEvent(step_index=3, timestamp_epoch_ms=3000, step_length_m=0.8, heading_rad=0.0),
        ]
        wifi_scans = [
            # Wi-Fi scan arrives between step 1 and step 2
            WifiMeasurementEvent(
                scan_index=1, timestamp_epoch_ms=1500, x_m=1.2, y_m=0.0,
                building_id="BLDG-1", floor_id=1, confidence_score=0.8, matched_ap_count=5
            ),
        ]
        pipeline = FusionPipeline()
        init_pose = Pose2D(x_m=0.0, y_m=0.0, theta_rad=0.0, building_id="BLDG-1", floor_id=1)

        fused_states, summary = pipeline.run(pdr_steps, wifi_scans, initial_pose=init_pose)

        # Expected event order: Init(1000), Step 1(1000), Wi-Fi(1500), Step 2(2000), Step 3(3000)
        self.assertEqual(len(fused_states), 5)
        self.assertEqual(fused_states[0].status, FusionStatus.INITIALIZING)
        self.assertEqual(fused_states[1].status, FusionStatus.PDR_ONLY)
        self.assertEqual(fused_states[2].status, FusionStatus.WIFI_CORRECTED)
        self.assertEqual(fused_states[3].status, FusionStatus.PDR_ONLY)
        self.assertEqual(fused_states[4].status, FusionStatus.PDR_ONLY)

        self.assertEqual(summary.pdr_steps_count, 3)
        self.assertEqual(summary.wifi_scans_accepted, 1)
        self.assertEqual(summary.wifi_scans_rejected, 0)


class TestComparativeEvaluationFramework(unittest.TestCase):
    """Test synthetic benchmark generation, trajectory metrics, and outlier rejection."""

    def test_synthetic_benchmark_evaluation(self):
        evaluator = FusionEvaluator()
        gt_points, pdr_steps, wifi_events = evaluator.generate_controlled_synthetic_trajectory(
            duration_s=60.0,
            pdr_drift_rate_deg_min=30.0,
            inject_outlier=True,
        )
        self.assertGreater(len(gt_points), 30)
        self.assertGreater(len(wifi_events), 5)

        # Run pipeline
        pipeline = FusionPipeline()
        init_pose = Pose2D(x_m=0.0, y_m=0.0, theta_rad=0.0, building_id="BLDG-1", floor_id=1)
        fused_states, summary = pipeline.run(pdr_steps, wifi_events, initial_pose=init_pose)

        # Verify outlier was rejected by the chi2 gate
        self.assertGreater(summary.wifi_scans_rejected, 0)
        self.assertIn("STATISTICAL_GATE_EXCEEDED", summary.rejection_breakdown)

        # Build PDR-only coordinates
        pdr_coords = []
        cx, cy = 0.0, 0.0
        for s in pdr_steps:
            cx += s.step_length_m * math.cos(s.heading_rad)
            cy += s.step_length_m * math.sin(s.heading_rad)
            pdr_coords.append((s.timestamp_epoch_ms, cx, cy))

        report = evaluator.evaluate_against_ground_truth(
            fused_states=fused_states,
            ground_truth=gt_points,
            pdr_states=pdr_coords,
            wifi_measurements=wifi_events,
            is_synthetic_fixture=True,
        )

        self.assertTrue(report.is_synthetic_fixture)
        # Fused trajectory should achieve superior endpoint accuracy compared to drifting PDR
        self.assertLess(report.fused_metrics.endpoint_error_m, report.pdr_only_metrics.endpoint_error_m)
        self.assertGreater(report.fused_metrics.uncertainty_95_coverage, 0.70)


class TestSensorFusionAPIEndpoints(unittest.TestCase):
    """Integration test suite for FastAPI /datasets/{id}/fusion endpoints."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = DatasetConfig(base_dir=self.temp_dir)
        self.registry = DatasetRegistry(self.config)

        from backend.app.services.wifi_survey_service import WifiSurveyService
        from backend.app.services.dataset_ingestion import DatasetIngestionService
        from backend.app.api.routes import datasets

        self.wifi_service = WifiSurveyService(
            config=self.config,
            registry=self.registry,
            storage_base_dir=self.temp_dir / "radio_maps",
        )
        self.fusion_service = SensorFusionService(
            config=self.config,
            registry=self.registry,
            wifi_service=self.wifi_service,
        )

        datasets._ingestion_service = DatasetIngestionService(self.config, self.registry)
        datasets._wifi_survey_service = self.wifi_service
        datasets._fusion_service = self.fusion_service

        self.app = create_app()
        self.client = TestClient(self.app)

        self.dataset_id = "test-fusion-ds"
        extracted_path = self.config.extracted_path / self.dataset_id
        extracted_path.mkdir(parents=True, exist_ok=True)

        # Register dataset in registry
        self.registry.register_dataset({
            "dataset_id": self.dataset_id,
            "session_id": "SES-FUSION-01",
            "raw_filename": "survey.zip",
            "extracted_path": str(extracted_path),
            "entry_count": 5,
            "file_size_bytes": 1024,
            "manifest": {"metadata": {"session_id": "SES-FUSION-01"}},
        })

        # Create dummy PDR steps parquet
        pdr_dir = self.config.base_dir.parent / "processed" / "multimodal" / self.dataset_id / "pdr"
        pdr_dir.mkdir(parents=True, exist_ok=True)
        df_steps = pd.DataFrame([
            {"step_index": 1, "timestamp_ms": 1000, "step_length_m": 0.75, "heading_rad": 0.0, "sigma_step_length_m": 0.08, "sigma_heading_rad": 0.05},
            {"step_index": 2, "timestamp_ms": 1600, "step_length_m": 0.75, "heading_rad": 0.0, "sigma_step_length_m": 0.08, "sigma_heading_rad": 0.05},
            {"step_index": 3, "timestamp_ms": 2200, "step_length_m": 0.75, "heading_rad": 0.0, "sigma_step_length_m": 0.08, "sigma_heading_rad": 0.05},
        ])
        df_steps.to_parquet(pdr_dir / "pdr_steps.parquet", index=False)

        # Create dummy Wi-Fi observations
        wifi_df = pd.DataFrame([
            {
                "observation_id": "W-01", "session_id": "SES-FUSION-01", "original_timestamp": 1200,
                "time_source": "system", "normalized_epoch_ms": 1200, "normalized_iso": "2026-11-18T10:00:00Z",
                "sync_method": "none", "uncertainty_ms": 10, "is_fresh": True, "is_mock": False,
                "bssid": "00:11:22:33:44:01", "ssid": "AP-1", "rssi_dbm": -60.0,
                "frequency_mhz": 2412, "channel": 1, "channel_band": "2.4GHz",
                "building": "BLDG-1", "floor": 1, "room": "R101"
            }
        ])
        wifi_df.to_csv(extracted_path / "wifi_observations.csv", index=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_fusion_and_get_trajectory(self):
        # 1. Run fusion
        run_payload = {
            "site_id": "SITE-FUSION-TEST",
            "building_id": "BLDG-1",
            "floor_id": 1,
            "initial_x_m": 0.0,
            "initial_y_m": 0.0,
            "initial_heading_deg": 0.0,
            "chi2_threshold": 5.991,
            "enable_outlier_gating": True,
            "default_wifi_sigma_m": 4.0,
            "radio_map_version": "v1",
        }
        res_run = self.client.post(f"/api/datasets/{self.dataset_id}/fusion/run", json=run_payload)
        self.assertEqual(res_run.status_code, 200)
        data_run = res_run.json()
        self.assertEqual(data_run["dataset_id"], self.dataset_id)
        self.assertGreater(data_run["total_points"], 1)

        # 2. Get status
        res_status = self.client.get(f"/api/datasets/{self.dataset_id}/fusion/status")
        self.assertEqual(res_status.status_code, 200)
        self.assertTrue(res_status.json()["is_available"])

        # 3. Get trajectory
        res_traj = self.client.get(f"/api/datasets/{self.dataset_id}/fusion/trajectory?limit=100")
        self.assertEqual(res_traj.status_code, 200)
        self.assertEqual(len(res_traj.json()["trajectory"]), data_run["total_points"])

    def test_evaluate_fusion_endpoint(self):
        eval_payload = {
            "site_id": "SITE-FUSION-TEST",
            "chi2_threshold": 5.991,
            "use_synthetic_fixture": True,
        }
        res_eval = self.client.post(f"/api/datasets/{self.dataset_id}/fusion/evaluate", json=eval_payload)
        self.assertEqual(res_eval.status_code, 200)
        rep = res_eval.json()["report"]
        self.assertTrue(rep["is_synthetic_fixture"])
        self.assertIn("fused", rep)
        self.assertIn("pdr_only", rep)


if __name__ == "__main__":
    unittest.main()
