"""Comprehensive unit and API integration test suite for the Pedestrian Dead Reckoning (PDR) Engine (Task 21)."""

import io
import json
import math
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
from backend.app.services.pdr_service import PDRService
from backend.app.services.preprocessing_service import PreprocessingService
from src.multimodal_preprocessing.config import MultimodalPreprocessingConfig
from src.multimodal_preprocessing.pipeline import MultimodalPreprocessingPipeline
from src.pdr.config import PDRConfig
from src.pdr.coordinate_frames import (
    project_to_horizontal_plane,
    quaternion_from_android_rotation_vector,
    quaternion_to_euler_angles,
    quaternion_to_rotation_matrix,
    rotate_body_to_world,
    rotate_world_to_body,
)
from src.pdr.filtering import (
    butterworth_bandpass,
    butterworth_lowpass,
    exponential_smoothing_online,
)
from src.pdr.gravity import GravityCompResult, GravityCompensator
from src.pdr.heading import HeadingEstimator, StepHeading
from src.pdr.orientation import OrientationEstimator, OrientationResult
from src.pdr.pipeline import PDRPipeline
from src.pdr.step_detection import StepDetectionStats, StepDetector
from src.pdr.step_length import StepLengthEstimator
from src.pdr.trajectory import TrajectoryReconstructor, TrajectorySummary
from src.pdr.uncertainty import PDRUncertaintyPropagator, StepCovariance
from tests.test_dataset_ingestion import build_synthetic_survey_zip


class TestPDRCoordinateFramesAndFiltering(unittest.TestCase):
    """Unit tests for coordinate frame transforms, quaternion math, and digital filters."""

    def test_quaternion_and_rotation_matrix(self):
        """Verifies quaternion normalization and SO(3) rotation matrix properties."""
        # 90 degree rotation around Z axis: q = [cos(45 deg), 0, 0, sin(45 deg)]
        angle = math.pi / 2.0
        qz = math.sin(angle / 2.0)
        qw = math.cos(angle / 2.0)
        q = np.array([qw, 0.0, 0.0, qz])

        R = quaternion_to_rotation_matrix(q)
        # R should be orthogonal: R * R^T = I
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-7)
        # Rotating vector [1, 0, 0] by 90 deg around Z should yield [0, 1, 0]
        v_in = np.array([1.0, 0.0, 0.0])
        v_out = rotate_body_to_world(v_in, R)
        np.testing.assert_allclose(v_out, [0.0, 1.0, 0.0], atol=1e-7)

        # Euler angles extraction
        roll, pitch, yaw = quaternion_to_euler_angles(q)
        self.assertAlmostEqual(roll, 0.0, places=5)
        self.assertAlmostEqual(pitch, 0.0, places=5)
        self.assertAlmostEqual(yaw, math.pi / 2.0, places=5)

    def test_horizontal_plane_projection(self):
        """Verifies decomposition of 3D acceleration into vertical and horizontal components."""
        g_unit = np.array([0.0, 0.0, -1.0])  # Gravity pointing downwards (-Z)
        v = np.array([2.0, 3.0, 4.0])        # Arbitrary vector
        v_horiz, v_vert = project_to_horizontal_plane(v, g_unit)

        # Horizontal component must lie strictly in XY plane
        np.testing.assert_allclose(v_horiz, [2.0, 3.0, 0.0], atol=1e-7)
        # Vertical component must match +Z value (upward)
        self.assertAlmostEqual(v_vert, 4.0, places=5)

    def test_signal_filtering_and_gap_reset(self):
        """Verifies exponential smoothing filter resets across sampling gaps without bleeding."""
        t_ms = np.array([1000, 1100, 1200, 2500, 2600, 2700])  # Gap between 1200 and 2500 (1300 ms)
        vals = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 100.0])

        smoothed = exponential_smoothing_online(vals, t_ms, cutoff_hz=0.5, gap_threshold_ms=500.0)
        # At gap point t=2500ms, value should reset directly to 100.0 without sluggish transition from 10.0
        self.assertEqual(smoothed[3], 100.0)


class TestPDRKinematicsAndStepDetection(unittest.TestCase):
    """Unit tests for gravity separation, step detection, step length, and heading."""

    def setUp(self):
        self.config = PDRConfig()

    def test_gravity_compensation_with_low_pass(self):
        """Verifies low-pass gravity separation extracts static ~9.8 m/s^2 from sensor stream."""
        # Simulated phone standing vertically: z=9.81, x=0, y=0 with minor 1 Hz hand tremor
        t_ms = np.arange(1000, 5000, 20)  # 50 Hz, 4 seconds
        x = np.sin(2 * np.pi * 1.0 * t_ms / 1000.0) * 0.2
        y = np.cos(2 * np.pi * 1.0 * t_ms / 1000.0) * 0.2
        z = np.ones_like(t_ms) * 9.80665

        accel_df = pd.DataFrame({
            "observation_id": [f"A-{i}" for i in range(len(t_ms))],
            "normalized_epoch_ms": t_ms,
            "x": x, "y": y, "z": z
        })

        compensator = GravityCompensator(self.config)
        res = compensator.process(accel_df)

        self.assertEqual(res.method_used, "LOW_PASS_ACCELEROMETER")
        self.assertIn("gravity_z", res.df.columns)
        self.assertIn("linear_accel_z", res.df.columns)
        # Estimated gravity magnitude should be close to 9.81 m/s^2
        mean_gz = float(res.df["gravity_z"].mean())
        self.assertAlmostEqual(mean_gz, 9.80665, delta=0.5)

    def test_synthetic_walking_sequence_step_detection(self):
        """Generates a synthetic 10-step walking signal and verifies exact detection and refractory rejection."""
        # Walking cadence: 1.8 Hz (one step every ~555 ms)
        # 10 steps = ~5.55 seconds of walking at 50 Hz sampling (278 samples)
        fs = 50.0
        duration_s = 6.0
        t_s = np.arange(0, duration_s, 1.0 / fs)
        t_ms = (t_s * 1000.0).astype(int) + 1000

        # Vertical acceleration pattern: periodic sinusoidal heel-strike bursts (1.8 Hz)
        cadence_hz = 1.8
        vert_signal = 1.6 * np.sin(2 * np.pi * cadence_hz * t_s)

        # Inject a 3.2 Hz stutter burst across 1.2s to test refractory rejection (< 330 ms)
        stutter_mask = (t_s >= 2.0) & (t_s <= 3.2)
        vert_signal[stutter_mask] = 1.6 * np.sin(2 * np.pi * 3.2 * t_s[stutter_mask])

        df = pd.DataFrame({
            "observation_id": [f"ACC-{i}" for i in range(len(t_ms))],
            "normalized_epoch_ms": t_ms,
            "vertical_accel": vert_signal,
            "x": np.zeros_like(t_s),
            "y": np.zeros_like(t_s),
            "z": vert_signal + 9.80665
        })

        detector = StepDetector(self.config)
        steps_df, stats = detector.process(df, session_id="SES-SYNTH-WALK")

        # 6.0 seconds at 1.8 Hz is approximately 10 to 11 steps
        self.assertGreaterEqual(len(steps_df), 9)
        self.assertLessEqual(len(steps_df), 12)
        # Refractory period must have rejected bounce duplicates
        self.assertGreater(stats.total_rejected_refractory, 0)
        # Mean cadence should be near 1.8 Hz * 60 = 108 SPM
        self.assertAlmostEqual(stats.mean_cadence_spm, 108.0, delta=15.0)

    def test_weinberg_step_length_estimation(self):
        """Verifies Weinberg formula L = K * (amax - amin)^0.25 and clamping to physiological bounds."""
        steps_df = pd.DataFrame([
            # Small bounce: range = 1.0 m/s^2 -> L = 0.42 * 1.0^0.25 = 0.42 m
            {"step_id": "S1", "accel_range_mps2": 1.0},
            # Normal walking bounce: range = 4.0 m/s^2 -> L = 0.42 * (4.0^0.25) = 0.42 * 1.414 = 0.594 m
            {"step_id": "S2", "accel_range_mps2": 4.0},
            # Massive acceleration spike: range = 100.0 m/s^2 -> clamped to max 1.20 m
            {"step_id": "S3", "accel_range_mps2": 100.0},
            # Near zero acceleration: range = 0.001 m/s^2 -> clamped to min 0.30 m
            {"step_id": "S4", "accel_range_mps2": 0.001},
        ])

        estimator = StepLengthEstimator(self.config)
        out_df = estimator.process(steps_df)

        self.assertAlmostEqual(out_df["step_length_m"].iloc[0], 0.42, delta=0.01)
        self.assertAlmostEqual(out_df["step_length_m"].iloc[1], 0.594, delta=0.02)
        self.assertEqual(out_df["step_length_m"].iloc[2], 1.20)
        self.assertEqual(out_df["step_length_m"].iloc[3], 0.30)

    def test_2d_trajectory_turn_and_covariance_growth(self):
        """Verifies 2D trajectory reconstruction on a right-angle path with monotonic uncertainty growth."""
        # 10 steps total:
        # Steps 1 to 5: Heading 0 rad (East, +X) with length 0.8 m
        # Steps 6 to 10: Heading pi/2 rad (North, +Y) with length 0.8 m
        t_base = 1000
        steps = []
        for i in range(10):
            steps.append({
                "step_id": f"STEP-{i+1:05d}",
                "timestamp_ms": t_base + i * 600,
                "step_length_m": 0.8,
                "step_length_uncertainty_m": 0.08,
                "confidence": 0.9,
            })
        steps_df = pd.DataFrame(steps)

        # Orientation DataFrame
        orientations = []
        for i in range(10):
            t = t_base + i * 600
            yaw = 0.0 if i < 5 else (math.pi / 2.0)
            orientations.append({
                "normalized_epoch_ms": t,
                "yaw_rad": yaw,
                "heading_source": "ROTATION_VECTOR",
                "orientation_confidence": 0.95,
                "magnetic_disturbance": False
            })
        orientation_df = pd.DataFrame(orientations)

        reconstructor = TrajectoryReconstructor(self.config)
        traj_df, summary = reconstructor.process(steps_df, orientation_df, session_id="SES-L-TURN")

        # Total distance: 10 steps * 0.8 m = 8.0 m
        self.assertAlmostEqual(summary.total_distance_m, 8.0, delta=0.1)
        # Position after step 5 (index 5): x ~ 5 * 0.8 = 4.0 m, y ~ 0.0 m
        step5_pt = traj_df[traj_df["step_id"] == "STEP-00005"].iloc[0]
        self.assertAlmostEqual(step5_pt["rel_x_m"], 4.0, delta=0.05)
        self.assertAlmostEqual(step5_pt["rel_y_m"], 0.0, delta=0.05)

        # Final position after step 10: x ~ 4.0 m, y ~ 5 * 0.8 = 4.0 m
        final_pt = traj_df.iloc[-1]
        self.assertAlmostEqual(final_pt["rel_x_m"], 4.0, delta=0.05)
        self.assertAlmostEqual(final_pt["rel_y_m"], 4.0, delta=0.05)

        # Monotonically increasing uncertainty radius
        uncertainties = traj_df["pos_uncertainty_radius_m"].values
        for i in range(1, len(uncertainties)):
            self.assertGreaterEqual(uncertainties[i], uncertainties[i - 1])


def build_synthetic_walking_survey_zip(
    session_id: str = "SES-A1B2C3D4",
    session_name: str = "PDR Walking Survey",
    duration_s: float = 6.0,
    fs_hz: float = 50.0,
    cadence_hz: float = 1.8
) -> bytes:
    """Builds an in-memory survey ZIP containing sinusoidal walking acceleration and rotation vectors."""
    import csv
    import hashlib
    import zipfile
    from tests.test_dataset_ingestion import (
        EXPECTED_ANNOTATION_HEADER,
        EXPECTED_IMU_HEADER,
        EXPECTED_PHOTO_HEADER,
        EXPECTED_WIFI_HEADER,
        create_test_image_bytes,
    )

    buf = io.BytesIO()
    timestamp_ms = 1731855600000

    clock_anchor = {
        "wallClockEpochMs": timestamp_ms,
        "bootTimeNs": 123456789000,
        "anchorIsoUtc": "2026-11-17T15:00:00.000Z"
    }

    n_samples = int(duration_s * fs_hz)
    t_s = np.arange(0, duration_s, 1.0 / fs_hz)

    # 1. Wi-Fi
    wifi_buf = io.StringIO()
    wifi_writer = csv.writer(wifi_buf)
    wifi_writer.writerow(EXPECTED_WIFI_HEADER)
    wifi_writer.writerow([
        "WIFI-001", session_id, "123456789000", "CLOCK_BOOTTIME_NANOS",
        str(timestamp_ms), "2026-11-17T15:00:00.000Z", "BOOTTIME_OFFSET", "30.0",
        "true", "false", "00:11:22:33:44:55", "IndoorAI-AP1",
        "-65", "2412", "1", "2.4GHz", "BLDG-01", "FL-02", "LAB-201"
    ])
    wifi_content = wifi_buf.getvalue().encode("utf-8")

    # 2. IMU observations (walking accelerometer)
    imu_buf = io.StringIO()
    imu_writer = csv.writer(imu_buf)
    imu_writer.writerow(EXPECTED_IMU_HEADER)

    accel_z = 9.80665 + 1.8 * np.sin(2 * np.pi * cadence_hz * t_s)
    accel_x = 0.2 * np.cos(2 * np.pi * cadence_hz * t_s)
    accel_y = 0.1 * np.sin(2 * np.pi * cadence_hz * t_s)

    rec_idx = 0
    for i in range(n_samples):
        t_sample_ms = timestamp_ms + int(t_s[i] * 1000.0)
        t_sample_ns = 123456789000 + int(t_s[i] * 1_000_000_000.0)

        # Accel
        rec_idx += 1
        az = accel_z[i]
        ax = accel_x[i]
        ay = accel_y[i]
        mag = float(np.sqrt(ax**2 + ay**2 + az**2))
        imu_writer.writerow([
            f"IMU-REC-{rec_idx:04d}", session_id, str(t_sample_ns), "CLOCK_BOOTTIME_NANOS",
            str(t_sample_ms), "2026-11-17T15:00:00.000Z", "BOOTTIME_OFFSET", "3.0",
            "ACCELEROMETER", f"{ax:.4f}", f"{ay:.4f}", f"{az:.4f}", f"{mag:.4f}",
            "SENSOR_STATUS_ACCURACY_HIGH", "SENSOR_DELAY_GAME", "false"
        ])

        # Rotation vector (facing North: yaw=0)
        rec_idx += 1
        imu_writer.writerow([
            f"IMU-REC-{rec_idx:04d}", session_id, str(t_sample_ns), "CLOCK_BOOTTIME_NANOS",
            str(t_sample_ms), "2026-11-17T15:00:00.000Z", "BOOTTIME_OFFSET", "3.0",
            "ROTATION_VECTOR", "0.0", "0.0", "0.0", "1.0",
            "SENSOR_STATUS_ACCURACY_HIGH", "SENSOR_DELAY_GAME", "false"
        ])

    imu_content = imu_buf.getvalue().encode("utf-8")

    # 3. Photo
    photo_buf = io.StringIO()
    photo_writer = csv.writer(photo_buf)
    photo_writer.writerow(EXPECTED_PHOTO_HEADER)
    img_data = create_test_image_bytes()
    photo_writer.writerow([
        "IMG-REC-001", session_id, str(timestamp_ms), "CLOCK_REALTIME_MILLIS",
        str(timestamp_ms), "2026-11-17T15:00:00.000Z", "DIRECT_REALTIME", "50.0",
        "photo_001.jpg", "/app_storage/photo_001.jpg", str(len(img_data)), "100", "100",
        "BLDG-01", "FL-02", "LAB-201", "CORRIDOR", "Corridor",
        "North corridor test", "", "true", "0", "false",
        "NO_FACES_DETECTED", "false"
    ])
    photo_content = photo_buf.getvalue().encode("utf-8")

    # 4. Annotations
    ann_buf = io.StringIO()
    ann_writer = csv.writer(ann_buf)
    ann_writer.writerow(EXPECTED_ANNOTATION_HEADER)
    ann_writer.writerow([
        "ANN-REC-001", session_id, str(timestamp_ms), "CLOCK_REALTIME_MILLIS",
        str(timestamp_ms), "2026-11-17T15:00:00.000Z", "DIRECT_REALTIME", "50.0",
        "CORRIDOR", "Corridor", "Walking corridor test", "BLDG-01", "FL-02", "LAB-201",
        "", "false"
    ])
    ann_content = ann_buf.getvalue().encode("utf-8")

    # 5. session.json
    session_dict = {
        "sessionId": session_id,
        "sessionName": session_name,
        "createdAtMs": timestamp_ms,
        "startTimeMs": timestamp_ms,
        "endTimeMs": timestamp_ms + int(duration_s * 1000),
        "isMock": False,
        "deviceModel": "Pixel 7 Pro",
        "appVersion": "1.0.0",
        "clockSynchronizationAnchor": clock_anchor,
        "buildingId": "BLDG-01",
        "floorId": "FL-02",
        "notes": "Synthetic walking survey for PDR integration tests",
        "totalObservations": 1 + rec_idx + 1 + 1,
        "totalRealObservations": 1 + rec_idx + 1 + 1,
        "totalMockObservations": 0,
        "exportFormatVersion": "1.0.0",
        "integrityHash": "dummy"
    }
    session_content = json.dumps(session_dict, indent=2).encode("utf-8")

    file_payloads = {
        "session.json": session_content,
        "wifi_observations.csv": wifi_content,
        "imu_observations.csv": imu_content,
        "photo_metadata.csv": photo_content,
        "annotations.csv": ann_content,
        "images/photo_001.jpg": img_data,
    }

    manifest_files = []
    for fn, data in file_payloads.items():
        sha = hashlib.sha256(data).hexdigest()
        manifest_files.append({"filename": fn, "sizeBytes": len(data), "sha256": sha})

    manifest_dict = {
        "exportVersion": "1.0.0",
        "sessionId": session_id,
        "exportedAtMs": timestamp_ms + int(duration_s * 1000) + 5000,
        "exportedAtIso": "2026-11-17T15:01:05.000Z",
        "recordCounts": {
            "wifiCount": 1,
            "imuCount": rec_idx,
            "photoCount": 1,
            "annotationCount": 1,
            "totalObservations": 1 + rec_idx + 1 + 1
        },
        "files": manifest_files
    }
    manifest_content = json.dumps(manifest_dict, indent=2).encode("utf-8")
    file_payloads["manifest.json"] = manifest_content

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn, payload in file_payloads.items():
            zf.writestr(fn, payload)

    return buf.getvalue()


class TestPDRPipelineAndFastAPI(unittest.TestCase):
    """Integration tests for end-to-end PDR pipeline, artifacts, and REST endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="test_pdr_pipeline_")
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
            get_pdr_service,
            get_preprocessing_service,
        )
        cls.app.dependency_overrides[get_ingestion_service] = lambda: cls.ingestion_service
        cls.prep_service = PreprocessingService(config=cls.config, registry=cls.registry)
        cls.app.dependency_overrides[get_preprocessing_service] = lambda: cls.prep_service
        cls.pdr_service = PDRService(config=cls.config, registry=cls.registry)
        cls.app.dependency_overrides[get_pdr_service] = lambda: cls.pdr_service
        cls.test_client = TestClient(cls.app)

        # Ingest a valid survey ZIP with walking IMU observations
        zip_bytes = build_synthetic_walking_survey_zip(
            session_id="SES-A1B2C3D4",
            session_name="PDR Integration Walking Survey",
            duration_s=6.0,
            fs_hz=50.0,
            cadence_hz=1.8
        )
        cls.dataset_id, cls.report, cls.entry = cls.ingestion_service.ingest_dataset_zip(
            io.BytesIO(zip_bytes),
            original_filename="pdr_walking_survey.zip"
        )

        # Run Task 20 preprocessing first to generate sensor_cleaned.parquet
        cls.prep_service.preprocess_dataset(cls.dataset_id)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_pdr_pipeline_artifacts_and_manifest(self):
        """Verifies full PDR pipeline generates all 5 Parquet files, report, and manifest."""
        processed_dir = self.config.base_dir.parent / "processed" / "multimodal" / self.dataset_id
        pipeline_cfg = PDRConfig(
            base_output_dir=self.config.base_dir.parent / "processed" / "multimodal"
        )
        pipeline = PDRPipeline(pipeline_cfg)
        report, manifest = pipeline.run(self.dataset_id, processed_dir=processed_dir)

        self.assertIn(report.status, ["COMPLETED", "WARNINGS"])
        pdr_out_dir = processed_dir / "pdr"

        # Verify all 5 Parquet files exist
        self.assertTrue((pdr_out_dir / "gravity_compensated.parquet").is_file())
        self.assertTrue((pdr_out_dir / "orientation_estimates.parquet").is_file())
        self.assertTrue((pdr_out_dir / "detected_steps.parquet").is_file())
        self.assertTrue((pdr_out_dir / "step_features.parquet").is_file())
        self.assertTrue((pdr_out_dir / "relative_trajectory.parquet").is_file())
        self.assertTrue((pdr_out_dir / "pdr_report.json").is_file())
        self.assertTrue((pdr_out_dir / "pdr_manifest.json").is_file())

        # Verify reading Parquet outputs
        traj_df = pd.read_parquet(pdr_out_dir / "relative_trajectory.parquet")
        self.assertIn("rel_x_m", traj_df.columns)
        self.assertIn("rel_y_m", traj_df.columns)

    def test_fastapi_pdr_endpoints(self):
        """Verifies POST /pdr, GET /pdr, GET /pdr/trajectory, and GET /pdr/steps."""
        # 1. Trigger PDR execution via API
        req_payload = {
            "step_detection_signal": "vertical",
            "weinberg_k": 0.45,
            "min_step_interval_s": 0.30
        }
        res_post = self.test_client.post(f"/api/datasets/{self.dataset_id}/pdr", json=req_payload)
        self.assertEqual(res_post.status_code, 200)
        data_post = res_post.json()
        self.assertEqual(data_post["dataset_id"], self.dataset_id)
        self.assertIn("report", data_post)

        # 2. Query PDR status
        res_status = self.test_client.get(f"/api/datasets/{self.dataset_id}/pdr")
        self.assertEqual(res_status.status_code, 200)
        data_status = res_status.json()
        self.assertTrue(data_status["is_computed"])
        self.assertIsNotNone(data_status["report"])

        # 3. Query trajectory points
        res_traj = self.test_client.get(f"/api/datasets/{self.dataset_id}/pdr/trajectory")
        self.assertEqual(res_traj.status_code, 200)
        data_traj = res_traj.json()
        self.assertIn("points", data_traj)
        self.assertGreaterEqual(data_traj["total_points"], 1)

        # 4. Query step records
        res_steps = self.test_client.get(f"/api/datasets/{self.dataset_id}/pdr/steps")
        self.assertEqual(res_steps.status_code, 200)
        data_steps = res_steps.json()
        self.assertIn("steps", data_steps)

        # 5. Query full PDR report
        res_rep = self.test_client.get(f"/api/datasets/{self.dataset_id}/pdr/report")
        self.assertEqual(res_rep.status_code, 200)
        data_rep = res_rep.json()
        self.assertIn("movement_summary", data_rep)


if __name__ == "__main__":
    unittest.main()
