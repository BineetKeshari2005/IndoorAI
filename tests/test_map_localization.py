"""Unit and integration test suite for Task 27: Map-Constrained Multimodal Localization using a Particle Filter."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import create_app
from src.computer_vision.models import (
    LandmarkCategory,
    LandmarkRecord,
    VerificationStatus,
)
from src.map_localization.evaluation import (
    MapLocalizationEvaluationReport,
    MapLocalizationEvaluator,
    TrajectoryComparisonMetrics,
    generate_controlled_benchmark_trajectory,
)
from src.map_localization.initialization import ParticleInitializer
from src.map_localization.landmark_observation import LandmarkMeasurementModel
from src.map_localization.map_constraints import MapConstraintEvaluator
from src.map_localization.measurement_model import WifiMeasurementModel
from src.map_localization.models import (
    FilterStatus,
    ObservationType,
    ParticleFilterConfig,
    ParticleFilterState,
    ParticleFilterSummary,
    ResampleAlgorithm,
)
from src.map_localization.motion_model import PdrMotionModel
from src.map_localization.particle import Particle, wrap_heading_rad
from src.map_localization.particle_filter import MapConstrainedParticleFilter
from src.map_localization.pipeline import MapLocalizationPipeline
from src.map_localization.resampling import Resampler
from src.mapping.evaluation import generate_controlled_synthetic_building_map
from src.mapping.geometry import Point2D, Polygon2D
from src.mapping.models import BuildingMap, Door, FloorMap, MapVerificationStatus, Room
from src.sensor_fusion.coordinates import Pose2D
from src.sensor_fusion.models import FusedState, PdrStepEvent, WifiMeasurementEvent


class TestParticlePrimitives(unittest.TestCase):
    """Unit tests for particle representation and geometry helper math."""

    def test_particle_attributes_and_copy(self):
        p1 = Particle(x_m=5.0, y_m=10.0, theta_rad=math.pi / 2, floor_id=1, building_id="BLDG_A", weight=0.002)
        p2 = p1.copy()
        self.assertEqual(p1.x_m, p2.x_m)
        self.assertEqual(p1.y_m, p2.y_m)
        self.assertEqual(p1.theta_rad, p2.theta_rad)
        self.assertEqual(p1.floor_id, p2.floor_id)

        # Modifying copy should not change original
        p2.x_m = 12.0
        self.assertEqual(p1.x_m, 5.0)
        self.assertEqual(p2.x_m, 12.0)

    def test_heading_wraparound(self):
        self.assertAlmostEqual(wrap_heading_rad(0.0), 0.0)
        self.assertAlmostEqual(wrap_heading_rad(math.pi), math.pi)
        self.assertAlmostEqual(wrap_heading_rad(-math.pi), -math.pi)
        self.assertAlmostEqual(wrap_heading_rad(3 * math.pi), math.pi)
        self.assertAlmostEqual(wrap_heading_rad(-3 * math.pi), -math.pi)
        self.assertAlmostEqual(wrap_heading_rad(2 * math.pi + 0.5), 0.5)

    def test_particle_distance_calculation(self):
        p = Particle(x_m=3.0, y_m=4.0)
        self.assertAlmostEqual(p.distance_to(0.0, 0.0), 5.0)
        self.assertAlmostEqual(p.distance_to(3.0, 4.0), 0.0)


class TestParticleInitializer(unittest.TestCase):
    """Unit tests for particle initialization strategies."""

    def setUp(self):
        self.config = ParticleFilterConfig(particle_count=200, random_seed=42)
        self.initializer = ParticleInitializer(config=self.config)
        self.bmap = generate_controlled_synthetic_building_map()

    def test_initialize_from_known_location(self):
        particles = self.initializer.initialize_from_known_location(
            x_m=10.0,
            y_m=5.0,
            theta_rad=0.0,
            floor_id=0,
            building_id="SYNTH_BLDG_01",
            pos_std_m=0.5,
            building_map=self.bmap,
        )
        self.assertEqual(len(particles), 200)
        xs = [p.x_m for p in particles]
        ys = [p.y_m for p in particles]
        self.assertAlmostEqual(float(np.mean(xs)), 10.0, delta=0.2)
        self.assertAlmostEqual(float(np.mean(ys)), 5.0, delta=0.2)
        self.assertTrue(all(p.floor_id == 0 for p in particles))

    def test_initialize_from_wifi_fix(self):
        wifi = WifiMeasurementEvent(
            scan_index=1,
            timestamp_epoch_ms=1000,
            building_id="SYNTH_BLDG_01",
            floor_id=0,
            x_m=8.0,
            y_m=6.0,
            covariance=np.array([[4.0, 0.0], [0.0, 4.0]]),
        )
        particles = self.initializer.initialize_from_wifi_fix(wifi, building_map=self.bmap)
        self.assertEqual(len(particles), 200)
        xs = [p.x_m for p in particles]
        self.assertAlmostEqual(float(np.mean(xs)), 8.0, delta=0.5)

    def test_initialize_from_region(self):
        particles = self.initializer.initialize_from_region(
            min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, floor_id=0, building_id="SYNTH_BLDG_01"
        )
        self.assertEqual(len(particles), 200)
        for p in particles:
            self.assertTrue(0.0 <= p.x_m <= 10.0)
            self.assertTrue(0.0 <= p.y_m <= 10.0)


class TestPdrMotionModel(unittest.TestCase):
    """Unit tests for kinematic step propagation."""

    def setUp(self):
        self.config = ParticleFilterConfig(sigma_step_length_m=0.05, sigma_heading_rad=0.02, random_seed=42)
        self.motion_model = PdrMotionModel(config=self.config)

    def test_predict_step_forward(self):
        particles = [Particle(x_m=0.0, y_m=0.0, theta_rad=0.0, weight=1.0) for _ in range(50)]
        step = PdrStepEvent(
            step_index=1,
            timestamp_epoch_ms=1000,
            step_length_m=1.0,
            heading_rad=0.0,
            heading_delta_rad=0.0,
        )
        updated = self.motion_model.predict_step(particles, step)
        self.assertEqual(len(updated), 50)
        xs = [p.x_m for p in updated]
        ys = [p.y_m for p in updated]
        self.assertAlmostEqual(float(np.mean(xs)), 1.0, delta=0.05)
        self.assertAlmostEqual(float(np.mean(ys)), 0.0, delta=0.05)

    def test_stationary_diffusion(self):
        particles = [Particle(x_m=5.0, y_m=5.0, theta_rad=0.0, weight=1.0) for _ in range(20)]
        diffused = self.motion_model.apply_stationary_diffusion(particles, dt_s=1.0)
        xs = [p.x_m for p in diffused]
        self.assertAlmostEqual(float(np.mean(xs)), 5.0, delta=0.1)


class TestMapConstraintEvaluator(unittest.TestCase):
    """Unit tests for wall collision rejection and doorway passage."""

    def setUp(self):
        self.config = ParticleFilterConfig(
            wall_crossing_penalty_factor=0.0,
            allow_unknown_areas=False,
        )
        self.evaluator = MapConstraintEvaluator(config=self.config)

        # Create a single floor map with a wall and a door
        floor = FloorMap(
            floor_id=0,
            building_id="BLDG_TEST",
            floor_number=0,
            label="Ground Floor",
            boundary=Polygon2D.create_rectangle(min_x=0.0, min_y=0.0, max_x=20.0, max_y=20.0),
        )
        # Room inside: x in [0, 10], y in [0, 10]
        # Wall is along x=10 from y=0 to y=10
        room = Room(
            room_id="RM_01",
            name="Office",
            floor_id=0,
            building_id="BLDG_TEST",
            polygon=Polygon2D.create_rectangle(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0),
            verification_status=MapVerificationStatus.VERIFIED,
        )
        # Door opening centered at x=10, y=5
        door = Door(
            door_id="DOOR_01",
            building_id="BLDG_TEST",
            floor_id=0,
            name="Office Door",
            position=Point2D(10.0, 5.0),
            connects_from_id="RM_01",
            connects_to_id="CORR_01",
        )
        floor.rooms[room.room_id] = room
        floor.doors[door.door_id] = door

        self.bmap = BuildingMap(building_id="BLDG_TEST", name="Test Building", floors={0: floor})

    def test_solid_wall_collision_rejection(self):
        # Step from (9.0, 2.0) to (11.0, 2.0) crosses the wall at x=10, y=2 (far from door at y=5)
        old_p = [Particle(x_m=9.0, y_m=2.0, floor_id=0, weight=1.0)]
        new_p = [Particle(x_m=11.0, y_m=2.0, floor_id=0, weight=1.0)]

        constrained, rejected = self.evaluator.apply_constraints(old_p, new_p, self.bmap)
        self.assertEqual(rejected, 1)
        self.assertFalse(constrained[0].is_alive)
        self.assertEqual(constrained[0].weight, 0.0)

    def test_doorway_crossing_passage_permitted(self):
        # Step from (9.0, 5.0) to (11.0, 5.0) crosses through the door opening at x=10, y=5!
        old_p = [Particle(x_m=9.0, y_m=5.0, floor_id=0, weight=1.0)]
        new_p = [Particle(x_m=11.0, y_m=5.0, floor_id=0, weight=1.0)]

        constrained, rejected = self.evaluator.apply_constraints(old_p, new_p, self.bmap)
        self.assertEqual(rejected, 0)
        self.assertTrue(constrained[0].is_alive)
        self.assertGreater(constrained[0].weight, 0.0)

    def test_outer_floor_boundary_rejection(self):
        # Step moving outside floor boundary (e.g. x=21.0 > 20.0)
        old_p = [Particle(x_m=19.0, y_m=10.0, floor_id=0, weight=1.0)]
        new_p = [Particle(x_m=22.0, y_m=10.0, floor_id=0, weight=1.0)]

        constrained, rejected = self.evaluator.apply_constraints(old_p, new_p, self.bmap)
        self.assertEqual(rejected, 1)
        self.assertFalse(constrained[0].is_alive)


class TestMeasurementModels(unittest.TestCase):
    """Unit tests for Wi-Fi and landmark measurement updates."""

    def test_wifi_measurement_likelihood(self):
        model = WifiMeasurementModel()
        particles = [
            Particle(x_m=5.0, y_m=5.0, floor_id=0, weight=0.5),   # close
            Particle(x_m=25.0, y_m=25.0, floor_id=0, weight=0.5), # far
        ]
        wifi = WifiMeasurementEvent(
            scan_index=1,
            timestamp_epoch_ms=1000,
            building_id="BLDG_TEST",
            floor_id=0,
            x_m=5.0,
            y_m=5.0,
            covariance=np.array([[4.0, 0.0], [0.0, 4.0]]),
        )
        updated = model.update_weights(particles, wifi)
        self.assertGreater(updated[0].log_weight, updated[1].log_weight)

    def test_landmark_proximity_update(self):
        model = LandmarkMeasurementModel()
        particles = [
            Particle(x_m=10.0, y_m=10.0, floor_id=0, weight=0.5),
            Particle(x_m=30.0, y_m=30.0, floor_id=0, weight=0.5),
        ]
        # Verified landmark
        landmark = LandmarkRecord(
            landmark_id="LM_01",
            building_id="BLDG_TEST",
            floor_id=0,
            category=LandmarkCategory.DOOR,
            verified_coordinates_m=(10.0, 10.0),
            verification_status=VerificationStatus.MANUALLY_VERIFIED,
        )
        updated = model.update_weights(particles, landmark)
        self.assertGreater(updated[0].log_weight, updated[1].log_weight)

    def test_unverified_landmark_ignored(self):
        model = LandmarkMeasurementModel()
        particles = [Particle(x_m=10.0, y_m=10.0, floor_id=0, weight=0.5)]
        unverified_lm = LandmarkRecord(
            landmark_id="LM_UNVERIFIED",
            building_id="BLDG_TEST",
            floor_id=0,
            category=LandmarkCategory.DOOR,
            verified_coordinates_m=(10.0, 10.0),
            verification_status=VerificationStatus.UNVERIFIED,
        )
        updated = model.update_weights(particles, unverified_lm)
        self.assertEqual(particles[0].log_weight, updated[0].log_weight)


class TestResampling(unittest.TestCase):
    """Unit tests for systematic resampling and effective sample size."""

    def setUp(self):
        self.resampler = Resampler(config=ParticleFilterConfig(random_seed=42))

    def test_normalize_weights_log_sum_exp(self):
        particles = [
            Particle(log_weight=-10.0),
            Particle(log_weight=-10.0),
        ]
        norm, n_eff = self.resampler.normalize_weights(particles)
        self.assertAlmostEqual(norm[0].weight, 0.5)
        self.assertAlmostEqual(norm[1].weight, 0.5)
        self.assertAlmostEqual(n_eff, 2.0)

    def test_systematic_resampling_diversity(self):
        particles = [
            Particle(x_m=1.0, y_m=1.0, weight=0.9),
            Particle(x_m=10.0, y_m=10.0, weight=0.1),
        ]
        resampled, count = self.resampler.systematic_resample(particles, n_particles=100)
        self.assertEqual(len(resampled), 100)
        self.assertAlmostEqual(resampled[0].weight, 0.01)
        # Majority should be from the first particle (x=1.0)
        x_ones = sum(1 for p in resampled if p.x_m < 3.0)
        self.assertGreater(x_ones, 80)


class TestMapConstrainedParticleFilterEngine(unittest.TestCase):
    """Integration test of complete MapConstrainedParticleFilter state estimation."""

    def setUp(self):
        self.bmap = generate_controlled_synthetic_building_map()
        self.config = ParticleFilterConfig(particle_count=300, random_seed=42)
        self.pf = MapConstrainedParticleFilter(config=self.config, building_map=self.bmap)

    def test_filter_lifecycle(self):
        # 1. Initialize
        state0 = self.pf.initialize(known_location=(5.0, 5.0, 0.0, 0, "SYNTH_BLDG_01"), timestamp_epoch_ms=1000)
        self.assertEqual(state0.status, FilterStatus.INITIALIZING)
        self.assertAlmostEqual(state0.x_m, 5.0, delta=0.3)
        self.assertAlmostEqual(state0.y_m, 5.0, delta=0.3)

        # 2. Predict Step
        step = PdrStepEvent(
            step_index=1,
            timestamp_epoch_ms=1500,
            step_length_m=0.75,
            heading_rad=0.0,
            heading_delta_rad=0.0,
        )
        state1 = self.pf.predict_step(step)
        self.assertIn(state1.status, (FilterStatus.TRACKING, FilterStatus.MAP_CONSTRAINED))
        self.assertAlmostEqual(state1.x_m, 5.75, delta=0.3)

        # 3. Update Wi-Fi
        wifi = WifiMeasurementEvent(
            scan_index=1,
            timestamp_epoch_ms=2000,
            building_id="SYNTH_BLDG_01",
            floor_id=0,
            x_m=6.0,
            y_m=5.0,
            covariance=np.array([[2.0, 0.0], [0.0, 2.0]]),
        )
        state2 = self.pf.update_wifi(wifi)
        self.assertEqual(state2.observation_type, ObservationType.WIFI_FIX)
        self.assertAlmostEqual(state2.x_m, 6.0, delta=0.5)


class TestMapLocalizationEvaluator(unittest.TestCase):
    """Unit tests for comparative benchmark evaluation."""

    def test_synthetic_benchmark_evaluation(self):
        bmap, gt_points, pdr_steps, wifi_fixes, unconstrained_states = generate_controlled_benchmark_trajectory(
            step_count=30, noise_seed=42
        )

        pipeline = MapLocalizationPipeline(config=ParticleFilterConfig(particle_count=250), building_map=bmap)
        gt_first = gt_points[0]
        gt_x = gt_first["x"] if isinstance(gt_first, dict) else gt_first[1]
        gt_y = gt_first["y"] if isinstance(gt_first, dict) else gt_first[2]
        init_pose = Pose2D(x_m=float(gt_x), y_m=float(gt_y), theta_rad=0.0, floor_id=0)

        pf_states, summary = pipeline.run(
            pdr_steps=pdr_steps,
            wifi_measurements=wifi_fixes,
            initial_pose=init_pose,
            building_map=bmap,
            run_ekf_baseline=True,
        )

        evaluator = MapLocalizationEvaluator()
        report = evaluator.evaluate_against_ground_truth(
            pf_states=pf_states,
            ground_truth=gt_points,
            unconstrained_states=unconstrained_states,
            building_map=bmap,
            evaluation_id="TEST-BENCH-01",
        )

        self.assertIsInstance(report, MapLocalizationEvaluationReport)
        self.assertGreater(len(report.metrics), 0)
        self.assertIn("particle_filter", report.metrics)
        self.assertIn("unconstrained_baseline", report.metrics)

        pf_metrics = report.metrics["particle_filter"]
        self.assertGreaterEqual(pf_metrics.total_points, 10)
        self.assertLess(pf_metrics.mean_error_m, 20.0)
        self.assertGreaterEqual(report.wall_violation_reduction_pct, 0.0)


class TestMapLocalizationAPI(unittest.TestCase):
    """Integration test suite for Map Localization FastAPI endpoints."""

    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_get_map_localization_status(self):
        res = self.client.get("/api/datasets/DS-TEST-MAP/map-localization/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("is_available", data)

    def test_run_map_localization_endpoint(self):
        payload = {
            "building_id": "SYNTH_BLDG_01",
            "floor_id": 0,
            "particle_count": 150,
            "resample_threshold_ratio": 0.5,
            "sigma_step_length_m": 0.10,
            "sigma_heading_rad": 0.087,
            "wall_crossing_penalty_factor": 0.0,
            "allow_unknown_areas": True,
            "random_seed": 42,
        }
        res = self.client.post("/api/datasets/DS-TEST-MAP/map-localization/run", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_points", data)
        self.assertIn("summary", data)
        self.assertIn("trajectory", data)
        self.assertGreater(data["total_points"], 0)

        # Retrieve computed trajectory
        traj_res = self.client.get("/api/datasets/DS-TEST-MAP/map-localization/trajectory")
        self.assertEqual(traj_res.status_code, 200)
        traj_data = traj_res.json()
        self.assertEqual(len(traj_data["trajectory"]), data["total_points"])

    def test_evaluate_map_localization_endpoint(self):
        eval_payload = {
            "use_synthetic_fixture": True,
            "particle_count": 150,
            "allow_unknown_areas": True,
        }
        res = self.client.post("/api/datasets/DS-TEST-MAP/map-localization/evaluate", json=eval_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("report", data)
        self.assertIn("metrics", data["report"])


if __name__ == "__main__":
    unittest.main()
