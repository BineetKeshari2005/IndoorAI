"""Comprehensive test suite for site-specific Wi-Fi radio map and fingerprint localization (Task 23)."""

from __future__ import annotations

import json
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
from backend.app.services.wifi_survey_service import WifiSurveyService
from src.wifi_localization.evaluator import WifiLocalizationEvaluator
from src.wifi_localization.knn_localizer import KnnLocalizer
from src.wifi_localization.models import (
    ApObservation,
    ApStatistics,
    EvaluationMetrics,
    LocalizationPrediction,
    LocationRadioProfile,
    RadioMap,
    RadioMapMetadata,
    ReferenceLocation,
    SurveySweep,
)
from src.wifi_localization.normalization import (
    classify_channel_band,
    clean_ap_observation,
    deduplicate_sweep_observations,
    is_valid_rssi,
    normalize_bssid,
)
from src.wifi_localization.probabilistic_localizer import ProbabilisticLocalizer
from src.wifi_localization.radio_map_builder import RadioMapBuilder
from src.wifi_localization.storage import (
    delete_radio_map,
    list_radio_maps,
    load_radio_map,
    load_reference_locations,
    save_radio_map,
    save_reference_locations,
)


class TestWifiNormalization(unittest.TestCase):
    """Test BSSID sanitization, RSSI bounds checking, and sweep deduplication."""

    def test_normalize_bssid(self):
        self.assertEqual(normalize_bssid("00:1A:2B:3C:4D:5E"), "00:1a:2b:3c:4d:5e")
        self.assertEqual(normalize_bssid("00-1a-2b-3c-4d-5e"), "00:1a:2b:3c:4d:5e")
        self.assertEqual(normalize_bssid("AA:BB:CC:DD:EE:FF"), "aa:bb:cc:dd:ee:ff")
        # Invalid MAC formats
        self.assertIsNone(normalize_bssid("invalid-mac"))
        self.assertIsNone(normalize_bssid(""))
        self.assertIsNone(normalize_bssid("00:11:22:33:44"))
        self.assertIsNone(normalize_bssid("00:11:22:33:44:55:66"))

    def test_is_valid_rssi(self):
        self.assertTrue(is_valid_rssi(-65.0))
        self.assertTrue(is_valid_rssi(-105.0))
        self.assertTrue(is_valid_rssi(-20.0))
        # Out of bounds
        self.assertFalse(is_valid_rssi(-115.0))
        self.assertFalse(is_valid_rssi(-10.0))
        self.assertFalse(is_valid_rssi(100.0))  # Corrupt positive value
        self.assertFalse(is_valid_rssi(None))
        self.assertFalse(is_valid_rssi("not-a-number"))

    def test_classify_channel_band(self):
        self.assertEqual(classify_channel_band(2412), "2.4GHz")
        self.assertEqual(classify_channel_band(2462), "2.4GHz")
        self.assertEqual(classify_channel_band(5180), "5GHz")
        self.assertEqual(classify_channel_band(5825), "5GHz")
        self.assertEqual(classify_channel_band(6100), "6GHz")
        self.assertEqual(classify_channel_band(900), "unknown")
        self.assertEqual(classify_channel_band(None), "unknown")

    def test_clean_ap_observation(self):
        raw_valid = {
            "bssid": "00:11:22:33:44:55",
            "rssi_dbm": -68.5,
            "ssid": "Campus-Secure",
            "frequency_mhz": 5200,
        }
        obs = clean_ap_observation(raw_valid)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.bssid, "00:11:22:33:44:55")
        self.assertEqual(obs.rssi_dbm, -68.5)
        self.assertEqual(obs.channel_band, "5GHz")

        # Invalid RSSI rejected
        self.assertIsNone(clean_ap_observation({"bssid": "00:11:22:33:44:55", "rssi_dbm": 50}))
        # Invalid BSSID rejected
        self.assertIsNone(clean_ap_observation({"bssid": "bad", "rssi_dbm": -70}))

    def test_deduplicate_sweep_observations(self):
        obs1 = ApObservation(bssid="aa:bb:cc:dd:ee:01", rssi_dbm=-75.0)
        obs2 = ApObservation(bssid="aa:bb:cc:dd:ee:01", rssi_dbm=-65.0)  # Stronger
        obs3 = ApObservation(bssid="aa:bb:cc:dd:ee:02", rssi_dbm=-80.0)

        deduped = deduplicate_sweep_observations([obs1, obs2, obs3])
        self.assertEqual(len(deduped), 2)
        by_bssid = {o.bssid: o.rssi_dbm for o in deduped}
        self.assertEqual(by_bssid["aa:bb:cc:dd:ee:01"], -65.0)
        self.assertEqual(by_bssid["aa:bb:cc:dd:ee:02"], -80.0)


class TestRadioMapBuilder(unittest.TestCase):
    """Test multi-sweep aggregation, visibility rates, and distribution statistics."""

    def setUp(self):
        self.site_id = "SITE-TEST"
        self.loc1 = ReferenceLocation(
            location_id="LOC-01",
            site_id=self.site_id,
            building_id="BLDG-A",
            floor_id=1,
            x_m=10.0,
            y_m=20.0,
            room_id="ROOM-101",
        )
        self.loc2 = ReferenceLocation(
            location_id="LOC-02",
            site_id=self.site_id,
            building_id="BLDG-A",
            floor_id=1,
            x_m=30.0,
            y_m=20.0,
            room_id="ROOM-102",
        )
        self.locations = {"LOC-01": self.loc1, "LOC-02": self.loc2}

    def test_group_records_into_sweeps(self):
        builder = RadioMapBuilder(site_id=self.site_id)
        records = [
            # Sweep 1 at LOC-01
            {"session_id": "SES-1", "location_id": "LOC-01", "timestamp_epoch_ms": 1000, "bssid": "00:11:22:33:44:01", "rssi_dbm": -60.0},
            {"session_id": "SES-1", "location_id": "LOC-01", "timestamp_epoch_ms": 1050, "bssid": "00:11:22:33:44:02", "rssi_dbm": -70.0},
            # Sweep 2 at LOC-01 (>500ms gap)
            {"session_id": "SES-1", "location_id": "LOC-01", "timestamp_epoch_ms": 2000, "bssid": "00:11:22:33:44:01", "rssi_dbm": -62.0},
            # Sweep 3 at LOC-02 (different location)
            {"session_id": "SES-1", "location_id": "LOC-02", "timestamp_epoch_ms": 2100, "bssid": "00:11:22:33:44:03", "rssi_dbm": -55.0},
        ]
        sweeps = builder.group_raw_records_into_sweeps(records, sweep_gap_ms=500)
        self.assertEqual(len(sweeps), 3)
        self.assertEqual(sweeps[0].location_id, "LOC-01")
        self.assertEqual(len(sweeps[0].observations), 2)
        self.assertEqual(sweeps[1].location_id, "LOC-01")
        self.assertEqual(len(sweeps[1].observations), 1)
        self.assertEqual(sweeps[2].location_id, "LOC-02")

    def test_build_radio_map_aggregation(self):
        builder = RadioMapBuilder(site_id=self.site_id, version="v1")
        # 3 sweeps at LOC-01
        sw1 = SurveySweep(
            sweep_id="SW-1", session_id="S1", location_id="LOC-01", timestamp_epoch_ms=1000,
            observations=[
                ApObservation("00:11:22:33:44:01", -60.0),
                ApObservation("00:11:22:33:44:02", -70.0),
            ]
        )
        sw2 = SurveySweep(
            sweep_id="SW-2", session_id="S1", location_id="LOC-01", timestamp_epoch_ms=2000,
            observations=[
                ApObservation("00:11:22:33:44:01", -64.0),
                ApObservation("00:11:22:33:44:02", -72.0),
            ]
        )
        sw3 = SurveySweep(
            sweep_id="SW-3", session_id="S1", location_id="LOC-01", timestamp_epoch_ms=3000,
            observations=[
                ApObservation("00:11:22:33:44:01", -62.0),
                # AP 02 missing in sweep 3
            ]
        )

        radio_map = builder.build_from_sweeps([sw1, sw2, sw3], self.locations)

        self.assertIn("LOC-01", radio_map.profiles)
        prof = radio_map.profiles["LOC-01"]
        self.assertEqual(prof.total_sweeps, 3)

        # AP 01 was seen in all 3 sweeps
        stat1 = prof.ap_stats["00:11:22:33:44:01"]
        self.assertEqual(stat1.observation_count, 3)
        self.assertEqual(stat1.visibility_rate, 1.0)
        self.assertAlmostEqual(stat1.mean_rssi, -62.0, places=1)
        self.assertAlmostEqual(stat1.median_rssi, -62.0, places=1)

        # AP 02 was seen in 2 of 3 sweeps
        stat2 = prof.ap_stats["00:11:22:33:44:02"]
        self.assertEqual(stat2.observation_count, 2)
        self.assertAlmostEqual(stat2.visibility_rate, 2 / 3, places=2)
        self.assertAlmostEqual(stat2.mean_rssi, -71.0, places=1)

        # Metadata
        self.assertEqual(radio_map.metadata.total_locations, 1)
        self.assertEqual(radio_map.metadata.unique_bssids, 2)


class TestLocalizationEngines(unittest.TestCase):
    """Test Weighted k-NN and Probabilistic localizers against known synthetic radio maps."""

    def setUp(self):
        self.site_id = "SITE-CAMPUS"
        self.loc_a = ReferenceLocation("LOC-A", self.site_id, "BLDG-1", 1, 0.0, 0.0)
        self.loc_b = ReferenceLocation("LOC-B", self.site_id, "BLDG-1", 1, 10.0, 0.0)
        self.loc_c = ReferenceLocation("LOC-C", self.site_id, "BLDG-1", 2, 0.0, 10.0)

        # Profiles
        prof_a = LocationRadioProfile(
            location_id="LOC-A", site_id=self.site_id, building_id="BLDG-1", floor_id=1, x_m=0.0, y_m=0.0, total_sweeps=5,
            ap_stats={
                "00:00:00:00:00:01": ApStatistics("00:00:00:00:00:01", 5, 5, 1.0, -50.0, -50.0, 2.0, -52.0, -48.0),
                "00:00:00:00:00:02": ApStatistics("00:00:00:00:00:02", 5, 5, 1.0, -80.0, -80.0, 2.0, -82.0, -78.0),
            },
            mean_rssi_vector={"00:00:00:00:00:01": -50.0, "00:00:00:00:00:02": -80.0},
        )
        prof_b = LocationRadioProfile(
            location_id="LOC-B", site_id=self.site_id, building_id="BLDG-1", floor_id=1, x_m=10.0, y_m=0.0, total_sweeps=5,
            ap_stats={
                "00:00:00:00:00:01": ApStatistics("00:00:00:00:00:01", 5, 5, 1.0, -80.0, -80.0, 2.0, -82.0, -78.0),
                "00:00:00:00:00:02": ApStatistics("00:00:00:00:00:02", 5, 5, 1.0, -50.0, -50.0, 2.0, -52.0, -48.0),
            },
            mean_rssi_vector={"00:00:00:00:00:01": -80.0, "00:00:00:00:00:02": -50.0},
        )
        prof_c = LocationRadioProfile(
            location_id="LOC-C", site_id=self.site_id, building_id="BLDG-1", floor_id=2, x_m=0.0, y_m=10.0, total_sweeps=5,
            ap_stats={
                "00:00:00:00:00:03": ApStatistics("00:00:00:00:00:03", 5, 5, 1.0, -45.0, -45.0, 2.0, -47.0, -43.0),
            },
            mean_rssi_vector={"00:00:00:00:00:03": -45.0},
        )

        meta = RadioMapMetadata(self.site_id, "v1", "2026-11-18T00:00:00Z", 3, 15, 3, missing_penalty_rssi=-105.0)
        self.radio_map = RadioMap(
            metadata=meta,
            locations={"LOC-A": self.loc_a, "LOC-B": self.loc_b, "LOC-C": self.loc_c},
            profiles={"LOC-A": prof_a, "LOC-B": prof_b, "LOC-C": prof_c},
            all_bssids=["00:00:00:00:00:01", "00:00:00:00:00:02", "00:00:00:00:00:03"],
        )

    def test_knn_exact_match(self):
        localizer = KnnLocalizer(self.radio_map, k=1)
        # Query near LOC-A
        query = {"00:00:00:00:00:01": -51.0, "00:00:00:00:00:02": -79.0}
        pred = localizer.localize(query)

        self.assertEqual(pred.predicted_building_id, "BLDG-1")
        self.assertEqual(pred.predicted_floor_id, 1)
        self.assertAlmostEqual(pred.predicted_x_m, 0.0, delta=1.0)
        self.assertAlmostEqual(pred.predicted_y_m, 0.0, delta=1.0)
        self.assertFalse(pred.is_low_confidence)
        self.assertGreater(pred.confidence, 0.5)

    def test_knn_interpolation(self):
        localizer = KnnLocalizer(self.radio_map, k=2)
        # Query midway between LOC-A (x=0) and LOC-B (x=10)
        query = {"00:00:00:00:00:01": -65.0, "00:00:00:00:00:02": -65.0}
        pred = localizer.localize(query)

        self.assertEqual(pred.predicted_floor_id, 1)
        # Coordinates should interpolate between 0 and 10 (~5.0)
        self.assertAlmostEqual(pred.predicted_x_m, 5.0, delta=1.0)
        self.assertAlmostEqual(pred.predicted_y_m, 0.0, delta=1.0)

    def test_knn_missing_ap_handling(self):
        localizer = KnnLocalizer(self.radio_map, k=1)
        # AP 03 belongs to LOC-C on floor 2
        query = {"00:00:00:00:00:03": -46.0}
        pred = localizer.localize(query)
        self.assertEqual(pred.predicted_floor_id, 2)
        self.assertEqual(pred.candidate_matches[0].location_id, "LOC-C")

    def test_probabilistic_localizer(self):
        localizer = ProbabilisticLocalizer(self.radio_map, k=1)
        # Query matching LOC-B
        query = {"00:00:00:00:00:01": -79.0, "00:00:00:00:00:02": -51.0}
        pred = localizer.localize(query)

        self.assertEqual(pred.predicted_building_id, "BLDG-1")
        self.assertEqual(pred.predicted_floor_id, 1)
        self.assertAlmostEqual(pred.predicted_x_m, 10.0, delta=1.0)
        self.assertEqual(pred.candidate_matches[0].location_id, "LOC-B")

    def test_low_confidence_and_empty_query(self):
        localizer = KnnLocalizer(self.radio_map, k=1)
        # Completely unknown AP
        query = {"ff:ff:ff:ff:ff:ff": -60.0}
        pred = localizer.localize(query)
        self.assertTrue(pred.is_low_confidence)
        self.assertTrue(len(pred.warnings) > 0)

        # Empty query
        pred_empty = localizer.localize({})
        self.assertTrue(pred_empty.is_low_confidence)


class TestSessionLevelEvaluator(unittest.TestCase):
    """Test leakage-resistant evaluation under strict session separation."""

    def setUp(self):
        self.site_id = "SITE-EVAL"
        self.loc1 = ReferenceLocation("LOC-1", self.site_id, "BLDG-1", 1, 0.0, 0.0)
        self.loc2 = ReferenceLocation("LOC-2", self.site_id, "BLDG-1", 1, 20.0, 0.0)
        self.locations = {"LOC-1": self.loc1, "LOC-2": self.loc2}

        # Session A sweeps (TRAIN)
        self.sw_a1 = SurveySweep("SW-A1", "SES-TRAIN", "LOC-1", 1000, [ApObservation("00:11:22:33:44:01", -50.0)])
        self.sw_a2 = SurveySweep("SW-A2", "SES-TRAIN", "LOC-2", 2000, [ApObservation("00:11:22:33:44:02", -50.0)])

        # Session B sweeps (TEST)
        self.sw_b1 = SurveySweep("SW-B1", "SES-TEST", "LOC-1", 3000, [ApObservation("00:11:22:33:44:01", -52.0)])
        self.sw_b2 = SurveySweep("SW-B2", "SES-TEST", "LOC-2", 4000, [ApObservation("00:11:22:33:44:02", -51.0)])

        self.all_sweeps = [self.sw_a1, self.sw_a2, self.sw_b1, self.sw_b2]

    def test_strict_session_separation_no_leakage(self):
        evaluator = WifiLocalizationEvaluator(algorithm="weighted_knn", k=1)
        metrics = evaluator.evaluate_session_split(
            sweeps=self.all_sweeps,
            locations=self.locations,
            train_session_ids=["SES-TRAIN"],
            test_session_ids=["SES-TEST"],
            site_id=self.site_id,
        )

        self.assertEqual(metrics.total_test_samples, 2)
        self.assertEqual(metrics.building_accuracy, 1.0)
        self.assertEqual(metrics.floor_accuracy, 1.0)
        self.assertEqual(metrics.location_id_accuracy, 1.0)
        self.assertAlmostEqual(metrics.mean_coordinate_error_m, 0.0, delta=1.0)

    def test_data_leakage_detection(self):
        evaluator = WifiLocalizationEvaluator()
        # Same session in both train and test must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            evaluator.evaluate_session_split(
                sweeps=self.all_sweeps,
                locations=self.locations,
                train_session_ids=["SES-A"],
                test_session_ids=["SES-A", "SES-B"],
            )
        self.assertIn("Data leakage detected", str(ctx.exception))


class TestRadioMapStorage(unittest.TestCase):
    """Test filesystem persistence, JSON serialization, and retrieval."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.site_id = "SITE-STORE"
        self.loc = ReferenceLocation("LOC-1", self.site_id, "B1", 1, 5.0, 5.0)
        self.meta = RadioMapMetadata(self.site_id, "v1", "2026-11-18T00:00:00Z", 1, 2, 1)
        self.radio_map = RadioMap(metadata=self.meta, locations={"LOC-1": self.loc}, profiles={}, all_bssids=["00:11:22:33:44:55"])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_radio_map(self):
        save_path = save_radio_map(self.radio_map, base_dir=self.temp_dir)
        self.assertTrue((save_path / "radio_map.json").is_file())

        loaded = load_radio_map(self.site_id, "v1", base_dir=self.temp_dir)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.metadata.site_id, self.site_id)
        self.assertIn("LOC-1", loaded.locations)

        maps_list = list_radio_maps(base_dir=self.temp_dir)
        self.assertEqual(len(maps_list), 1)

        deleted = delete_radio_map(self.site_id, "v1", base_dir=self.temp_dir)
        self.assertTrue(deleted)
        self.assertIsNone(load_radio_map(self.site_id, "v1", base_dir=self.temp_dir))


class TestWifiSurveyEndpoints(unittest.TestCase):
    """Integration tests for FastAPI datasets Wi-Fi survey endpoints."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = DatasetConfig(base_dir=self.temp_dir)
        self.registry = DatasetRegistry(self.config)
        self.wifi_service = WifiSurveyService(
            config=self.config,
            registry=self.registry,
            storage_base_dir=self.temp_dir / "radio_maps"
        )

        app = create_app()
        # Override dependency
        from backend.app.api.routes import datasets
        datasets._wifi_survey_service = self.wifi_service
        self.client = TestClient(app)

        # Create dummy survey dataset with wifi_observations.csv
        self.dataset_id = "test-survey-dataset"
        extracted_path = self.config.extracted_path / self.dataset_id
        extracted_path.mkdir(parents=True, exist_ok=True)

        wifi_df = pd.DataFrame([
            {
                "observation_id": "W-01", "session_id": "SES-101", "original_timestamp": 1000,
                "time_source": "system", "normalized_epoch_ms": 1000, "normalized_iso": "2026-11-18T10:00:00Z",
                "sync_method": "none", "uncertainty_ms": 10, "is_fresh": True, "is_mock": False,
                "bssid": "00:11:22:33:44:01", "ssid": "TestAP-1", "rssi_dbm": -60.0,
                "frequency_mhz": 2412, "channel": 1, "channel_band": "2.4GHz",
                "building": "BLDG-1", "floor": 1, "room": "R101"
            },
            {
                "observation_id": "W-02", "session_id": "SES-101", "original_timestamp": 1050,
                "time_source": "system", "normalized_epoch_ms": 1050, "normalized_iso": "2026-11-18T10:00:00Z",
                "sync_method": "none", "uncertainty_ms": 10, "is_fresh": True, "is_mock": False,
                "bssid": "00:11:22:33:44:02", "ssid": "TestAP-2", "rssi_dbm": -75.0,
                "frequency_mhz": 5180, "channel": 36, "channel_band": "5GHz",
                "building": "BLDG-1", "floor": 1, "room": "R101"
            },
            # Second sweep in session 102 (for evaluation)
            {
                "observation_id": "W-03", "session_id": "SES-102", "original_timestamp": 5000,
                "time_source": "system", "normalized_epoch_ms": 5000, "normalized_iso": "2026-11-18T10:01:00Z",
                "sync_method": "none", "uncertainty_ms": 10, "is_fresh": True, "is_mock": False,
                "bssid": "00:11:22:33:44:01", "ssid": "TestAP-1", "rssi_dbm": -62.0,
                "frequency_mhz": 2412, "channel": 1, "channel_band": "2.4GHz",
                "building": "BLDG-1", "floor": 1, "room": "R101"
            },
        ])
        wifi_df.to_csv(extracted_path / "wifi_observations.csv", index=False)

        # Register in catalog
        self.registry.register_dataset({
            "dataset_id": self.dataset_id,
            "session_id": "SES-101",
            "raw_filename": "survey.zip",
            "extracted_path": str(extracted_path),
            "entry_count": 3,
            "file_size_bytes": 1024,
            "manifest": {"metadata": {"session_id": "SES-101"}},
        })

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_and_list_locations(self):
        payload = {
            "location_id": "LOC-TEST-01",
            "site_id": "SITE-UNIT-TEST",
            "building_id": "BLDG-1",
            "floor_id": 1,
            "x_m": 12.5,
            "y_m": 25.0,
            "room_id": "R101",
            "description": "Hallway junction",
            "is_verified": True,
            "metadata": {},
        }
        res = self.client.post(f"/api/datasets/{self.dataset_id}/wifi-survey/locations", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["location_id"], "LOC-TEST-01")

        list_res = self.client.get(f"/api/datasets/{self.dataset_id}/wifi-survey/locations?site_id=SITE-UNIT-TEST")
        self.assertEqual(list_res.status_code, 200)
        self.assertEqual(list_res.json()["total_locations"], 1)

    def test_build_radio_map_and_localize(self):
        # 1. Build radio map
        build_payload = {
            "site_id": "SITE-BUILD-TEST",
            "version": "v1",
            "missing_penalty_rssi": -105.0,
            "min_sweeps_per_location": 1,
            "min_visibility_rate": 0.0,
            "auto_register_locations": True,
        }
        b_res = self.client.post(f"/api/datasets/{self.dataset_id}/wifi-survey/fingerprints/build", json=build_payload)
        self.assertEqual(b_res.status_code, 200)
        b_data = b_res.json()
        self.assertEqual(b_data["metadata"]["site_id"], "SITE-BUILD-TEST")
        self.assertGreaterEqual(b_data["metadata"]["total_sweeps"], 1)

        # 2. Query radio map
        map_res = self.client.get(f"/api/datasets/{self.dataset_id}/wifi-survey/radio-map?site_id=SITE-BUILD-TEST&version=v1")
        self.assertEqual(map_res.status_code, 200)

        # 3. Localize query fingerprint
        loc_payload = {
            "site_id": "SITE-BUILD-TEST",
            "version": "v1",
            "observations": {
                "00:11:22:33:44:01": -61.0,
                "00:11:22:33:44:02": -74.0,
            },
            "algorithm": "weighted_knn",
            "k": 1,
            "distance_metric": "euclidean",
        }
        pred_res = self.client.post(f"/api/datasets/{self.dataset_id}/wifi-survey/localize", json=loc_payload)
        self.assertEqual(pred_res.status_code, 200)
        pred_data = pred_res.json()
        self.assertEqual(pred_data["predicted_building_id"], "BLDG-1")
        self.assertEqual(pred_data["predicted_floor_id"], 1)
        self.assertFalse(pred_data["is_low_confidence"])
        self.assertEqual(len(pred_data["candidate_matches"]), 1)

    def test_evaluate_wifi_survey(self):
        eval_payload = {
            "site_id": "SITE-BUILD-TEST",
            "algorithm": "weighted_knn",
            "k": 1,
            "distance_metric": "euclidean",
            "train_session_ids": ["SES-101"],
            "test_session_ids": ["SES-102"],
        }
        eval_res = self.client.post(f"/api/datasets/{self.dataset_id}/wifi-survey/evaluate", json=eval_payload)
        self.assertEqual(eval_res.status_code, 200)
        metrics = eval_res.json()["metrics"]
        self.assertEqual(metrics["total_test_samples"], 1)
        self.assertEqual(metrics["building_accuracy"], 1.0)
        self.assertEqual(metrics["floor_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
