"""Comprehensive unit and integration tests for Real-Time Multimodal Streaming (Task 29).

Tests event validation, deduplication, bounded buffer backpressure, WebSocket communication,
reconnection replay, session isolation, Wi-Fi/PDR/Particle Filter dispatch, live navigation,
privacy sanitization constraints, and REST API endpoints.
"""

from __future__ import annotations

import math
import time
import unittest
import uuid

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.realtime.event_buffer import (
    BackpressurePolicy,
    BoundedEventBuffer,
)
from backend.app.services.realtime.event_models import (
    EventProcessingStatus,
    RealtimeEvent,
    RealtimeEventType,
    RejectionReason,
)
from backend.app.services.realtime.event_validator import EventValidator
from backend.app.services.realtime.realtime_pipeline import (
    RealtimePipeline,
    get_realtime_pipeline,
)
from backend.app.services.realtime.replay_manager import ReplayManager
from backend.app.services.realtime.session_stream import SessionStream
from src.mapping.evaluation import generate_controlled_synthetic_building_map
from src.navigation.models import PlannedRoute, RoutePreference, RoutingAlgorithm
from src.navigation.pipeline import NavigationPipeline


class TestRealtimeStreaming(unittest.TestCase):
    """Test suite covering real-time ingestion, streaming, and engine integration."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.pipeline = get_realtime_pipeline()
        self.session_id = f"TEST-SES-{uuid.uuid4().hex[:8].upper()}"
        self.validator = EventValidator()

    def test_event_validation_valid_payloads(self) -> None:
        """Verifies that valid Wi-Fi, IMU, and lifecycle events pass validation."""
        # 1. Valid Wi-Fi scan
        wifi_ev = RealtimeEvent(
            event_id="EV-WIFI-01",
            session_id=self.session_id,
            event_type=RealtimeEventType.WIFI_SCAN,
            device_id="DEV-01",
            payload={
                "bssid_scans": [
                    {"bssid": "aa:bb:cc:dd:ee:01", "rssi_dbm": -65.0, "frequency_mhz": 2412},
                    {"bssid": "aa:bb:cc:dd:ee:02", "rssi_dbm": -72.5, "frequency_mhz": 5180},
                ]
            },
        )
        valid, reason, msg = self.validator.validate_event(wifi_ev)
        self.assertTrue(valid)
        self.assertEqual(reason, RejectionReason.NONE)

        # 2. Valid IMU batch
        imu_ev = RealtimeEvent(
            event_id="EV-IMU-01",
            session_id=self.session_id,
            event_type=RealtimeEventType.IMU_BATCH,
            device_id="DEV-01",
            payload={
                "samples": [
                    {"sensor_type": "ACCELEROMETER", "x": 0.1, "y": 0.2, "z": 9.81},
                    {"sensor_type": "GYROSCOPE", "x": 0.01, "y": -0.02, "z": 0.05},
                ]
            },
        )
        valid, reason, msg = self.validator.validate_event(imu_ev)
        self.assertTrue(valid)

    def test_event_validation_invalid_payloads(self) -> None:
        """Verifies safe rejection of malformed, NaN/Inf, and out-of-bounds payloads."""
        # NaN in accelerometer
        nan_ev = RealtimeEvent(
            event_id="EV-NAN-01",
            session_id=self.session_id,
            event_type=RealtimeEventType.IMU_BATCH,
            device_id="DEV-01",
            payload={"samples": [{"sensor_type": "ACCELEROMETER", "x": float("nan"), "y": 0.0, "z": 9.8}]},
        )
        valid, reason, msg = self.validator.validate_event(nan_ev)
        self.assertFalse(valid)
        self.assertEqual(reason, RejectionReason.NAN_OR_INF_DETECTED)

        # RSSI out of physical bounds (+10 dBm)
        oob_wifi = RealtimeEvent(
            event_id="EV-OOB-01",
            session_id=self.session_id,
            event_type=RealtimeEventType.WIFI_SCAN,
            device_id="DEV-01",
            payload={"bssid_scans": [{"bssid": "00:11:22:33:44:55", "rssi_dbm": 15.0}]},
        )
        valid, reason, msg = self.validator.validate_event(oob_wifi)
        self.assertFalse(valid)
        self.assertEqual(reason, RejectionReason.OUT_OF_BOUNDS)

        # Unreasonable future timestamp (> 1 hour ahead)
        future_ev = RealtimeEvent(
            event_id="EV-FUT-01",
            session_id=self.session_id,
            event_type=RealtimeEventType.HEARTBEAT,
            device_id="DEV-01",
            client_timestamp_utc=int((time.time() + 10000) * 1000),
            payload={},
        )
        valid, reason, msg = self.validator.validate_event(future_ev)
        self.assertFalse(valid)
        self.assertEqual(reason, RejectionReason.TIMESTAMP_ANOMALY)

    def test_privacy_sanitization_enforcement(self) -> None:
        """Enforces that photo events lacking verified privacy sanitization are rejected."""
        unsanitized_photo = RealtimeEvent(
            event_id="EV-PHOTO-UNSANITIZED",
            session_id=self.session_id,
            event_type=RealtimeEventType.PHOTO_CAPTURE,
            device_id="DEV-01",
            payload={
                "photo_id": "PHOTO-001",
                "filename": "corridor_door.jpg",
                "exif_sanitized": False,  # Not sanitized!
            },
        )
        valid, reason, msg = self.validator.validate_event(unsanitized_photo)
        self.assertFalse(valid)
        self.assertEqual(reason, RejectionReason.MALFORMED_PAYLOAD)

        # Sanitized photo passes
        sanitized_photo = RealtimeEvent(
            event_id="EV-PHOTO-SANITIZED",
            session_id=self.session_id,
            event_type=RealtimeEventType.PHOTO_CAPTURE,
            device_id="DEV-01",
            payload={
                "photo_id": "PHOTO-002",
                "filename": "corridor_door_clean.jpg",
                "exif_sanitized": True,
            },
        )
        valid, reason, msg = self.validator.validate_event(sanitized_photo)
        self.assertTrue(valid)

    def test_duplicate_event_idempotency(self) -> None:
        """Verifies duplicate events are identified and acknowledged as DUPLICATE without reprocessing."""
        ev = RealtimeEvent(
            event_id=f"EV-DUP-{uuid.uuid4().hex[:6]}",
            session_id=self.session_id,
            event_type=RealtimeEventType.HEARTBEAT,
            device_id="DEV-01",
            payload={"battery": 95},
        )

        ack1, _ = self.pipeline.ingest_event(ev)
        self.assertEqual(ack1.status, EventProcessingStatus.ACCEPTED)

        # Ingest same event again
        ack2, _ = self.pipeline.ingest_event(ev)
        self.assertEqual(ack2.status, EventProcessingStatus.DUPLICATE)
        self.assertEqual(ack2.rejection_reason, RejectionReason.DUPLICATE_EVENT_ID)

    def test_bounded_buffer_and_backpressure_drop_oldest(self) -> None:
        """Verifies BoundedEventBuffer enforces max capacity and drops oldest events."""
        buf = BoundedEventBuffer(session_id="BUF_TEST", max_capacity=3, policy=BackpressurePolicy.DROP_OLDEST)
        self.assertEqual(buf.size, 0)

        for i in range(5):
            ev = RealtimeEvent(
                event_id=f"E{i}",
                session_id="BUF_TEST",
                event_type=RealtimeEventType.HEARTBEAT,
                device_id="D1",
                payload={},
            )
            buf.push(ev)

        self.assertEqual(buf.size, 3)
        self.assertEqual(buf.total_pushed, 5)
        self.assertEqual(buf.total_dropped, 2)
        # Oldest (E0, E1) dropped; remaining: E2, E3, E4
        self.assertEqual(buf.pop().event_id, "E2")
        self.assertEqual(buf.pop().event_id, "E3")
        self.assertEqual(buf.pop().event_id, "E4")
        self.assertIsNone(buf.pop())

    def test_bounded_buffer_backpressure_reject_new(self) -> None:
        """Verifies BoundedEventBuffer reject_new policy rejects incoming events when full."""
        buf = BoundedEventBuffer(session_id="BUF_TEST_REJ", max_capacity=2, policy=BackpressurePolicy.REJECT_NEW)
        ev1 = RealtimeEvent(event_id="E1", session_id="S", event_type=RealtimeEventType.HEARTBEAT, device_id="D", payload={})
        ev2 = RealtimeEvent(event_id="E2", session_id="S", event_type=RealtimeEventType.HEARTBEAT, device_id="D", payload={})
        ev3 = RealtimeEvent(event_id="E3", session_id="S", event_type=RealtimeEventType.HEARTBEAT, device_id="D", payload={})

        acc1, _, _ = buf.push(ev1)
        acc2, _, _ = buf.push(ev2)
        acc3, reason3, _ = buf.push(ev3)

        self.assertTrue(acc1)
        self.assertTrue(acc2)
        self.assertFalse(acc3)
        self.assertEqual(reason3, RejectionReason.BACKPRESSURE_DROP)
        self.assertEqual(buf.size, 2)

    def test_batch_ingestion_rest_endpoint(self) -> None:
        """Tests REST POST batch event ingestion endpoint."""
        events_payload = [
            {"event_id": f"B1-{uuid.uuid4().hex[:4]}", "event_type": "HEARTBEAT", "payload": {}},
            {"event_id": f"B2-{uuid.uuid4().hex[:4]}", "event_type": "ORIENTATION_UPDATE", "payload": {"heading_rad": 1.2}},
            {"event_id": f"B3-{uuid.uuid4().hex[:4]}", "event_type": "WIFI_SCAN", "payload": {
                "bssid_scans": [{"bssid": "aa:bb:cc:dd:ee:11", "rssi_dbm": -60.0}]
            }},
        ]

        resp = self.client.post(
            f"/api/realtime/sessions/{self.session_id}/events/batch",
            json={"events": events_payload},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["session_id"], self.session_id)
        self.assertEqual(data["total_events"], 3)
        self.assertEqual(data["accepted"], 3)
        self.assertEqual(len(data["acknowledgements"]), 3)
        self.assertIsNotNone(data["acknowledgements"][0]["sequence_number"])

    def test_websocket_streaming_flow(self) -> None:
        """Tests full WebSocket connection lifecycle: connect, status, ping/pong, observation ack, disconnect."""
        with self.client.websocket_connect(f"/api/realtime/sessions/{self.session_id}/stream") as ws:
            # 1. First message upon connection is STREAM_STATUS
            status_init = ws.receive_json()
            self.assertEqual(status_init["type"], "STREAM_STATUS")
            self.assertEqual(status_init["session_id"], self.session_id)

            # 2. Ping / Pong
            ws.send_json({"type": "PING"})
            pong = ws.receive_json()
            self.assertEqual(pong["type"], "PONG")

            # 3. Send valid observation event
            ev_id = f"WS-EV-{uuid.uuid4().hex[:6]}"
            ws.send_json({
                "event_id": ev_id,
                "session_id": self.session_id,
                "event_type": "ORIENTATION_UPDATE",
                "device_id": "DEVICE-WS",
                "payload": {"heading_rad": 0.785, "azimuth_deg": 45.0},
            })

            # 4. Receive immediate server ACK
            ack = ws.receive_json()
            self.assertEqual(ack["type"], "ACK")
            self.assertEqual(ack["event_id"], ev_id)
            self.assertEqual(ack["status"], "ACCEPTED")

    def test_reconnection_and_replay_manager(self) -> None:
        """Tests ReplayManager ability to reconstruct update history strictly after a client's last ACK."""
        replay = ReplayManager(session_id="REPLAY_TEST", max_history_size=50)

        ev1 = RealtimeEvent(event_id="E1", session_id="S", event_type=RealtimeEventType.HEARTBEAT, device_id="D", payload={})
        ev2 = RealtimeEvent(event_id="E2", session_id="S", event_type=RealtimeEventType.WIFI_SCAN, device_id="D", payload={})
        ev3 = RealtimeEvent(event_id="E3", session_id="S", event_type=RealtimeEventType.IMU_BATCH, device_id="D", payload={})

        seq1 = replay.record_processed_item(ev1, [{"type": "UPDATE", "val": 1}])
        seq2 = replay.record_processed_item(ev2, [{"type": "UPDATE", "val": 2}])
        seq3 = replay.record_processed_item(ev3, [{"type": "UPDATE", "val": 3}])

        # Client reconnects having acknowledged E1 (seq 1) -> should receive E2 and E3
        items, is_gap = replay.get_replay_since(last_ack_event_id="E1")
        self.assertFalse(is_gap)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["event_id"], "E2")
        self.assertEqual(items[1]["event_id"], "E3")

    def test_session_isolation(self) -> None:
        """Verifies events submitted to session A never leak into session B."""
        stream_a = self.pipeline.get_or_create_stream("SESSION_ISOLATION_A")
        stream_b = self.pipeline.get_or_create_stream("SESSION_ISOLATION_B")

        ev_a = RealtimeEvent(
            event_id="EV-A-01",
            session_id="SESSION_ISOLATION_A",
            event_type=RealtimeEventType.HEARTBEAT,
            device_id="DEV-A",
            payload={"msg": "hello from A"},
        )
        self.pipeline.ingest_event(ev_a)

        self.assertEqual(stream_a.events_received, 1)
        self.assertEqual(stream_b.events_received, 0)
        self.assertTrue(stream_a.is_event_duplicate("EV-A-01"))
        self.assertFalse(stream_b.is_event_duplicate("EV-A-01"))

    def test_unverified_landmark_anti_fabrication(self) -> None:
        """Verifies unverified landmarks do not generate confirmed localization fixes."""
        stream = self.pipeline.get_or_create_stream(self.session_id)
        stream.latest_localization = None

        unverified_ev = RealtimeEvent(
            event_id="EV-LM-UNVERIFIED",
            session_id=self.session_id,
            event_type=RealtimeEventType.LANDMARK_OBSERVATION,
            device_id="DEV-01",
            payload={
                "landmark_type": "DOOR",
                "is_verified": False,
                "x_m": 25.0,
                "y_m": 15.0,
            },
        )
        ack, updates = self.pipeline.ingest_event(unverified_ev)
        self.assertEqual(ack.status, EventProcessingStatus.ACCEPTED)
        # Updates should be empty because unverified landmark cannot serve as anchor
        self.assertEqual(len(updates), 0)
        self.assertIsNone(stream.latest_localization)

    def test_verified_landmark_updates_anchor(self) -> None:
        """Verifies verified landmark updates localization anchor."""
        verified_ev = RealtimeEvent(
            event_id="EV-LM-VERIFIED",
            session_id=self.session_id,
            event_type=RealtimeEventType.LANDMARK_OBSERVATION,
            device_id="DEV-01",
            payload={
                "landmark_type": "DOOR",
                "is_verified": True,
                "x_m": 20.0,
                "y_m": 10.0,
                "floor_id": 1,
            },
        )
        ack, updates = self.pipeline.ingest_event(verified_ev)
        self.assertEqual(ack.status, EventProcessingStatus.ACCEPTED)
        self.assertEqual(len(updates), 1)
        loc_up = updates[0]
        self.assertEqual(loc_up["type"], "LOCALIZATION_UPDATE")
        self.assertEqual(loc_up["source"], "VERIFIED_LANDMARK")
        self.assertEqual(loc_up["x_m"], 20.0)
        self.assertEqual(loc_up["y_m"], 10.0)

    def test_live_navigation_tracking_and_flapping_prevention(self) -> None:
        """Tests live navigation advancement and anti-flapping check on low-confidence fixes."""
        bldg = generate_controlled_synthetic_building_map()
        nav_pip = NavigationPipeline(building_map=bldg)
        # Plan route from room_101 to room_102
        dest_mgr = nav_pip.destinations
        dests = dest_mgr.destinations
        dest = dests[0]

        planned_route, _ = nav_pip.plan_route(
            origin_coords=(4.0, 4.0, 1),
            destination_id=dest.destination_id,
        )
        self.assertIsNotNone(planned_route)

        session_obj = nav_pip.start_session(route=planned_route, dataset_id="default")
        stream = self.pipeline.get_or_create_stream(self.session_id)
        self.pipeline.attach_navigation_session(
            session_id=self.session_id,
            nav_session_manager=nav_pip.sessions,
            nav_session_id=session_obj.session_id,
        )

        # 1. Inject fix with LOW confidence -> should NOT generate navigation update
        from backend.app.services.realtime.event_models import LiveLocalizationUpdateMessage
        low_conf_loc = LiveLocalizationUpdateMessage(
            session_id=self.session_id,
            timestamp_ms=int(time.time() * 1000),
            x_m=4.5,
            y_m=4.5,
            floor_id=1,
            heading_rad=0.0,
            uncertainty_radius_m=8.0,
            confidence_level="LOW",
            source="WEAK_WIFI",
        )
        nav_up_low = self.pipeline.processor._update_navigation(low_conf_loc, stream)
        self.assertIsNone(nav_up_low)

        # 2. Inject fix with MEDIUM or HIGH confidence -> advances navigation
        high_conf_loc = LiveLocalizationUpdateMessage(
            session_id=self.session_id,
            timestamp_ms=int(time.time() * 1000),
            x_m=4.5,
            y_m=4.5,
            floor_id=1,
            heading_rad=0.0,
            uncertainty_radius_m=1.5,
            confidence_level="HIGH",
            source="VERIFIED_FIX",
        )
        nav_up_high = self.pipeline.processor._update_navigation(high_conf_loc, stream)
        self.assertIsNotNone(nav_up_high)
        self.assertEqual(nav_up_high.to_dict()["type"], "NAVIGATION_UPDATE")
        self.assertFalse(nav_up_high.arrival_status)

    def test_session_lifecycle_control_api(self) -> None:
        """Tests stream pause, resume, reset, and stop via REST control endpoint."""
        # 1. Pause
        r_pause = self.client.post(
            f"/api/realtime/sessions/{self.session_id}/control",
            json={"action": "PAUSE"},
        )
        self.assertEqual(r_pause.status_code, 200)
        self.assertFalse(r_pause.json()["is_active"])

        # Ingestion while paused should be rejected
        ev = RealtimeEvent(
            event_id=f"EV-PAUSE-{uuid.uuid4().hex[:4]}",
            session_id=self.session_id,
            event_type=RealtimeEventType.HEARTBEAT,
            device_id="D1",
            payload={},
        )
        ack, _ = self.pipeline.ingest_event(ev)
        self.assertEqual(ack.status, EventProcessingStatus.REJECTED)
        self.assertEqual(ack.rejection_reason, RejectionReason.SESSION_INACTIVE)

        # 2. Resume
        r_resume = self.client.post(
            f"/api/realtime/sessions/{self.session_id}/control",
            json={"action": "RESUME"},
        )
        self.assertEqual(r_resume.status_code, 200)
        self.assertTrue(r_resume.json()["is_active"])

        # 3. Status retrieval
        r_status = self.client.get(f"/api/realtime/sessions/{self.session_id}/status")
        self.assertEqual(r_status.status_code, 200)
        self.assertTrue(r_status.json()["is_active"])

    def test_simulation_step_injection(self) -> None:
        """Tests safe development simulator endpoint."""
        sim_resp = self.client.post(
            f"/api/realtime/sessions/{self.session_id}/simulate",
            json={
                "x_m": 12.0,
                "y_m": 8.0,
                "floor_id": 1,
                "heading_rad": 1.57,
                "uncertainty_radius_m": 1.5,
            },
        )
        self.assertEqual(sim_resp.status_code, 200)
        data = sim_resp.json()
        self.assertEqual(data["ack"]["status"], "ACCEPTED")
        self.assertGreaterEqual(len(data["updates"]), 1)
        loc = data["updates"][0]
        self.assertEqual(loc["type"], "LOCALIZATION_UPDATE")
        self.assertEqual(loc["x_m"], 12.0)
        self.assertEqual(loc["y_m"], 8.0)
        self.assertEqual(loc["floor_id"], 1)


if __name__ == "__main__":
    unittest.main()
