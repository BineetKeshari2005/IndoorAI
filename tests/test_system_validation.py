"""Final System Validation, Security, and Release Readiness Tests (Task 30).

Validates end-to-end multimodal pipeline execution, RFC 7519 HMAC-SHA256 JWT security,
session ownership enforcement, cross-session isolation, concurrency, anisotropic
uncertainty modeling, and performance benchmark SLAs.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import time
import unittest
import uuid

from fastapi.testclient import TestClient

from backend.app.core.security import (
    JWTManager,
    SessionAccessController,
    TokenExpiredError,
    TokenInvalidError,
    UserRole,
    get_access_controller,
    get_jwt_manager,
)
from backend.app.main import app
from backend.app.services.realtime.event_models import (
    RealtimeEvent,
    RealtimeEventType,
)
from backend.app.services.realtime.realtime_pipeline import (
    RealtimePipeline,
    get_realtime_pipeline,
)
from src.evaluation.system_benchmark import SystemPerformanceBenchmark
from src.map_localization.models import FilterStatus, ObservationType, ParticleFilterState
from src.map_localization.particle_filter import MapConstrainedParticleFilter
from src.mapping.evaluation import generate_controlled_synthetic_building_map
from src.navigation.models import PlannedRoute, RoutePreference, RoutingAlgorithm
from src.navigation.pipeline import NavigationPipeline


class TestSystemSecurityAndValidation(unittest.TestCase):
    """Exhaustive validation test suite for release hardening."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.pipeline = get_realtime_pipeline()
        self.jwt_mgr = get_jwt_manager()
        self.access_ctrl = get_access_controller()
        self.access_ctrl.clear()

    def test_jwt_token_creation_and_verification(self) -> None:
        """Verifies RFC 7519 HMAC-SHA256 token issuance and claim extraction."""
        token = self.jwt_mgr.create_token(
            user_id="USER-ALICE-100",
            username="Alice Surveyor",
            role=UserRole.SURVEYOR,
            allowed_sessions=["SES-AUDIT-01", "SES-AUDIT-02"],
            device_id="PIXEL-8-PRO",
            expires_in_seconds=3600,
        )

        self.assertIsInstance(token, str)
        parts = token.split(".")
        self.assertEqual(len(parts), 3, "JWT must contain header, payload, and signature.")

        claims = self.jwt_mgr.verify_token(token)
        self.assertEqual(claims.sub, "USER-ALICE-100")
        self.assertEqual(claims.username, "Alice Surveyor")
        self.assertEqual(claims.role, UserRole.SURVEYOR)
        self.assertEqual(claims.allowed_sessions, ["SES-AUDIT-01", "SES-AUDIT-02"])
        self.assertEqual(claims.device_id, "PIXEL-8-PRO")
        self.assertGreater(claims.exp, int(time.time()))

    def test_jwt_expired_token_rejection(self) -> None:
        """Verifies that expired tokens raise TokenExpiredError and return 401."""
        expired_token = self.jwt_mgr.create_token(
            user_id="USER-BOB-200",
            username="Bob Expired",
            role=UserRole.NAVIGATOR,
            expires_in_seconds=-60,  # Expired 60s ago
        )

        with self.assertRaises(TokenExpiredError):
            self.jwt_mgr.verify_token(expired_token)

        # REST endpoint auth verification with expired token
        resp = self.client.get(
            "/api/auth/verify",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        self.assertEqual(resp.status_code, 401)
        self.assertIn("expired", resp.json()["detail"].lower())

    def test_jwt_tampered_signature_rejection(self) -> None:
        """Verifies that tampered payloads or forged signatures are rejected."""
        valid_token = self.jwt_mgr.create_token(
            user_id="USER-CHARLIE-300",
            username="Charlie Authentic",
            role=UserRole.SURVEYOR,
        )
        header_b64, payload_b64, sig_b64 = valid_token.split(".")

        # Corrupt signature
        corrupt_sig = "A" + sig_b64[1:]
        tampered_token = f"{header_b64}.{payload_b64}.{corrupt_sig}"

        with self.assertRaises(TokenInvalidError):
            self.jwt_mgr.verify_token(tampered_token)

        resp = self.client.get(
            "/api/auth/verify",
            headers={"Authorization": f"Bearer {tampered_token}"},
        )
        self.assertEqual(resp.status_code, 401)

    def test_cross_session_authorization_isolation(self) -> None:
        """Ensures User A cannot access or control User B's streaming session."""
        session_a = f"SES-A-{uuid.uuid4().hex[:6]}"
        session_b = f"SES-B-{uuid.uuid4().hex[:6]}"

        # Token A restricted to session_a
        token_a = self.jwt_mgr.create_token(
            user_id="USER-A",
            username="User A",
            role=UserRole.SURVEYOR,
            allowed_sessions=[session_a],
        )

        # Token B restricted to session_b
        token_b = self.jwt_mgr.create_token(
            user_id="USER-B",
            username="User B",
            role=UserRole.SURVEYOR,
            allowed_sessions=[session_b],
        )

        # Register session_a ownership
        self.access_ctrl.register_session(session_a, owner_user_id="USER-A")
        self.access_ctrl.register_session(session_b, owner_user_id="USER-B")

        # User A accessing session_a status -> OK
        resp_a_on_a = self.client.get(
            f"/api/realtime/sessions/{session_a}/status",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        self.assertEqual(resp_a_on_a.status_code, 200)

        # User B attempting to access session_a status -> 403 Forbidden
        resp_b_on_a = self.client.get(
            f"/api/realtime/sessions/{session_a}/status",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        self.assertEqual(resp_b_on_a.status_code, 403)
        self.assertTrue(
            "access denied" in resp_b_on_a.json()["detail"].lower()
            or "not authorized" in resp_b_on_a.json()["detail"].lower()
        )

        # User B attempting to post events to session_a -> 403 Forbidden
        batch_payload = {
            "events": [
                {
                    "event_id": "EV-MALICIOUS-01",
                    "session_id": session_a,
                    "event_type": "IMU_BATCH",
                    "device_id": "DEVICE-B",
                    "payload": {"samples": [{"sensor_type": "ACCELEROMETER", "x": 0, "y": 0, "z": 9.8}]},
                }
            ]
        }
        resp_b_post_a = self.client.post(
            f"/api/realtime/sessions/{session_a}/events/batch",
            json=batch_payload,
            headers={"Authorization": f"Bearer {token_b}"},
        )
        self.assertEqual(resp_b_post_a.status_code, 403)

        # Admin token can access session_a -> 200 OK
        admin_token = self.jwt_mgr.create_token(
            user_id="USER-ADMIN",
            username="Admin Overseer",
            role=UserRole.ADMIN,
        )
        resp_admin_on_a = self.client.get(
            f"/api/realtime/sessions/{session_a}/status",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(resp_admin_on_a.status_code, 200)

    def test_multi_session_concurrency_and_data_isolation(self) -> None:
        """Verifies multi-session concurrent ingestion has zero cross-talk or leakage."""
        num_sessions = 3
        events_per_session = 25
        sessions = [f"SES-ISOLATE-{i}-{uuid.uuid4().hex[:4]}" for i in range(num_sessions)]

        for sid in sessions:
            self.pipeline.get_or_create_stream(sid)

        def _push_worker(sid: str, start_val: float) -> int:
            accepted = 0
            for k in range(events_per_session):
                ev = RealtimeEvent(
                    event_id=f"EV-{sid}-{k}",
                    session_id=sid,
                    event_type=RealtimeEventType.IMU_BATCH,
                    device_id="DEV-ISO",
                    payload={
                        "samples": [{"sensor_type": "ACCELEROMETER", "x": start_val, "y": 0, "z": 9.8}],
                        "step": {"step_length_m": 0.70, "heading_rad": 0.0},
                    },
                    sequence_number=k,
                )
                ack, _ = self.pipeline.ingest_event(ev)
                if ack.status.value == "ACCEPTED":
                    accepted += 1
            return accepted

        with ThreadPoolExecutor(max_workers=num_sessions) as ex:
            futures = [ex.submit(_push_worker, sessions[i], float(i * 10)) for i in range(num_sessions)]
            results = [f.result() for f in futures]

        self.assertEqual(results, [events_per_session] * num_sessions)

        # Verify stream stats independence
        for sid in sessions:
            stream = self.pipeline.get_stream(sid)
            self.assertIsNotNone(stream)
            self.assertEqual(stream.events_received, events_per_session)
            self.assertEqual(stream.events_processed, events_per_session)
            self.assertEqual(stream.events_rejected, 0)
            self.pipeline.remove_stream(sid)

    def test_end_to_end_multimodal_navigation_pipeline(self) -> None:
        """Validates end-to-end integration: synthetic ingestion -> PF -> map constraints -> live navigation."""
        sid = f"SES-E2E-{uuid.uuid4().hex[:6]}"
        bldg_map = generate_controlled_synthetic_building_map()

        # 1. Setup Navigation Pipeline
        nav_pip = NavigationPipeline(building_map=bldg_map)
        planned_route, _ = nav_pip.plan_route(
            origin_coords=(2.0, 2.5, 0),
            destination_coords=(20.0, 2.5, 0),
            algorithm=RoutingAlgorithm.ASTAR,
        )
        self.assertIsNotNone(planned_route)

        # 2. Setup Particle Filter
        pf = MapConstrainedParticleFilter(building_map=bldg_map)
        pf.initialize(known_location=(2.0, 2.5, 0.0, 0, bldg_map.building_id))

        # 3. Attach components to RealtimePipeline
        self.pipeline.attach_building_map(sid, bldg_map)
        self.pipeline.attach_particle_filter(sid, pf)

        session_obj = nav_pip.start_session(route=planned_route, dataset_id="default")
        self.pipeline.attach_navigation_session(
            session_id=sid,
            nav_session_manager=nav_pip.sessions,
            nav_session_id=session_obj.session_id,
        )

        # Also test attach-navigation REST endpoint
        attach_resp = self.client.post(
            f"/api/realtime/sessions/{sid}/attach-navigation",
            json={"navigation_session_id": session_obj.session_id},
        )
        self.assertEqual(attach_resp.status_code, 200)

        # 4. Stream consecutive steps along the corridor
        stream = self.pipeline.get_stream(sid)
        self.assertIsNotNone(stream)

        for step_idx in range(5):
            x_step = 2.0 + (step_idx + 1) * 2.0
            ev = RealtimeEvent(
                event_id=f"EV-E2E-STEP-{step_idx}",
                session_id=sid,
                event_type=RealtimeEventType.IMU_BATCH,
                device_id="E2E-PHONE",
                payload={
                    "samples": [{"sensor_type": "ACCELEROMETER", "x": 0.0, "y": 0.0, "z": 9.81}],
                    "step": {"step_length_m": 2.0, "heading_rad": 0.0},
                    "x_m": x_step,
                    "y_m": 2.5,
                    "floor_id": 0,
                },
            )
            ack, updates = self.pipeline.ingest_event(ev)
            self.assertEqual(ack.status.value, "ACCEPTED")

        # Verify state updates occurred
        self.assertIsNotNone(stream.latest_localization)
        self.assertIsNotNone(stream.latest_navigation)
        self.assertEqual(stream.latest_localization.floor_id, 0)
        self.assertGreater(stream.cumulative_pdr_distance_m, 0.0)

        self.pipeline.remove_stream(sid)

    def test_particle_filter_anisotropic_uncertainty_attributes(self) -> None:
        """Verifies anisotropic covariance (var_x, var_y, cov_xy) in ParticleFilterState."""
        state = ParticleFilterState(
            timestamp_epoch_ms=1000,
            building_id="BLDG-TEST",
            floor_id=0,
            x_m=10.0,
            y_m=5.0,
            theta_rad=0.0,
            status=FilterStatus.TRACKING,
            particle_count=200,
            effective_sample_size=180.0,
            particle_spread_m=1.5,
            position_uncertainty_m=3.67,
            var_x=1.8,
            var_y=0.45,
            cov_xy=0.05,
        )

        d = state.to_dict()
        self.assertEqual(d["var_x"], 1.8)
        self.assertEqual(d["var_y"], 0.45)
        self.assertEqual(d["cov_xy"], 0.05)

        restored = ParticleFilterState.from_dict(d)
        self.assertEqual(restored.var_x, 1.8)
        self.assertEqual(restored.var_y, 0.45)
        self.assertEqual(restored.cov_xy, 0.05)

    def test_performance_benchmark_sla_targets(self) -> None:
        """Runs the system benchmark harness and asserts strict SLA compliance."""
        bench = SystemPerformanceBenchmark(pipeline=self.pipeline)
        summary = bench.run_full_suite()

        self.assertIn("sla_compliance", summary)
        sla = summary["sla_compliance"]
        self.assertTrue(sla["latency_median_passed"], f"Median latency target breached: {sla['latency_median_achieved_ms']}ms")
        self.assertTrue(sla["latency_p95_passed"], f"P95 latency target breached: {sla['latency_p95_achieved_ms']}ms")
        self.assertTrue(sla["throughput_passed"], f"Throughput target breached: {sla['throughput_achieved_events_per_sec']} ev/s")
        self.assertTrue(sla["success_rate_passed"], f"Success rate target breached: {sla['success_rate_achieved_pct']}%")


if __name__ == "__main__":
    unittest.main()
