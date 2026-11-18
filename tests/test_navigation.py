"""Unit and integration test suite for Task 28 — Indoor Navigation & Route Planning."""

import json
import math
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app
from src.mapping.evaluation import generate_controlled_synthetic_building_map
from src.mapping.models import (
    BuildingMap,
    ConnectionType,
    Door,
    DoorType,
    FloorMap,
    MapVerificationStatus,
    Point2D,
    Polygon2D,
    Room,
    VerticalTransition,
)
from src.mapping.topology import IndoorTopologyGraph, TopologyNode, TopologyNodeType
from src.navigation.accessibility import AccessibilityAuditor, AccessibilityAuditStatus
from src.navigation.destination import DestinationManager
from src.navigation.evaluation import NavigationEvaluator
from src.navigation.instruction_generator import InstructionGenerator
from src.navigation.models import (
    DestinationCategory,
    InstructionType,
    NavigationProgress,
    NavigationStatus,
    PlannedRoute,
    RouteCostWeights,
    RoutePreference,
    RoutingAlgorithm,
)
from src.navigation.navigation_session import NavigationSessionManager
from src.navigation.path_geometry import PathGeometryBuilder
from src.navigation.pipeline import NavigationPipeline
from src.navigation.progress_tracker import ProgressTracker
from src.navigation.rerouting import ReroutingEngine
from src.navigation.route_graph import RouteGraph
from src.navigation.route_planner import RoutePlanner
from src.navigation.route_validator import RouteValidator
from src.navigation.turn_detection import TurnDetectionThresholds, TurnDetector


class TestIndoorNavigation(unittest.TestCase):
    """Test suite verifying all Task 28 requirements."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.bmap = generate_controlled_synthetic_building_map()
        self.dest_manager = DestinationManager(self.bmap)
        self.route_graph = RouteGraph.from_building_map(self.bmap)
        self.planner = RoutePlanner(building_map=self.bmap, route_graph=self.route_graph)
        self.pipeline = NavigationPipeline(building_map=self.bmap, storage_dir=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_destination_indexing_and_search(self):
        """Verifies destination indexing, categorisation, and fuzzy keyword search."""
        dests = self.dest_manager.destinations
        self.assertGreaterEqual(len(dests), 6)

        # Room search
        res_lab = self.dest_manager.search(query="Robotics", limit=10)
        self.assertTrue(any(d.destination_id == "DEST-RM-room_102" for d in res_lab))

        # Room number search
        res_num = self.dest_manager.search(query="201", limit=10)
        self.assertTrue(any(d.destination_id == "DEST-RM-room_201" for d in res_num))

        # Filter by floor
        res_f1 = self.dest_manager.search(floor_id=1, limit=20)
        for d in res_f1:
            self.assertEqual(d.floor_id, 1)

        # Filter by category
        res_cat = self.dest_manager.search(category=DestinationCategory.VERTICAL_TRANSITION)
        self.assertTrue(any("elevator" in d.destination_id for d in res_cat))

    def test_dijkstra_same_floor_routing(self):
        """Verifies Dijkstra route planning between rooms on the same floor."""
        route, err = self.planner.plan_route(
            origin_node_id="NODE-RM-room_101",
            destination_id="DEST-RM-room_102",
            algorithm=RoutingAlgorithm.DIJKSTRA,
            preference=RoutePreference.SHORTEST,
        )
        self.assertIsNotNone(route, f"Routing failed: {err}")
        self.assertEqual(route.origin_name, "Reception & Lobby")
        self.assertEqual(route.destination_name, "Robotics Research Lab")
        self.assertEqual(route.origin_floor_id, 0)
        self.assertEqual(route.destination_floor_id, 0)
        self.assertFalse(route.is_multi_floor)
        self.assertGreater(route.total_distance_m, 5.0)
        self.assertGreaterEqual(len(route.waypoints), 3)
        self.assertGreaterEqual(len(route.instructions), 2)

    def test_astar_same_floor_routing_and_optimality(self):
        """Verifies A* algorithm produces an optimal path identical in distance to Dijkstra."""
        route_dijkstra, _ = self.planner.plan_route(
            origin_node_id="NODE-RM-room_101",
            destination_id="DEST-RM-room_102",
            algorithm=RoutingAlgorithm.DIJKSTRA,
        )
        route_astar, _ = self.planner.plan_route(
            origin_node_id="NODE-RM-room_101",
            destination_id="DEST-RM-room_102",
            algorithm=RoutingAlgorithm.ASTAR,
        )
        self.assertIsNotNone(route_dijkstra)
        self.assertIsNotNone(route_astar)
        self.assertAlmostEqual(route_dijkstra.total_distance_m, route_astar.total_distance_m, places=2)

    def test_heuristic_admissibility(self):
        """Verifies that the Euclidean + vertical displacement heuristic h(u, v) <= h*(u, v)."""
        evaluator = NavigationEvaluator(self.planner)
        is_admissible, viol = evaluator.check_heuristic_admissibility(sample_pairs_count=15)
        self.assertTrue(is_admissible, f"Heuristic admissibility violated: {viol}")
        self.assertEqual(len(viol), 0)

    def test_multi_floor_route_planning(self):
        """Verifies multi-floor routing via stairs and elevators from Floor 0 to Floor 1."""
        # 1. Default routing takes stairs (shorter path in synthetic building)
        route_stairs, err = self.planner.plan_route(
            origin_node_id="NODE-RM-room_101",
            destination_id="DEST-RM-room_202",
            algorithm=RoutingAlgorithm.ASTAR,
        )
        self.assertIsNotNone(route_stairs, f"Multi-floor planning failed: {err}")
        self.assertTrue(route_stairs.is_multi_floor)
        self.assertEqual(route_stairs.floors_traversed, [0, 1])

        stair_ins = [i for i in route_stairs.instructions if i.instruction_type == InstructionType.TAKE_STAIRS]
        self.assertGreaterEqual(len(stair_ins), 1)

        # 2. Accessible routing avoids stairs and takes elevator
        route_elev, err2 = self.planner.plan_route(
            origin_node_id="NODE-RM-room_101",
            destination_id="DEST-RM-room_202",
            algorithm=RoutingAlgorithm.ASTAR,
            preference=RoutePreference.AVOID_STAIRS,
        )
        self.assertIsNotNone(route_elev, f"Multi-floor accessible planning failed: {err2}")
        self.assertTrue(route_elev.is_multi_floor)
        elev_ins = [i for i in route_elev.instructions if i.instruction_type == InstructionType.USE_ELEVATOR]
        self.assertGreaterEqual(len(elev_ins), 1)
        self.assertIn("Elevator", elev_ins[0].text)

    def test_coordinate_snapping_routing(self):
        """Verifies routing starting from arbitrary coordinates (x, y, floor) snapped to nearest node."""
        route, err = self.planner.plan_route(
            origin_coords=(7.0, 5.0, 0),
            destination_coords=(19.0, 5.0, 0),
            algorithm=RoutingAlgorithm.ASTAR,
        )
        self.assertIsNotNone(route, f"Coordinate routing failed: {err}")
        self.assertEqual(route.origin_floor_id, 0)
        self.assertEqual(route.destination_floor_id, 0)
        self.assertGreater(route.total_distance_m, 10.0)

    def test_wall_penetration_rejection(self):
        """Verifies that a route piercing a solid room wall without a door is rejected."""
        validator = RouteValidator(self.bmap)
        # Create an artificial waypoint sequence that directly crosses the solid wall of room_101
        # Room 101 wall is from (3.0, 2.0) to (11.0, 2.0).
        # Crossing at x=7.0 from y=1.0 (outside) to y=3.0 (inside) penetrates solid wall without door
        w1 = self.planner.path_builder.create_waypoint("n1", 0, 7.0, 1.0, label="Outside")
        w2 = self.planner.path_builder.create_waypoint("n2", 0, 7.0, 3.0, label="Inside")
        res = validator.validate_path_geometry([w1, w2])
        self.assertFalse(res.is_valid)
        self.assertEqual(res.error_code, "WALL_COLLISION_DETECTED")

    def test_accessibility_auditor(self):
        """Verifies that accessibility auditor validates compliant vs unknown paths."""
        auditor = AccessibilityAuditor(self.bmap)
        route, _ = self.planner.plan_route(
            origin_node_id="NODE-RM-room_101",
            destination_id="DEST-RM-room_201",
            preference=RoutePreference.AVOID_STAIRS,
        )
        self.assertIsNotNone(route)
        res = auditor.audit_route(route.waypoints, preference=RoutePreference.ACCESSIBLE)
        self.assertEqual(res.status, AccessibilityAuditStatus.ACCESSIBLE)

    def test_turn_detection_and_instructions(self):
        """Verifies angular threshold classification for turn directives."""
        detector = TurnDetector()
        generator = InstructionGenerator(detector)

        # 3 waypoints forming a 90-degree right turn
        # Move East from (0,0) to (10,0), then turn South to (10,-10)
        w0 = self.planner.path_builder.create_waypoint("w0", 0, 0.0, 0.0, label="Start")
        w1 = self.planner.path_builder.create_waypoint("w1", 0, 10.0, 0.0, label="Turn Corner")
        w2 = self.planner.path_builder.create_waypoint("w2", 0, 10.0, -10.0, label="End")

        instructions = generator.generate_instructions([w0, w1, w2], destination_name="Lab")
        self.assertEqual(len(instructions), 3)
        self.assertEqual(instructions[0].instruction_type, InstructionType.START)
        self.assertEqual(instructions[1].instruction_type, InstructionType.TURN_RIGHT)
        self.assertEqual(instructions[2].instruction_type, InstructionType.ARRIVE_DESTINATION)

    def test_progress_tracking_and_arrival(self):
        """Verifies progress tracker advances waypoints and registers arrival."""
        route, _ = self.planner.plan_route(
            origin_node_id="NODE-RM-room_101",
            destination_id="DEST-RM-room_102",
        )
        tracker = ProgressTracker(route)

        # Initial pose at origin
        w0 = route.waypoints[0]
        prog = tracker.update_position(w0.x_m, w0.y_m, w0.floor_id)
        self.assertEqual(prog.current_waypoint_index, 0)
        self.assertFalse(prog.is_off_route)
        self.assertEqual(prog.completion_percentage, 0.0)

        # Step to final waypoint
        w_end = route.waypoints[-1]
        prog_end = tracker.update_position(w_end.x_m, w_end.y_m, w_end.floor_id)
        self.assertTrue(tracker.is_arrived)
        self.assertAlmostEqual(prog_end.completion_percentage, 100.0, places=0)
        self.assertLessEqual(prog_end.remaining_distance_m, 1.5)

    def test_debounced_rerouting(self):
        """Verifies that deviation >4m triggers recalculation and respects 5s cooldown debounce."""
        route, _ = self.planner.plan_route(
            origin_node_id="NODE-RM-room_101",
            destination_id="DEST-RM-room_102",
        )
        engine = ReroutingEngine(self.planner, deviation_threshold_m=4.0, reroute_cooldown_s=5.0)

        # Small normal deviation (1.0m) -> no reroute
        w1 = route.waypoints[1]
        needs1, reason1 = engine.check_deviation(w1.x_m + 1.0, w1.y_m, w1.floor_id, route, current_wp_idx=1)
        self.assertFalse(needs1)

        # Large deviation (6.0m) -> triggers reroute
        dev_x, dev_y = w1.x_m + 6.0, w1.y_m + 6.0
        needs2, reason2 = engine.check_deviation(dev_x, dev_y, w1.floor_id, route, current_wp_idx=1)
        self.assertTrue(needs2)
        self.assertIn("CROSS_TRACK_DEVIATION", reason2)

        # Execute reroute
        now_ms = int(time.time() * 1000)
        new_route, event = engine.trigger_reroute(
            current_x_m=dev_x,
            current_y_m=dev_y,
            current_floor_id=w1.floor_id,
            active_route=route,
            reason=reason2,
            timestamp_epoch_ms=now_ms,
        )
        self.assertIsNotNone(new_route)
        self.assertIsNotNone(event)
        self.assertNotEqual(new_route.route_id, route.route_id)

        # Immediate secondary deviation within cooldown (<5s) should be debounced
        in_cd = engine.is_in_cooldown(now_ms + 2000)
        self.assertTrue(in_cd)

        # After cooldown (>5s) cooldown expires
        after_cd = engine.is_in_cooldown(now_ms + 6000)
        self.assertFalse(after_cd)

    def test_navigation_session_lifecycle(self):
        """Verifies NavigationSession creation, progress feed, and persistence."""
        sess_mgr = NavigationSessionManager(self.tmp_dir)
        route, _ = self.planner.plan_route(
            origin_node_id="NODE-RM-room_101",
            destination_id="DEST-RM-room_102",
        )
        session = sess_mgr.create_session(route=route, dataset_id="TEST_DS")
        self.assertEqual(session.status, NavigationStatus.CREATED)

        # Update progress
        w0 = route.waypoints[0]
        prog = NavigationProgress(
            current_x_m=w0.x_m,
            current_y_m=w0.y_m,
            current_floor_id=w0.floor_id,
            current_waypoint_index=0,
            next_instruction_index=0,
            distance_to_next_waypoint_m=0.0,
            remaining_distance_m=route.total_distance_m,
            completion_percentage=0.0,
            cross_track_deviation_m=0.0,
            is_off_route=False,
        )
        sess_mgr.record_progress(session.session_id, prog)
        self.assertEqual(session.status, NavigationStatus.ACTIVE)

        # Complete session
        completed = sess_mgr.complete_session(session.session_id)
        self.assertIsNotNone(completed)
        self.assertEqual(completed.status, NavigationStatus.COMPLETED)

        # Check saved session file
        sess_file = self.tmp_dir / f"{session.session_id}.json"
        self.assertTrue(sess_file.exists())
        with open(sess_file, "r") as f:
            data = json.load(f)
        self.assertEqual(data["session_id"], session.session_id)
        self.assertEqual(data["status"], "COMPLETED")

    def test_pipeline_end_to_end_evaluation(self):
        """Verifies end-to-end evaluation report generation."""
        report = self.pipeline.evaluate(evaluation_id="TEST_EVAL_01")
        self.assertGreaterEqual(report.evaluated_route_pairs, 1)
        self.assertTrue(report.heuristic_admissible)
        self.assertGreater(report.astar_mean_speedup, 0.0)
        self.assertEqual(len(report.simulation_metrics), 1)
        self.assertTrue(report.simulation_metrics[0].trip_completed)


class TestNavigationAPI(unittest.TestCase):
    """Integration test suite for FastAPI navigation endpoints."""

    def setUp(self):
        self.client = TestClient(app)
        self.dataset_id = "test_nav_dataset"

    def test_api_list_destinations(self):
        """Tests GET /api/datasets/{id}/navigation/destinations."""
        resp = self.client.get(f"/api/datasets/{self.dataset_id}/navigation/destinations?building_id=SYNTH_BLDG_01")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("destinations", data)
        self.assertGreaterEqual(data["total_destinations"], 3)

    def test_api_plan_route_and_get(self):
        """Tests POST /api/datasets/{id}/navigation/routes and GET route by ID."""
        req_payload = {
            "building_id": "SYNTH_BLDG_01",
            "origin_node_id": "NODE-RM-room_101",
            "destination_id": "DEST-RM-room_102",
            "algorithm": "ASTAR",
            "preference": "SHORTEST",
        }
        resp = self.client.post(f"/api/datasets/{self.dataset_id}/navigation/routes", json=req_payload)
        self.assertEqual(resp.status_code, 200, resp.text)
        route_data = resp.json()
        self.assertIn("route_id", route_data)
        self.assertGreater(route_data["total_distance_m"], 0.0)

        # GET by ID
        route_id = route_data["route_id"]
        get_resp = self.client.get(f"/api/datasets/{self.dataset_id}/navigation/routes/{route_id}")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["route_id"], route_id)

    def test_api_session_lifecycle(self):
        """Tests session creation, progress updates, and completion via REST API."""
        # Create session
        create_payload = {
            "building_id": "SYNTH_BLDG_01",
            "origin_node_id": "NODE-RM-room_101",
            "destination_id": "DEST-RM-room_102",
        }
        create_resp = self.client.post(f"/api/datasets/{self.dataset_id}/navigation/sessions", json=create_payload)
        self.assertEqual(create_resp.status_code, 201, create_resp.text)
        session_data = create_resp.json()
        session_id = session_data["session_id"]

        # GET session
        get_resp = self.client.get(f"/api/datasets/{self.dataset_id}/navigation/sessions/{session_id}")
        self.assertEqual(get_resp.status_code, 200)

        # Progress update
        prog_payload = {
            "x_m": 5.0,
            "y_m": 5.0,
            "floor_id": 0,
            "localization_status": "TRACKING",
        }
        prog_resp = self.client.post(f"/api/datasets/{self.dataset_id}/navigation/sessions/{session_id}/progress", json=prog_payload)
        self.assertEqual(prog_resp.status_code, 200)
        prog_data = prog_resp.json()
        self.assertIn("completion_percentage", prog_data)

        # Complete session
        comp_resp = self.client.post(f"/api/datasets/{self.dataset_id}/navigation/sessions/{session_id}/complete")
        self.assertEqual(comp_resp.status_code, 200)
        self.assertEqual(comp_resp.json()["status"], "COMPLETED")

    def test_api_evaluate(self):
        """Tests POST and GET navigation evaluation benchmark."""
        eval_payload = {
            "building_id": "SYNTH_BLDG_01",
            "use_synthetic_benchmark": True,
        }
        resp = self.client.post(f"/api/datasets/{self.dataset_id}/navigation/evaluate", json=eval_payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("report", data)
        self.assertTrue(data["report"]["heuristic_admissible"])


if __name__ == "__main__":
    unittest.main()
