"""Unit and integration test suite for Task 26: Indoor Map Generation & Spatial Representation."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import create_app
from src.mapping.evaluation import (
    MapEvaluationMetrics,
    MapEvaluator,
    generate_controlled_synthetic_building_map,
)
from src.mapping.geometry import (
    BoundingBox2D,
    LineSegment2D,
    Point2D,
    Polygon2D,
)
from src.mapping.map_builder import MapBuilder
from src.mapping.map_export import MapExporter
from src.mapping.map_storage import MapStorage
from src.mapping.map_validator import (
    MapValidationError,
    MapValidationReport,
    MapValidator,
    ValidationSeverity,
)
from src.mapping.models import (
    AccessibilityType,
    BuildingMap,
    ConnectionType,
    Corridor,
    Door,
    DoorType,
    FloorMap,
    MapVerificationStatus,
    Room,
    RoomCategory,
    SpatialConnection,
    VerticalTransition,
)
from src.mapping.pipeline import MappingPipeline
from src.mapping.topology import (
    IndoorTopologyGraph,
    TopologyEdge,
    TopologyNode,
    TopologyNodeType,
)


class TestMappingGeometry(unittest.TestCase):
    """Unit tests for 2D geometry primitives."""

    def test_point2d_basic_operations(self):
        p1 = Point2D(3.0, 4.0)
        p2 = Point2D(6.0, 8.0)
        self.assertAlmostEqual(p1.distance_to(p2), 5.0, places=4)
        self.assertEqual(p1.to_dict(), {"x": 3.0, "y": 4.0})
        self.assertEqual(Point2D.from_dict({"x": 3.0, "y": 4.0}), p1)

    def test_line_segment_operations(self):
        seg = LineSegment2D(Point2D(0.0, 0.0), Point2D(10.0, 0.0))
        self.assertAlmostEqual(seg.length, 10.0)

        # Midpoint distance
        p_mid = Point2D(5.0, 5.0)
        self.assertAlmostEqual(seg.distance_to_point(p_mid), 5.0)

        # Distance beyond endpoint
        p_far = Point2D(15.0, 0.0)
        self.assertAlmostEqual(seg.distance_to_point(p_far), 5.0)

        # Intersecting segment
        seg2 = LineSegment2D(Point2D(5.0, -5.0), Point2D(5.0, 5.0))
        self.assertTrue(seg.intersects_with(seg2))

        # Parallel non-intersecting
        seg3 = LineSegment2D(Point2D(0.0, 2.0), Point2D(10.0, 2.0))
        self.assertFalse(seg.intersects_with(seg3))

    def test_polygon_area_centroid_and_containment(self):
        # 10m x 10m square from (0,0) to (10,10)
        poly = Polygon2D.create_rectangle(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0)
        self.assertAlmostEqual(poly.area, 100.0)
        self.assertAlmostEqual(poly.centroid.x, 5.0)
        self.assertAlmostEqual(poly.centroid.y, 5.0)

        # Point inside
        self.assertTrue(poly.contains_point(Point2D(5.0, 5.0)))
        self.assertTrue(poly.contains_point(Point2D(1.0, 1.0)))

        # Point outside
        self.assertFalse(poly.contains_point(Point2D(15.0, 5.0)))
        self.assertFalse(poly.contains_point(Point2D(-1.0, 5.0)))

        # Point on boundary
        self.assertTrue(poly.contains_point(Point2D(0.0, 5.0)))

    def test_polygon_self_intersection_detection(self):
        # Normal square (no self-intersection)
        normal_sq = Polygon2D.create_rectangle(0, 0, 5, 5)
        self.assertFalse(normal_sq.has_self_intersections())

        # Self-intersecting figure-8 / hourglass polygon
        bowtie_poly = Polygon2D([
            Point2D(0.0, 0.0),
            Point2D(10.0, 10.0),
            Point2D(0.0, 10.0),
            Point2D(10.0, 0.0),
        ])
        self.assertTrue(bowtie_poly.has_self_intersections())


class TestMappingModels(unittest.TestCase):
    """Unit tests for indoor mapping domain entities and serialization."""

    def test_room_model_and_serialization(self):
        rect = Polygon2D.create_rectangle(0, 0, 8, 6)
        room = Room(
            room_id="room_101",
            building_id="BLDG_A",
            floor_id=0,
            name="Robotics Lab",
            polygon=rect,
            category=RoomCategory.LAB,
            room_number="101",
            entrance_door_ids=["door_d1"],
            verification_status=MapVerificationStatus.VERIFIED,
        )
        self.assertAlmostEqual(room.area_sq_m, 48.0)
        self.assertEqual(room.centroid, Point2D(4.0, 3.0))

        data = room.to_dict()
        restored = Room.from_dict(data)
        self.assertEqual(restored.room_id, "room_101")
        self.assertEqual(restored.category, RoomCategory.LAB)
        self.assertAlmostEqual(restored.area_sq_m, 48.0)

    def test_corridor_and_door_models(self):
        corridor = Corridor(
            corridor_id="corr_main",
            building_id="BLDG_A",
            floor_id=0,
            name="Central Hall",
            centerline=[Point2D(0.0, 0.0), Point2D(20.0, 0.0)],
            width_m=2.4,
            verification_status=MapVerificationStatus.VERIFIED,
        )
        self.assertAlmostEqual(corridor.length_m, 20.0)
        c_dict = corridor.to_dict()
        self.assertEqual(Corridor.from_dict(c_dict).corridor_id, "corr_main")

        door = Door(
            door_id="door_d1",
            building_id="BLDG_A",
            floor_id=0,
            name="Lab Entrance",
            position=Point2D(4.0, 0.0),
            connects_from_id="room_101",
            connects_to_id="corr_main",
            door_type=DoorType.SINGLE_SWING,
            width_m=0.95,
        )
        d_dict = door.to_dict()
        restored_door = Door.from_dict(d_dict)
        self.assertEqual(restored_door.door_id, "door_d1")
        self.assertEqual(restored_door.door_type, DoorType.SINGLE_SWING)

    def test_vertical_transition_and_connection(self):
        vt = VerticalTransition(
            transition_id="stair_1",
            building_id="BLDG_A",
            transition_type=ConnectionType.STAIRCASE,
            name="North Stairwell",
            floor_positions={
                0: Point2D(20.0, 10.0),
                1: Point2D(20.0, 10.0),
            },
            is_accessible=False,
        )
        self.assertEqual(vt.connected_floors, [0, 1])
        vt_dict = vt.to_dict()
        restored_vt = VerticalTransition.from_dict(vt_dict)
        self.assertEqual(restored_vt.transition_id, "stair_1")
        self.assertEqual(len(restored_vt.floor_positions), 2)

        conn = SpatialConnection(
            connection_id="conn_1",
            building_id="BLDG_A",
            source_id="room_101",
            destination_id="door_d1",
            source_floor_id=0,
            destination_floor_id=0,
            connection_type=ConnectionType.DOOR,
            distance_m=3.5,
        )
        self.assertFalse(conn.is_vertical)
        conn_dict = conn.to_dict()
        self.assertEqual(SpatialConnection.from_dict(conn_dict).distance_m, 3.5)


class TestTopologicalGraph(unittest.TestCase):
    """Unit tests for topological navigation graph construction and reachability."""

    def setUp(self):
        self.building_map = generate_controlled_synthetic_building_map()

    def test_graph_construction_from_building_map(self):
        graph = IndoorTopologyGraph.from_building_map(self.building_map)
        self.assertGreater(len(graph.nodes), 10)
        self.assertGreater(len(graph.edges), 10)

        # Check node types presence
        types_present = {n.node_type for n in graph.nodes.values()}
        self.assertIn(TopologyNodeType.ROOM, types_present)
        self.assertIn(TopologyNodeType.CORRIDOR, types_present)
        self.assertIn(TopologyNodeType.DOOR, types_present)
        self.assertIn(TopologyNodeType.VERTICAL_TRANSITION, types_present)

    def test_dijkstra_shortest_path_multi_floor(self):
        graph = IndoorTopologyGraph.from_building_map(self.building_map)

        # Find route from Room 101 (Floor 0) to Room 202 (Floor 1)
        path, cost = graph.find_shortest_path("room_101", "room_202")
        self.assertIsNotNone(path, "Path across floors should exist via stairwell or elevator")
        self.assertGreater(cost, 0.0)
        self.assertEqual(path[0], "room_101")
        self.assertEqual(path[-1], "room_202")

    def test_wheelchair_accessibility_filter(self):
        graph = IndoorTopologyGraph.from_building_map(self.building_map)

        # Normal path allows stairs and elevators
        path_any, _ = graph.find_shortest_path("room_101", "room_202", wheelchair_accessible_only=False)
        self.assertIsNotNone(path_any)

        # Wheelchair accessible path must route strictly via the elevator
        path_wheelchair, _ = graph.find_shortest_path("room_101", "room_202", wheelchair_accessible_only=True)
        self.assertIsNotNone(path_wheelchair)
        # Verify elevator node is in wheelchair path
        has_elevator = any("elevator" in node_id.lower() for node_id in path_wheelchair)
        self.assertTrue(has_elevator, "Accessible path must use elevator instead of stairs")


class TestMapValidator(unittest.TestCase):
    """Unit tests for geometric, topological, and structural validation."""

    def test_valid_synthetic_map_passes_validation(self):
        bmap = generate_controlled_synthetic_building_map()
        validator = MapValidator()
        report = validator.validate(bmap)
        self.assertTrue(report.is_valid, f"Synthetic building map should be valid: {report.errors}")
        self.assertEqual(report.error_count, 0)

    def test_corrupted_polygon_detected(self):
        bmap = generate_controlled_synthetic_building_map()
        # Corrupt Room 101 with self-intersecting polygon
        bowtie = Polygon2D([
            Point2D(0.0, 0.0),
            Point2D(10.0, 10.0),
            Point2D(0.0, 10.0),
            Point2D(10.0, 0.0),
        ])
        bmap.floors[0].rooms["room_101"].polygon = bowtie

        validator = MapValidator()
        report = validator.validate(bmap)
        self.assertFalse(report.is_valid)
        error_types = [e.error_type for e in report.detailed_errors]
        self.assertIn("SELF_INTERSECTING_POLYGON", error_types)

    def test_excessive_vertical_landing_shift_detected(self):
        bmap = generate_controlled_synthetic_building_map()
        # Shift Floor 1 elevator landing by 15 meters horizontally (> 8m limit)
        bmap.vertical_transitions["elevator_east"].floor_positions[1] = Point2D(50.0, 14.0)

        validator = MapValidator()
        report = validator.validate(bmap)
        self.assertFalse(report.is_valid)
        error_types = [e.error_type for e in report.detailed_errors]
        self.assertIn("VERTICAL_TRANSITION_MISALIGNMENT", error_types)


class TestMapStorageAndVersioning(unittest.TestCase):
    """Unit tests for atomic map storage and immutable version snapshot publishing."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage = MapStorage(base_dir=Path(self.temp_dir))
        self.synth_map = generate_controlled_synthetic_building_map()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_building_map(self):
        self.storage.save_building_map(self.synth_map)
        loaded = self.storage.load_building_map(self.synth_map.building_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.building_id, self.synth_map.building_id)
        self.assertEqual(loaded.floor_count, 2)
        self.assertEqual(loaded.total_rooms_count, 6)

    def test_publish_version_snapshots(self):
        self.storage.save_building_map(self.synth_map)
        v1_map = self.storage.publish_version(self.synth_map.building_id, version_tag="v1.0.0")
        self.assertEqual(v1_map.map_version, "v1.0.0")

        # Publish v2
        v2_map = self.storage.publish_version(self.synth_map.building_id, version_tag="v2.0.0")
        self.assertEqual(v2_map.map_version, "v2.0.0")

        versions = self.storage.list_versions(self.synth_map.building_id)
        self.assertEqual(len(versions), 2)
        v_names = [v["version"] for v in versions]
        self.assertIn("v1.0.0", v_names)
        self.assertIn("v2.0.0", v_names)


class TestMapExporter(unittest.TestCase):
    """Unit tests for JSON and RFC 7946 GeoJSON export functionality."""

    def setUp(self):
        self.bmap = generate_controlled_synthetic_building_map()
        self.exporter = MapExporter(self.bmap)

    def test_export_json_format(self):
        json_str = self.exporter.export_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["building_id"], self.bmap.building_id)
        self.assertIn("floors", parsed)

        # Single floor JSON export
        f0_json = self.exporter.export_json(floor_id=0)
        f0_parsed = json.loads(f0_json)
        self.assertEqual(f0_parsed["floor_id"], 0)

    def test_export_geojson_feature_collection(self):
        geojson = self.exporter.export_geojson(floor_id=0)
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertIn("features", geojson)
        self.assertGreater(len(geojson["features"]), 5)

        # Inspect feature types
        entity_types = {f["properties"]["entity_type"] for f in geojson["features"]}
        self.assertIn("floor_boundary", entity_types)
        self.assertIn("room", entity_types)
        self.assertIn("corridor_centerline", entity_types)
        self.assertIn("door", entity_types)
        self.assertIn("wifi_reference", entity_types)
        self.assertIn("landmark", entity_types)
        self.assertIn("topology_edge", entity_types)


class TestMappingPipelineAndEvaluator(unittest.TestCase):
    """Unit tests for end-to-end pipeline and evaluation metrics."""

    def test_map_evaluator_metrics(self):
        bmap = generate_controlled_synthetic_building_map()
        metrics = MapEvaluator.evaluate(bmap)
        self.assertEqual(metrics.total_floors, 2)
        self.assertEqual(metrics.total_rooms, 6)
        self.assertEqual(metrics.total_doors, 6)
        self.assertEqual(metrics.total_vertical_transitions, 2)
        self.assertTrue(metrics.is_fully_connected)
        self.assertGreater(metrics.total_surveyed_area_sq_m, 200.0)
        self.assertAlmostEqual(metrics.verification_ratio, 1.0, places=2)

    def test_incremental_builder_with_multimodal_data(self):
        builder = MapBuilder(building_id="TEST_BLDG", name="Test Complex")
        builder.add_floor(floor_id=0, label="Ground")
        builder.add_room(
            room_id="r1",
            floor_id=0,
            name="Office 1",
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
            category="OFFICE",
        )
        builder.import_wifi_reference_locations(
            floor_id=0,
            wifi_references=[{"reference_id": "wifi_1", "x": 5.0, "y": 5.0}],
            auto_associate_rooms=True,
        )
        bmap, report = builder.build(validate=False)
        self.assertEqual(bmap.floors[0].rooms["r1"].wifi_reference_ids, ["wifi_1"])


class TestMappingFastAPIEndpoints(unittest.TestCase):
    """Integration tests verifying FastAPI endpoints for mapping operations."""

    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_list_and_get_indoor_maps(self):
        # 1. List maps
        res_list = self.client.get("/api/datasets/default/maps/buildings")
        self.assertEqual(res_list.status_code, 200)
        buildings = res_list.json()
        self.assertGreater(len(buildings), 0)
        bldg_id = buildings[0]["building_id"]

        # 2. Get specific map
        res_get = self.client.get(f"/api/datasets/default/maps/{bldg_id}")
        self.assertEqual(res_get.status_code, 200)
        bmap_data = res_get.json()
        self.assertEqual(bmap_data["building_id"], bldg_id)
        self.assertIn("floors", bmap_data)

    def test_validate_and_evaluate_map(self):
        bldg_id = "SYNTH_BLDG_01"
        # Validate
        res_val = self.client.post(f"/api/datasets/default/maps/{bldg_id}/validate")
        self.assertEqual(res_val.status_code, 200)
        val_data = res_val.json()
        self.assertTrue(val_data["is_valid"])

        # Evaluate
        res_eval = self.client.post(f"/api/datasets/default/maps/{bldg_id}/evaluate")
        self.assertEqual(res_eval.status_code, 200)
        eval_data = res_eval.json()
        self.assertEqual(eval_data["total_floors"], 2)
        self.assertEqual(eval_data["total_rooms"], 6)

    def test_export_geojson_and_json(self):
        bldg_id = "SYNTH_BLDG_01"
        # GeoJSON
        res_geo = self.client.get(f"/api/datasets/default/maps/{bldg_id}/export/geojson?floor_id=0")
        self.assertEqual(res_geo.status_code, 200)
        geo_data = res_geo.json()
        self.assertEqual(geo_data["type"], "FeatureCollection")

        # JSON
        res_json = self.client.get(f"/api/datasets/default/maps/{bldg_id}/export/json")
        self.assertEqual(res_json.status_code, 200)

    def test_publish_version_lifecycle(self):
        bldg_id = "SYNTH_BLDG_01"
        res_pub = self.client.post(
            f"/api/datasets/default/maps/{bldg_id}/publish",
            json={"version_tag": "v1.2.0"},
        )
        self.assertEqual(res_pub.status_code, 200)

        res_vers = self.client.get(f"/api/datasets/default/maps/{bldg_id}/versions")
        self.assertEqual(res_vers.status_code, 200)
        vers_list = res_vers.json()
        self.assertTrue(any(v["version"] == "v1.2.0" for v in vers_list))


if __name__ == "__main__":
    unittest.main()
