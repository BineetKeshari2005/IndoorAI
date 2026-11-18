"""Graph-based indoor route planner implementing Dijkstra and A* search (Task 28)."""

from __future__ import annotations

import heapq
import math
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.logging import setup_logger
from src.mapping.models import BuildingMap
from src.mapping.topology import IndoorTopologyGraph
from src.navigation.accessibility import AccessibilityAuditor, AccessibilityAuditStatus
from src.navigation.destination import Destination, DestinationManager
from src.navigation.instruction_generator import InstructionGenerator
from src.navigation.models import (
    NavigationInstruction,
    NavigationWaypoint,
    PlannedRoute,
    RouteCostWeights,
    RoutePreference,
    RoutingAlgorithm,
)
from src.navigation.path_geometry import PathGeometryBuilder
from src.navigation.route_graph import RouteGraph
from src.navigation.route_validator import RouteValidator

logger = setup_logger("RoutePlanner")


class RoutePlanner:
    """Core indoor routing engine supporting Dijkstra and A* heuristic graph search."""

    def __init__(
        self,
        building_map: BuildingMap,
        weights: Optional[Any] = None,
        destination_manager: Optional[DestinationManager] = None,
        route_graph: Optional[RouteGraph] = None,
    ) -> None:
        self.building_map: BuildingMap = building_map
        self.topology: IndoorTopologyGraph = IndoorTopologyGraph.from_building_map(building_map)

        if isinstance(weights, RouteGraph):
            self.graph: RouteGraph = weights
            self.weights: RouteCostWeights = self.graph.weights
        elif route_graph is not None:
            self.graph = route_graph
            self.weights = route_graph.weights
        else:
            self.weights = weights if isinstance(weights, RouteCostWeights) else RouteCostWeights()
            self.graph = RouteGraph(self.topology, self.weights)

        self.destinations: DestinationManager = destination_manager or DestinationManager(building_map)
        self.instruction_gen: InstructionGenerator = InstructionGenerator()
        self.path_builder: PathGeometryBuilder = PathGeometryBuilder()
        self.validator: RouteValidator = RouteValidator(building_map)

    def set_building_map(self, building_map: BuildingMap) -> None:
        """Assigns an updated building map."""
        self.building_map = building_map
        self.topology = IndoorTopologyGraph.from_building_map(building_map)
        self.graph = RouteGraph(self.topology, self.weights)
        self.destinations.load_from_building_map(building_map)

    def plan_route(
        self,
        origin_node_id: Optional[str] = None,
        origin_coords: Optional[Tuple[float, float, int]] = None,  # (x, y, floor_id)
        destination_id: Optional[str] = None,
        destination_coords: Optional[Tuple[float, float, int]] = None,
        algorithm: RoutingAlgorithm = RoutingAlgorithm.ASTAR,
        preference: RoutePreference = RoutePreference.SHORTEST,
    ) -> Tuple[Optional[PlannedRoute], Optional[str]]:
        """Plans an end-to-end, validated indoor navigation route.

        Returns:
            Tuple of (PlannedRoute, None) on success, or (None, error_reason) on failure.
        """
        # 1. Resolve Origin Node
        start_node = origin_node_id
        origin_name = "Starting Location"
        origin_floor = 0

        if start_node and start_node in self.topology.nodes:
            origin_floor = self.topology.nodes[start_node].floor_id
            origin_name = self.topology.nodes[start_node].label or start_node
        elif origin_coords:
            ox, oy, ofloor = origin_coords
            start_node = PathGeometryBuilder.snap_coordinate_to_nearest_corridor(
                x=ox, y=oy, floor_id=ofloor, topology=self.topology
            )
            origin_floor = ofloor
            origin_name = f"Location ({ox:.1f}, {oy:.1f}) [F{ofloor}]"

        if not start_node or start_node not in self.topology.nodes:
            return None, "NO_VALID_ROUTE: Could not resolve a valid starting node in building topology."

        # 2. Resolve Target Node
        target_node = None
        target_name = "Destination"
        target_floor = 0
        resolved_dest_id = destination_id or "DEST-CUSTOM"

        if destination_id:
            dest_obj = self.destinations.get_destination(destination_id)
            if dest_obj:
                target_name = dest_obj.display_name
                target_floor = dest_obj.floor_id
                resolved_dest_id = dest_obj.destination_id
                # Map destination to node in topology
                target_node = self._resolve_destination_node(dest_obj)

        if not target_node and destination_coords:
            dx, dy, dfloor = destination_coords
            target_node = PathGeometryBuilder.snap_coordinate_to_nearest_corridor(
                x=dx, y=dy, floor_id=dfloor, topology=self.topology
            )
            target_floor = dfloor
            target_name = f"Target ({dx:.1f}, {dy:.1f}) [F{dfloor}]"

        if not target_node or target_node not in self.topology.nodes:
            return None, f"NO_VALID_ROUTE: Destination '{destination_id}' cannot be resolved in building graph."

        # 3. Execute Graph Search
        start_time = time.time()
        if algorithm == RoutingAlgorithm.ASTAR:
            node_path, total_cost = self._astar_search(start_node, target_node, preference=preference)
        else:
            node_path, total_cost = self._dijkstra_search(start_node, target_node, preference=preference)

        search_time_ms = (time.time() - start_time) * 1000.0

        if not node_path:
            return None, f"NO_VALID_ROUTE: No walkable path connects '{origin_name}' to '{target_name}' under preference {preference.value}."

        # 4. Construct Waypoints & Geometry
        waypoints = PathGeometryBuilder.build_waypoints(node_path, self.topology)
        total_dist_m = PathGeometryBuilder.calculate_path_length_m(waypoints)

        # 5. Route Validation (wall collisions, floor transitions, accessibility)
        expected_ver = self.building_map.metadata.get("version", "1.0.0")
        val_result = RouteValidator.validate_route(
            waypoints=waypoints,
            topology=self.topology,
            building_map=self.building_map,
            preference=preference,
            expected_map_version=expected_ver,
        )

        if not val_result.is_valid:
            logger.warning(f"Planned route failed validation: {val_result.reason}")
            return None, f"NO_VALID_ROUTE: {val_result.reason}"

        # 6. Generate Turn-by-Turn Instructions
        instructions = self.instruction_gen.generate_instructions(
            waypoints=waypoints,
            destination_name=target_name,
        )

        # 7. Compile Planned Route
        walk_speed = max(0.5, self.weights.walking_speed_mps)
        est_time_s = total_dist_m / walk_speed
        floors = sorted(list(set(w.floor_id for w in waypoints)))

        route = PlannedRoute(
            route_id=f"ROUTE-{uuid.uuid4().hex[:8].upper()}",
            building_id=self.building_map.building_id,
            origin_name=origin_name,
            destination_name=target_name,
            origin_floor_id=origin_floor,
            destination_floor_id=target_floor,
            destination_id=resolved_dest_id,
            total_distance_m=total_dist_m,
            estimated_time_s=round(est_time_s, 1),
            waypoints=waypoints,
            instructions=instructions,
            algorithm_used=algorithm,
            preference=preference,
            is_accessible=val_result.details.get("barrier_count", 0) == 0,
            floors_traversed=floors,
            map_version=expected_ver,
            creation_timestamp_ms=int(time.time() * 1000),
            metadata={
                "search_time_ms": round(search_time_ms, 2),
                "graph_cost": round(total_cost, 2),
                "node_count": len(node_path),
            },
        )

        logger.info(
            f"Successfully planned route {route.route_id}: {total_dist_m:.1f} m, {len(waypoints)} waypoints, "
            f"{len(instructions)} instructions using {algorithm.value}."
        )
        return route, None

    def _dijkstra_search(
        self,
        start_node: str,
        goal_node: str,
        preference: RoutePreference = RoutePreference.SHORTEST,
    ) -> Tuple[Optional[List[str]], float]:
        """Classic Dijkstra shortest path implementation with non-negative edge costs."""
        distances: Dict[str, float] = {nid: float("inf") for nid in self.topology.nodes}
        previous: Dict[str, Optional[str]] = {nid: None for nid in self.topology.nodes}
        distances[start_node] = 0.0

        pq: List[Tuple[float, str]] = [(0.0, start_node)]

        while pq:
            curr_dist, curr_node = heapq.heappop(pq)
            if curr_dist > distances[curr_node]:
                continue
            if curr_node == goal_node:
                break

            for neighbor_id, cost, _ in self.graph.get_neighbors(curr_node, preference=preference):
                new_dist = curr_dist + cost
                if new_dist < distances[neighbor_id]:
                    distances[neighbor_id] = new_dist
                    previous[neighbor_id] = curr_node
                    heapq.heappush(pq, (new_dist, neighbor_id))

        if distances[goal_node] == float("inf"):
            return None, float("inf")

        path = self._reconstruct_path(previous, goal_node)
        return path, distances[goal_node]

    def _astar_search(
        self,
        start_node: str,
        goal_node: str,
        preference: RoutePreference = RoutePreference.SHORTEST,
    ) -> Tuple[Optional[List[str]], float]:
        """A* graph search using an admissible Euclidean and multi-floor heuristic."""
        g_scores: Dict[str, float] = {nid: float("inf") for nid in self.topology.nodes}
        f_scores: Dict[str, float] = {nid: float("inf") for nid in self.topology.nodes}
        previous: Dict[str, Optional[str]] = {nid: None for nid in self.topology.nodes}

        g_scores[start_node] = 0.0
        f_scores[start_node] = self.graph.heuristic(start_node, goal_node)

        pq: List[Tuple[float, float, str]] = [(f_scores[start_node], 0.0, start_node)]

        while pq:
            curr_f, curr_g, curr_node = heapq.heappop(pq)
            if curr_g > g_scores[curr_node]:
                continue
            if curr_node == goal_node:
                break

            for neighbor_id, cost, _ in self.graph.get_neighbors(curr_node, preference=preference):
                tentative_g = g_scores[curr_node] + cost
                if tentative_g < g_scores[neighbor_id]:
                    g_scores[neighbor_id] = tentative_g
                    h_val = self.graph.heuristic(neighbor_id, goal_node)
                    f_val = tentative_g + h_val
                    f_scores[neighbor_id] = f_val
                    previous[neighbor_id] = curr_node
                    heapq.heappush(pq, (f_val, tentative_g, neighbor_id))

        if g_scores[goal_node] == float("inf"):
            return None, float("inf")

        path = self._reconstruct_path(previous, goal_node)
        return path, g_scores[goal_node]

    def _resolve_destination_node(self, dest: Destination) -> Optional[str]:
        """Finds the corresponding topological node ID for a destination."""
        # 1. Direct entity_id match
        for nid, node in self.topology.nodes.items():
            if node.entity_id == dest.entity_id and node.floor_id == dest.floor_id:
                return nid

        # 2. Check room node prefix
        rm_cand = f"NODE-RM-{dest.entity_id}"
        if rm_cand in self.topology.nodes:
            return rm_cand

        # 3. Check vertical transition candidate
        vt_cand = f"NODE-TRANS-{dest.entity_id}-F{dest.floor_id}"
        if vt_cand in self.topology.nodes:
            return vt_cand

        # 4. Snap to nearest node on that floor
        return PathGeometryBuilder.snap_coordinate_to_nearest_corridor(
            x=dest.x_m, y=dest.y_m, floor_id=dest.floor_id, topology=self.topology
        )

    @staticmethod
    def _reconstruct_path(previous: Dict[str, Optional[str]], goal_node: str) -> List[str]:
        path: List[str] = []
        curr: Optional[str] = goal_node
        while curr is not None:
            path.append(curr)
            curr = previous.get(curr)
        path.reverse()
        return path
