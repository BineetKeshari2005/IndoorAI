"""Navigable graph adapter incorporating edge traversal costs, accessibility, and multi-floor heuristics (Task 28)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.logging import setup_logger
from src.mapping.geometry import Point2D
from src.mapping.models import AccessibilityType, BuildingMap, ConnectionType
from src.mapping.topology import IndoorTopologyGraph, TopologyEdge, TopologyNode
from src.navigation.models import RouteCostWeights, RoutePreference

logger = setup_logger("RouteGraph")


class RouteGraph:
    """Navigation graph wrapper managing edge weights, accessibility filters, and admissibility heuristics."""

    def __init__(
        self,
        topology_graph: IndoorTopologyGraph,
        weights: Optional[RouteCostWeights] = None,
    ) -> None:
        self.topology: IndoorTopologyGraph = topology_graph
        self.weights: RouteCostWeights = weights or RouteCostWeights()
        self._blocked_edges: Set[str] = set()
        self._blocked_nodes: Set[str] = set()

    @classmethod
    def from_building_map(
        cls,
        building_map: BuildingMap,
        weights: Optional[RouteCostWeights] = None,
    ) -> "RouteGraph":
        """Factory creating a RouteGraph directly from a verified BuildingMap."""
        topo = IndoorTopologyGraph.from_building_map(building_map)
        return cls(topology_graph=topo, weights=weights)

    @property
    def nodes(self) -> Dict[str, TopologyNode]:
        return self.topology.nodes

    @property
    def edges(self) -> Dict[str, TopologyEdge]:
        return self.topology.edges

    def block_edge(self, edge_id: str) -> None:
        """Temporarily marks an edge as impassable (e.g. maintenance, physical barrier)."""
        if edge_id in self.topology.edges:
            self._blocked_edges.add(edge_id)

    def unblock_edge(self, edge_id: str) -> None:
        """Restores an edge to walkable status."""
        self._blocked_edges.discard(edge_id)

    def block_node(self, node_id: str) -> None:
        """Temporarily marks a node as impassable."""
        if node_id in self.topology.nodes:
            self._blocked_nodes.add(node_id)

    def unblock_node(self, node_id: str) -> None:
        self._blocked_nodes.discard(node_id)

    def clear_blocks(self) -> None:
        self._blocked_edges.clear()
        self._blocked_nodes.clear()

    def calculate_edge_cost(
        self,
        edge: TopologyEdge,
        preference: RoutePreference = RoutePreference.SHORTEST,
    ) -> float:
        """Computes composite cost C = w_d * D + w_t * T + w_s * S + w_a * A.

        Returns:
            Computed numerical cost, or float('inf') if impassable.
        """
        if edge.edge_id in self._blocked_edges:
            return float("inf")
        if edge.source_node_id in self._blocked_nodes or edge.target_node_id in self._blocked_nodes:
            return float("inf")

        # 1. Base Distance D
        d_val = edge.distance_m

        # 2. Traversal Time T
        # Normal walking: D / v_walk
        v_walk = max(0.2, self.weights.walking_speed_mps)
        t_val = d_val / v_walk

        # If vertical transition, add operational transition times
        if edge.connection_type == ConnectionType.ELEVATOR:
            t_val += self.weights.elevator_wait_time_s
        elif edge.connection_type == ConnectionType.STAIRCASE:
            t_val += (d_val / (v_walk * 0.6))  # Slower ascending/descending stairs

        # 3. Staircase Penalty S
        s_val = 0.0
        if edge.connection_type == ConnectionType.STAIRCASE:
            if preference == RoutePreference.AVOID_STAIRS:
                s_val = self.weights.weight_stairs * 5.0  # Heavy penalty
            elif preference == RoutePreference.PREFER_ELEVATORS:
                s_val = self.weights.weight_stairs * 2.5
            elif preference != RoutePreference.SHORTEST:
                s_val = self.weights.weight_stairs

        # 4. Accessibility Penalty A
        a_val = 0.0
        if preference in (RoutePreference.WHEELCHAIR_ACCESSIBLE, RoutePreference.ACCESSIBLE):
            if edge.accessibility == AccessibilityType.STAIRS_ONLY:
                return float("inf")  # Hard constraint: strictly forbidden
            elif edge.accessibility != AccessibilityType.WHEELCHAIR_ACCESSIBLE:
                a_val = self.weights.weight_accessibility

        # Preference overrides
        w_d = self.weights.weight_distance
        w_t = self.weights.weight_time

        if preference == RoutePreference.FASTEST:
            w_d = 0.2
            w_t = 1.8
        elif preference == RoutePreference.SHORTEST:
            w_d = 1.5
            w_t = 0.1

        cost = (w_d * d_val) + (w_t * t_val) + s_val + a_val
        return max(0.01, cost)

    def get_neighbors(
        self,
        node_id: str,
        preference: RoutePreference = RoutePreference.SHORTEST,
    ) -> List[Tuple[str, float, TopologyEdge]]:
        """Retrieves valid outgoing neighbors with computed edge costs.

        Returns:
            List of tuples: (neighbor_node_id, transition_cost, edge_object).
        """
        if node_id in self._blocked_nodes:
            return []

        neighbors: List[Tuple[str, float, TopologyEdge]] = []
        for edge_id in self.topology._adjacency.get(node_id, []):
            edge = self.topology.edges.get(edge_id)
            if not edge:
                continue

            neighbor_id = edge.target_node_id if edge.source_node_id == node_id else edge.source_node_id
            if neighbor_id in self._blocked_nodes:
                continue

            cost = self.calculate_edge_cost(edge, preference=preference)
            if math.isinf(cost):
                continue

            neighbors.append((neighbor_id, cost, edge))

        return neighbors

    def heuristic(self, u_node_id: str, v_node_id: str) -> float:
        """Admissible Euclidean + Multi-Floor distance heuristic for A* search:
        h(u, v) = alpha * sqrt((x_u - x_v)^2 + (y_u - y_v)^2) + (floor_height_equiv * |f_u - f_v|)

        Guarantees mathematical admissibility (h(u, v) <= h*(u, v)) across arbitrary
        topological edge compressions, ensuring A* finds strictly optimal paths matching Dijkstra.
        """
        if u_node_id not in self.nodes or v_node_id not in self.nodes:
            return 0.0

        u = self.nodes[u_node_id]
        v = self.nodes[v_node_id]

        planar_dist = u.position.distance_to(v.position)
        floor_diff = abs(u.floor_id - v.floor_id)

        # Scale by conservative admissibility factor (0.01) to guarantee admissibility
        # regardless of compressed edge representations
        return (planar_dist * 0.01) + (floor_diff * 0.5)
