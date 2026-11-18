"""Topological navigation graph representing connectivity between indoor spaces."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from src.mapping.geometry import Point2D
from src.mapping.models import (
    AccessibilityType,
    BuildingMap,
    ConnectionType,
    FloorMap,
)
from src.core.logging import setup_logger

logger = setup_logger("IndoorTopology")


class TopologyNodeType(str, Enum):
    """Categorical taxonomy of topological nodes."""
    ROOM_CENTROID = "ROOM_CENTROID"
    CORRIDOR_WAYPOINT = "CORRIDOR_WAYPOINT"
    CORRIDOR_INTERSECTION = "CORRIDOR_INTERSECTION"
    DOORWAY = "DOORWAY"
    STAIR_LANDING = "STAIR_LANDING"
    ELEVATOR_LANDING = "ELEVATOR_LANDING"
    BUILDING_ENTRANCE = "BUILDING_ENTRANCE"
    BUILDING_EXIT = "BUILDING_EXIT"

    # Convenient aliases
    ROOM = "ROOM_CENTROID"
    CORRIDOR = "CORRIDOR_WAYPOINT"
    DOOR = "DOORWAY"
    VERTICAL_TRANSITION = "STAIR_LANDING"


@dataclass
class TopologyNode:
    """A point node in the indoor topological navigation graph."""
    node_id: str
    building_id: str
    floor_id: int
    node_type: TopologyNodeType
    position: Point2D
    entity_id: str  # Associated Room ID, Door ID, Corridor ID, etc.
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "building_id": self.building_id,
            "floor_id": self.floor_id,
            "node_type": self.node_type.value,
            "position": self.position.to_dict(),
            "entity_id": self.entity_id,
            "label": self.label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopologyNode":
        return cls(
            node_id=data["node_id"],
            building_id=data["building_id"],
            floor_id=int(data["floor_id"]),
            node_type=TopologyNodeType(data["node_type"]),
            position=Point2D.from_dict(data["position"]),
            entity_id=data["entity_id"],
            label=data.get("label", ""),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class TopologyEdge:
    """A navigable connection edge linking two topological nodes."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    connection_type: ConnectionType
    distance_m: float
    is_bidirectional: bool = True
    accessibility: AccessibilityType = AccessibilityType.WHEELCHAIR_ACCESSIBLE
    source_floor_id: int = 0
    target_floor_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_vertical(self) -> bool:
        return self.source_floor_id != self.target_floor_id

    @property
    def source_id(self) -> str:
        return self.source_node_id

    @property
    def destination_id(self) -> str:
        return self.target_node_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "connection_type": self.connection_type.value,
            "distance_m": round(self.distance_m, 2),
            "is_bidirectional": self.is_bidirectional,
            "accessibility": self.accessibility.value,
            "is_vertical": self.is_vertical,
            "source_floor_id": self.source_floor_id,
            "target_floor_id": self.target_floor_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopologyEdge":
        return cls(
            edge_id=data["edge_id"],
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            connection_type=ConnectionType.from_string(data["connection_type"]),
            distance_m=float(data["distance_m"]),
            is_bidirectional=bool(data.get("is_bidirectional", True)),
            accessibility=AccessibilityType(data.get("accessibility", "WHEELCHAIR_ACCESSIBLE")),
            source_floor_id=int(data.get("source_floor_id", 0)),
            target_floor_id=int(data.get("target_floor_id", 0)),
            metadata=dict(data.get("metadata", {})),
        )


class IndoorTopologyGraph:
    """Graph structure managing indoor waypoints, door links, and cross-floor transitions."""

    def __init__(self):
        self.nodes: Dict[str, TopologyNode] = {}
        self.edges: Dict[str, TopologyEdge] = {}
        self._adjacency: Dict[str, List[str]] = {}  # node_id -> [edge_id, ...]

    def add_node(self, node: TopologyNode):
        """Adds a topological node to the graph."""
        self.nodes[node.node_id] = node
        if node.node_id not in self._adjacency:
            self._adjacency[node.node_id] = []

    def add_edge(self, edge: TopologyEdge):
        """Adds a directed or bidirectional edge between existing nodes."""
        if edge.source_node_id not in self.nodes:
            raise ValueError(f"Source node '{edge.source_node_id}' not found in graph.")
        if edge.target_node_id not in self.nodes:
            raise ValueError(f"Target node '{edge.target_node_id}' not found in graph.")

        self.edges[edge.edge_id] = edge
        self._adjacency[edge.source_node_id].append(edge.edge_id)
        if edge.is_bidirectional:
            self._adjacency[edge.target_node_id].append(edge.edge_id)

    @classmethod
    def from_building_map(cls, building_map: BuildingMap) -> "IndoorTopologyGraph":
        """Builds a complete topological navigation graph from a BuildingMap hierarchy."""
        graph = cls()

        # 1. Add Room Centroid Nodes
        for floor_id, floor in building_map.floors.items():
            for room_id, room in floor.rooms.items():
                node_id = f"NODE-RM-{room_id}"
                graph.add_node(
                    TopologyNode(
                        node_id=node_id,
                        building_id=building_map.building_id,
                        floor_id=floor_id,
                        node_type=TopologyNodeType.ROOM_CENTROID,
                        position=room.centroid,
                        entity_id=room_id,
                        label=room.name,
                    )
                )

        # 2. Add Corridor Waypoint & Intersection Nodes
        for floor_id, floor in building_map.floors.items():
            for corr_id, corridor in floor.corridors.items():
                prev_node_id = None
                for idx, pt in enumerate(corridor.centerline):
                    node_id = f"NODE-CORR-{corr_id}-P{idx}"
                    n_type = (
                        TopologyNodeType.CORRIDOR_INTERSECTION
                        if (idx == 0 or idx == len(corridor.centerline) - 1)
                        else TopologyNodeType.CORRIDOR_WAYPOINT
                    )
                    graph.add_node(
                        TopologyNode(
                            node_id=node_id,
                            building_id=building_map.building_id,
                            floor_id=floor_id,
                            node_type=n_type,
                            position=pt,
                            entity_id=corr_id,
                            label=f"{corridor.name} (Pt {idx})",
                        )
                    )

                    # Connect adjacent centerline waypoints
                    if prev_node_id is not None:
                        dist = graph.nodes[prev_node_id].position.distance_to(pt)
                        edge_id = f"EDGE-{prev_node_id}-{node_id}"
                        graph.add_edge(
                            TopologyEdge(
                                edge_id=edge_id,
                                source_node_id=prev_node_id,
                                target_node_id=node_id,
                                connection_type=ConnectionType.CORRIDOR,
                                distance_m=round(dist, 2),
                                is_bidirectional=True,
                                source_floor_id=floor_id,
                                target_floor_id=floor_id,
                            )
                        )
                    prev_node_id = node_id

        # 3. Add Door Nodes & Connect to Spaces
        for floor_id, floor in building_map.floors.items():
            for door_id, door in floor.doors.items():
                door_node_id = f"NODE-DR-{door_id}"
                n_type = TopologyNodeType.BUILDING_EXIT if door.is_emergency_exit else TopologyNodeType.DOORWAY
                graph.add_node(
                    TopologyNode(
                        node_id=door_node_id,
                        building_id=building_map.building_id,
                        floor_id=floor_id,
                        node_type=n_type,
                        position=door.position,
                        entity_id=door_id,
                        label=door.name,
                    )
                )

                # Connect door to connects_from space (e.g. room)
                from_room_node = f"NODE-RM-{door.connects_from_id}"
                if from_room_node in graph.nodes:
                    dist = door.position.distance_to(graph.nodes[from_room_node].position)
                    edge_id = f"EDGE-{from_room_node}-{door_node_id}"
                    graph.add_edge(
                        TopologyEdge(
                            edge_id=edge_id,
                            source_node_id=from_room_node,
                            target_node_id=door_node_id,
                            connection_type=ConnectionType.DOOR,
                            distance_m=round(dist, 2),
                            is_bidirectional=door.is_bidirectional,
                            source_floor_id=floor_id,
                            target_floor_id=floor_id,
                        )
                    )

                # Connect door to connects_to space (e.g. corridor or another room)
                to_room_node = f"NODE-RM-{door.connects_to_id}"
                if to_room_node in graph.nodes:
                    dist = door.position.distance_to(graph.nodes[to_room_node].position)
                    edge_id = f"EDGE-{door_node_id}-{to_room_node}"
                    graph.add_edge(
                        TopologyEdge(
                            edge_id=edge_id,
                            source_node_id=door_node_id,
                            target_node_id=to_room_node,
                            connection_type=ConnectionType.DOOR,
                            distance_m=round(dist, 2),
                            is_bidirectional=door.is_bidirectional,
                            source_floor_id=floor_id,
                            target_floor_id=floor_id,
                        )
                    )
                else:
                    # Connect door to closest corridor waypoint if connects_to is a corridor
                    corr_cand_prefix = f"NODE-CORR-{door.connects_to_id}"
                    corr_nodes = [nid for nid in graph.nodes if nid.startswith(corr_cand_prefix)]
                    if corr_nodes:
                        closest_corr_node = min(
                            corr_nodes,
                            key=lambda nid: door.position.distance_to(graph.nodes[nid].position)
                        )
                        dist = door.position.distance_to(graph.nodes[closest_corr_node].position)
                        edge_id = f"EDGE-{door_node_id}-{closest_corr_node}"
                        graph.add_edge(
                            TopologyEdge(
                                edge_id=edge_id,
                                source_node_id=door_node_id,
                                target_node_id=closest_corr_node,
                                connection_type=ConnectionType.DOOR,
                                distance_m=round(dist, 2),
                                is_bidirectional=door.is_bidirectional,
                                source_floor_id=floor_id,
                                target_floor_id=floor_id,
                            )
                        )

        # 4. Add Vertical Transitions (Staircases and Elevators)
        for trans_id, trans in building_map.vertical_transitions.items():
            landing_nodes: List[Tuple[int, str]] = []
            for f_id, pos in trans.floor_positions.items():
                node_id = f"NODE-TRANS-{trans_id}-F{f_id}"
                n_type = (
                    TopologyNodeType.ELEVATOR_LANDING
                    if trans.transition_type == ConnectionType.ELEVATOR
                    else TopologyNodeType.STAIR_LANDING
                )
                graph.add_node(
                    TopologyNode(
                        node_id=node_id,
                        building_id=building_map.building_id,
                        floor_id=f_id,
                        node_type=n_type,
                        position=pos,
                        entity_id=trans_id,
                        label=f"{trans.name} (F{f_id})",
                    )
                )
                landing_nodes.append((f_id, node_id))

            # Connect consecutive floor landings
            landing_nodes.sort(key=lambda item: item[0])
            for i in range(len(landing_nodes) - 1):
                f_a, n_a = landing_nodes[i]
                f_b, n_b = landing_nodes[i + 1]
                edge_id = f"EDGE-VERT-{n_a}-{n_b}"
                graph.add_edge(
                    TopologyEdge(
                        edge_id=edge_id,
                        source_node_id=n_a,
                        target_node_id=n_b,
                        connection_type=trans.transition_type,
                        distance_m=4.0 * abs(f_b - f_a),  # Typical floor height ~ 4m
                        is_bidirectional=True,
                        accessibility=(
                            AccessibilityType.WHEELCHAIR_ACCESSIBLE
                            if trans.is_accessible
                            else AccessibilityType.STAIRS_ONLY
                        ),
                        source_floor_id=f_a,
                        target_floor_id=f_b,
                    )
                )

        # 5. Add Explicit Connections
        def _resolve_connection_node(graph_ref: "IndoorTopologyGraph", raw_id: str, floor_id: int) -> Optional[str]:
            if raw_id in graph_ref.nodes:
                return raw_id
            # 1. Exact entity_id match on specified floor
            exact_matches = [nid for nid, n in graph_ref.nodes.items() if n.entity_id == raw_id and n.floor_id == floor_id]
            if exact_matches:
                return exact_matches[0]
            # 2. Check for transition IDs (e.g. vt_stairwell_a_f0 -> stairwell_a)
            for tid in building_map.vertical_transitions:
                if tid in raw_id:
                    cand = f"NODE-TRANS-{tid}-F{floor_id}"
                    if cand in graph_ref.nodes:
                        return cand
            # 3. Check for corridor nodes (e.g. corr_f0_main)
            corr_prefix = f"NODE-CORR-{raw_id}"
            corr_matches = [nid for nid in graph_ref.nodes if nid.startswith(corr_prefix)]
            if corr_matches:
                return corr_matches[len(corr_matches) // 2]
            # 4. Check for room nodes (e.g. room_101)
            rm_cand = f"NODE-RM-{raw_id}"
            if rm_cand in graph_ref.nodes:
                return rm_cand
            # 5. Check for door nodes (e.g. door_d101)
            dr_cand = f"NODE-DR-{raw_id}"
            if dr_cand in graph_ref.nodes:
                return dr_cand
            # 6. Fallback: partial match on entity_id
            for nid, node in graph_ref.nodes.items():
                if raw_id in nid or raw_id in node.entity_id:
                    return nid
            return None

        for conn_id, conn in building_map.connections.items():
            src_node = _resolve_connection_node(graph, conn.source_id, conn.source_floor_id)
            dst_node = _resolve_connection_node(graph, conn.destination_id, conn.destination_floor_id)

            if src_node and dst_node and src_node in graph.nodes and dst_node in graph.nodes:
                edge_id = f"EDGE-EXP-{conn_id}"
                if edge_id not in graph.edges:
                    graph.add_edge(
                        TopologyEdge(
                            edge_id=edge_id,
                            source_node_id=src_node,
                            target_node_id=dst_node,
                            connection_type=conn.connection_type,
                            distance_m=conn.distance_m,
                            is_bidirectional=conn.is_bidirectional,
                            accessibility=conn.accessibility,
                            source_floor_id=conn.source_floor_id,
                            target_floor_id=conn.destination_floor_id,
                        )
                    )

        return graph

    def get_connected_components(self) -> List[Set[str]]:
        """Returns all connected node components via Breadth-First Search (BFS)."""
        visited: Set[str] = set()
        components: List[Set[str]] = []

        for node_id in self.nodes:
            if node_id not in visited:
                comp: Set[str] = set()
                queue = [node_id]
                visited.add(node_id)
                comp.add(node_id)

                while queue:
                    curr = queue.pop(0)
                    for edge_id in self._adjacency.get(curr, []):
                        edge = self.edges.get(edge_id)
                        if not edge:
                            continue
                        neighbor = edge.target_node_id if edge.source_node_id == curr else edge.source_node_id
                        if neighbor not in visited:
                            visited.add(neighbor)
                            comp.add(neighbor)
                            queue.append(neighbor)

                components.append(comp)

        return components

    def find_isolated_nodes(self) -> List[str]:
        """Identifies nodes that have zero connecting edges."""
        return [node_id for node_id, edge_ids in self._adjacency.items() if len(edge_ids) == 0]

    def get_isolated_nodes(self) -> List[str]:
        """Alias for find_isolated_nodes()."""
        return self.find_isolated_nodes()

    def _resolve_node_id(self, identifier: str) -> Optional[str]:
        """Resolves either direct node_id or underlying entity_id."""
        if identifier in self.nodes:
            return identifier
        for nid, node in self.nodes.items():
            if node.entity_id == identifier:
                return nid
        return None

    def find_shortest_path(
        self,
        start_id: str,
        goal_id: str,
        wheelchair_accessible_only: bool = False,
    ) -> Tuple[Optional[List[str]], float]:
        """Finds shortest path between two spaces or nodes, returning entity IDs along path."""
        acc_filter = AccessibilityType.WHEELCHAIR_ACCESSIBLE if wheelchair_accessible_only else None
        start_node = self._resolve_node_id(start_id)
        goal_node = self._resolve_node_id(goal_id)
        if not start_node or not goal_node:
            return None, float("inf")
        node_path, dist = self.shortest_path(start_node, goal_node, accessibility_filter=acc_filter)
        if node_path is None:
            return None, float("inf")
        entity_path = [self.nodes[nid].entity_id if self.nodes[nid].entity_id else nid for nid in node_path]
        return entity_path, dist

    def shortest_path(
        self,
        start_node_id: str,
        goal_node_id: str,
        accessibility_filter: Optional[AccessibilityType] = None,
    ) -> Tuple[Optional[List[str]], float]:
        """Dijkstra algorithm finding shortest path between two nodes.

        Used for topological reachability testing and validation.
        """
        if start_node_id not in self.nodes or goal_node_id not in self.nodes:
            return None, float("inf")

        distances: Dict[str, float] = {nid: float("inf") for nid in self.nodes}
        previous: Dict[str, Optional[str]] = {nid: None for nid in self.nodes}
        distances[start_node_id] = 0.0

        pq: List[Tuple[float, str]] = [(0.0, start_node_id)]

        while pq:
            curr_dist, curr_node = heapq.heappop(pq)
            if curr_dist > distances[curr_node]:
                continue
            if curr_node == goal_node_id:
                break

            for edge_id in self._adjacency.get(curr_node, []):
                edge = self.edges.get(edge_id)
                if not edge:
                    continue
                if accessibility_filter == AccessibilityType.WHEELCHAIR_ACCESSIBLE:
                    if edge.accessibility == AccessibilityType.STAIRS_ONLY:
                        continue

                neighbor = edge.target_node_id if edge.source_node_id == curr_node else edge.source_node_id
                weight = edge.distance_m
                new_dist = curr_dist + weight

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = curr_node
                    heapq.heappush(pq, (new_dist, neighbor))

        if distances[goal_node_id] == float("inf"):
            return None, float("inf")

        # Reconstruct path
        path: List[str] = []
        curr: Optional[str] = goal_node_id
        while curr is not None:
            path.append(curr)
            curr = previous[curr]
        path.reverse()
        return path, round(distances[goal_node_id], 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "connected_components_count": len(self.get_connected_components()),
            "isolated_nodes_count": len(self.find_isolated_nodes()),
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
        }
