"""Map export utilities supporting JSON serialization and standard GeoJSON FeatureCollections."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.mapping.models import BuildingMap, FloorMap
from src.mapping.topology import IndoorTopologyGraph


class MapExporter:
    """Exports indoor spatial representations to JSON and RFC 7946 GeoJSON format."""

    def __init__(self, building_map: BuildingMap):
        self.building_map = building_map
        self.topology_graph = IndoorTopologyGraph.from_building_map(building_map)

    def export_json(self, floor_id: Optional[int] = None, indent: int = 2) -> str:
        """Export the building map or a single floor as a formatted JSON string."""
        if floor_id is not None:
            if floor_id not in self.building_map.floors:
                raise ValueError(f"Floor {floor_id} not found in building {self.building_map.building_id}")
            data = self.building_map.floors[floor_id].to_dict()
        else:
            data = self.building_map.to_dict()
        return json.dumps(data, indent=indent, default=str)

    def export_geojson(
        self,
        floor_id: Optional[int] = None,
        include_topology: bool = True,
        include_wifi_refs: bool = True,
        include_landmarks: bool = True,
    ) -> Dict[str, Any]:
        """Export spatial entities as an RFC 7946 GeoJSON FeatureCollection dictionary.

        Note: Coordinates are in BUILDING_LOCAL_CARTESIAN_ENU (+X East, +Y North, meters).
        """
        features: List[Dict[str, Any]] = []

        floors_to_export = (
            [self.building_map.floors[floor_id]]
            if (floor_id is not None and floor_id in self.building_map.floors)
            else list(self.building_map.floors.values())
        )

        for floor in floors_to_export:
            # 1. Floor boundary
            if floor.boundary:
                coords = [[p.x, p.y] for p in floor.boundary.vertices]
                if coords and coords[0] != coords[-1]:
                    coords.append(coords[0])
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords],
                    },
                    "properties": {
                        "entity_type": "floor_boundary",
                        "id": f"boundary_floor_{floor.floor_id}",
                        "floor_id": floor.floor_id,
                        "floor_number": floor.floor_number,
                        "label": floor.label,
                        "elevation_m": floor.elevation_m,
                        "area_sq_m": floor.total_floor_area_sq_m,
                    },
                })

            # 2. Rooms
            for r_id, room in floor.rooms.items():
                coords = [[p.x, p.y] for p in room.polygon.vertices]
                if coords and coords[0] != coords[-1]:
                    coords.append(coords[0])
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords],
                    },
                    "properties": {
                        "entity_type": "room",
                        "id": room.room_id,
                        "name": room.name,
                        "room_number": room.room_number,
                        "category": room.category.value,
                        "floor_id": room.floor_id,
                        "area_sq_m": round(room.area_sq_m, 2),
                        "verification_status": room.verification_status.value,
                        "centroid": [room.centroid.x, room.centroid.y],
                        "door_count": len(room.entrance_door_ids),
                    },
                })

            # 3. Corridors
            for c_id, corridor in floor.corridors.items():
                if corridor.polygon:
                    poly_coords = [[p.x, p.y] for p in corridor.polygon.vertices]
                    if poly_coords and poly_coords[0] != poly_coords[-1]:
                        poly_coords.append(poly_coords[0])
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [poly_coords],
                        },
                        "properties": {
                            "entity_type": "corridor_boundary",
                            "id": f"{corridor.corridor_id}_polygon",
                            "corridor_id": corridor.corridor_id,
                            "name": corridor.name,
                            "floor_id": corridor.floor_id,
                            "width_m": corridor.width_m,
                            "length_m": round(corridor.length_m, 2),
                            "verification_status": corridor.verification_status.value,
                        },
                    })

                centerline_coords = [[p.x, p.y] for p in corridor.centerline]
                if len(centerline_coords) >= 2:
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": centerline_coords,
                        },
                        "properties": {
                            "entity_type": "corridor_centerline",
                            "id": corridor.corridor_id,
                            "name": corridor.name,
                            "floor_id": corridor.floor_id,
                            "width_m": corridor.width_m,
                            "length_m": round(corridor.length_m, 2),
                            "verification_status": corridor.verification_status.value,
                        },
                    })

            # 4. Doors
            for d_id, door in floor.doors.items():
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [door.position.x, door.position.y],
                    },
                    "properties": {
                        "entity_type": "door",
                        "id": door.door_id,
                        "name": door.name,
                        "floor_id": door.floor_id,
                        "door_type": door.door_type.value,
                        "width_m": door.width_m,
                        "connects_from_id": door.connects_from_id,
                        "connects_to_id": door.connects_to_id,
                        "is_bidirectional": door.is_bidirectional,
                        "is_emergency_exit": door.is_emergency_exit,
                        "verification_status": door.verification_status.value,
                    },
                })

            # 5. Wi-Fi Reference Locations
            if include_wifi_refs and floor.wifi_reference_locations:
                for w_id, w_data in floor.wifi_reference_locations.items():
                    pos = w_data.get("position", {})
                    px = pos.get("x", 0.0)
                    py = pos.get("y", 0.0)
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [px, py],
                        },
                        "properties": {
                            "entity_type": "wifi_reference",
                            "id": w_id,
                            "label": w_data.get("label", w_id),
                            "floor_id": floor.floor_id,
                            "bssid_count": len(w_data.get("fingerprints", [])),
                        },
                    })

            # 6. Landmarks
            if include_landmarks and floor.landmarks:
                for lm_id, lm_data in floor.landmarks.items():
                    pos = lm_data.get("position", {})
                    px = pos.get("x", 0.0)
                    py = pos.get("y", 0.0)
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [px, py],
                        },
                        "properties": {
                            "entity_type": "landmark",
                            "id": lm_id,
                            "name": lm_data.get("name", lm_id),
                            "category": lm_data.get("category", "OTHER"),
                            "floor_id": floor.floor_id,
                            "verification_status": lm_data.get("verification_status", "UNVERIFIED"),
                        },
                    })

        # 7. Vertical Transitions (Stairwells and Elevators)
        for vt_id, vt in self.building_map.vertical_transitions.items():
            for f_id, pos in vt.floor_positions.items():
                if floor_id is not None and f_id != floor_id:
                    continue
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [pos.x, pos.y],
                    },
                    "properties": {
                        "entity_type": "vertical_transition",
                        "id": f"{vt.transition_id}_f{f_id}",
                        "transition_id": vt.transition_id,
                        "name": vt.name,
                        "transition_type": vt.transition_type.value,
                        "floor_id": f_id,
                        "connected_floors": vt.connected_floors,
                        "is_accessible": vt.is_accessible,
                        "verification_status": vt.verification_status.value,
                    },
                })

        # 8. Topological Navigation Graph Edges
        if include_topology:
            for edge in self.topology_graph.edges.values():
                src_node = self.topology_graph.nodes.get(edge.source_id)
                dst_node = self.topology_graph.nodes.get(edge.destination_id)
                if not src_node or not dst_node:
                    continue

                if floor_id is not None:
                    if src_node.floor_id != floor_id or dst_node.floor_id != floor_id:
                        continue

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [src_node.position.x, src_node.position.y],
                            [dst_node.position.x, dst_node.position.y],
                        ],
                    },
                    "properties": {
                        "entity_type": "topology_edge",
                        "id": edge.edge_id,
                        "source_id": edge.source_id,
                        "destination_id": edge.destination_id,
                        "source_type": src_node.node_type.value,
                        "destination_type": dst_node.node_type.value,
                        "distance_m": round(edge.distance_m, 2),
                        "connection_type": edge.connection_type.value,
                        "accessibility": edge.accessibility.value,
                        "is_bidirectional": edge.is_bidirectional,
                    },
                })

        return {
            "type": "FeatureCollection",
            "metadata": {
                "building_id": self.building_map.building_id,
                "building_name": self.building_map.name,
                "map_version": self.building_map.map_version,
                "coordinate_frame": {
                    "frame_id": self.building_map.coordinate_frame.frame_id,
                    "units": self.building_map.coordinate_frame.units,
                    "x_axis_description": self.building_map.coordinate_frame.x_axis_description,
                    "y_axis_description": self.building_map.coordinate_frame.y_axis_description,
                },
                "floor_id_filter": floor_id,
                "feature_count": len(features),
            },
            "features": features,
        }
