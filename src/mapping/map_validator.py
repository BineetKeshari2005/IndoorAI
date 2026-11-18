"""Map validation engine enforcing geometric sanity, referential integrity, and topological connectivity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from src.mapping.geometry import Point2D, Polygon2D
from src.mapping.models import (
    BuildingMap,
    ConnectionType,
    FloorMap,
    MapVerificationStatus,
)
from src.mapping.topology import IndoorTopologyGraph
from src.core.logging import setup_logger

logger = setup_logger("MapValidator")


from enum import Enum


class ValidationSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class MapValidationError:
    """Detailed validation issue description."""
    error_type: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    entity_id: Optional[str] = None
    floor_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "entity_id": self.entity_id,
            "floor_id": self.floor_id,
        }


@dataclass
class MapValidationReport:
    """Structured audit report evaluating geometric, referential, and topological validity."""
    is_valid: bool
    building_id: str
    map_version: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    detailed_errors: List[MapValidationError] = field(default_factory=list)
    detailed_warnings: List[MapValidationError] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        return f"Validation {status}: {self.error_count} error(s), {self.warning_count} warning(s)."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "building_id": self.building_id,
            "map_version": self.map_version,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "summary": self.summary,
            "errors": self.errors,
            "warnings": self.warnings,
            "detailed_errors": [e.to_dict() for e in self.detailed_errors],
            "detailed_warnings": [w.to_dict() for w in self.detailed_warnings],
            "statistics": self.statistics,
            "timestamp": self.timestamp,
        }


class MapValidator:
    """Performs rigorous multi-stage validation on BuildingMap structures."""

    def __init__(self, max_vertical_planar_displacement_m: float = 8.0):
        self.max_vertical_planar_displacement_m = max_vertical_planar_displacement_m

    def validate(self, building_map: BuildingMap) -> MapValidationReport:
        """Executes full validation suite on the building map."""
        errors: List[str] = []
        warnings: List[str] = []
        detailed_errors: List[MapValidationError] = []
        detailed_warnings: List[MapValidationError] = []

        def add_err(etype: str, msg: str, ent_id: Optional[str] = None, fid: Optional[int] = None):
            errors.append(msg)
            detailed_errors.append(MapValidationError(error_type=etype, message=msg, severity=ValidationSeverity.ERROR, entity_id=ent_id, floor_id=fid))

        def add_warn(wtype: str, msg: str, ent_id: Optional[str] = None, fid: Optional[int] = None):
            warnings.append(msg)
            detailed_warnings.append(MapValidationError(error_type=wtype, message=msg, severity=ValidationSeverity.WARNING, entity_id=ent_id, floor_id=fid))

        # 1. Coordinate Frame Verification
        if building_map.coordinate_frame.frame_id != "BUILDING_LOCAL_CARTESIAN_ENU":
            add_warn(
                "NON_STANDARD_COORDINATE_FRAME",
                f"Non-standard coordinate frame '{building_map.coordinate_frame.frame_id}'. Recommended: 'BUILDING_LOCAL_CARTESIAN_ENU'."
            )
        if building_map.coordinate_frame.units != "meters":
            add_err("INVALID_UNITS", f"Invalid coordinate units '{building_map.coordinate_frame.units}'. Must be 'meters'.")

        # 2. Floor References and Levels
        if not building_map.floors:
            add_err("MISSING_FLOORS", f"Building '{building_map.building_id}' contains 0 floor definitions.")

        known_floor_ids = set(building_map.floors.keys())
        known_room_ids: Set[str] = set()
        known_door_ids: Set[str] = set()
        known_corridor_ids: Set[str] = set()

        total_area = 0.0
        surveyed_area = 0.0

        for f_id, floor in building_map.floors.items():
            if floor.building_id != building_map.building_id:
                add_err(
                    "BUILDING_ID_MISMATCH",
                    f"Floor {f_id} has mismatched building_id '{floor.building_id}' (expected '{building_map.building_id}').",
                    fid=f_id,
                )

            # Check boundary polygon
            if floor.boundary:
                if not floor.boundary.is_valid_polygon:
                    add_err("INVALID_FLOOR_BOUNDARY", f"Floor {f_id} has an invalid or self-intersecting boundary polygon.", fid=f_id)
                total_area += floor.boundary.area

            # Check Rooms
            for r_id, room in floor.rooms.items():
                if r_id in known_room_ids:
                    add_err("DUPLICATE_ROOM_ID", f"Duplicate room_id '{r_id}' detected across floors.", ent_id=r_id, fid=f_id)
                known_room_ids.add(r_id)

                if room.floor_id != f_id:
                    add_err("FLOOR_ID_MISMATCH", f"Room '{r_id}' declared floor_id {room.floor_id} != parent floor {f_id}.", ent_id=r_id, fid=f_id)

                # Geometric validation
                if not room.polygon.is_valid_polygon:
                    add_err("SELF_INTERSECTING_POLYGON", f"Room '{r_id}' has an invalid or self-intersecting polygon.", ent_id=r_id, fid=f_id)
                elif room.polygon.area < 0.5:
                    add_warn("SMALL_ROOM_AREA", f"Room '{r_id}' area ({room.polygon.area:.2f} m^2) is unusually small.", ent_id=r_id, fid=f_id)

                # Check containment within floor boundary if boundary exists
                if floor.boundary and floor.boundary.is_valid_polygon:
                    if not floor.boundary.contains_point(room.centroid):
                        add_warn("ROOM_OUTSIDE_FLOOR_BOUNDARY", f"Room '{r_id}' centroid falls outside floor {f_id} boundary.", ent_id=r_id, fid=f_id)

                surveyed_area += room.area_sq_m

            # Check Corridors
            for c_id, corridor in floor.corridors.items():
                if c_id in known_corridor_ids:
                    add_err("DUPLICATE_CORRIDOR_ID", f"Duplicate corridor_id '{c_id}' detected.", ent_id=c_id, fid=f_id)
                known_corridor_ids.add(c_id)

                if len(corridor.centerline) < 2:
                    add_err("CORRIDOR_TOO_SHORT", f"Corridor '{c_id}' centerline has < 2 waypoints.", ent_id=c_id, fid=f_id)
                if corridor.length_m < 0.5:
                    add_warn("SHORT_CORRIDOR", f"Corridor '{c_id}' length ({corridor.length_m:.2f} m) is unusually short.", ent_id=c_id, fid=f_id)

            # Check Doors
            for d_id, door in floor.doors.items():
                if d_id in known_door_ids:
                    add_err("DUPLICATE_DOOR_ID", f"Duplicate door_id '{d_id}' detected.", ent_id=d_id, fid=f_id)
                known_door_ids.add(d_id)

                # Check connectivity references
                from_id = door.connects_from_id
                to_id = door.connects_to_id

                from_valid = (from_id in floor.rooms or from_id in floor.corridors)
                to_valid = (to_id in floor.rooms or to_id in floor.corridors)

                if not from_valid:
                    add_warn("DOOR_DANGLING_FROM", f"Door '{d_id}' connects_from_id '{from_id}' not found in floor {f_id} rooms/corridors.", ent_id=d_id, fid=f_id)
                if not to_valid:
                    add_warn("DOOR_DANGLING_TO", f"Door '{d_id}' connects_to_id '{to_id}' not found in floor {f_id} rooms/corridors.", ent_id=d_id, fid=f_id)

        # 3. Vertical Transitions Check
        for trans_id, trans in building_map.vertical_transitions.items():
            if trans.building_id != building_map.building_id:
                add_err("VERTICAL_TRANSITION_BUILDING_MISMATCH", f"Vertical transition '{trans_id}' has mismatched building_id.", ent_id=trans_id)

            if len(trans.floor_positions) < 2:
                add_err("VERTICAL_TRANSITION_TOO_FEW_FLOORS", f"Vertical transition '{trans_id}' connects fewer than 2 floors.", ent_id=trans_id)

            # Verify all connected floors exist
            for connected_fid in trans.floor_positions:
                if connected_fid not in known_floor_ids:
                    add_err("VERTICAL_TRANSITION_INVALID_FLOOR", f"Vertical transition '{trans_id}' references non-existent floor {connected_fid}.", ent_id=trans_id)

            # Planar coordinate consistency check across landings
            positions = list(trans.floor_positions.values())
            for i in range(len(positions) - 1):
                p_a = positions[i]
                p_b = positions[i + 1]
                displacement = p_a.distance_to(p_b)
                if displacement > self.max_vertical_planar_displacement_m:
                    add_err(
                        "VERTICAL_TRANSITION_MISALIGNMENT",
                        f"Vertical transition '{trans_id}' has excessive planar displacement ({displacement:.1f}m > {self.max_vertical_planar_displacement_m}m) between landings.",
                        ent_id=trans_id,
                    )

        # 4. Topological Graph Connectivity & Reachability
        try:
            graph = IndoorTopologyGraph.from_building_map(building_map)
            isolated = graph.find_isolated_nodes()
            if isolated:
                add_warn("ISOLATED_TOPOLOGICAL_NODES", f"Found {len(isolated)} isolated topological nodes with 0 connections: {isolated[:3]}...")

            components = graph.get_connected_components()
            if len(components) > len(building_map.floors):
                add_warn(
                    "DISCONNECTED_COMPONENTS",
                    f"Topological graph has {len(components)} disconnected components across {len(building_map.floors)} floors."
                )
        except Exception as e:
            add_err("TOPOLOGY_GRAPH_ERROR", f"Topological graph construction failed: {e}")

        # 5. Compile Statistics
        stats = {
            "floor_count": len(building_map.floors),
            "room_count": len(known_room_ids),
            "corridor_count": len(known_corridor_ids),
            "door_count": len(known_door_ids),
            "vertical_transition_count": len(building_map.vertical_transitions),
            "explicit_connection_count": len(building_map.connections),
            "total_floor_area_sq_m": round(total_area, 2),
            "surveyed_room_area_sq_m": round(surveyed_area, 2),
            "coverage_ratio": round(surveyed_area / total_area, 3) if total_area > 0 else 1.0,
        }

        is_valid = (len(errors) == 0)
        report = MapValidationReport(
            is_valid=is_valid,
            building_id=building_map.building_id,
            map_version=building_map.map_version,
            errors=errors,
            warnings=warnings,
            detailed_errors=detailed_errors,
            detailed_warnings=detailed_warnings,
            statistics=stats,
        )

        if not is_valid:
            logger.warning("Map validation for %s failed with %d errors.", building_map.building_id, len(errors))
        else:
            logger.info("Map validation for %s passed (%d warnings).", building_map.building_id, len(warnings))

        return report
