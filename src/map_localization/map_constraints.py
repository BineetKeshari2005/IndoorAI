"""Map-constraint engine enforcing physical wall, boundary, and floor transition rules."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from src.map_localization.models import ParticleFilterConfig
from src.map_localization.particle import Particle
from src.mapping.geometry import LineSegment2D, Point2D, Polygon2D
from src.mapping.models import BuildingMap, FloorMap, MapVerificationStatus


class MapConstraintEvaluator:
    """Evaluates spatial validity of particle step transitions against building geometry."""

    def __init__(self, config: Optional[ParticleFilterConfig] = None):
        self.config = config or ParticleFilterConfig()
        # Cache: (building_id, floor_id) -> list of (wall_segment, list_of_doors_on_segment)
        self._wall_cache: Dict[Tuple[str, int], List[Tuple[LineSegment2D, List[Point2D]]]] = {}

    def clear_cache(self) -> None:
        """Clears cached pre-compiled wall geometry."""
        self._wall_cache.clear()

    def _get_walls_for_floor(
        self,
        building_map: BuildingMap,
        floor_id: int,
    ) -> List[Tuple[LineSegment2D, List[Point2D]]]:
        """Extracts verified solid wall segments and associated door portals for a floor."""
        cache_key = (building_map.building_id, floor_id)
        if cache_key in self._wall_cache:
            return self._wall_cache[cache_key]

        if floor_id not in building_map.floors:
            return []

        floor = building_map.floors[floor_id]
        wall_entries: List[Tuple[LineSegment2D, List[Point2D]]] = []

        # Gather door positions on this floor
        door_positions: List[Point2D] = [d.position for d in floor.doors.values()]

        # 1. Process room polygon edges
        for r_id, room in floor.rooms.items():
            # Only enforce hard constraints for verified or partially surveyed room boundaries
            if room.verification_status in (MapVerificationStatus.VERIFIED, MapVerificationStatus.PARTIALLY_SURVEYED):
                edges = room.polygon.edges()
                for edge in edges:
                    # Find doors lying on or very near this wall segment (within 1.2m)
                    doors_on_wall = [
                        dp for dp in door_positions
                        if edge.distance_to_point(dp) <= 1.2
                    ]
                    wall_entries.append((edge, doors_on_wall))

        self._wall_cache[cache_key] = wall_entries
        return wall_entries

    def is_step_valid(
        self,
        p_old: Particle,
        p_new: Particle,
        building_map: Optional[BuildingMap],
    ) -> bool:
        """Checks if a particle's trajectory step from p_old to p_new crosses any solid walls."""
        if not building_map:
            return True

        floor_id = p_new.floor_id
        if floor_id not in building_map.floors:
            return self.config.allow_unknown_areas

        floor = building_map.floors[floor_id]

        # 1. Floor boundary containment check
        if floor.boundary and floor.boundary.is_valid_polygon:
            pt_new = Point2D(p_new.x, p_new.y)
            if not floor.boundary.contains_point(pt_new):
                return False

        # 2. Wall crossing check
        step_seg = LineSegment2D(Point2D(p_old.x, p_old.y), Point2D(p_new.x, p_new.y))
        wall_entries = self._get_walls_for_floor(building_map, floor_id)

        for wall_seg, doors_on_wall in wall_entries:
            if step_seg.intersects(wall_seg):
                # Line crossed a room wall. Check if crossing occurred at a door portal!
                if doors_on_wall:
                    # Permitted if the step passes within threshold of an opening
                    crossed_at_door = any(
                        step_seg.distance_to_point(dp) <= 1.2 for dp in doors_on_wall
                    )
                    if crossed_at_door:
                        continue  # Passed through a door!
                # Penetrated a solid wall without passing through a door!
                return False

        return True

    def can_transition_floor(
        self,
        p: Particle,
        target_floor_id: int,
        building_map: Optional[BuildingMap],
    ) -> bool:
        """Determines if a particle is near a vertical transition landing allowing floor change."""
        if p.floor_id == target_floor_id:
            return True

        if not building_map:
            return True

        cur_floor = p.floor_id
        pt = Point2D(p.x, p.y)
        max_r = self.config.max_vertical_transition_radius_m

        for vt in building_map.vertical_transitions.values():
            # Check if this staircase or elevator connects cur_floor and target_floor_id
            if cur_floor in vt.floor_positions and target_floor_id in vt.floor_positions:
                landing_pos = vt.floor_positions[cur_floor]
                if pt.distance_to(landing_pos) <= max_r:
                    return True

        return False

    def apply_constraints(
        self,
        old_particles: List[Particle],
        new_particles: List[Particle],
        building_map: Optional[BuildingMap],
    ) -> Tuple[List[Particle], int]:
        """Applies spatial constraints to the updated particle population.

        Particles that cross solid walls or exit floor boundaries have their weights zeroed
        or penalized, while valid particles retain their likelihoods.
        """
        if not building_map:
            return new_particles, 0

        rejected_count = 0
        constrained_particles: List[Particle] = []

        penalty = self.config.wall_crossing_penalty_factor

        for i in range(len(new_particles)):
            p_old = old_particles[i]
            p_new = new_particles[i]

            if not p_new.is_alive:
                constrained_particles.append(p_new)
                continue

            valid = self.is_step_valid(p_old, p_new, building_map)
            if not valid:
                rejected_count += 1
                p_copy = p_new.copy()
                if penalty <= 0.0:
                    p_copy.is_alive = False
                    p_copy.weight = 0.0
                    p_copy.log_weight = -1e9
                else:
                    p_copy.weight *= penalty
                    p_copy.log_weight += math.log(max(1e-12, penalty))
                constrained_particles.append(p_copy)
            else:
                constrained_particles.append(p_new)

        return constrained_particles, rejected_count
