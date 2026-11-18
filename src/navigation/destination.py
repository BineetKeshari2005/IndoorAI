"""Destination directory and search engine for verified indoor navigation targets (Task 28)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import setup_logger
from src.mapping.models import (
    AccessibilityType,
    BuildingMap,
    ConnectionType,
    FloorMap,
    MapVerificationStatus,
    Room,
    RoomCategory,
    VerticalTransition,
)
from src.navigation.models import Destination, DestinationCategory

logger = setup_logger("DestinationManager")


class DestinationManager:
    """Extracts, indexes, and searches verified destination targets across indoor facilities."""

    def __init__(self, building_map: Optional[BuildingMap] = None) -> None:
        self.building_map: Optional[BuildingMap] = None
        self._destinations: Dict[str, Destination] = {}
        if building_map:
            self.load_from_building_map(building_map)

    def load_from_building_map(self, building_map: BuildingMap) -> int:
        """Extracts and registers all verified destinations from a BuildingMap hierarchy."""
        self.building_map = building_map
        self._destinations.clear()
        map_ver = building_map.metadata.get("version", "1.0.0")

        # 1. Extract Room destinations
        for floor_id, floor in building_map.floors.items():
            for room_id, room in floor.rooms.items():
                # Enforce anti-fabrication invariant: only verified or partially surveyed rooms
                if room.verification_status == MapVerificationStatus.UNVERIFIED:
                    logger.debug(f"Skipping unverified room '{room_id}' as navigation destination.")
                    continue

                cat = self._map_room_category_to_destination(room.category)
                dest_id = f"DEST-RM-{room_id}"
                display_name = room.name if room.name else f"Room {room.room_number or room_id}"
                aliases = [room.name.lower()]
                if room.room_number:
                    aliases.append(room.room_number.lower())
                    aliases.append(f"room {room.room_number.lower()}")

                self._destinations[dest_id] = Destination(
                    destination_id=dest_id,
                    building_id=building_map.building_id,
                    floor_id=floor_id,
                    display_name=display_name,
                    category=cat,
                    entity_id=room_id,
                    verified_coordinates_m=(room.centroid.x, room.centroid.y),
                    verification_status=room.verification_status,
                    room_number=room.room_number,
                    is_accessible=True,
                    map_version=map_ver,
                    aliases=aliases,
                    metadata={"room_type": room.category.value, "doors": room.entrance_door_ids},
                )

        # 2. Extract Vertical Transition destinations (Elevators, Staircases, Ramps)
        for vt_id, vt in building_map.vertical_transitions.items():
            if vt.verification_status == MapVerificationStatus.UNVERIFIED:
                continue

            v_cat = DestinationCategory.ELEVATOR if vt.transition_type == ConnectionType.ELEVATOR else DestinationCategory.STAIRCASE
            is_acc = getattr(vt, "is_accessible", False)

            positions = getattr(vt, "floor_positions", {}) or {}
            for fl_id, landing_pt in positions.items():
                dest_id = f"DEST-VT-{vt_id}-F{fl_id}"
                type_name = "Elevator" if vt.transition_type == ConnectionType.ELEVATOR else "Stairwell"
                display_name = f"{vt.name} (Floor {fl_id})" if vt.name else f"{type_name} {vt_id} (Floor {fl_id})"

                self._destinations[dest_id] = Destination(
                    destination_id=dest_id,
                    building_id=building_map.building_id,
                    floor_id=fl_id,
                    display_name=display_name,
                    category=v_cat,
                    entity_id=vt_id,
                    verified_coordinates_m=(landing_pt.x, landing_pt.y),
                    verification_status=vt.verification_status,
                    is_accessible=is_acc,
                    map_version=map_ver,
                    aliases=[vt.name.lower(), type_name.lower()],
                    metadata={"transition_type": vt.transition_type.value, "connected_floors": vt.connected_floors},
                )

        # 3. Extract Verified Building Exits and Entrances
        for floor_id, floor in building_map.floors.items():
            for door_id, door in floor.doors.items():
                if door.verification_status == MapVerificationStatus.UNVERIFIED:
                    continue

                if door.is_emergency_exit:
                    dest_id = f"DEST-EXIT-{door_id}"
                    self._destinations[dest_id] = Destination(
                        destination_id=dest_id,
                        building_id=building_map.building_id,
                        floor_id=floor_id,
                        display_name=f"Emergency Exit ({door.name or door_id})",
                        category=DestinationCategory.EXIT,
                        entity_id=door_id,
                        verified_coordinates_m=(door.position.x, door.position.y),
                        verification_status=door.verification_status,
                        is_accessible=True,
                        map_version=map_ver,
                        aliases=["exit", "emergency exit", door.name.lower() if door.name else ""],
                    )

        # 4. Extract Verified Landmarks from floors
        for floor_id, floor in building_map.floors.items():
            for lm_id, lm_dict in floor.landmarks.items():
                # Only verified visual landmarks
                is_ver = lm_dict.get("is_verified", False) or lm_dict.get("verification_status") in ("VERIFIED", "MANUALLY_VERIFIED")
                if not is_ver:
                    continue

                coords = lm_dict.get("coordinates") or lm_dict.get("verified_coordinates_m") or [0.0, 0.0]
                dest_id = f"DEST-LM-{lm_id}"
                disp_name = lm_dict.get("name") or f"Landmark {lm_id}"
                self._destinations[dest_id] = Destination(
                    destination_id=dest_id,
                    building_id=building_map.building_id,
                    floor_id=floor_id,
                    display_name=disp_name,
                    category=DestinationCategory.LANDMARK,
                    entity_id=lm_id,
                    verified_coordinates_m=(float(coords[0]), float(coords[1])),
                    verification_status=MapVerificationStatus.VERIFIED,
                    is_accessible=True,
                    map_version=map_ver,
                    aliases=[disp_name.lower()],
                )

        logger.info(f"Loaded {len(self._destinations)} verified destinations for building '{building_map.building_id}'.")
        return len(self._destinations)

    @property
    def destinations(self) -> List[Destination]:
        """Returns all indexed destinations."""
        return list(self._destinations.values())

    def search(
        self,
        query: str = "",
        floor_id: Optional[int] = None,
        category: Optional[DestinationCategory] = None,
        limit: int = 20,
    ) -> List[Destination]:
        """Convenience alias for search_destinations."""
        return self.search_destinations(query=query, floor_id=floor_id, category=category, limit=limit)

    def get_destination(self, destination_id: str) -> Optional[Destination]:
        """Retrieves a destination by its destination_id or entity_id."""
        if destination_id in self._destinations:
            return self._destinations[destination_id]
        # Search by entity_id
        for d in self._destinations.values():
            if d.entity_id == destination_id or f"DEST-RM-{d.entity_id}" == destination_id:
                return d
        return None

    def list_destinations(
        self,
        floor_id: Optional[int] = None,
        category: Optional[DestinationCategory] = None,
        accessible_only: bool = False,
    ) -> List[Destination]:
        """Returns all registered destinations matching optional filters."""
        results = list(self._destinations.values())
        if floor_id is not None:
            results = [d for d in results if d.floor_id == floor_id]
        if category is not None:
            if category == DestinationCategory.VERTICAL_TRANSITION:
                results = [d for d in results if d.category in (DestinationCategory.VERTICAL_TRANSITION, DestinationCategory.ELEVATOR, DestinationCategory.STAIRCASE)]
            else:
                results = [d for d in results if d.category == category]
        if accessible_only:
            results = [d for d in results if d.is_accessible]
        return results

    def search_destinations(
        self,
        query: str,
        floor_id: Optional[int] = None,
        category: Optional[DestinationCategory] = None,
        limit: int = 10,
    ) -> List[Destination]:
        """Searches destinations by free text, room number, or category with ranking."""
        q_clean = query.strip().lower()
        if not q_clean:
            return self.list_destinations(floor_id=floor_id, category=category)[:limit]

        scored_candidates: List[Tuple[float, Destination]] = []

        for dest in self._destinations.values():
            if floor_id is not None and dest.floor_id != floor_id:
                continue
            if category is not None:
                if category == DestinationCategory.VERTICAL_TRANSITION:
                    if dest.category not in (DestinationCategory.VERTICAL_TRANSITION, DestinationCategory.ELEVATOR, DestinationCategory.STAIRCASE):
                        continue
                elif dest.category != category:
                    continue

            score = 0.0
            name_clean = dest.display_name.lower()
            rm_clean = (dest.room_number or "").lower()

            # Exact match checks
            if q_clean == name_clean or q_clean == rm_clean:
                score += 100.0
            elif q_clean in name_clean:
                score += 50.0 + (len(q_clean) / max(1, len(name_clean))) * 20.0
            elif rm_clean and q_clean in rm_clean:
                score += 60.0
            elif any(q_clean in alias for alias in dest.aliases):
                score += 40.0
            elif dest.category.value.lower() == q_clean:
                score += 30.0

            # Token overlap
            tokens = set(re.findall(r"\w+", q_clean))
            dest_tokens = set(re.findall(r"\w+", name_clean))
            overlap = tokens.intersection(dest_tokens)
            if overlap:
                score += len(overlap) * 15.0

            if score > 0.0:
                scored_candidates.append((score, dest))

        # Sort descending by match score
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        return [dest for _, dest in scored_candidates[:limit]]

    @staticmethod
    def _map_room_category_to_destination(cat: RoomCategory) -> DestinationCategory:
        mapping = {
            RoomCategory.OFFICE: DestinationCategory.OFFICE,
            RoomCategory.LAB: DestinationCategory.LABORATORY,
            RoomCategory.CLASSROOM: DestinationCategory.CLASSROOM,
            RoomCategory.RESTROOM: DestinationCategory.RESTROOM,
            RoomCategory.CONFERENCE: DestinationCategory.CONFERENCE,
            RoomCategory.STAIRWELL: DestinationCategory.STAIRCASE,
            RoomCategory.ELEVATOR_SHAFT: DestinationCategory.ELEVATOR,
            RoomCategory.LOBBY: DestinationCategory.ROOM,
            RoomCategory.OTHER: DestinationCategory.ROOM,
        }
        return mapping.get(cat, DestinationCategory.ROOM)
