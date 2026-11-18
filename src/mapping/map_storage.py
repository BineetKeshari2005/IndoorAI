"""Atomic filesystem persistence and versioning for indoor maps."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.mapping.models import BuildingMap, FloorMap, SpatialConnection
from src.core.logging import setup_logger

logger = setup_logger("MapStorage")


class MapStorage:
    """Manages atomic saving, loading, versioning, and archiving of indoor building maps."""

    def __init__(self, base_storage_dir: Optional[Path] = None, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or base_storage_dir or Path("data/indoor_maps")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_building_dir(self, building_id: str) -> Path:
        clean_id = building_id.strip().replace("/", "_").replace("..", "")
        bldg_dir = self.base_dir / clean_id
        bldg_dir.mkdir(parents=True, exist_ok=True)
        return bldg_dir

    def save_building_map(self, building_map: BuildingMap, atomic: bool = True) -> BuildingMap:
        """Persists a complete BuildingMap to disk atomically."""
        bldg_dir = self._get_building_dir(building_map.building_id)
        target_dir = bldg_dir.parent / f"{bldg_dir.name}_tmp_{datetime.now().timestamp()}" if atomic else bldg_dir

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            floors_dir = target_dir / "floors"
            floors_dir.mkdir(parents=True, exist_ok=True)

            # 1. Save Building Root & Metadata
            building_meta = building_map.to_dict()
            floors_payload = building_meta.pop("floors", {})
            conn_payload = building_meta.pop("connections", {})

            with open(target_dir / "building.json", "w", encoding="utf-8") as f:
                json.dump(building_meta, f, indent=2)

            # 2. Save Individual Floors
            for f_id, f_data in floors_payload.items():
                floor_file = floors_dir / f"floor_{f_id}.json"
                with open(floor_file, "w", encoding="utf-8") as f:
                    json.dump(f_data, f, indent=2)

            # 3. Save Connections
            with open(target_dir / "connections.json", "w", encoding="utf-8") as f:
                json.dump({"connections": conn_payload}, f, indent=2)

            # 4. Atomic Swap
            if atomic:
                # Remove old files in bldg_dir and copy over
                # Note: preserve versions directory if present in bldg_dir
                versions_dir = bldg_dir / "versions"
                if versions_dir.exists():
                    target_versions = target_dir / "versions"
                    if not target_versions.exists():
                        shutil.copytree(versions_dir, target_versions)

                # Swap directories
                for item in bldg_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir() and item.name != "versions":
                        shutil.rmtree(item)

                for item in target_dir.iterdir():
                    dst = bldg_dir / item.name
                    if item.is_file():
                        shutil.copy2(item, dst)
                    elif item.is_dir() and item.name != "versions":
                        if dst.exists():
                            shutil.rmtree(dst)
                        shutil.copytree(item, dst)

                shutil.rmtree(target_dir, ignore_errors=True)

            logger.info("Saved building map '%s' (version %s)", building_map.building_id, building_map.map_version)
            return building_map
        except Exception as e:
            if atomic and target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            raise IOError(f"Failed to persist building map '{building_map.building_id}': {e}") from e

    def load_building_map(self, building_id: str) -> Optional[BuildingMap]:
        """Loads and reconstructs a BuildingMap from disk."""
        bldg_dir = self._get_building_dir(building_id)
        bldg_file = bldg_dir / "building.json"
        if not bldg_file.is_file():
            return None

        try:
            with open(bldg_file, "r", encoding="utf-8") as f:
                bldg_data = json.load(f)

            # Load Floors
            floors_dir = bldg_dir / "floors"
            floors_dict: Dict[int, Any] = {}
            if floors_dir.is_dir():
                for f_file in floors_dir.glob("floor_*.json"):
                    with open(f_file, "r", encoding="utf-8") as f:
                        f_data = json.load(f)
                        f_id = int(f_data.get("floor_id", 0))
                        floors_dict[f_id] = f_data

            bldg_data["floors"] = floors_dict

            # Load Connections
            conn_file = bldg_dir / "connections.json"
            if conn_file.is_file():
                with open(conn_file, "r", encoding="utf-8") as f:
                    c_data = json.load(f)
                    bldg_data["connections"] = c_data.get("connections", {})

            return BuildingMap.from_dict(bldg_data)
        except Exception as e:
            logger.error("Failed to load building map '%s': %s", building_id, e)
            return None

    def publish_version(
        self,
        building_id: str,
        new_version: Optional[str] = None,
        version_tag: Optional[str] = None,
        release_notes: str = "",
        publisher: str = "survey_operator",
    ) -> Optional[BuildingMap]:
        """Publishes a new immutable version of the building map."""
        bmap = self.load_building_map(building_id)
        if not bmap:
            return None

        ver = version_tag or new_version or "1.1.0"
        bmap.map_version = ver
        bmap.updated_at = datetime.now(timezone.utc).isoformat()
        bmap.metadata["last_published_version"] = ver
        bmap.metadata["release_notes"] = release_notes
        bmap.metadata["published_by"] = publisher

        # Save main current map
        self.save_building_map(bmap)

        # Archive into versions/v{version}/
        bldg_dir = self._get_building_dir(building_id)
        folder_name = ver if ver.startswith("v") else f"v{ver}"
        ver_dir = bldg_dir / "versions" / folder_name
        ver_dir.mkdir(parents=True, exist_ok=True)

        with open(ver_dir / "map_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": ver,
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "release_notes": release_notes,
                    "publisher": publisher,
                    "building_map": bmap.to_dict(),
                },
                f,
                indent=2,
            )

        logger.info("Published map '%s' version %s", building_id, ver)
        return bmap

    def list_buildings(self) -> List[Dict[str, Any]]:
        """Lists metadata summaries for all buildings present in map storage."""
        if not self.base_dir.exists():
            return []
        summaries: List[Dict[str, Any]] = []
        for d in sorted(self.base_dir.iterdir()):
            if d.is_dir():
                bldg_file = d / "building.json"
                if bldg_file.is_file():
                    try:
                        with open(bldg_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        # Count floors
                        floors_dir = d / "floors"
                        floor_count = len(list(floors_dir.glob("floor_*.json"))) if floors_dir.is_dir() else 0
                        summaries.append({
                            "building_id": data.get("building_id", d.name),
                            "name": data.get("name", d.name),
                            "map_version": data.get("map_version", "1.0.0"),
                            "floor_count": floor_count or data.get("floor_count", 0),
                            "total_rooms_count": data.get("total_rooms_count", 0),
                            "total_doors_count": data.get("total_doors_count", 0),
                            "updated_at": data.get("updated_at", ""),
                        })
                    except Exception as e:
                        logger.warning("Error reading building %s metadata: %s", d.name, e)
        return summaries

    def list_building_ids(self) -> List[str]:
        """Lists all building IDs present in map storage."""
        return [b["building_id"] for b in self.list_buildings()]

    def list_versions(self, building_id: str) -> List[Dict[str, Any]]:
        """Lists historical published versions for a building."""
        bldg_dir = self._get_building_dir(building_id)
        ver_dir = bldg_dir / "versions"
        if not ver_dir.is_dir():
            return []

        versions: List[Dict[str, Any]] = []
        for v_sub in sorted(ver_dir.iterdir()):
            if v_sub.is_dir():
                snap = v_sub / "map_snapshot.json"
                if snap.is_file():
                    try:
                        with open(snap, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            versions.append({
                                "version": data.get("version", v_sub.name),
                                "published_at": data.get("published_at"),
                                "release_notes": data.get("release_notes", ""),
                                "publisher": data.get("publisher", ""),
                            })
                    except Exception:
                        pass
        return versions
