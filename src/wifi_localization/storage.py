"""Storage and serialization utilities for site-specific Wi-Fi radio maps and reference locations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.logging import setup_logger
from src.wifi_localization.models import RadioMap, ReferenceLocation

logger = setup_logger("RadioMapStorage")

DEFAULT_STORAGE_DIR = Path("data/radio_maps")


def get_radio_map_dir(site_id: str, version: str, base_dir: Optional[Path] = None) -> Path:
    """Get the target directory path for a site and version radio map."""
    root = base_dir or DEFAULT_STORAGE_DIR
    return root / site_id / version


def save_radio_map(radio_map: RadioMap, base_dir: Optional[Path] = None) -> Path:
    """Serialize and save a compiled radio map to JSON storage.

    Args:
        radio_map: The RadioMap instance to save.
        base_dir: Optional override for the root storage directory.

    Returns:
        Path to the saved directory.
    """
    target_dir = get_radio_map_dir(radio_map.metadata.site_id, radio_map.metadata.version, base_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save full radio map
    full_path = target_dir / "radio_map.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(radio_map.to_dict(), f, indent=2)

    # 2. Save metadata separately for fast indexing
    meta_path = target_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(radio_map.metadata.to_dict(), f, indent=2)

    # 3. Save locations separately
    loc_path = target_dir / "locations.json"
    with open(loc_path, "w", encoding="utf-8") as f:
        loc_data = {k: v.to_dict() for k, v in radio_map.locations.items()}
        json.dump(loc_data, f, indent=2)

    logger.info(f"Saved radio map '{radio_map.metadata.version}' for site '{radio_map.metadata.site_id}' to {target_dir}")
    return target_dir


def load_radio_map(site_id: str, version: str = "v1", base_dir: Optional[Path] = None) -> Optional[RadioMap]:
    """Load a radio map from disk.

    Args:
        site_id: Site identifier.
        version: Version string (defaults to 'v1').
        base_dir: Optional storage directory override.

    Returns:
        Loaded RadioMap or None if not found.
    """
    target_file = get_radio_map_dir(site_id, version, base_dir) / "radio_map.json"
    if not target_file.is_file():
        logger.warning(f"Radio map file not found: {target_file}")
        return None

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return RadioMap.from_dict(data)
    except Exception as e:
        logger.error(f"Failed to load radio map from {target_file}: {e}")
        return None


def list_radio_maps(site_id: Optional[str] = None, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List available radio maps on disk.

    Args:
        site_id: Optional site filter.
        base_dir: Optional storage root override.

    Returns:
        List of metadata dictionaries.
    """
    root = base_dir or DEFAULT_STORAGE_DIR
    if not root.is_dir():
        return []

    results: List[Dict[str, Any]] = []
    site_dirs = [root / site_id] if site_id else [p for p in root.iterdir() if p.is_dir()]

    for s_dir in site_dirs:
        if not s_dir.is_dir():
            continue
        cur_site = s_dir.name
        for v_dir in s_dir.iterdir():
            if not v_dir.is_dir():
                continue
            meta_file = v_dir / "metadata.json"
            if meta_file.is_file():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    results.append(meta)
                except Exception as e:
                    logger.warning(f"Error reading metadata from {meta_file}: {e}")

    return sorted(results, key=lambda m: (m.get("site_id", ""), m.get("version", "")))


def delete_radio_map(site_id: str, version: str, base_dir: Optional[Path] = None) -> bool:
    """Delete a radio map directory from disk."""
    import shutil

    target_dir = get_radio_map_dir(site_id, version, base_dir)
    if not target_dir.is_dir():
        return False
    shutil.rmtree(target_dir, ignore_errors=True)
    logger.info(f"Deleted radio map {site_id}/{version}")
    return True


def save_reference_locations(
    site_id: str,
    locations: Dict[str, ReferenceLocation],
    base_dir: Optional[Path] = None,
) -> Path:
    """Persist site-level reference location registry."""
    root = base_dir or DEFAULT_STORAGE_DIR
    site_dir = root / site_id
    site_dir.mkdir(parents=True, exist_ok=True)
    target_file = site_dir / "reference_locations.json"
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump({k: v.to_dict() for k, v in locations.items()}, f, indent=2)
    logger.info(f"Saved {len(locations)} reference locations to {target_file}")
    return target_file


def load_reference_locations(
    site_id: str,
    base_dir: Optional[Path] = None,
) -> Dict[str, ReferenceLocation]:
    """Load site-level reference location registry."""
    root = base_dir or DEFAULT_STORAGE_DIR
    target_file = root / site_id / "reference_locations.json"
    if not target_file.is_file():
        return {}
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: ReferenceLocation.from_dict(v) for k, v in data.items()}
    except Exception as e:
        logger.error(f"Failed to load reference locations from {target_file}: {e}")
        return {}
