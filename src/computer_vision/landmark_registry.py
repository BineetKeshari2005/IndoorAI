"""Persistent registry and lifecycle manager for indoor landmarks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.computer_vision.models import (
    LandmarkCategory,
    LandmarkDetection,
    LandmarkRecord,
    VerificationStatus,
)
from src.core.logging import setup_logger

logger = setup_logger("LandmarkRegistry")


class LandmarkRegistry:
    """Manages persistent registration, manual verification, and querying of indoor landmarks."""

    def __init__(self, base_storage_dir: Optional[Path] = None):
        self.base_dir = base_storage_dir or Path("data/landmarks")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # In-memory index: building_id -> {landmark_id: LandmarkRecord}
        self._landmarks: Dict[str, Dict[str, LandmarkRecord]] = {}
        self._load_all_buildings()

    def _get_building_file(self, building_id: str) -> Path:
        clean_bldg = building_id.strip().replace("/", "_").replace("..", "")
        bldg_dir = self.base_dir / clean_bldg
        bldg_dir.mkdir(parents=True, exist_ok=True)
        return bldg_dir / "landmarks.json"

    def _load_all_buildings(self):
        """Loads all existing landmarks from disk."""
        if not self.base_dir.exists():
            return
        for bldg_dir in self.base_dir.iterdir():
            if bldg_dir.is_dir():
                json_file = bldg_dir / "landmarks.json"
                if json_file.is_file():
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            bldg_id = bldg_dir.name
                            self._landmarks[bldg_id] = {}
                            for item in data.get("landmarks", []):
                                rec = LandmarkRecord.from_dict(item)
                                self._landmarks[bldg_id][rec.landmark_id] = rec
                    except Exception as e:
                        logger.warning("Failed to load landmarks from %s: %s", json_file, e)

    def _save_building(self, building_id: str):
        """Persists landmarks of a specific building to disk atomically."""
        bldg_file = self._get_building_file(building_id)
        records = list(self._landmarks.get(building_id, {}).values())
        payload = {
            "building_id": building_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(records),
            "landmarks": [r.to_dict() for r in records],
        }

        temp_file = bldg_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            temp_file.replace(bldg_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise IOError(f"Failed to save landmarks for building '{building_id}': {e}") from e

    def register_detection(
        self,
        detection: LandmarkDetection,
        name: str = "",
        description: str = "",
        auto_verify: bool = False,
    ) -> LandmarkRecord:
        """Registers a detected landmark candidate into the registry.

        Default verification status is UNVERIFIED unless auto_verify is explicitly True.
        """
        building_id = detection.building_id or "UNASSIGNED"
        floor_id = detection.floor_id if detection.floor_id is not None else 0

        if building_id not in self._landmarks:
            self._landmarks[building_id] = {}

        # Generate unique structured landmark ID
        seq = len(self._landmarks[building_id]) + 1
        cat_prefix = detection.category.value[:3]
        landmark_id = f"LM-{building_id}-F{floor_id:02d}-{cat_prefix}-{seq:04d}"

        ocr_txt = detection.ocr_record.text if detection.ocr_record else None
        now_iso = datetime.now(timezone.utc).isoformat()
        status = VerificationStatus.MANUALLY_VERIFIED if auto_verify else VerificationStatus.UNVERIFIED

        templates = [detection.descriptor_vector] if detection.descriptor_vector else []

        record = LandmarkRecord(
            landmark_id=landmark_id,
            building_id=building_id,
            floor_id=floor_id,
            category=detection.category,
            name=name or f"{detection.category.value.title()} ({landmark_id})",
            description=description,
            reference_location_id=detection.reference_location_id,
            source_image_ids=[detection.photo_id] if detection.photo_id else [],
            associated_detection_ids=[detection.detection_id],
            ocr_text=ocr_txt,
            verification_status=status,
            created_at=now_iso,
            updated_at=now_iso,
            descriptor_templates=templates,
        )

        self._landmarks[building_id][landmark_id] = record
        self._save_building(building_id)
        logger.info("Registered landmark %s (%s) in %s F%d", landmark_id, record.category.value, building_id, floor_id)
        return record

    def verify_landmark(
        self,
        landmark_id: str,
        building_id: str,
        verified_name: Optional[str] = None,
        coordinates_m: Optional[Tuple[float, float]] = None,
        description: Optional[str] = None,
        status: VerificationStatus = VerificationStatus.MANUALLY_VERIFIED,
    ) -> Optional[LandmarkRecord]:
        """Manually marks a landmark as verified or rejected, with optional coordinates."""
        bldg_map = self._landmarks.get(building_id)
        if not bldg_map or landmark_id not in bldg_map:
            # Check all buildings
            for b_id, mapping in self._landmarks.items():
                if landmark_id in mapping:
                    bldg_map = mapping
                    building_id = b_id
                    break

        if not bldg_map or landmark_id not in bldg_map:
            return None

        record = bldg_map[landmark_id]
        record.verification_status = status
        if verified_name:
            record.name = verified_name
        if coordinates_m is not None:
            record.verified_coordinates_m = (float(coordinates_m[0]), float(coordinates_m[1]))
        if description:
            record.description = description
        record.updated_at = datetime.now(timezone.utc).isoformat()

        self._save_building(building_id)
        logger.info("Updated landmark %s status to %s", landmark_id, status.value)
        return record

    def associate_detection(
        self,
        landmark_id: str,
        building_id: str,
        detection: LandmarkDetection,
    ) -> Optional[LandmarkRecord]:
        """Associates a new photo detection to an existing physical landmark record."""
        bldg_map = self._landmarks.get(building_id)
        if not bldg_map or landmark_id not in bldg_map:
            return None

        record = bldg_map[landmark_id]
        if detection.photo_id and detection.photo_id not in record.source_image_ids:
            record.source_image_ids.append(detection.photo_id)
        if detection.detection_id and detection.detection_id not in record.associated_detection_ids:
            record.associated_detection_ids.append(detection.detection_id)
        if detection.descriptor_vector:
            record.descriptor_templates.append(detection.descriptor_vector)

        record.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_building(building_id)
        return record

    def get_landmark(self, landmark_id: str, building_id: Optional[str] = None) -> Optional[LandmarkRecord]:
        """Retrieves a single landmark by ID."""
        if building_id and building_id in self._landmarks:
            return self._landmarks[building_id].get(landmark_id)

        for bldg_records in self._landmarks.values():
            if landmark_id in bldg_records:
                return bldg_records[landmark_id]
        return None

    def list_landmarks(
        self,
        building_id: Optional[str] = None,
        floor_id: Optional[int] = None,
        category: Optional[LandmarkCategory] = None,
        status: Optional[VerificationStatus] = None,
    ) -> List[LandmarkRecord]:
        """Filters landmarks by building, floor, category, or verification status."""
        results: List[LandmarkRecord] = []
        target_buildings = [building_id] if (building_id and building_id in self._landmarks) else list(self._landmarks.keys())

        for b_id in target_buildings:
            for rec in self._landmarks.get(b_id, {}).values():
                if floor_id is not None and rec.floor_id != floor_id:
                    continue
                if category is not None and rec.category != category:
                    continue
                if status is not None and rec.verification_status != status:
                    continue
                results.append(rec)

        return results

    def clear(self):
        """Clears in-memory cache (primarily for isolated test fixtures)."""
        self._landmarks.clear()
