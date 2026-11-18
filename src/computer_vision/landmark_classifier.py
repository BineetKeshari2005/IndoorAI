"""Landmark classification verification, taxonomy normalization, and metadata validation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.computer_vision.models import (
    LandmarkCategory,
    LandmarkDetection,
    VerificationStatus,
)
from src.core.logging import setup_logger

logger = setup_logger("LandmarkClassifier")


class LandmarkClassifier:
    """Classifies and validates indoor landmark candidates against survey metadata."""

    @staticmethod
    def validate_category(category_str: str) -> LandmarkCategory:
        """Parses and validates category against supported taxonomy."""
        return LandmarkCategory.from_string(category_str)

    @classmethod
    def sanitize_detection_metadata(
        cls,
        detection: LandmarkDetection,
        survey_metadata: Optional[Dict[str, Any]] = None,
    ) -> LandmarkDetection:
        """Ensures building and floor IDs are inherited exclusively from verified survey provenance.

        Strict Principle: Never invent building or floor labels based on visual appearance.
        """
        if survey_metadata:
            # Overwrite with verified ground-truth metadata from session
            if "building" in survey_metadata and survey_metadata["building"]:
                detection.building_id = str(survey_metadata["building"]).strip()
            if "floor" in survey_metadata and survey_metadata["floor"] is not None:
                try:
                    detection.floor_id = int(survey_metadata["floor"])
                except (ValueError, TypeError):
                    pass
            if "room" in survey_metadata and survey_metadata["room"]:
                detection.reference_location_id = str(survey_metadata["room"]).strip()

        return detection

    @classmethod
    def filter_by_categories(
        cls,
        detections: List[LandmarkDetection],
        allowed_categories: Optional[List[LandmarkCategory]] = None,
    ) -> List[LandmarkDetection]:
        """Filters detections to only requested landmark categories."""
        if not allowed_categories:
            return detections
        allowed_set = set(allowed_categories)
        return [d for d in detections if d.category in allowed_set]
