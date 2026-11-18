"""Domain data models and schemas for indoor landmark detection and visual recognition."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class LandmarkCategory(str, Enum):
    """Taxonomy of indoor landmarks for indoor navigation and semantic mapping."""
    DOOR = "DOOR"
    ROOM_ENTRANCE = "ROOM_ENTRANCE"
    STAIRCASE = "STAIRCASE"
    ELEVATOR = "ELEVATOR"
    CORRIDOR = "CORRIDOR"
    INTERSECTION = "INTERSECTION"
    ROOM_NUMBER_SIGN = "ROOM_NUMBER_SIGN"
    DIRECTION_SIGN = "DIRECTION_SIGN"
    EXIT_SIGN = "EXIT_SIGN"
    DISTINCTIVE_STRUCTURE = "DISTINCTIVE_STRUCTURE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, val: str) -> "LandmarkCategory":
        """Safely parse category string."""
        clean = (val or "").strip().upper().replace(" ", "_")
        for member in cls:
            if member.value == clean or member.name == clean:
                return member
        return cls.UNKNOWN


class VerificationStatus(str, Enum):
    """Verification state for landmark records."""
    UNVERIFIED = "UNVERIFIED"
    MANUALLY_VERIFIED = "MANUALLY_VERIFIED"
    REJECTED = "REJECTED"


class VisualMatchStatus(str, Enum):
    """Categorical confidence status of visual descriptor matching."""
    STRONG_MATCH = "STRONG_MATCH"
    WEAK_MATCH = "WEAK_MATCH"
    NO_MATCH = "NO_MATCH"


@dataclass
class BoundingBox:
    """Normalized bounding box coordinates in [0.0, 1.0] relative to image dimensions.

    Conventions:
        x_min, y_min: top-left corner (0.0 <= x <= 1.0, 0.0 <= y <= 1.0)
        x_max, y_max: bottom-right corner
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self):
        # Clamp to [0.0, 1.0] and ensure valid rectangle
        self.x_min = max(0.0, min(1.0, float(self.x_min)))
        self.y_min = max(0.0, min(1.0, float(self.y_min)))
        self.x_max = max(0.0, min(1.0, float(self.x_max)))
        self.y_max = max(0.0, min(1.0, float(self.y_max)))
        if self.x_max < self.x_min:
            self.x_min, self.x_max = self.x_max, self.x_min
        if self.y_max < self.y_min:
            self.y_min, self.y_max = self.y_max, self.y_min

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_pixel_coords(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to absolute integer pixel rectangle (px_min, py_min, px_max, py_max)."""
        px_min = int(round(self.x_min * img_width))
        py_min = int(round(self.y_min * img_height))
        px_max = int(round(self.x_max * img_width))
        py_max = int(round(self.y_max * img_height))
        return (px_min, py_min, px_max, py_max)

    @classmethod
    def from_pixel_coords(
        cls,
        px_min: int,
        py_min: int,
        px_max: int,
        py_max: int,
        img_width: int,
        img_height: int
    ) -> "BoundingBox":
        """Construct normalized BoundingBox from integer pixel coordinates."""
        w = max(1, img_width)
        h = max(1, img_height)
        return cls(
            x_min=px_min / w,
            y_min=py_min / h,
            x_max=px_max / w,
            y_max=py_max / h,
        )

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union (IoU) with another bounding box."""
        inter_xmin = max(self.x_min, other.x_min)
        inter_ymin = max(self.y_min, other.y_min)
        inter_xmax = min(self.x_max, other.x_max)
        inter_ymax = min(self.y_max, other.y_max)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 1e-9:
            return 0.0
        return float(inter_area / union_area)

    def to_dict(self) -> Dict[str, float]:
        return {
            "x_min": round(self.x_min, 4),
            "y_min": round(self.y_min, 4),
            "x_max": round(self.x_max, 4),
            "y_max": round(self.y_max, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox":
        return cls(
            x_min=float(data.get("x_min", 0.0)),
            y_min=float(data.get("y_min", 0.0)),
            x_max=float(data.get("x_max", 1.0)),
            y_max=float(data.get("y_max", 1.0)),
        )


@dataclass
class OcrTextRecord:
    """Text recognized from a signboard or room plate."""
    ocr_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    confidence: float = 0.0
    bounding_box: Optional[BoundingBox] = None
    is_manually_verified: bool = False
    source_image_id: str = ""
    timestamp_epoch_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ocr_id": self.ocr_id,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "is_manually_verified": self.is_manually_verified,
            "source_image_id": self.source_image_id,
            "timestamp_epoch_ms": self.timestamp_epoch_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OcrTextRecord":
        bb_data = data.get("bounding_box")
        bb = BoundingBox.from_dict(bb_data) if bb_data else None
        return cls(
            ocr_id=data.get("ocr_id", str(uuid.uuid4())),
            text=data.get("text", ""),
            confidence=float(data.get("confidence", 0.0)),
            bounding_box=bb,
            is_manually_verified=bool(data.get("is_manually_verified", False)),
            source_image_id=data.get("source_image_id", ""),
            timestamp_epoch_ms=int(data.get("timestamp_epoch_ms", 0)),
        )


@dataclass
class LandmarkDetection:
    """A detected landmark candidate extracted from a survey photograph."""
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: LandmarkCategory = LandmarkCategory.UNKNOWN
    bounding_box: BoundingBox = field(default_factory=lambda: BoundingBox(0, 0, 1, 1))
    confidence: float = 0.0
    photo_id: str = ""
    dataset_id: str = ""
    session_id: str = ""
    timestamp_epoch_ms: int = 0
    building_id: Optional[str] = None
    floor_id: Optional[int] = None
    reference_location_id: Optional[str] = None
    ocr_record: Optional[OcrTextRecord] = None
    descriptor_vector: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "category": self.category.value,
            "bounding_box": self.bounding_box.to_dict(),
            "confidence": round(self.confidence, 4),
            "photo_id": self.photo_id,
            "dataset_id": self.dataset_id,
            "session_id": self.session_id,
            "timestamp_epoch_ms": self.timestamp_epoch_ms,
            "building_id": self.building_id,
            "floor_id": self.floor_id,
            "reference_location_id": self.reference_location_id,
            "ocr_record": self.ocr_record.to_dict() if self.ocr_record else None,
            "has_descriptor": self.descriptor_vector is not None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LandmarkDetection":
        bb_data = data.get("bounding_box", {})
        bb = BoundingBox.from_dict(bb_data) if bb_data else BoundingBox(0, 0, 1, 1)
        ocr_data = data.get("ocr_record")
        ocr = OcrTextRecord.from_dict(ocr_data) if ocr_data else None
        return cls(
            detection_id=data.get("detection_id", str(uuid.uuid4())),
            category=LandmarkCategory.from_string(data.get("category", "UNKNOWN")),
            bounding_box=bb,
            confidence=float(data.get("confidence", 0.0)),
            photo_id=data.get("photo_id", ""),
            dataset_id=data.get("dataset_id", ""),
            session_id=data.get("session_id", ""),
            timestamp_epoch_ms=int(data.get("timestamp_epoch_ms", 0)),
            building_id=data.get("building_id"),
            floor_id=int(data["floor_id"]) if data.get("floor_id") is not None else None,
            reference_location_id=data.get("reference_location_id"),
            ocr_record=ocr,
            descriptor_vector=data.get("descriptor_vector"),
        )


@dataclass
class LandmarkRecord:
    """A persistently registered physical indoor landmark."""
    landmark_id: str
    building_id: str
    floor_id: int
    category: LandmarkCategory
    name: str = ""
    description: str = ""
    reference_location_id: Optional[str] = None
    verified_coordinates_m: Optional[Tuple[float, float]] = None  # (x_m, y_m)
    source_image_ids: List[str] = field(default_factory=list)
    associated_detection_ids: List[str] = field(default_factory=list)
    ocr_text: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    descriptor_templates: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "landmark_id": self.landmark_id,
            "building_id": self.building_id,
            "floor_id": self.floor_id,
            "category": self.category.value,
            "name": self.name,
            "description": self.description,
            "reference_location_id": self.reference_location_id,
            "verified_coordinates_m": list(self.verified_coordinates_m) if self.verified_coordinates_m else None,
            "source_image_ids": self.source_image_ids,
            "associated_detection_ids": self.associated_detection_ids,
            "ocr_text": self.ocr_text,
            "verification_status": self.verification_status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "template_count": len(self.descriptor_templates),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LandmarkRecord":
        coords = None
        if data.get("verified_coordinates_m"):
            raw_c = data["verified_coordinates_m"]
            coords = (float(raw_c[0]), float(raw_c[1]))
        return cls(
            landmark_id=data["landmark_id"],
            building_id=data["building_id"],
            floor_id=int(data["floor_id"]),
            category=LandmarkCategory.from_string(data.get("category", "UNKNOWN")),
            name=data.get("name", ""),
            description=data.get("description", ""),
            reference_location_id=data.get("reference_location_id"),
            verified_coordinates_m=coords,
            source_image_ids=data.get("source_image_ids", []),
            associated_detection_ids=data.get("associated_detection_ids", []),
            ocr_text=data.get("ocr_text"),
            verification_status=VerificationStatus(data.get("verification_status", "UNVERIFIED")),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            descriptor_templates=data.get("descriptor_templates", []),
        )


@dataclass
class VisualMatchCandidate:
    """Candidate match when comparing a query image against a registered landmark."""
    landmark_id: str
    similarity_score: float  # [0.0, 1.0]
    inlier_count: int
    total_query_features: int
    match_status: VisualMatchStatus
    landmark_name: str
    category: LandmarkCategory
    building_id: str
    floor_id: int
    verified_coordinates_m: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "landmark_id": self.landmark_id,
            "similarity_score": round(self.similarity_score, 4),
            "inlier_count": self.inlier_count,
            "total_query_features": self.total_query_features,
            "match_status": self.match_status.value,
            "landmark_name": self.landmark_name,
            "category": self.category.value,
            "building_id": self.building_id,
            "floor_id": self.floor_id,
            "verified_coordinates_m": list(self.verified_coordinates_m) if self.verified_coordinates_m else None,
        }


@dataclass
class VisualMatchReport:
    """Result of visual landmark matching for a given query image."""
    query_image_id: str
    match_status: VisualMatchStatus
    candidates: List[VisualMatchCandidate] = field(default_factory=list)
    best_candidate_id: Optional[str] = None
    similarity_score: float = 0.0
    timestamp_epoch_ms: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_image_id": self.query_image_id,
            "match_status": self.match_status.value,
            "candidates": [c.to_dict() for c in self.candidates],
            "best_candidate_id": self.best_candidate_id,
            "similarity_score": round(self.similarity_score, 4),
            "timestamp_epoch_ms": self.timestamp_epoch_ms,
            "notes": self.notes,
        }


@dataclass
class CategoryEvaluationMetrics:
    """Performance metrics for an individual landmark category."""
    category: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    def compute(self):
        prec_denom = self.true_positives + self.false_positives
        rec_denom = self.true_positives + self.false_negatives
        self.precision = round(self.true_positives / prec_denom, 4) if prec_denom > 0 else 0.0
        self.recall = round(self.true_positives / rec_denom, 4) if rec_denom > 0 else 0.0
        f1_denom = self.precision + self.recall
        self.f1_score = round(2.0 * self.precision * self.recall / f1_denom, 4) if f1_denom > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }


@dataclass
class VisionEvaluationReport:
    """Comprehensive evaluation metrics for indoor landmark detection and matching."""
    evaluation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_synthetic_fixture: bool = False
    total_images_evaluated: int = 0
    total_ground_truth_boxes: int = 0
    total_predictions: int = 0
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0
    mean_iou: float = 0.0
    per_category: Dict[str, CategoryEvaluationMetrics] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    ocr_exact_match_rate: Optional[float] = None
    ocr_character_error_rate: Optional[float] = None
    matching_precision: Optional[float] = None
    matching_recall: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "is_synthetic_fixture": self.is_synthetic_fixture,
            "total_images_evaluated": self.total_images_evaluated,
            "total_ground_truth_boxes": self.total_ground_truth_boxes,
            "total_predictions": self.total_predictions,
            "overall_precision": round(self.overall_precision, 4),
            "overall_recall": round(self.overall_recall, 4),
            "overall_f1": round(self.overall_f1, 4),
            "mean_iou": round(self.mean_iou, 4),
            "per_category": {k: v.to_dict() for k, v in self.per_category.items()},
            "confusion_matrix": self.confusion_matrix,
            "ocr_exact_match_rate": round(self.ocr_exact_match_rate, 4) if self.ocr_exact_match_rate is not None else None,
            "ocr_character_error_rate": round(self.ocr_character_error_rate, 4) if self.ocr_character_error_rate is not None else None,
            "matching_precision": round(self.matching_precision, 4) if self.matching_precision is not None else None,
            "matching_recall": round(self.matching_recall, 4) if self.matching_recall is not None else None,
            "notes": self.notes,
        }
