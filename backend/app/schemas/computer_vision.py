"""Pydantic request and response schemas for computer vision and indoor landmark APIs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class BoundingBoxSchema(BaseModel):
    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., ge=0.0, le=1.0)
    height: float = Field(..., ge=0.0, le=1.0)


class OcrTextSchema(BaseModel):
    ocr_id: str
    text: str
    confidence: float
    bounding_box: Optional[BoundingBoxSchema] = None
    is_manually_verified: bool = False
    source_image_id: str = ""
    timestamp_epoch_ms: int = 0


class LandmarkDetectionSchema(BaseModel):
    detection_id: str
    category: str
    bounding_box: BoundingBoxSchema
    confidence: float
    photo_id: str
    dataset_id: str
    session_id: str
    timestamp_epoch_ms: int
    building_id: Optional[str] = None
    floor_id: Optional[int] = None
    reference_location_id: Optional[str] = None
    ocr_record: Optional[OcrTextSchema] = None
    has_descriptor: bool = False


class VisionAnalyzeRequest(BaseModel):
    confidence_threshold: float = Field(0.40, ge=0.10, le=0.99, description="Minimum detection confidence score")
    enable_ocr: bool = Field(True, description="Whether to run OCR on detected signboards")
    auto_register: bool = Field(False, description="Whether to automatically register detections as unverified landmarks")


class VisionAnalyzeResponse(BaseModel):
    dataset_id: str
    session_id: str
    total_photos_scanned: int
    photos_processed_valid: int
    photos_blocked_privacy: int
    total_landmarks_detected: int
    detections: List[LandmarkDetectionSchema]


class LandmarkRecordSchema(BaseModel):
    landmark_id: str
    building_id: str
    floor_id: int
    category: str
    name: str
    description: str
    reference_location_id: Optional[str] = None
    verified_coordinates_m: Optional[List[float]] = None
    source_image_ids: List[str]
    associated_detection_ids: List[str]
    ocr_text: Optional[str] = None
    verification_status: str
    created_at: str
    updated_at: str
    template_count: int


class LandmarkVerifyRequest(BaseModel):
    landmark_id: str
    building_id: str
    status: str = Field("MANUALLY_VERIFIED", description="UNVERIFIED, MANUALLY_VERIFIED, or REJECTED")
    verified_name: Optional[str] = None
    coordinates_m: Optional[List[float]] = None  # [x_m, y_m]
    description: Optional[str] = None


class VisualMatchCandidateSchema(BaseModel):
    landmark_id: str
    similarity_score: float
    inlier_count: int
    total_query_features: int
    match_status: str
    landmark_name: str
    category: str
    building_id: str
    floor_id: int
    verified_coordinates_m: Optional[List[float]] = None


class VisualMatchRequest(BaseModel):
    photo_filename: str = Field(..., description="Filename of photo in dataset (e.g. photo_001.jpg)")
    building_id: Optional[str] = None
    floor_id: Optional[int] = None
    max_candidates: int = Field(5, ge=1, le=20)


class VisualMatchResponse(BaseModel):
    query_image_id: str
    match_status: str
    candidates: List[VisualMatchCandidateSchema]
    best_candidate_id: Optional[str] = None
    similarity_score: float
    timestamp_epoch_ms: int = 0
    notes: str


class VisionEvaluateRequest(BaseModel):
    is_synthetic: bool = Field(True, description="Run evaluation on controlled synthetic benchmark fixture")
    iou_threshold: float = Field(0.50, ge=0.10, le=0.95)


class CategoryEvaluationSchema(BaseModel):
    category: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float


class VisionEvaluateResponse(BaseModel):
    evaluation_id: str
    is_synthetic_fixture: bool
    total_images_evaluated: int
    total_ground_truth_boxes: int
    total_predictions: int
    overall_precision: float
    overall_recall: float
    overall_f1: float
    mean_iou: float
    per_category: Dict[str, CategoryEvaluationSchema]
    confusion_matrix: Dict[str, Dict[str, int]]
    ocr_exact_match_rate: Optional[float] = None
    ocr_character_error_rate: Optional[float] = None
    matching_precision: Optional[float] = None
    matching_recall: Optional[float] = None
    notes: str
