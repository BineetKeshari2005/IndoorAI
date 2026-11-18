"""Backend service layer for computer vision and indoor landmark detection operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pandas as pd

from backend.app.core.dataset_config import DatasetConfig, default_dataset_config
from backend.app.services.dataset_registry import DatasetRegistry
from backend.app.schemas.computer_vision import (
    BoundingBoxSchema,
    CategoryEvaluationSchema,
    LandmarkDetectionSchema,
    LandmarkRecordSchema,
    LandmarkVerifyRequest,
    OcrTextSchema,
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
    VisionEvaluateRequest,
    VisionEvaluateResponse,
    VisualMatchCandidateSchema,
    VisualMatchRequest,
    VisualMatchResponse,
)
from src.computer_vision import (
    BoundingBox,
    LandmarkCategory,
    LandmarkDetection,
    LandmarkRecord,
    LandmarkRegistry,
    SafeImageLoader,
    VerificationStatus,
    VisionEvaluator,
    VisionPipeline,
    VisualLandmarkMatcher,
)
from src.core.logging import setup_logger

logger = setup_logger("ComputerVisionService")


class ComputerVisionService:
    """Orchestrates dataset computer vision analysis, landmark registry, matching, and evaluation."""

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        registry: Optional[DatasetRegistry] = None,
        landmark_registry: Optional[LandmarkRegistry] = None,
    ):
        self.config = config or default_dataset_config
        self.registry = registry or DatasetRegistry(self.config)
        self.landmarks_dir = self.config.base_dir / "landmarks"
        self.landmarks_dir.mkdir(parents=True, exist_ok=True)
        self.landmark_registry = landmark_registry or LandmarkRegistry(base_storage_dir=self.landmarks_dir)
        self.pipeline = VisionPipeline(registry=self.landmark_registry)

    def analyze_dataset(
        self,
        dataset_id: str,
        request_params: Optional[VisionAnalyzeRequest] = None,
    ) -> VisionAnalyzeResponse:
        """Executes landmark detection and sign OCR across all valid photos in a dataset session."""
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset with ID '{dataset_id}' not found in registry.")

        params = request_params or VisionAnalyzeRequest()
        extracted_dir = self.config.extracted_path / dataset_id
        if not extracted_dir.is_dir():
            raise ValueError(f"Extracted dataset directory not found at {extracted_dir}")

        # Update pipeline detector confidence
        self.pipeline.detector.confidence_threshold = params.confidence_threshold
        self.pipeline.detector.enable_ocr = params.enable_ocr

        session_id = dataset.get("session_id", "UNKNOWN") if isinstance(dataset, dict) else getattr(dataset, "session_id", "UNKNOWN")

        summary = self.pipeline.process_survey_photos(
            extracted_dataset_dir=extracted_dir,
            dataset_id=dataset_id,
            session_id=session_id,
            auto_register=params.auto_register,
        )

        detections_data = summary.get("detections", [])
        detection_schemas: List[LandmarkDetectionSchema] = []

        for d in detections_data:
            bb = BoundingBoxSchema(**d["bounding_box"])
            ocr = OcrTextSchema(**d["ocr_record"]) if d.get("ocr_record") else None
            detection_schemas.append(
                LandmarkDetectionSchema(
                    detection_id=d["detection_id"],
                    category=d["category"],
                    bounding_box=bb,
                    confidence=d["confidence"],
                    photo_id=d["photo_id"],
                    dataset_id=d["dataset_id"],
                    session_id=d["session_id"],
                    timestamp_epoch_ms=d["timestamp_epoch_ms"],
                    building_id=d.get("building_id"),
                    floor_id=d.get("floor_id"),
                    reference_location_id=d.get("reference_location_id"),
                    ocr_record=ocr,
                    has_descriptor=d.get("has_descriptor", False),
                )
            )

        # Save detections parquet if detections present
        if detections_data:
            try:
                flattened = []
                for d in detections_data:
                    flattened.append({
                        "detection_id": d["detection_id"],
                        "category": d["category"],
                        "confidence": d["confidence"],
                        "photo_id": d["photo_id"],
                        "dataset_id": d["dataset_id"],
                        "session_id": d["session_id"],
                        "timestamp_epoch_ms": d["timestamp_epoch_ms"],
                        "building_id": d.get("building_id", ""),
                        "floor_id": d.get("floor_id"),
                        "reference_location_id": d.get("reference_location_id", ""),
                        "x_min": d["bounding_box"]["x_min"],
                        "y_min": d["bounding_box"]["y_min"],
                        "x_max": d["bounding_box"]["x_max"],
                        "y_max": d["bounding_box"]["y_max"],
                        "ocr_text": d["ocr_record"]["text"] if d.get("ocr_record") else "",
                    })
                df = pd.DataFrame(flattened)
                parquet_out = extracted_dir / "vision_detections.parquet"
                df.to_parquet(parquet_out, index=False)
            except Exception as e:
                logger.warning("Could not persist vision_detections.parquet: %s", e)

        return VisionAnalyzeResponse(
            dataset_id=dataset_id,
            session_id=session_id,
            total_photos_scanned=summary["total_photos_scanned"],
            photos_processed_valid=summary["photos_processed_valid"],
            photos_blocked_privacy=summary["photos_blocked_privacy"],
            total_landmarks_detected=summary["total_landmarks_detected"],
            detections=detection_schemas,
        )

    def get_detections(
        self,
        dataset_id: str,
        category: Optional[str] = None,
    ) -> List[LandmarkDetectionSchema]:
        """Retrieves previously computed landmark detections for a dataset."""
        extracted_dir = self.config.extracted_path / dataset_id
        summary_file = extracted_dir / "vision_analysis_summary.json"

        if not summary_file.is_file():
            # Trigger analysis with defaults
            resp = self.analyze_dataset(dataset_id)
            detections = resp.detections
        else:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            detections = []
            for d in data.get("detections", []):
                bb = BoundingBoxSchema(**d["bounding_box"])
                ocr = OcrTextSchema(**d["ocr_record"]) if d.get("ocr_record") else None
                detections.append(
                    LandmarkDetectionSchema(
                        detection_id=d["detection_id"],
                        category=d["category"],
                        bounding_box=bb,
                        confidence=d["confidence"],
                        photo_id=d["photo_id"],
                        dataset_id=d["dataset_id"],
                        session_id=d["session_id"],
                        timestamp_epoch_ms=d["timestamp_epoch_ms"],
                        building_id=d.get("building_id"),
                        floor_id=d.get("floor_id"),
                        reference_location_id=d.get("reference_location_id"),
                        ocr_record=ocr,
                        has_descriptor=d.get("has_descriptor", False),
                    )
                )

        if category:
            clean_cat = category.strip().upper()
            detections = [d for d in detections if d.category == clean_cat]

        return detections

    def list_landmarks(
        self,
        building_id: Optional[str] = None,
        floor_id: Optional[int] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[LandmarkRecordSchema]:
        """Queries persistent landmarks in the landmark registry."""
        cat_enum = LandmarkCategory.from_string(category) if category else None
        stat_enum = None
        if status:
            try:
                stat_enum = VerificationStatus(status.strip().upper())
            except ValueError:
                pass

        records = self.landmark_registry.list_landmarks(
            building_id=building_id,
            floor_id=floor_id,
            category=cat_enum,
            status=stat_enum,
        )

        return [
            LandmarkRecordSchema(
                landmark_id=r.landmark_id,
                building_id=r.building_id,
                floor_id=r.floor_id,
                category=r.category.value,
                name=r.name,
                description=r.description,
                reference_location_id=r.reference_location_id,
                verified_coordinates_m=list(r.verified_coordinates_m) if r.verified_coordinates_m else None,
                source_image_ids=r.source_image_ids,
                associated_detection_ids=r.associated_detection_ids,
                ocr_text=r.ocr_text,
                verification_status=r.verification_status.value,
                created_at=r.created_at,
                updated_at=r.updated_at,
                template_count=len(r.descriptor_templates),
            )
            for r in records
        ]

    def verify_landmark(self, request_params: LandmarkVerifyRequest) -> Optional[LandmarkRecordSchema]:
        """Manually verifies or rejects a landmark record."""
        try:
            status_enum = VerificationStatus(request_params.status.strip().upper())
        except ValueError:
            status_enum = VerificationStatus.MANUALLY_VERIFIED

        coords = None
        if request_params.coordinates_m and len(request_params.coordinates_m) >= 2:
            coords = (float(request_params.coordinates_m[0]), float(request_params.coordinates_m[1]))

        updated = self.landmark_registry.verify_landmark(
            landmark_id=request_params.landmark_id,
            building_id=request_params.building_id,
            verified_name=request_params.verified_name,
            coordinates_m=coords,
            description=request_params.description,
            status=status_enum,
        )

        if not updated:
            return None

        return LandmarkRecordSchema(
            landmark_id=updated.landmark_id,
            building_id=updated.building_id,
            floor_id=updated.floor_id,
            category=updated.category.value,
            name=updated.name,
            description=updated.description,
            reference_location_id=updated.reference_location_id,
            verified_coordinates_m=list(updated.verified_coordinates_m) if updated.verified_coordinates_m else None,
            source_image_ids=updated.source_image_ids,
            associated_detection_ids=updated.associated_detection_ids,
            ocr_text=updated.ocr_text,
            verification_status=updated.verification_status.value,
            created_at=updated.created_at,
            updated_at=updated.updated_at,
            template_count=len(updated.descriptor_templates),
        )

    def match_photo(self, dataset_id: str, request_params: VisualMatchRequest) -> VisualMatchResponse:
        """Compares a photo in a dataset against registered landmarks."""
        extracted_dir = self.config.extracted_path / dataset_id
        images_dir = extracted_dir / "photos"
        if not images_dir.is_dir():
            images_dir = extracted_dir / "images"

        clean_fn = Path(request_params.photo_filename).name
        target_img = images_dir / clean_fn
        if not target_img.is_file():
            raise ValueError(f"Photo '{clean_fn}' not found in dataset images directory.")

        # Load and validate
        photo_csv = extracted_dir / "photo_metadata.csv"
        from src.computer_vision.image_validation import ImagePrivacyValidator
        meta_table = ImagePrivacyValidator.load_photo_metadata_table(photo_csv)
        meta = meta_table.get(clean_fn, {})

        bgr_img, val_res = SafeImageLoader.load_bgr(
            image_path=target_img,
            photo_id=clean_fn,
            enforce_sanitization=True,
            metadata_dict=meta,
        )

        rep = self.pipeline.matcher.match_query_image(
            query_bgr=bgr_img,
            query_image_id=clean_fn,
            building_id=request_params.building_id,
            floor_id=request_params.floor_id,
            max_candidates=request_params.max_candidates,
        )

        cand_schemas = [
            VisualMatchCandidateSchema(
                landmark_id=c.landmark_id,
                similarity_score=c.similarity_score,
                inlier_count=c.inlier_count,
                total_query_features=c.total_query_features,
                match_status=c.match_status.value,
                landmark_name=c.landmark_name,
                category=c.category.value,
                building_id=c.building_id,
                floor_id=c.floor_id,
                verified_coordinates_m=list(c.verified_coordinates_m) if c.verified_coordinates_m else None,
            )
            for c in rep.candidates
        ]

        return VisualMatchResponse(
            query_image_id=rep.query_image_id,
            match_status=rep.match_status.value,
            candidates=cand_schemas,
            best_candidate_id=rep.best_candidate_id,
            similarity_score=rep.similarity_score,
            timestamp_epoch_ms=rep.timestamp_epoch_ms,
            notes=rep.notes,
        )

    def evaluate_vision(self, request_params: VisionEvaluateRequest) -> VisionEvaluateResponse:
        """Evaluates detection, OCR, and visual matching against ground-truth benchmarks."""
        evaluator = VisionEvaluator(iou_threshold=request_params.iou_threshold)

        if request_params.is_synthetic:
            preds, gt, matches, gt_matches = evaluator.generate_controlled_synthetic_vision_fixture()
            report = evaluator.evaluate_detections(
                predictions_by_photo=preds,
                ground_truth_by_photo=gt,
                is_synthetic_fixture=True,
                matching_reports=matches,
                ground_truth_matches=gt_matches,
                notes="Controlled synthetic vision benchmark fixture (strictly partitioned).",
            )
        else:
            # When real manual annotations are not available, return clean un-fabricated report
            report = evaluator.evaluate_detections(
                predictions_by_photo={},
                ground_truth_by_photo={},
                is_synthetic_fixture=False,
                notes="No real manually labeled image ground-truth callset available for dataset.",
            )

        cat_dict = {}
        for cat_name, m in report.per_category.items():
            cat_dict[cat_name] = CategoryEvaluationSchema(
                category=m.category,
                true_positives=m.true_positives,
                false_positives=m.false_positives,
                false_negatives=m.false_negatives,
                precision=m.precision,
                recall=m.recall,
                f1_score=m.f1_score,
            )

        return VisionEvaluateResponse(
            evaluation_id=report.evaluation_id,
            is_synthetic_fixture=report.is_synthetic_fixture,
            total_images_evaluated=report.total_images_evaluated,
            total_ground_truth_boxes=report.total_ground_truth_boxes,
            total_predictions=report.total_predictions,
            overall_precision=report.overall_precision,
            overall_recall=report.overall_recall,
            overall_f1=report.overall_f1,
            mean_iou=report.mean_iou,
            per_category=cat_dict,
            confusion_matrix=report.confusion_matrix,
            ocr_exact_match_rate=report.ocr_exact_match_rate,
            ocr_character_error_rate=report.ocr_character_error_rate,
            matching_precision=report.matching_precision,
            matching_recall=report.matching_recall,
            notes=report.notes,
        )
