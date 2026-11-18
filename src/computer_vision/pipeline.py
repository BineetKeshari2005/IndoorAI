"""End-to-end computer vision pipeline for indoor landmark detection and visual recognition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.computer_vision.image_loader import ImageLoadingError, SafeImageLoader
from src.computer_vision.image_validation import ImagePrivacyValidator, ImageValidationResult
from src.computer_vision.landmark_classifier import LandmarkClassifier
from src.computer_vision.landmark_detector import LandmarkDetector
from src.computer_vision.landmark_matching import VisualLandmarkMatcher
from src.computer_vision.landmark_registry import LandmarkRegistry
from src.computer_vision.models import (
    LandmarkCategory,
    LandmarkDetection,
    LandmarkRecord,
    VisualMatchReport,
)
from src.computer_vision.ocr_engine import SignboardOcrEngine
from src.computer_vision.preprocessing import ImagePreprocessor
from src.core.logging import setup_logger

logger = setup_logger("VisionPipeline")


class VisionPipeline:
    """Orchestrates privacy verification, image loading, landmark detection, OCR, and registry matching."""

    def __init__(
        self,
        registry: Optional[LandmarkRegistry] = None,
        detector: Optional[LandmarkDetector] = None,
        matcher: Optional[VisualLandmarkMatcher] = None,
        ocr_engine: Optional[SignboardOcrEngine] = None,
        confidence_threshold: float = 0.40,
        enable_ocr: bool = True,
        enforce_sanitization: bool = True,
    ):
        self.enforce_sanitization = enforce_sanitization
        self.registry = registry or LandmarkRegistry()
        self.ocr_engine = ocr_engine or (SignboardOcrEngine() if enable_ocr else None)
        self.detector = detector or LandmarkDetector(
            confidence_threshold=confidence_threshold,
            enable_ocr=enable_ocr,
            ocr_engine=self.ocr_engine,
        )
        self.matcher = matcher or VisualLandmarkMatcher(registry=self.registry)

    def process_single_image(
        self,
        image_path: Union[str, Path],
        photo_id: str = "",
        dataset_id: str = "",
        session_id: str = "",
        timestamp_epoch_ms: int = 0,
        metadata_dict: Optional[Dict[str, Any]] = None,
        auto_register: bool = False,
    ) -> Tuple[List[LandmarkDetection], Optional[VisualMatchReport], ImageValidationResult]:
        """Runs the complete vision pipeline on a single survey image."""
        path_obj = Path(image_path)
        pid = photo_id or path_obj.stem

        # 1. Safe Load & Privacy Validation
        try:
            bgr_img, val_result = SafeImageLoader.load_bgr(
                image_path=path_obj,
                photo_id=pid,
                enforce_sanitization=self.enforce_sanitization,
                metadata_dict=metadata_dict,
            )
        except ImageLoadingError as e:
            logger.warning("Image loading blocked by privacy or corruption: %s", e)
            val_res = ImageValidationResult(
                is_valid=False,
                photo_id=pid,
                errors=[str(e)],
                session_id=session_id,
                dataset_id=dataset_id,
            )
            return [], None, val_res

        # 2. Extract verified survey metadata
        bldg = metadata_dict.get("building") if metadata_dict else None
        flr = None
        if metadata_dict and metadata_dict.get("floor") is not None:
            try:
                flr = int(metadata_dict["floor"])
            except (ValueError, TypeError):
                pass
        room = metadata_dict.get("room") if metadata_dict else None

        # 3. Detect Landmarks
        detections = self.detector.detect(
            bgr_img=bgr_img,
            photo_id=pid,
            dataset_id=dataset_id,
            session_id=session_id,
            timestamp_epoch_ms=timestamp_epoch_ms,
            building_id=bldg,
            floor_id=flr,
            reference_location_id=room,
        )

        # 4. Optional Visual Matching against Landmark Registry
        match_report = self.matcher.match_query_image(
            query_bgr=bgr_img,
            query_image_id=pid,
            building_id=bldg,
            floor_id=flr,
        )

        # 5. Optional Auto-Registration into Registry (as UNVERIFIED)
        if auto_register:
            for det in detections:
                self.registry.register_detection(det, auto_verify=False)

        return detections, match_report, val_result

    def process_survey_photos(
        self,
        extracted_dataset_dir: Path,
        dataset_id: str,
        session_id: str,
        auto_register: bool = False,
    ) -> Dict[str, Any]:
        """Processes all photos in a survey dataset session respecting photo_metadata.csv."""
        photo_csv = extracted_dataset_dir / "photo_metadata.csv"
        images_dir = extracted_dataset_dir / "photos"
        if not images_dir.is_dir():
            images_dir = extracted_dataset_dir / "images"

        metadata_table = ImagePrivacyValidator.load_photo_metadata_table(photo_csv)

        total_photos = 0
        valid_photos = 0
        blocked_privacy = 0
        total_detections: List[LandmarkDetection] = []
        matching_reports: List[VisualMatchReport] = []

        if images_dir.is_dir():
            for img_file in sorted(images_dir.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    total_photos += 1
                    meta = metadata_table.get(img_file.name) or metadata_table.get(img_file.stem) or {}
                    ts = int(meta.get("normalized_epoch_ms", 0)) if meta.get("normalized_epoch_ms") else 0

                    dets, match_rep, val_res = self.process_single_image(
                        image_path=img_file,
                        photo_id=img_file.name,
                        dataset_id=dataset_id,
                        session_id=session_id,
                        timestamp_epoch_ms=ts,
                        metadata_dict=meta,
                        auto_register=auto_register,
                    )

                    if val_res.is_valid:
                        valid_photos += 1
                        total_detections.extend(dets)
                        if match_rep:
                            matching_reports.append(match_rep)
                    else:
                        blocked_privacy += 1

        summary = {
            "dataset_id": dataset_id,
            "session_id": session_id,
            "total_photos_scanned": total_photos,
            "photos_processed_valid": valid_photos,
            "photos_blocked_privacy": blocked_privacy,
            "total_landmarks_detected": len(total_detections),
            "detections": [d.to_dict() for d in total_detections],
            "matching_reports": [m.to_dict() for m in matching_reports],
        }

        # Persist summary
        out_file = extracted_dataset_dir / "vision_analysis_summary.json"
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.warning("Could not write vision analysis summary to %s: %s", out_file, e)

        return summary
