"""Indoor landmark detector combining deep neural detection and architectural heuristics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.computer_vision.feature_extraction import VisualFeatureExtractor
from src.computer_vision.models import (
    BoundingBox,
    LandmarkCategory,
    LandmarkDetection,
)
from src.computer_vision.ocr_engine import SignboardOcrEngine
from src.computer_vision.preprocessing import ImagePreprocessor
from src.core.logging import setup_logger

logger = setup_logger("LandmarkDetector")


class LandmarkDetector:
    """Detects indoor architectural landmarks and signboards from photographs."""

    def __init__(
        self,
        confidence_threshold: float = 0.40,
        nms_iou_threshold: float = 0.45,
        model_weights_path: Optional[str] = None,
        enable_ocr: bool = True,
        ocr_engine: Optional[SignboardOcrEngine] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.model_weights_path = model_weights_path
        self.enable_ocr = enable_ocr
        self.ocr_engine = ocr_engine or (SignboardOcrEngine() if enable_ocr else None)
        self.feature_extractor = VisualFeatureExtractor()

        # Try loading fine-tuned YOLO weights if provided
        self._yolo_model = None
        if self.model_weights_path and Path(self.model_weights_path).is_file():
            self._load_yolo_model()

    def _load_yolo_model(self):
        """Attempts to load fine-tuned Ultralytics YOLO model."""
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO(self.model_weights_path)
            logger.info("Loaded custom YOLO weights from %s", self.model_weights_path)
        except Exception as e:
            logger.warning("Failed to load custom YOLO weights: %s", e)
            self._yolo_model = None

    def detect(
        self,
        bgr_img: np.ndarray,
        photo_id: str = "",
        dataset_id: str = "",
        session_id: str = "",
        timestamp_epoch_ms: int = 0,
        building_id: Optional[str] = None,
        floor_id: Optional[int] = None,
        reference_location_id: Optional[str] = None,
    ) -> List[LandmarkDetection]:
        """Detects indoor landmarks in an image."""
        if bgr_img is None or bgr_img.size == 0 or bgr_img.shape[0] < 16 or bgr_img.shape[1] < 16:
            return []

        h, w = bgr_img.shape[:2]
        raw_detections: List[LandmarkDetection] = []

        # 1. Neural model detection if weights loaded
        if self._yolo_model is not None:
            try:
                results = self._yolo_model.predict(
                    source=bgr_img,
                    conf=self.confidence_threshold,
                    iou=self.nms_iou_threshold,
                    verbose=False,
                )
                for res in results:
                    boxes = res.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            cls_name = res.names.get(cls_id, "UNKNOWN")
                            conf = float(box.conf[0].item())
                            xyxy = box.xyxy[0].tolist()
                            norm_bbox = BoundingBox.from_pixel_coords(
                                int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), w, h
                            )
                            raw_detections.append(
                                LandmarkDetection(
                                    category=LandmarkCategory.from_string(cls_name),
                                    bounding_box=norm_bbox,
                                    confidence=conf,
                                    photo_id=photo_id,
                                    dataset_id=dataset_id,
                                    session_id=session_id,
                                    timestamp_epoch_ms=timestamp_epoch_ms,
                                    building_id=building_id,
                                    floor_id=floor_id,
                                    reference_location_id=reference_location_id,
                                )
                            )
            except Exception as e:
                logger.warning("YOLO inference failed: %s; falling back to heuristic detector", e)

        # 2. Structural & Architectural Feature Detector (Fallback & Complementary)
        if not raw_detections:
            heuristic_dets = self._detect_architectural_structures(
                bgr_img=bgr_img,
                photo_id=photo_id,
                dataset_id=dataset_id,
                session_id=session_id,
                timestamp_epoch_ms=timestamp_epoch_ms,
                building_id=building_id,
                floor_id=floor_id,
                reference_location_id=reference_location_id,
            )
            raw_detections.extend(heuristic_dets)

        # 3. Filter by confidence threshold
        filtered = [d for d in raw_detections if d.confidence >= self.confidence_threshold]

        # 4. Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
        nms_detections = self._apply_nms(filtered, self.nms_iou_threshold)

        # 5. Extract Descriptors and OCR for Signboards
        for det in nms_detections:
            crop = ImagePreprocessor.crop_box(bgr_img, det.bounding_box)
            # Global HSV descriptor
            det.descriptor_vector = self.feature_extractor.extract_global_embedding(crop)

            # OCR trigger for signboard classes
            if self.enable_ocr and self.ocr_engine is not None and det.category in (
                LandmarkCategory.ROOM_NUMBER_SIGN,
                LandmarkCategory.DIRECTION_SIGN,
                LandmarkCategory.EXIT_SIGN,
            ):
                ocr_res = self.ocr_engine.recognize_text(
                    crop,
                    source_image_id=photo_id,
                    timestamp_epoch_ms=timestamp_epoch_ms,
                    relative_bbox=det.bounding_box,
                )
                det.ocr_record = ocr_res

        return nms_detections

    def _detect_architectural_structures(
        self,
        bgr_img: np.ndarray,
        photo_id: str,
        dataset_id: str,
        session_id: str,
        timestamp_epoch_ms: int,
        building_id: Optional[str],
        floor_id: Optional[int],
        reference_location_id: Optional[str],
    ) -> List[LandmarkDetection]:
        """Detects architectural indoor elements using morphological, edge, and contour analysis."""
        h, w = bgr_img.shape[:2]
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 140)

        detections: List[LandmarkDetection] = []

        # A. Detect Vertical Doors & Room Entrances
        door_dets = self._find_door_structures(
            edges=edges,
            bgr_img=bgr_img,
            w=w,
            h=h,
            photo_id=photo_id,
            dataset_id=dataset_id,
            session_id=session_id,
            timestamp_epoch_ms=timestamp_epoch_ms,
            building_id=building_id,
            floor_id=floor_id,
            reference_location_id=reference_location_id,
        )
        detections.extend(door_dets)

        # B. Detect Corridor Lines & Perspective Vanishing Points
        corridor_dets = self._find_corridor_structures(
            edges=edges,
            w=w,
            h=h,
            photo_id=photo_id,
            dataset_id=dataset_id,
            session_id=session_id,
            timestamp_epoch_ms=timestamp_epoch_ms,
            building_id=building_id,
            floor_id=floor_id,
            reference_location_id=reference_location_id,
        )
        detections.extend(corridor_dets)

        # C. Detect Signboards (Room signs, Exit signs)
        sign_dets = self._find_signboard_structures(
            enhanced_gray=enhanced,
            bgr_img=bgr_img,
            w=w,
            h=h,
            photo_id=photo_id,
            dataset_id=dataset_id,
            session_id=session_id,
            timestamp_epoch_ms=timestamp_epoch_ms,
            building_id=building_id,
            floor_id=floor_id,
            reference_location_id=reference_location_id,
        )
        detections.extend(sign_dets)

        return detections

    def _find_door_structures(
        self,
        edges: np.ndarray,
        bgr_img: np.ndarray,
        w: int,
        h: int,
        photo_id: str,
        dataset_id: str,
        session_id: str,
        timestamp_epoch_ms: int,
        building_id: Optional[str],
        floor_id: Optional[int],
        reference_location_id: Optional[str],
    ) -> List[LandmarkDetection]:
        """Finds door-like vertical rectangular frames."""
        results = []
        # Morphological vertical dilation to connect vertical door jamb edges
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        v_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)

        contours, _ = cv2.findContours(v_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = ch / float(max(1, cw))
            # Typical indoor doors have aspect ratio between 1.5 and 3.5, and reasonable height
            if 1.4 <= aspect <= 3.8 and ch > h * 0.25 and cw > w * 0.08:
                norm_box = BoundingBox.from_pixel_coords(x, y, x + cw, y + ch, w, h)
                conf = min(0.85, 0.45 + (ch / h) * 0.3)
                results.append(
                    LandmarkDetection(
                        category=LandmarkCategory.DOOR,
                        bounding_box=norm_box,
                        confidence=round(conf, 4),
                        photo_id=photo_id,
                        dataset_id=dataset_id,
                        session_id=session_id,
                        timestamp_epoch_ms=timestamp_epoch_ms,
                        building_id=building_id,
                        floor_id=floor_id,
                        reference_location_id=reference_location_id,
                    )
                )
        return results

    def _find_corridor_structures(
        self,
        edges: np.ndarray,
        w: int,
        h: int,
        photo_id: str,
        dataset_id: str,
        session_id: str,
        timestamp_epoch_ms: int,
        building_id: Optional[str],
        floor_id: Optional[int],
        reference_location_id: Optional[str],
    ) -> List[LandmarkDetection]:
        """Identifies corridor perspective lines converging towards the center."""
        results = []
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=int(h * 0.25), maxLineGap=20)
        if lines is not None and len(lines) >= 6:
            # Check for perspective convergence
            left_slanted = 0
            right_slanted = 0
            for line in lines:
                l = line.flatten()
                if len(l) < 4:
                    continue
                x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
                dx = x2 - x1
                dy = y2 - y1
                if abs(dx) > 10 and abs(dy) > 20:
                    slope = dy / float(dx)
                    if slope > 0.4:
                        right_slanted += 1
                    elif slope < -0.4:
                        left_slanted += 1

            if left_slanted >= 2 and right_slanted >= 2:
                # Corridor detected in central view
                norm_box = BoundingBox(x_min=0.10, y_min=0.10, x_max=0.90, y_max=0.90)
                conf = min(0.88, 0.50 + 0.04 * (left_slanted + right_slanted))
                results.append(
                    LandmarkDetection(
                        category=LandmarkCategory.CORRIDOR,
                        bounding_box=norm_box,
                        confidence=round(conf, 4),
                        photo_id=photo_id,
                        dataset_id=dataset_id,
                        session_id=session_id,
                        timestamp_epoch_ms=timestamp_epoch_ms,
                        building_id=building_id,
                        floor_id=floor_id,
                        reference_location_id=reference_location_id,
                    )
                )
        return results

    def _find_signboard_structures(
        self,
        enhanced_gray: np.ndarray,
        bgr_img: np.ndarray,
        w: int,
        h: int,
        photo_id: str,
        dataset_id: str,
        session_id: str,
        timestamp_epoch_ms: int,
        building_id: Optional[str],
        floor_id: Optional[int],
        reference_location_id: Optional[str],
    ) -> List[LandmarkDetection]:
        """Finds rectangular plaques, room numbers, and exit signboards."""
        results = []
        # Threshold to detect high-contrast plaque bounds
        thresh = cv2.adaptiveThreshold(
            enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / float(max(1, ch))
            # Signboards are typically horizontal rectangles mounted at eye level (y in upper/middle half)
            if 1.2 <= aspect <= 5.0 and (w * 0.05 < cw < w * 0.55) and (h * 0.03 < ch < h * 0.35):
                if y < h * 0.70:  # Mounted above floor level
                    crop_bgr = bgr_img[y:y+ch, x:x+cw]
                    # Classify sign type based on color / position
                    category = self._classify_sign_color(crop_bgr, y, h)
                    norm_box = BoundingBox.from_pixel_coords(x, y, x + cw, y + ch, w, h)
                    conf = 0.65
                    results.append(
                        LandmarkDetection(
                            category=category,
                            bounding_box=norm_box,
                            confidence=round(conf, 4),
                            photo_id=photo_id,
                            dataset_id=dataset_id,
                            session_id=session_id,
                            timestamp_epoch_ms=timestamp_epoch_ms,
                            building_id=building_id,
                            floor_id=floor_id,
                            reference_location_id=reference_location_id,
                        )
                    )
        return results

    @staticmethod
    def _classify_sign_color(crop: np.ndarray, y: int, img_h: int) -> LandmarkCategory:
        """Heuristically identifies Exit vs Direction vs Room signs based on color and mount height."""
        if crop.size == 0:
            return LandmarkCategory.ROOM_NUMBER_SIGN

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Green mask (Exit signs)
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_frac = float(np.count_nonzero(green_mask) / crop.size)
        if green_frac > 0.08 or (y < img_h * 0.25):  # High mounted or green
            return LandmarkCategory.EXIT_SIGN

        # Blue mask (often directional signs)
        blue_mask = cv2.inRange(hsv, (90, 50, 50), (135, 255, 255))
        blue_frac = float(np.count_nonzero(blue_mask) / crop.size)
        if blue_frac > 0.10:
            return LandmarkCategory.DIRECTION_SIGN

        return LandmarkCategory.ROOM_NUMBER_SIGN

    def _apply_nms(self, detections: List[LandmarkDetection], iou_thresh: float) -> List[LandmarkDetection]:
        """Performs Non-Maximum Suppression (NMS) to eliminate duplicate detections."""
        if not detections:
            return []

        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep: List[LandmarkDetection] = []

        while sorted_dets:
            current = sorted_dets.pop(0)
            keep.append(current)

            remaining = []
            for other in sorted_dets:
                # If same category or high overlap, suppress
                iou = current.bounding_box.iou(other.bounding_box)
                if iou >= iou_thresh:
                    # Suppress duplicate
                    continue
                remaining.append(other)
            sorted_dets = remaining

        return keep
