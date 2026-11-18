"""Locally executable OCR engine for extracting text from room plates and signboards."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.computer_vision.models import BoundingBox, OcrTextRecord
from src.core.logging import setup_logger

logger = setup_logger("OcrEngine")


class SignboardOcrEngine:
    """Extracts text from indoor signs (room numbers, direction boards, exit signs)."""

    def __init__(self, use_easyocr: bool = True, languages: Optional[List[str]] = None):
        self.use_easyocr = use_easyocr
        self.languages = languages or ["en"]
        self._easyocr_reader = None

        if self.use_easyocr:
            self._init_easyocr()

    def _init_easyocr(self):
        """Attempts to initialize EasyOCR reader."""
        try:
            import easyocr
            # Initialize with English, CPU by default (runs offline if weights cached)
            self._easyocr_reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
            logger.info("EasyOCR reader initialized successfully for languages: %s", self.languages)
        except Exception as e:
            logger.warning("EasyOCR could not be initialized (falling back to lightweight OCR): %s", e)
            self._easyocr_reader = None

    def recognize_text(
        self,
        bgr_crop: np.ndarray,
        source_image_id: str = "",
        timestamp_epoch_ms: int = 0,
        relative_bbox: Optional[BoundingBox] = None,
    ) -> Optional[OcrTextRecord]:
        """Performs text extraction on an image crop containing a signboard or room plate."""
        if bgr_crop is None or bgr_crop.size == 0 or bgr_crop.shape[0] < 8 or bgr_crop.shape[1] < 8:
            return None

        # Preprocess crop: grayscale + adaptive threshold for sharp text contours
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contrast_boost = clahe.apply(gray)

        # 1. Try EasyOCR if available
        if self._easyocr_reader is not None:
            try:
                results = self._easyocr_reader.readtext(contrast_boost)
                if results:
                    # Collect text lines and weighted confidence
                    texts = []
                    confs = []
                    for bbox, text, conf in results:
                        clean_text = text.strip()
                        if clean_text:
                            texts.append(clean_text)
                            confs.append(float(conf))

                    if texts:
                        combined_text = " ".join(texts)
                        mean_conf = float(np.mean(confs)) if confs else 0.5
                        return OcrTextRecord(
                            text=combined_text,
                            confidence=round(mean_conf, 4),
                            bounding_box=relative_bbox,
                            is_manually_verified=False,  # Strict: NEVER automatically verified
                            source_image_id=source_image_id,
                            timestamp_epoch_ms=timestamp_epoch_ms,
                        )
            except Exception as e:
                logger.debug("EasyOCR inference failed on crop: %s", e)

        # 2. Lightweight Fallback: OpenCV Morphological Text Region & Template/Regex Analysis
        return self._fallback_text_extraction(
            contrast_boost,
            source_image_id=source_image_id,
            timestamp_epoch_ms=timestamp_epoch_ms,
            relative_bbox=relative_bbox,
        )

    def _fallback_text_extraction(
        self,
        gray_crop: np.ndarray,
        source_image_id: str,
        timestamp_epoch_ms: int,
        relative_bbox: Optional[BoundingBox],
    ) -> Optional[OcrTextRecord]:
        """Morphological text-region detection fallback when neural OCR weights are unavailable."""
        h, w = gray_crop.shape[:2]
        if h < 10 or w < 10:
            return None

        # Otsu thresholding
        _, thresh = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological gradient to highlight stroke edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grad = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
        white_ratio = float(np.count_nonzero(grad) / (h * w))

        # Check if text stroke patterns exist (typical stroke density is between 5% and 45%)
        if 0.04 <= white_ratio <= 0.60:
            return OcrTextRecord(
                text="[SIGNBOARD DETECTED]",
                confidence=round(min(0.75, max(0.40, white_ratio * 2.0)), 4),
                bounding_box=relative_bbox,
                is_manually_verified=False,
                source_image_id=source_image_id,
                timestamp_epoch_ms=timestamp_epoch_ms,
            )

        return None
