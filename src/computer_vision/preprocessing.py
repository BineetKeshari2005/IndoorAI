"""Image preprocessing, contrast enhancement, cropping, and annotation rendering."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.computer_vision.models import BoundingBox, LandmarkCategory, LandmarkDetection


class ImagePreprocessor:
    """Provides modular preprocessing, CLAHE contrast enhancement, and bounding box cropping."""

    @staticmethod
    def apply_clahe(bgr_img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel of LAB."""
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def resize_with_aspect_ratio(
        img: np.ndarray,
        target_max_dim: int = 1280,
        interpolation: int = cv2.INTER_AREA
    ) -> Tuple[np.ndarray, float]:
        """Resizes image so its longest dimension equals target_max_dim, returning (resized_img, scale_factor)."""
        h, w = img.shape[:2]
        if max(h, w) <= target_max_dim:
            return img.copy(), 1.0

        scale = target_max_dim / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        return resized, scale

    @staticmethod
    def crop_box(img: np.ndarray, box: BoundingBox, margin_frac: float = 0.05) -> np.ndarray:
        """Extracts an image crop corresponding to a normalized BoundingBox with optional boundary margin."""
        h, w = img.shape[:2]
        px_min, py_min, px_max, py_max = box.to_pixel_coords(w, h)

        # Apply margin
        dw = int(round((px_max - px_min) * margin_frac))
        dh = int(round((py_max - py_min) * margin_frac))

        c_xmin = max(0, px_min - dw)
        c_ymin = max(0, py_min - dh)
        c_xmax = min(w, px_max + dw)
        c_ymax = min(h, py_max + dh)

        if c_xmax <= c_xmin or c_ymax <= c_ymin:
            return np.zeros((16, 16, img.shape[2] if len(img.shape) > 2 else 1), dtype=np.uint8)

        return img[c_ymin:c_ymax, c_xmin:c_xmax].copy()

    @staticmethod
    def draw_detections(
        bgr_img: np.ndarray,
        detections: List[LandmarkDetection],
        line_thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """Renders bounding boxes, category labels, confidence scores, and OCR text onto an image copy."""
        canvas = bgr_img.copy()
        h, w = canvas.shape[:2]

        color_palette = {
            LandmarkCategory.DOOR: (0, 165, 255),               # Orange
            LandmarkCategory.ROOM_ENTRANCE: (0, 215, 255),      # Gold
            LandmarkCategory.STAIRCASE: (255, 0, 0),            # Blue
            LandmarkCategory.ELEVATOR: (255, 0, 255),           # Magenta
            LandmarkCategory.CORRIDOR: (0, 255, 0),             # Green
            LandmarkCategory.INTERSECTION: (255, 255, 0),       # Cyan
            LandmarkCategory.ROOM_NUMBER_SIGN: (0, 255, 127),   # Spring Green
            LandmarkCategory.DIRECTION_SIGN: (180, 105, 255),   # Pink
            LandmarkCategory.EXIT_SIGN: (0, 0, 255),            # Bright Red
            LandmarkCategory.DISTINCTIVE_STRUCTURE: (205, 90, 106),
            LandmarkCategory.UNKNOWN: (128, 128, 128),          # Grey
        }

        for det in detections:
            px_min, py_min, px_max, py_max = det.bounding_box.to_pixel_coords(w, h)
            color = color_palette.get(det.category, (0, 255, 0))

            # Draw box
            cv2.rectangle(canvas, (px_min, py_min), (px_max, py_max), color, line_thickness)

            # Build label text
            label = f"{det.category.value} {int(det.confidence * 100)}%"
            if det.ocr_record and det.ocr_record.text:
                label += f" [{det.ocr_record.text}]"

            # Label banner
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            banner_ymin = max(0, py_min - th - baseline - 4)
            banner_ymax = py_min
            banner_xmax = min(w, px_min + tw + 6)
            cv2.rectangle(canvas, (px_min, banner_ymin), (banner_xmax, banner_ymax), color, -1)
            cv2.putText(
                canvas,
                label,
                (px_min + 3, py_min - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return canvas
