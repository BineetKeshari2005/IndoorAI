"""Safe image loading, orientation normalization, and array conversions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.computer_vision.image_validation import ImagePrivacyValidator, ImageValidationResult
from src.core.logging import setup_logger

logger = setup_logger("ImageLoader")


class ImageLoadingError(ValueError):
    """Raised when an image cannot be safely decoded or fails privacy checks."""
    pass


class SafeImageLoader:
    """Safely loads images into NumPy/OpenCV arrays respecting privacy constraints."""

    @classmethod
    def load_rgb(
        cls,
        image_path: Union[str, Path],
        photo_id: str = "",
        enforce_sanitization: bool = True,
        metadata_dict: Optional[dict] = None,
    ) -> Tuple[np.ndarray, ImageValidationResult]:
        """Loads an image as an RGB uint8 NumPy array with orientation normalization."""
        path_obj = Path(image_path)
        pid = photo_id or path_obj.stem

        # 1. Privacy & integrity validation
        val_result = ImagePrivacyValidator.validate_photo_file(
            image_path=path_obj,
            photo_id=pid,
            metadata_dict=metadata_dict,
            enforce_sanitization=enforce_sanitization,
        )

        if not val_result.is_valid:
            err_msg = "; ".join(val_result.errors)
            raise ImageLoadingError(f"Cannot load image '{pid}': {err_msg}")

        # 2. PIL-based load with EXIF orientation correction
        try:
            with Image.open(val_result.file_path) as pil_img:
                # Correct orientation based on EXIF tag if present
                transposed = ImageOps.exif_transpose(pil_img)
                # Convert to RGB (handles RGBA, Greyscale, Paletted)
                rgb_img = transposed.convert("RGB")
                arr = np.array(rgb_img, dtype=np.uint8)
                return arr, val_result
        except Exception as e:
            raise ImageLoadingError(f"Failed to decode image array for '{pid}': {e}") from e

    @classmethod
    def load_bgr(
        cls,
        image_path: Union[str, Path],
        photo_id: str = "",
        enforce_sanitization: bool = True,
        metadata_dict: Optional[dict] = None,
    ) -> Tuple[np.ndarray, ImageValidationResult]:
        """Loads an image as a BGR uint8 NumPy array (standard OpenCV format)."""
        rgb_arr, val_result = cls.load_rgb(
            image_path=image_path,
            photo_id=photo_id,
            enforce_sanitization=enforce_sanitization,
            metadata_dict=metadata_dict,
        )
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        return bgr_arr, val_result
