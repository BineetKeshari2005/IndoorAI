"""Privacy and integrity validation for survey photographs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from src.core.logging import setup_logger

logger = setup_logger("ImageValidation")

ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
MAGIC_NUMBERS: Dict[str, bytes] = {
    "jpeg": b"\xff\xd8\xff",
    "png": b"\x89PNG\r\n\x1a\n",
    "webp": b"RIFF",
}

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
MIN_FILE_SIZE_BYTES = 512                # 512 bytes
MIN_DIMENSION_PX = 32
MAX_DIMENSION_PX = 8192


@dataclass
class ImageValidationResult:
    """Outcome of privacy, integrity, and dimensional validation on a photo."""
    is_valid: bool
    photo_id: str
    file_path: Optional[Path] = None
    width_px: int = 0
    height_px: int = 0
    file_size_bytes: int = 0
    is_exif_sanitized: bool = False
    face_detection_status: str = "UNKNOWN"
    session_id: Optional[str] = None
    dataset_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ImagePrivacyValidator:
    """Validates survey photograph integrity, privacy compliance, and dimensional bounds."""

    @classmethod
    def validate_photo_file(
        cls,
        image_path: Path,
        photo_id: str,
        session_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        metadata_dict: Optional[Dict[str, Any]] = None,
        enforce_sanitization: bool = True,
    ) -> ImageValidationResult:
        """Thoroughly inspects image file on disk against privacy constraints and corruption."""
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Path & Existence Checks
        if not isinstance(image_path, Path):
            image_path = Path(image_path)

        # Check path traversal
        try:
            resolved = image_path.resolve()
            if ".." in str(image_path) or not resolved.exists():
                errors.append(f"Image path does not exist or contains invalid traversal: {image_path}")
                return ImageValidationResult(
                    is_valid=False,
                    photo_id=photo_id,
                    errors=errors,
                    session_id=session_id,
                    dataset_id=dataset_id,
                )
        except Exception as e:
            errors.append(f"Invalid image path resolution: {e}")
            return ImageValidationResult(is_valid=False, photo_id=photo_id, errors=errors)

        if not resolved.is_file():
            errors.append(f"Target is not a regular file: {resolved}")
            return ImageValidationResult(is_valid=False, photo_id=photo_id, errors=errors)

        # 2. Extension & Size Check
        ext = resolved.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"Disallowed image extension '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

        file_size = resolved.stat().st_size
        if file_size < MIN_FILE_SIZE_BYTES:
            errors.append(f"File size {file_size} bytes is below minimum {MIN_FILE_SIZE_BYTES} bytes.")
        elif file_size > MAX_FILE_SIZE_BYTES:
            errors.append(f"File size {file_size} bytes exceeds maximum limit {MAX_FILE_SIZE_BYTES} bytes.")

        # 3. Magic Number Validation
        try:
            with open(resolved, "rb") as f:
                header = f.read(16)
            is_valid_magic = False
            if ext in (".jpg", ".jpeg") and header.startswith(MAGIC_NUMBERS["jpeg"]):
                is_valid_magic = True
            elif ext == ".png" and header.startswith(MAGIC_NUMBERS["png"]):
                is_valid_magic = True
            elif ext == ".webp" and header.startswith(MAGIC_NUMBERS["webp"]):
                is_valid_magic = True

            if not is_valid_magic:
                errors.append(f"Image header magic bytes do not match declared extension '{ext}'.")
        except Exception as e:
            errors.append(f"Failed to read image file header: {e}")

        # 4. Dimensional and Decode Validation
        w_px, h_px = 0, 0
        try:
            with Image.open(resolved) as img:
                img.verify()  # Fast structural verification

            # Re-open for dimension inspection after verify()
            with Image.open(resolved) as img:
                w_px, h_px = img.size

            if w_px < MIN_DIMENSION_PX or h_px < MIN_DIMENSION_PX:
                errors.append(f"Image dimensions ({w_px}x{h_px}) are below minimum {MIN_DIMENSION_PX}px.")
            if w_px > MAX_DIMENSION_PX or h_px > MAX_DIMENSION_PX:
                errors.append(f"Image dimensions ({w_px}x{h_px}) exceed maximum {MAX_DIMENSION_PX}px.")
        except Exception as e:
            errors.append(f"Image decode failed (corrupted file): {e}")

        # 5. Privacy & EXIF Sanitization Verification
        is_sanitized = False
        face_status = "UNKNOWN"

        if metadata_dict:
            raw_san = str(metadata_dict.get("is_exif_sanitized", "")).strip().lower()
            if raw_san in ("true", "1", "yes"):
                is_sanitized = True
            face_status = str(metadata_dict.get("face_detection_status", "UNKNOWN")).strip()

        if enforce_sanitization and not is_sanitized:
            errors.append(
                f"Privacy check failed: Photo '{photo_id}' is not confirmed as EXIF-sanitized "
                f"(is_exif_sanitized != True). Visual landmark processing blocked."
            )

        is_valid = (len(errors) == 0)
        return ImageValidationResult(
            is_valid=is_valid,
            photo_id=photo_id,
            file_path=resolved,
            width_px=w_px,
            height_px=h_px,
            file_size_bytes=file_size,
            is_exif_sanitized=is_sanitized,
            face_detection_status=face_status,
            session_id=session_id,
            dataset_id=dataset_id,
            errors=errors,
            warnings=warnings,
        )

    @classmethod
    def load_photo_metadata_table(cls, csv_path: Path) -> Dict[str, Dict[str, Any]]:
        """Parses photo_metadata.csv into a dictionary keyed by filename or photo_id."""
        records: Dict[str, Dict[str, Any]] = {}
        if not csv_path.is_file():
            return records

        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get("photo_id", "").strip()
                    fn = row.get("filename", "").strip()
                    if pid:
                        records[pid] = row
                    if fn:
                        records[fn] = row
        except Exception as e:
            logger.warning("Failed to parse photo metadata table %s: %s", csv_path, e)

        return records
