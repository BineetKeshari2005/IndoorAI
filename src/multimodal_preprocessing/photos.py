"""Photo metadata verification, privacy filtering, and visual audit tracking (Task 20)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.core.logging import setup_logger
from src.multimodal_preprocessing.config import MultimodalPreprocessingConfig

logger = setup_logger("MultimodalPhotoPreprocessing")


@dataclass
class PhotoProcessingStats:
    """Summary metrics of photo metadata processing and privacy gating."""

    records_input: int = 0
    records_approved: int = 0
    records_excluded_unsanitized: int = 0
    records_excluded_missing_file: int = 0
    records_mock: int = 0


class PhotoMetadataProcessor:
    """Processes indoor landmark photo metadata, validates file existence, and gates on EXIF sanitization."""

    def __init__(self, config: Optional[MultimodalPreprocessingConfig] = None, images_dir: Optional[Path] = None):
        self.config = config or MultimodalPreprocessingConfig()
        self.images_dir = Path(images_dir) if images_dir else None

    def process(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, PhotoProcessingStats]:
        """Processes photo metadata without mutating image files.

        Returns:
            Tuple of:
            - approved_df: Validated, sanitization-verified photo records ready for visual analysis.
            - excluded_df: Photos rejected due to unverified privacy or missing disk files.
            - stats: Processing statistics.
        """
        stats = PhotoProcessingStats(records_input=len(raw_df))
        if raw_df.empty:
            logger.info("Photo metadata dataframe is empty; returning empty results.")
            empty_df = pd.DataFrame(columns=[
                "photo_id", "observation_id", "session_id", "normalized_epoch_ms", "normalized_iso",
                "time_source", "sync_method", "uncertainty_ms", "filename", "file_size_bytes",
                "image_width", "image_height", "file_format", "landmark_type_code",
                "landmark_type_name", "building", "floor", "room", "description",
                "client_reported_sanitized", "visual_audit_status", "is_mock"
            ])
            return empty_df, empty_df.copy(), stats

        df = raw_df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        # 1. Normalize IDs
        if "photo_id" not in df.columns and "observation_id" in df.columns:
            df["photo_id"] = df["observation_id"]
        elif "photo_id" in df.columns and "observation_id" not in df.columns:
            df["observation_id"] = df["photo_id"]

        # 2. Check file existence on disk if images_dir is available
        def file_exists(filename: Any) -> bool:
            if not self.images_dir or not isinstance(filename, str):
                return True
            clean_fn = Path(filename).name
            target = self.images_dir / clean_fn
            return target.is_file()

        df["file_exists_on_disk"] = df["filename"].apply(file_exists)
        missing_file_mask = ~df["file_exists_on_disk"]
        stats.records_excluded_missing_file = int(missing_file_mask.sum())

        # 3. Check client-reported EXIF sanitization
        def is_sanitized(val: Any) -> bool:
            if pd.isna(val):
                return False
            return str(val).strip().lower() in ("true", "1")

        sanitized_col = df.get("is_exif_sanitized", pd.Series([False] * len(df)))
        df["client_reported_sanitized"] = sanitized_col.apply(is_sanitized)
        unsanitized_mask = ~df["client_reported_sanitized"]
        stats.records_excluded_unsanitized = int(unsanitized_mask.sum())

        # 4. Independent Visual Audit Status:
        # Client EXIF removal does NOT guarantee visual privacy (faces, monitors, badges).
        # Mark visual audit as PENDING_AUDIT unless explicitly audited in metadata.
        df["visual_audit_status"] = "PENDING_AUDIT"

        # 5. Extract file format
        def get_format(fn: Any) -> str:
            if isinstance(fn, str):
                ext = Path(fn).suffix.lower().lstrip(".")
                return ext if ext else "unknown"
            return "unknown"

        df["file_format"] = df["filename"].apply(get_format)

        # 6. Apply rejection criteria
        exclusion_mask = pd.Series(False, index=df.index)
        if self.config.filter_unsanitized_photos:
            exclusion_mask = exclusion_mask | unsanitized_mask
        if self.config.require_image_file_exists and self.images_dir:
            exclusion_mask = exclusion_mask | missing_file_mask

        excluded_df = df[exclusion_mask].copy()
        approved_df = df[~exclusion_mask].copy()

        # 7. Normalize types and missing labels
        standard_cols = [
            "photo_id", "observation_id", "session_id", "normalized_epoch_ms", "normalized_iso",
            "time_source", "sync_method", "uncertainty_ms", "filename", "file_size_bytes",
            "image_width", "image_height", "file_format", "landmark_type_code",
            "landmark_type_name", "building", "floor", "room", "description",
            "client_reported_sanitized", "visual_audit_status", "is_mock"
        ]
        for col in standard_cols:
            if col not in approved_df.columns:
                approved_df[col] = "UNKNOWN" if col in ("building", "floor", "room", "landmark_type_name") else None

        stats.records_approved = len(approved_df)
        if "is_mock" in approved_df.columns:
            stats.records_mock = int(approved_df["is_mock"].astype(str).str.lower().isin(["true", "1"]).sum())

        approved_df = approved_df[standard_cols].copy()
        logger.info(
            "Photo Preprocessing: %d input -> %d approved, %d excluded unsanitized, %d missing files.",
            stats.records_input, stats.records_approved, stats.records_excluded_unsanitized, stats.records_excluded_missing_file
        )
        return approved_df, excluded_df, stats
