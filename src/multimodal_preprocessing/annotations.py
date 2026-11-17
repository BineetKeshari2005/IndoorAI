"""Surveyor manual annotation normalization and ground-truth validation (Task 20)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.core.logging import setup_logger
from src.multimodal_preprocessing.config import MultimodalPreprocessingConfig

logger = setup_logger("MultimodalAnnotationPreprocessing")

CANONICAL_ANNOTATION_TYPES = {
    "CORRIDOR": "Corridor / Hallway Path",
    "DOOR": "Door / Entryway",
    "ROOM_ENTRANCE": "Room Entrance",
    "STAIRCASE": "Staircase / Stairs",
    "ELEVATOR": "Elevator / Lift",
    "LANDMARK": "Identifiable Architectural Landmark",
    "REFERENCE_POINT": "Survey Calibration Reference Point",
    "MANUAL_LOCATION_CORRECTION": "Surveyor Manual Position Correction",
    "CUSTOM": "Custom Surveyor Note",
}


@dataclass
class AnnotationProcessingStats:
    """Summary metrics of manual annotation processing."""

    records_input: int = 0
    records_retained: int = 0
    records_removed_invalid_time: int = 0
    type_counts: Dict[str, int] = field(default_factory=dict)
    records_mock: int = 0


class AnnotationProcessor:
    """Standardizes spatial and architectural annotations while preserving surveyor provenance."""

    def __init__(self, config: Optional[MultimodalPreprocessingConfig] = None):
        self.config = config or MultimodalPreprocessingConfig()

    def process(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, AnnotationProcessingStats]:
        """Validates and standardizes annotations into a clean tabular structure.

        Returns:
            Tuple of:
            - cleaned_df: Standardized annotation records.
            - stats: Processing statistics.
        """
        stats = AnnotationProcessingStats(records_input=len(raw_df))
        if raw_df.empty:
            logger.info("Annotation dataframe is empty; returning empty results.")
            empty_df = pd.DataFrame(columns=[
                "annotation_id", "observation_id", "session_id", "normalized_epoch_ms",
                "normalized_iso", "time_source", "sync_method", "uncertainty_ms",
                "type_code", "type_name", "description", "building", "floor", "room",
                "related_observation_id", "label_provenance", "is_mock"
            ])
            return empty_df, stats

        df = raw_df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        # 1. Normalize IDs
        if "annotation_id" not in df.columns and "observation_id" in df.columns:
            df["annotation_id"] = df["observation_id"]
        elif "annotation_id" in df.columns and "observation_id" not in df.columns:
            df["observation_id"] = df["annotation_id"]

        # 2. Filter invalid timestamps
        if "normalized_epoch_ms" in df.columns:
            df["normalized_epoch_ms"] = pd.to_numeric(df["normalized_epoch_ms"], errors="coerce")
            valid_time = df["normalized_epoch_ms"].notna() & (df["normalized_epoch_ms"] > 0)
            stats.records_removed_invalid_time = int((~valid_time).sum())
            df = df[valid_time].copy()
            df["normalized_epoch_ms"] = df["normalized_epoch_ms"].astype("int64")

        # 3. Canonicalize type codes
        def canonicalize_type(val: Any) -> str:
            if not isinstance(val, str) or not val.strip():
                return "CUSTOM"
            clean = val.strip().upper().replace(" ", "_").replace("-", "_")
            if clean in CANONICAL_ANNOTATION_TYPES:
                return clean
            # Match partial keywords
            for k in CANONICAL_ANNOTATION_TYPES:
                if k in clean or clean in k:
                    return k
            return "CUSTOM"

        if "type_code" in df.columns and "type_name" in df.columns:
            raw_type = df["type_code"].fillna(df["type_name"])
        elif "type_code" in df.columns:
            raw_type = df["type_code"]
        elif "type_name" in df.columns:
            raw_type = df["type_name"]
        else:
            raw_type = pd.Series(["CUSTOM"] * len(df), index=df.index)

        df["type_code"] = raw_type.apply(canonicalize_type)
        df["type_name"] = df["type_code"].map(CANONICAL_ANNOTATION_TYPES)

        # 4. Explicit surveyor provenance
        df["label_provenance"] = "SURVEYOR_MANUAL"

        # 5. Ensure all standard columns exist and clean fields
        standard_cols = [
            "annotation_id", "observation_id", "session_id", "normalized_epoch_ms",
            "normalized_iso", "time_source", "sync_method", "uncertainty_ms",
            "type_code", "type_name", "description", "building", "floor", "room",
            "related_observation_id", "label_provenance", "is_mock"
        ]
        for col in standard_cols:
            if col not in df.columns:
                if col in ("building", "floor", "room"):
                    df[col] = "UNKNOWN"
                elif col in ("time_source", "sync_method"):
                    df[col] = "UNKNOWN"
                elif col == "is_mock":
                    df[col] = False
                elif col == "uncertainty_ms":
                    df[col] = 0.0
                elif col == "session_id":
                    df[col] = "UNKNOWN"
                elif col == "normalized_iso":
                    df[col] = ""
                else:
                    df[col] = ""
            else:
                if col in ("description", "building", "floor", "room", "related_observation_id"):
                    df[col] = df[col].astype(str).replace("nan", "").str.strip()

        # 6. Track statistics
        stats.records_retained = len(df)
        stats.type_counts = df["type_code"].value_counts().to_dict()
        if "is_mock" in df.columns:
            stats.records_mock = int(df["is_mock"].astype(str).str.lower().isin(["true", "1"]).sum())

        cleaned_df = df[standard_cols].sort_values("normalized_epoch_ms").reset_index(drop=True)
        logger.info(
            "Annotation Preprocessing: %d input -> %d retained across %d types.",
            stats.records_input, stats.records_retained, len(stats.type_counts)
        )
        return cleaned_df, stats
