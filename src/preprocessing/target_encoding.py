"""Target encoding and dataset partitioning for IndoorAI.

Separates localization targets (classification & regression), contextual metadata,
and input Wi-Fi features. Encodes categorical targets consistently and guarantees
that target or metadata columns never leak into input feature matrices.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.logging import setup_logger

logger = setup_logger("TargetEncoder")

CLASSIFICATION_TARGETS = ["BUILDINGID", "FLOOR"]
REGRESSION_TARGETS = ["LONGITUDE", "LATITUDE"]
METADATA_COLUMNS = [
    "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP",
    "original_row_index", "sample_weight"
]


@dataclass
class TargetEncodingMetadata:
    """Mapping records for categorical target encoders."""
    building_to_idx: Dict[int, int]
    idx_to_building: Dict[int, int]
    floor_to_idx: Dict[int, int]
    idx_to_floor: Dict[int, int]
    building_floor_to_idx: Dict[str, int]
    idx_to_building_floor: Dict[int, str]
    num_buildings: int
    num_floors: int
    num_building_floor_classes: int


def fit_target_encoders(df_train: pd.DataFrame) -> TargetEncodingMetadata:
    """Learn canonical categorical encodings strictly from training labels.

    Args:
        df_train: Raw training DataFrame.

    Returns:
        TargetEncodingMetadata containing bidirectional label mappings.
    """
    logger.info("Fitting target encodings strictly on training labels...")

    # Building encoding
    buildings = sorted(int(b) for b in df_train["BUILDINGID"].unique())
    bld_to_idx = {bld: i for i, bld in enumerate(buildings)}
    idx_to_bld = {i: bld for bld, i in bld_to_idx.items()}

    # Floor encoding
    floors = sorted(int(f) for f in df_train["FLOOR"].unique())
    flr_to_idx = {flr: i for i, flr in enumerate(floors)}
    idx_to_flr = {i: flr for flr, i in flr_to_idx.items()}

    # Hierarchical composite (Building_Floor) encoding
    bld_flr_pairs = sorted(
        df_train[["BUILDINGID", "FLOOR"]].drop_duplicates().apply(tuple, axis=1)
    )
    bld_flr_to_idx = {f"{int(b)}_{int(f)}": i for i, (b, f) in enumerate(bld_flr_pairs)}
    idx_to_bld_flr = {i: bf for bf, i in bld_flr_to_idx.items()}

    return TargetEncodingMetadata(
        building_to_idx=bld_to_idx,
        idx_to_building=idx_to_bld,
        floor_to_idx=flr_to_idx,
        idx_to_floor=idx_to_flr,
        building_floor_to_idx=bld_flr_to_idx,
        idx_to_building_floor=idx_to_bld_flr,
        num_buildings=len(bld_to_idx),
        num_floors=len(flr_to_idx),
        num_building_floor_classes=len(bld_flr_to_idx),
    )


def extract_targets_and_metadata(
    df: pd.DataFrame,
    encoder_meta: TargetEncodingMetadata,
    save_encoders_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract encoded targets DataFrame y and separated metadata DataFrame.

    Target columns:
    - BUILDINGID (raw label)
    - FLOOR (raw label)
    - BUILDING_INDEX (encoded 0..2)
    - FLOOR_INDEX (encoded 0..4)
    - BUILDING_FLOOR_INDEX (hierarchical composite class 0..12)
    - LONGITUDE (metric continuous)
    - LATITUDE (metric continuous)

    Metadata columns:
    - SPACEID, RELATIVEPOSITION, USERID, PHONEID, TIMESTAMP, original_row_index, sample_weight

    Args:
        df: Input DataFrame.
        encoder_meta: Canonical target encoding metadata.
        save_encoders_path: Optional path to save target_encoders.json.

    Returns:
        Tuple of (y_df, metadata_df).
    """
    y_data: Dict[str, Any] = {}

    # Regression targets
    y_data["LONGITUDE"] = df["LONGITUDE"].values.astype(np.float64)
    y_data["LATITUDE"] = df["LATITUDE"].values.astype(np.float64)

    # Classification targets (raw labels)
    y_data["BUILDINGID"] = df["BUILDINGID"].values.astype(np.int64)
    y_data["FLOOR"] = df["FLOOR"].values.astype(np.int64)

    # Classification targets (canonical 0-indexed integer classes)
    y_data["BUILDING_INDEX"] = df["BUILDINGID"].map(encoder_meta.building_to_idx).values.astype(np.int64)
    y_data["FLOOR_INDEX"] = df["FLOOR"].map(encoder_meta.floor_to_idx).values.astype(np.int64)

    # Composite building-floor target
    composite_keys = df.apply(lambda r: f"{int(r['BUILDINGID'])}_{int(r['FLOOR'])}", axis=1)
    y_data["BUILDING_FLOOR_INDEX"] = composite_keys.map(encoder_meta.building_floor_to_idx).values.astype(np.int64)

    y_df = pd.DataFrame(y_data, index=df.index)

    # Metadata extraction
    meta_cols = [c for c in METADATA_COLUMNS if c in df.columns]
    metadata_df = df[meta_cols].copy()

    # Ensure sample_weight is present
    if "sample_weight" not in metadata_df.columns:
        metadata_df["sample_weight"] = 1.0

    if save_encoders_path:
        save_encoders_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert integer keys to string for JSON serialization
        json_dict = asdict(encoder_meta)
        with open(save_encoders_path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=2)
        logger.info("Saved target encoders to %s", save_encoders_path)

    return y_df, metadata_df


def verify_no_leakage(
    X: pd.DataFrame,
    y: pd.DataFrame,
    metadata: pd.DataFrame,
    split_name: str = "dataset"
) -> bool:
    """Assert programmatically that no target or metadata columns leak into feature matrix X.

    Raises:
        AssertionError if any forbidden column is present in X.
    """
    forbidden_cols = set(CLASSIFICATION_TARGETS) | set(REGRESSION_TARGETS) | set(METADATA_COLUMNS)
    forbidden_cols.update(["BUILDING_INDEX", "FLOOR_INDEX", "BUILDING_FLOOR_INDEX"])

    leaked = set(X.columns) & forbidden_cols
    if leaked:
        err_msg = f"CRITICAL DATA LEAKAGE DETECTED in {split_name}! Features contain forbidden columns: {leaked}"
        logger.error(err_msg)
        raise AssertionError(err_msg)

    # Check for NaN / Infs in X
    if X.isna().any().any():
        err_msg = f"Feature matrix {split_name} contains unexpected NaN values!"
        logger.error(err_msg)
        raise AssertionError(err_msg)

    logger.info("Leakage Verification Passed for %s (%d rows, %d features). Zero target/metadata leakage.", split_name, len(X), len(X.columns))
    return True
