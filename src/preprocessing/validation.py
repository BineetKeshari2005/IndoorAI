"""Data validation module for UJIIndoorLoc training and validation datasets.

Performs comprehensive schema, integrity, categorical domain, hierarchical building-floor,
signal boundary, and coordinate checks across raw dataset partitions.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is in sys.path
_current_dir = Path(__file__).resolve().parent
_repo_root = _current_dir.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd

from src.core.config import get_settings
from src.core.logging import setup_logger

logger = setup_logger("DataValidation")

EXPECTED_TOTAL_COLUMNS = 529
EXPECTED_NUM_WAPS = 520
EXPECTED_CONTEXT_COLUMNS = [
    "LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID",
    "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"
]

# Valid building and floor hierarchical taxonomy in UJI campus
# Building 0 and Building 1 have floors 0-3; Building 2 has floors 0-4
VALID_BUILDING_FLOOR_PAIRS: Set[Tuple[int, int]] = {
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
}


@dataclass
class PartitionValidationResult:
    """Detailed validation outcome for a single dataset partition."""
    partition_name: str
    row_count: int
    column_count: int
    wap_count: int
    context_count: int
    null_count: int
    duplicate_rows_count: int
    constant_columns_count: int
    constant_columns: List[str]
    empty_columns: List[str]
    invalid_building_ids: List[int]
    invalid_floor_ids: List[int]
    invalid_building_floor_pairs: List[List[int]]
    unexpected_rssi_count: int
    invalid_coordinate_rows: int
    is_valid: bool
    warnings: List[str]
    errors: List[str]


@dataclass
class FullValidationReport:
    """Combined validation audit across train and validation datasets."""
    schema_consistency_match: bool
    differing_columns: List[str]
    differing_order: bool
    training: PartitionValidationResult
    validation: PartitionValidationResult
    overall_passed: bool


def validate_partition(df: pd.DataFrame, name: str) -> PartitionValidationResult:
    """Validate a single dataset partition against physical and schema invariants."""
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Column counts & partition
    col_count = len(df.columns)
    wap_cols = [c for c in df.columns if c.startswith("WAP")]
    context_cols = [c for c in df.columns if not c.startswith("WAP")]

    if col_count != EXPECTED_TOTAL_COLUMNS:
        errors.append(f"Expected {EXPECTED_TOTAL_COLUMNS} columns, found {col_count}")
    if len(wap_cols) != EXPECTED_NUM_WAPS:
        errors.append(f"Expected {EXPECTED_NUM_WAPS} WAP columns, found {len(wap_cols)}")
    
    missing_ctx = set(EXPECTED_CONTEXT_COLUMNS) - set(context_cols)
    if missing_ctx:
        errors.append(f"Missing required context columns: {list(missing_ctx)}")

    # 2. Missing values (NaN / Nulls)
    null_count = int(df.isna().sum().sum())
    if null_count > 0:
        errors.append(f"Found {null_count} unexpected NaN/Null cells")

    # 3. Duplicate rows check
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        warnings.append(f"Found {dup_count} exact duplicate rows (should be handled by duplicate strategy)")

    # 4. Empty and constant columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    if empty_cols:
        errors.append(f"Found {len(empty_cols)} completely empty columns: {empty_cols}")
    if constant_cols:
        warnings.append(f"Found {len(constant_cols)} constant columns (e.g. unobserved WAPs in this split)")

    # 5. Building & Floor hierarchical validation
    invalid_blds: List[int] = []
    invalid_flrs: List[int] = []
    invalid_bld_flr_pairs: List[List[int]] = []

    if "BUILDINGID" in df.columns and "FLOOR" in df.columns:
        valid_bld_set = {0, 1, 2}
        valid_flr_set = {0, 1, 2, 3, 4}
        
        found_blds = set(df["BUILDINGID"].unique())
        found_flrs = set(df["FLOOR"].unique())
        
        invalid_blds = [int(b) for b in (found_blds - valid_bld_set)]
        invalid_flrs = [int(f) for f in (found_flrs - valid_flr_set)]
        
        if invalid_blds:
            errors.append(f"Invalid BUILDINGID values detected: {invalid_blds}")
        if invalid_flrs:
            errors.append(f"Invalid FLOOR values detected: {invalid_flrs}")

        # Check joint (BUILDINGID, FLOOR) pairs
        observed_pairs = set(df[["BUILDINGID", "FLOOR"]].drop_duplicates().apply(tuple, axis=1))
        invalid_pairs = observed_pairs - VALID_BUILDING_FLOOR_PAIRS
        if invalid_pairs:
            invalid_bld_flr_pairs = [[int(b), int(f)] for b, f in invalid_pairs]
            errors.append(f"Invalid hierarchical (BUILDINGID, FLOOR) combinations detected: {invalid_bld_flr_pairs}")

    # 6. RSSI Signal Range (Must be 100 or in [-104, 0] dBm)
    unexpected_rssi_count = 0
    if wap_cols:
        wap_mat = df[wap_cols].values
        # Signal is invalid if != 100 AND (RSSI > 0 OR RSSI < -104)
        invalid_mask = (wap_mat != 100) & ((wap_mat > 0) | (wap_mat < -104))
        unexpected_rssi_count = int(invalid_mask.sum())
        if unexpected_rssi_count > 0:
            errors.append(f"Found {unexpected_rssi_count} RSSI measurements outside physically valid bounds [-104, 0] dBm")

    # 7. Coordinates Sanity (Must be finite numbers)
    invalid_coord_rows = 0
    if "LONGITUDE" in df.columns and "LATITUDE" in df.columns:
        invalid_coords = df["LONGITUDE"].isna() | df["LATITUDE"].isna() | np.isinf(df["LONGITUDE"]) | np.isinf(df["LATITUDE"])
        invalid_coord_rows = int(invalid_coords.sum())
        if invalid_coord_rows > 0:
            errors.append(f"Found {invalid_coord_rows} rows with invalid/infinite coordinates")

    is_valid = len(errors) == 0

    return PartitionValidationResult(
        partition_name=name,
        row_count=len(df),
        column_count=col_count,
        wap_count=len(wap_cols),
        context_count=len(context_cols),
        null_count=null_count,
        duplicate_rows_count=dup_count,
        constant_columns_count=len(constant_cols),
        constant_columns=constant_cols,
        empty_columns=empty_cols,
        invalid_building_ids=invalid_blds,
        invalid_floor_ids=invalid_flrs,
        invalid_building_floor_pairs=invalid_bld_flr_pairs,
        unexpected_rssi_count=unexpected_rssi_count,
        invalid_coordinate_rows=invalid_coord_rows,
        is_valid=is_valid,
        warnings=warnings,
        errors=errors,
    )


def validate_datasets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    save_report_path: Optional[Path] = None
) -> FullValidationReport:
    """Validate both training and validation splits and verify cross-split schema consistency.

    Args:
        df_train: Raw training DataFrame.
        df_val: Raw validation DataFrame.
        save_report_path: Optional destination to write JSON validation report.

    Returns:
        FullValidationReport dataclass.
    """
    logger.info("Starting comprehensive data validation audit...")

    train_res = validate_partition(df_train, "training")
    val_res = validate_partition(df_val, "validation")

    # Cross-split schema consistency
    differing_columns = list(set(df_train.columns) ^ set(df_val.columns))
    differing_order = list(df_train.columns) != list(df_val.columns)
    schema_consistency_match = (len(differing_columns) == 0) and not differing_order

    overall_passed = train_res.is_valid and val_res.is_valid and schema_consistency_match

    report = FullValidationReport(
        schema_consistency_match=schema_consistency_match,
        differing_columns=differing_columns,
        differing_order=differing_order,
        training=train_res,
        validation=val_res,
        overall_passed=overall_passed,
    )

    if save_report_path:
        save_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_report_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("Saved validation report to %s", save_report_path)

    if overall_passed:
        logger.info("Data validation passed successfully with 0 critical errors.")
    else:
        logger.error("Data validation encountered critical errors. Check report.")

    return report

