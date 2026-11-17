"""Duplicate record handler for IndoorAI preprocessing.

Implements configurable duplicate handling strategies:
- KEEP_ALL: Preserves all records without alteration (default baseline).
- REMOVE_EXACT: Drops redundant identical duplicate rows, preserving first occurrence.
- SAMPLE_WEIGHT: Retains unique fingerprints with a frequency-based sample_weight column.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.core.logging import setup_logger

logger = setup_logger("DuplicateHandler")


class DuplicateStrategy(str, Enum):
    """Supported duplicate handling policies."""
    KEEP_ALL = "keep_all"
    REMOVE_EXACT = "remove_exact"
    SAMPLE_WEIGHT = "sample_weight"


@dataclass
class DuplicateHandlingReport:
    """Audit report of duplicate strategy execution."""
    strategy: str
    input_rows: int
    output_rows: int
    duplicates_detected: int
    rows_removed: int
    sample_weight_column_added: bool
    notes: str


def handle_duplicates(
    df: pd.DataFrame,
    strategy: str = "keep_all",
    subset_columns: Optional[List[str]] = None,
    save_report_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, DuplicateHandlingReport]:
    """Apply the chosen duplicate handling strategy to a DataFrame.

    Preserves original row tracking via 'original_row_index' and never mutates input.

    Args:
        df: Input DataFrame (training split).
        strategy: 'keep_all', 'remove_exact', or 'sample_weight'.
        subset_columns: Optional subset of columns to define duplicates (defaults to all columns).
        save_report_path: Optional path to save JSON audit report.

    Returns:
        Tuple of (processed_df, DuplicateHandlingReport).
    """
    resolved_strategy = strategy.lower().strip()
    valid_strategies = [s.value for s in DuplicateStrategy]
    if resolved_strategy not in valid_strategies:
        raise ValueError(
            f"Unknown duplicate strategy '{strategy}'. Valid options: {valid_strategies}"
        )

    # Identify content columns to evaluate duplicates (excluding tracking fields)
    content_cols = subset_columns or [c for c in df.columns if c not in ("original_row_index", "sample_weight")]

    # Determine duplicates on content columns before adding tracking fields
    redundant_mask = df.duplicated(subset=content_cols, keep="first")
    num_redundant = int(redundant_mask.sum())
    input_rows = len(df)

    # Work on a defensive copy and preserve original row index for complete traceability
    df_work = df.copy()
    if "original_row_index" not in df_work.columns:
        df_work["original_row_index"] = df.index.values

    logger.info(
        "Applying duplicate strategy '%s'. Detected %d redundant duplicate rows out of %d total.",
        resolved_strategy, num_redundant, input_rows
    )

    if resolved_strategy == DuplicateStrategy.KEEP_ALL.value:
        df_work["sample_weight"] = 1.0
        output_rows = len(df_work)
        rows_removed = 0
        weight_added = True
        notes = "All records preserved. Default sample_weight=1.0 assigned to every row."

    elif resolved_strategy == DuplicateStrategy.REMOVE_EXACT.value:
        df_work = df_work[~redundant_mask].reset_index(drop=True)
        df_work["sample_weight"] = 1.0
        output_rows = len(df_work)
        rows_removed = input_rows - output_rows
        weight_added = True
        notes = f"Removed {rows_removed} exact redundant duplicate rows. Unique records assigned sample_weight=1.0."

    elif resolved_strategy == DuplicateStrategy.SAMPLE_WEIGHT.value:
        # Group by all content columns (excluding tracking original_row_index)
        check_cols = subset_columns or [c for c in df.columns if c != "original_row_index"]
        
        # Calculate frequency per unique row
        group_counts = df_work.groupby(check_cols, dropna=False).size().rename("sample_weight")
        df_work = (
            df_work.drop_duplicates(subset=check_cols, keep="first")
            .drop(columns=["sample_weight"], errors="ignore")
            .merge(group_counts, on=check_cols, how="left")
            .reset_index(drop=True)
        )
        output_rows = len(df_work)
        rows_removed = input_rows - output_rows
        weight_added = True
        notes = (
            f"Consolidated duplicate rows into unique fingerprints. "
            f"Assigned sample_weight equal to frequency count (1 to {int(df_work['sample_weight'].max())})."
        )

    report = DuplicateHandlingReport(
        strategy=resolved_strategy,
        input_rows=input_rows,
        output_rows=output_rows,
        duplicates_detected=num_redundant,
        rows_removed=rows_removed,
        sample_weight_column_added=weight_added,
        notes=notes,
    )

    if save_report_path:
        save_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_report_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("Saved duplicate handling report to %s", save_report_path)

    return df_work, report
