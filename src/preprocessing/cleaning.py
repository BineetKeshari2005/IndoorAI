"""Access Point feature quality assessment and configurable feature selection.

Audits zero-variance and low-visibility WAP features using strictly training-set statistics,
generates the 'ap_feature_quality.csv' report, and provides configurable feature selection
while defaulting to preserving the full 520-WAP schema.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.logging import setup_logger

logger = setup_logger("FeatureQualityCleaner")

UNAVAILABLE_WAP_VALUE = 100
DEFAULT_LOW_VISIBILITY_THRESHOLD_PCT = 0.1  # 0.1% detection threshold (~20 samples in train)


def audit_ap_feature_quality(
    df_train: pd.DataFrame,
    df_val: Optional[pd.DataFrame] = None,
    low_visibility_pct: float = DEFAULT_LOW_VISIBILITY_THRESHOLD_PCT,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """Analyze quality and variance of all 520 WAPs using training set data.

    Validation counts are recorded purely for informational reporting, never to influence
    feature selection.

    Args:
        df_train: Training DataFrame.
        df_val: Optional validation DataFrame for informational tracking.
        low_visibility_pct: Percentage threshold below which an active AP is flagged low visibility.
        save_path: Optional path to save CSV report.

    Returns:
        DataFrame containing per-WAP feature quality metrics.
    """
    wap_cols = [c for c in df_train.columns if c.startswith("WAP") and not c.endswith("_VISIBLE")]
    wap_cols = sorted(wap_cols)
    total_train = len(df_train)
    total_val = len(df_val) if df_val is not None else 0

    logger.info("Auditing feature quality for %d WAPs across %d training samples...", len(wap_cols), total_train)
    records = []

    for wap in wap_cols:
        t_series = df_train[wap]
        t_valid = (t_series != UNAVAILABLE_WAP_VALUE)
        t_count = int(t_valid.sum())
        t_pct = round((t_count / total_train) * 100.0, 4)

        if df_val is not None and wap in df_val.columns:
            v_series = df_val[wap]
            v_count = int((v_series != UNAVAILABLE_WAP_VALUE).sum())
            v_pct = round((v_count / total_val) * 100.0, 4)
        else:
            v_count, v_pct = 0, 0.0

        is_constant_train = (t_count == 0)
        is_low_vis_train = (0 < t_pct < low_visibility_pct)

        records.append({
            "wap_id": wap,
            "train_detected_count": t_count,
            "train_detected_pct": t_pct,
            "val_detected_count": v_count,
            "val_detected_pct": v_pct,
            "is_constant_train": is_constant_train,
            "is_low_visibility_train": is_low_vis_train,
            "variance_in_train": float(t_series.var()) if not is_constant_train else 0.0,
        })

    df_quality = pd.DataFrame(records)

    num_constant = int(df_quality["is_constant_train"].sum())
    num_low_vis = int(df_quality["is_low_visibility_train"].sum())
    logger.info(
        "Quality Audit: Found %d zero-variance constant WAPs in training data, %d low-visibility WAPs (< %.2f%%).",
        num_constant, num_low_vis, low_visibility_pct
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_quality.to_csv(save_path, index=False)
        logger.info("Saved AP feature quality report to %s", save_path)

    return df_quality


def select_features(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    drop_zero_variance: bool = False,
    ap_quality_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    """Configurably filter features based on training variance.

    Defaults to drop_zero_variance=False to preserve the complete 520-WAP schema.
    If drop_zero_variance is enabled, dropped columns are strictly derived from X_train.

    Args:
        X_train: Training feature DataFrame.
        X_val: Optional validation feature DataFrame.
        drop_zero_variance: If True, drop columns that have zero variance in X_train.
        ap_quality_df: Optional precomputed quality DataFrame.

    Returns:
        Tuple of (filtered_X_train, filtered_X_val, retained_column_names).
    """
    if not drop_zero_variance:
        logger.info("Preserving complete feature schema (%d features, zero-variance columns retained).", len(X_train.columns))
        return X_train, X_val, list(X_train.columns)

    # Identify zero-variance columns strictly from X_train
    if ap_quality_df is not None:
        constant_waps = set(ap_quality_df[ap_quality_df["is_constant_train"]]["wap_id"])
    else:
        constant_waps = set(X_train.columns[X_train.nunique() <= 1])

    # Also drop visibility mask features associated with dropped WAPs if present
    cols_to_drop = [
        col for col in X_train.columns
        if col in constant_waps or any(col == f"{w}_VISIBLE" for w in constant_waps)
    ]

    logger.info("Dropping %d zero-variance features based strictly on training data.", len(cols_to_drop))
    X_train_filtered = X_train.drop(columns=cols_to_drop)
    X_val_filtered = X_val.drop(columns=cols_to_drop) if X_val is not None else None

    return X_train_filtered, X_val_filtered, list(X_train_filtered.columns)

