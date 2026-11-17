"""Access Point-level statistics calculation and relative signal feature generation.

Computes training-only access point summary statistics (detection frequency,
mean/median/std RSSI) and generates configurable relative differential features
(e.g., AP signal relative to the sample's strongest or mean signal).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.logging import setup_logger

logger = setup_logger("APFeatures")

DEFAULT_UNAVAILABLE_SENTINEL = 100


def compute_training_ap_statistics(
    df_train: pd.DataFrame,
    wap_prefix: str = "WAP",
    sentinel: int = DEFAULT_UNAVAILABLE_SENTINEL,
    save_csv_path: Optional[Path] = None
) -> pd.DataFrame:
    """Calculate dataset-level AP statistics exclusively on the training partition.

    Args:
        df_train: Raw training DataFrame containing WAP columns.
        wap_prefix: Prefix identifying WAP columns (default 'WAP').
        sentinel: Unavailable AP marker (default 100).
        save_csv_path: Optional path to save 'ap_statistics.csv'.

    Returns:
        DataFrame with columns [ap_name, train_detection_count, train_detection_rate,
        mean_rssi, median_rssi, std_rssi, min_rssi, max_rssi, is_constant_100, status].
    """
    wap_cols = [c for c in df_train.columns if c.startswith(wap_prefix) and not c.endswith("_VISIBLE")]
    wap_cols = sorted(wap_cols)
    total_samples = len(df_train)

    logger.info("Computing AP-level statistics across %d training fingerprints (%d WAPs)...", total_samples, len(wap_cols))
    records = []

    for wap in wap_cols:
        series = df_train[wap]
        valid_mask = series != sentinel
        detected_count = int(valid_mask.sum())
        detection_rate = round(detected_count / total_samples, 6)
        detection_pct = round(detection_rate * 100.0, 4)

        if detected_count > 0:
            valid_vals = series[valid_mask]
            s_mean = round(float(valid_vals.mean()), 3)
            s_med = float(valid_vals.median())
            s_std = round(float(valid_vals.std()), 3) if detected_count > 1 else 0.0
            s_min = float(valid_vals.min())
            s_max = float(valid_vals.max())
            status = "ACTIVE"
            is_const = False
        else:
            s_mean = None
            s_med = None
            s_std = None
            s_min = None
            s_max = None
            status = "CONSTANT_UNAVAILABLE"
            is_const = True

        records.append({
            "ap_name": wap,
            "train_detection_count": detected_count,
            "train_detection_rate": detection_rate,
            "train_detection_pct": detection_pct,
            "mean_rssi": s_mean,
            "median_rssi": s_med,
            "std_rssi": s_std,
            "min_rssi": s_min,
            "max_rssi": s_max,
            "is_constant_100": is_const,
            "status": status,
        })

    ap_stats_df = pd.DataFrame(records)

    if save_csv_path:
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        ap_stats_df.to_csv(save_csv_path, index=False)
        logger.info("Saved training AP statistics artifact to %s", save_csv_path)

    return ap_stats_df


def compute_relative_signal_features(
    df: pd.DataFrame,
    sample_stats_df: Optional[pd.DataFrame] = None,
    wap_prefix: str = "WAP",
    sentinel: int = DEFAULT_UNAVAILABLE_SENTINEL,
    diff_to_max: bool = True,
    diff_to_mean: bool = False,
    default_unavailable_diff: float = 0.0,
    default_unavailable_val: Optional[float] = None
) -> pd.DataFrame:
    """Compute relative signal differences for each detected WAP.

    Calculates:
    - diff_to_max: (RSSI_i - strongest_rssi_1) when detected, else default_unavailable_diff.
    - diff_to_mean: (RSSI_i - rssi_mean) when detected, else default_unavailable_diff.

    Args:
        df: DataFrame containing raw WAP columns.
        sample_stats_df: Precomputed per-sample statistics containing 'strongest_rssi_1' and 'rssi_mean'.
                         If None, computed automatically from df.
        wap_prefix: WAP column prefix.
        sentinel: Value 100 denoting unavailable AP.
        diff_to_max: Whether to compute difference relative to strongest AP.
        diff_to_mean: Whether to compute difference relative to sample mean.
        default_unavailable_diff: Value assigned to unavailable APs in differential features.
        default_unavailable_val: Deprecated alias for default_unavailable_diff.

    Returns:
        DataFrame of relative features aligned with df.index.
    """
    if default_unavailable_val is not None:
        default_unavailable_diff = default_unavailable_val

    if sample_stats_df is None:
        from src.features.signal_statistics import extract_signal_statistics_from_dataframe, SignalStatsConfig
        sample_stats_df = extract_signal_statistics_from_dataframe(
            df=df,
            wap_prefix=wap_prefix,
            config=SignalStatsConfig(unavailable_sentinel=sentinel)
        )

    wap_cols = [c for c in df.columns if c.startswith(wap_prefix) and not c.endswith("_VISIBLE")]
    wap_cols = sorted(wap_cols)

    wap_mat = df[wap_cols].values.astype(np.float32)
    valid_mask = (wap_mat != sentinel)

    out_frames: List[pd.DataFrame] = []

    if diff_to_max:
        strongest = sample_stats_df["strongest_rssi_1"].values[:, None]
        # Diff is strictly non-positive (0.0 for strongest, negative for weaker APs)
        diff_max_mat = np.where(valid_mask, wap_mat - strongest, default_unavailable_diff)
        col_names = [f"{c}_DIFF_TO_MAX" for c in wap_cols]
        out_frames.append(pd.DataFrame(diff_max_mat, columns=col_names, index=df.index))

    if diff_to_mean:
        mean_rssi = sample_stats_df["rssi_mean"].values[:, None]
        diff_mean_mat = np.where(valid_mask, wap_mat - mean_rssi, default_unavailable_diff)
        col_names = [f"{c}_DIFF_TO_MEAN" for c in wap_cols]
        out_frames.append(pd.DataFrame(diff_mean_mat, columns=col_names, index=df.index))

    if not out_frames:
        return pd.DataFrame(index=df.index)

    return pd.concat(out_frames, axis=1)
