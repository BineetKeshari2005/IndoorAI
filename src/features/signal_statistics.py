"""Per-sample Wi-Fi signal statistics and environment descriptor extractor.

Computes visibility counts, signal strength moments, strongest access point signals,
and signal quality distribution bands for each individual fingerprint.
Guarantees that unavailable access points are excluded from RSSI moment calculations
and handles zero-visible-AP edge cases safely.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.logging import setup_logger

logger = setup_logger("SignalStatistics")

DEFAULT_UNAVAILABLE_SENTINEL = 100
DEFAULT_RSSI_FLOOR_DBM = -105.0
DEFAULT_STRONG_THRESHOLD_DBM = -60.0
DEFAULT_MEDIUM_THRESHOLD_DBM = -80.0

SUMMARY_FEATURE_NAMES = [
    "visible_ap_count",
    "visible_ap_ratio",
    "unavailable_ap_count",
    "rssi_mean",
    "rssi_median",
    "rssi_std",
    "rssi_min",
    "rssi_max",
    "rssi_range",
    "rssi_q25",
    "rssi_q75",
    "strongest_rssi_1",
    "strongest_rssi_2",
    "strongest_rssi_3",
    "strongest_rssi_5",
    "strong_signal_count",
    "medium_signal_count",
    "weak_signal_count",
]


@dataclass
class SignalStatsConfig:
    """Configuration parameters for per-sample signal feature extraction."""
    unavailable_sentinel: int = DEFAULT_UNAVAILABLE_SENTINEL
    rssi_floor_dbm: float = DEFAULT_RSSI_FLOOR_DBM
    strong_threshold_dbm: float = DEFAULT_STRONG_THRESHOLD_DBM
    medium_threshold_dbm: float = DEFAULT_MEDIUM_THRESHOLD_DBM


def compute_sample_signal_statistics(
    wap_matrix: np.ndarray,
    config: Optional[SignalStatsConfig] = None,
    total_waps: Optional[int] = None
) -> pd.DataFrame:
    """Compute 18 per-sample signal environment summary features for a WAP array.

    Args:
        wap_matrix: 2D numpy array of shape (N_samples, N_waps) with raw RSSI values in dBm
                    (where unavailable APs are equal to config.unavailable_sentinel, e.g. 100).
        config: Configuration containing thresholds and defaults.
        total_waps: Total access point dimension for calculating visible_ap_ratio.

    Returns:
        DataFrame with 18 summary statistics columns and N_samples rows.
    """
    cfg = config or SignalStatsConfig()
    n_samples, n_features = wap_matrix.shape
    num_total_waps = total_waps or n_features

    sentinel = cfg.unavailable_sentinel
    default_floor = cfg.rssi_floor_dbm
    strong_thresh = cfg.strong_threshold_dbm
    med_thresh = cfg.medium_threshold_dbm

    # Pre-allocate output feature matrix
    out = np.zeros((n_samples, len(SUMMARY_FEATURE_NAMES)), dtype=np.float32)

    logger.info("Computing per-sample Wi-Fi signal statistics across %d fingerprints (%d WAPs)...", n_samples, n_features)

    for i in range(n_samples):
        row = wap_matrix[i]
        # Isolate genuine physical RSSI measurements
        valid = row[row != sentinel]
        count = len(valid)

        # Visibility metrics
        vis_count = float(count)
        vis_ratio = vis_count / float(num_total_waps)
        unavail_count = float(num_total_waps - count)

        if count == 0:
            # Safe default fallback for blind measurements (no detected APs)
            r_mean = default_floor
            r_med = default_floor
            r_std = 0.0
            r_min = default_floor
            r_max = default_floor
            r_range = 0.0
            r_q25 = default_floor
            r_q75 = default_floor
            s1 = default_floor
            s2 = default_floor
            s3 = default_floor
            s5 = default_floor
            c_strong = 0.0
            c_med = 0.0
            c_weak = 0.0
        else:
            # Sorted in descending order: strongest signal first
            sorted_desc = np.sort(valid)[::-1]
            r_max = float(sorted_desc[0])
            r_min = float(sorted_desc[-1])
            r_range = r_max - r_min
            r_mean = float(np.mean(valid))
            r_med = float(np.median(valid))
            r_std = float(np.std(valid)) if count > 1 else 0.0
            r_q25 = float(np.percentile(valid, 25))
            r_q75 = float(np.percentile(valid, 75))

            s1 = r_max
            s2 = float(sorted_desc[1]) if count >= 2 else default_floor
            s3 = float(sorted_desc[2]) if count >= 3 else default_floor
            s5 = float(sorted_desc[4]) if count >= 5 else default_floor

            c_strong = float((valid >= strong_thresh).sum())
            c_med = float(((valid >= med_thresh) & (valid < strong_thresh)).sum())
            c_weak = float((valid < med_thresh).sum())

        out[i, :] = [
            vis_count, vis_ratio, unavail_count,
            r_mean, r_med, r_std, r_min, r_max, r_range, r_q25, r_q75,
            s1, s2, s3, s5,
            c_strong, c_med, c_weak
        ]

    stats_df = pd.DataFrame(out, columns=SUMMARY_FEATURE_NAMES)
    return stats_df


def extract_signal_statistics_from_dataframe(
    df: pd.DataFrame,
    wap_prefix: str = "WAP",
    config: Optional[SignalStatsConfig] = None
) -> pd.DataFrame:
    """Extract signal statistics from a DataFrame containing raw WAP columns.

    Args:
        df: Input DataFrame with raw WAP columns.
        wap_prefix: Prefix used for WAP column identification (default 'WAP').
        config: SignalStatsConfig.

    Returns:
        DataFrame with the 18 summary statistics aligned with df.index.
    """
    wap_cols = [c for c in df.columns if c.startswith(wap_prefix) and not c.endswith("_VISIBLE")]
    wap_cols = sorted(wap_cols)
    if not wap_cols:
        raise ValueError(f"No WAP columns starting with '{wap_prefix}' found in DataFrame.")

    wap_mat = df[wap_cols].values
    stats_df = compute_sample_signal_statistics(wap_mat, config=config, total_waps=len(wap_cols))
    stats_df.index = df.index
    return stats_df

