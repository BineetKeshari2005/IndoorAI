"""Wi-Fi signal preprocessing and representation engineering for IndoorAI.

Handles unavailable Wi-Fi access point observations (value 100),
replaces unobserved APs with a configurable physical noise floor (default -105 dBm),
generates explicit binary visibility masks for each WAP, and provides both
'rssi_only' and 'rssi_with_visibility' feature representations.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.logging import setup_logger

logger = setup_logger("WifiPreprocessing")

DEFAULT_UNAVAILABLE_SENTINEL = 100
DEFAULT_RSSI_FLOOR_DBM = -105.0


def extract_wap_columns(df: pd.DataFrame) -> List[str]:
    """Return all columns starting with 'WAP' in deterministic alphabetical/column order."""
    wap_cols = [c for c in df.columns if c.startswith("WAP") and not c.endswith("_VISIBLE")]
    return sorted(wap_cols)


def create_visibility_mask(
    df: pd.DataFrame,
    wap_cols: Optional[List[str]] = None,
    sentinel: int = DEFAULT_UNAVAILABLE_SENTINEL,
    suffix: str = "_VISIBLE"
) -> pd.DataFrame:
    """Create a binary detection mask DataFrame for all specified WAP columns.

    Args:
        df: Input DataFrame containing raw WAP columns.
        wap_cols: List of WAP column names (auto-detected if None).
        sentinel: Value denoting an unavailable AP (default 100).
        suffix: Suffix for binary indicator features (default '_VISIBLE').

    Returns:
        DataFrame of int8 binary indicators where 1=detected, 0=unavailable.
    """
    cols = wap_cols or extract_wap_columns(df)
    wap_mat = df[cols].values
    binary_mat = (wap_mat != sentinel).astype(np.int8)
    mask_col_names = [f"{c}{suffix}" for c in cols]
    return pd.DataFrame(binary_mat, columns=mask_col_names, index=df.index)


def process_rssi_matrix(
    df: pd.DataFrame,
    wap_cols: Optional[List[str]] = None,
    sentinel: int = DEFAULT_UNAVAILABLE_SENTINEL,
    rssi_floor: float = DEFAULT_RSSI_FLOOR_DBM
) -> pd.DataFrame:
    """Replace unavailable sentinel observations (100) with a physical RSSI noise floor.

    Genuine measurements are strictly preserved.

    Args:
        df: Input DataFrame containing raw WAP columns.
        wap_cols: List of WAP column names.
        sentinel: Sentinel representing missing AP (default 100).
        rssi_floor: Configurable floor value in dBm (default -105.0 dBm).

    Returns:
        DataFrame containing cleaned continuous RSSI values.
    """
    cols = wap_cols or extract_wap_columns(df)
    wap_mat = df[cols].values.astype(np.float32)
    cleaned_mat = np.where(wap_mat == sentinel, rssi_floor, wap_mat)
    return pd.DataFrame(cleaned_mat, columns=cols, index=df.index)


def generate_wifi_features(
    df: pd.DataFrame,
    representation: str = "rssi_with_visibility",
    wap_cols: Optional[List[str]] = None,
    sentinel: int = DEFAULT_UNAVAILABLE_SENTINEL,
    rssi_floor: float = DEFAULT_RSSI_FLOOR_DBM
) -> pd.DataFrame:
    """Generate the designated Wi-Fi feature matrix representation.

    Supported representations:
    1. 'rssi_only': Continuous RSSI values with unavailable APs set to rssi_floor (520 columns).
    2. 'rssi_with_visibility': Continuous RSSI values concatenated with binary *_VISIBLE masks (1040 columns).

    Args:
        df: Input DataFrame.
        representation: 'rssi_only' or 'rssi_with_visibility'.
        wap_cols: Optional list of WAP columns.
        sentinel: Unavailable AP sentinel (default 100).
        rssi_floor: Configurable floor in dBm (default -105.0 dBm).

    Returns:
        Processed Wi-Fi feature DataFrame.
    """
    resolved_rep = representation.lower().strip()
    valid_reps = ["rssi_only", "rssi_with_visibility"]
    if resolved_rep not in valid_reps:
        raise ValueError(
            f"Unknown feature representation '{representation}'. Valid options: {valid_reps}"
        )

    cols = wap_cols or extract_wap_columns(df)
    logger.info(
        "Generating '%s' feature representation for %d WAPs across %d samples (floor: %.1f dBm)...",
        resolved_rep, len(cols), len(df), rssi_floor
    )

    rssi_df = process_rssi_matrix(df, wap_cols=cols, sentinel=sentinel, rssi_floor=rssi_floor)

    if resolved_rep == "rssi_only":
        return rssi_df

    visibility_df = create_visibility_mask(df, wap_cols=cols, sentinel=sentinel)
    combined_df = pd.concat([rssi_df, visibility_df], axis=1)
    logger.info("Combined representation generated: %d total features.", len(combined_df.columns))
    return combined_df

