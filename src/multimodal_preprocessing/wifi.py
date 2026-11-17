"""Wi-Fi fingerprint cleaning, feature extraction, and signal representation (Task 20)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logging import setup_logger
from src.multimodal_preprocessing.config import MultimodalPreprocessingConfig

logger = setup_logger("MultimodalWifiPreprocessing")

MAC_REGEX = re.compile(r"^([0-9a-fA-F]{2}[:-]){5}([0-9a-fA-F]{2})$")


@dataclass
class WifiCleaningStats:
    """Statistical summary of Wi-Fi cleaning stage."""

    records_input: int = 0
    records_retained: int = 0
    records_removed_missing_bssid: int = 0
    records_removed_invalid_rssi: int = 0
    records_removed_invalid_freq: int = 0
    records_removed_duplicates: int = 0
    records_removed_stale: int = 0
    records_removed_invalid_time: int = 0
    unique_bssids_found: int = 0
    total_sweeps_formed: int = 0


class WifiPreprocessor:
    """Cleans raw Wi-Fi observations and extracts structured fingerprint features."""

    def __init__(self, config: Optional[MultimodalPreprocessingConfig] = None):
        self.config = config or MultimodalPreprocessingConfig()

    def process(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, WifiCleaningStats, Dict[str, Any]]:
        """Cleans Wi-Fi dataframe, groups into sweeps, and extracts sweep features.

        Returns:
            Tuple of:
            - cleaned_df: Standardized individual AP observation records.
            - features_df: Aggregated per-sweep fingerprint feature vectors.
            - stats: Cleaning statistics object.
            - bssid_stats: Dict of per-BSSID session statistics.
        """
        stats = WifiCleaningStats(records_input=len(raw_df))
        if raw_df.empty:
            logger.info("Wi-Fi raw dataframe is empty; returning empty results.")
            empty_cleaned = pd.DataFrame(columns=[
                "observation_id", "sweep_id", "session_id", "normalized_epoch_ms", "normalized_iso",
                "time_source", "sync_method", "uncertainty_ms", "is_fresh", "is_mock",
                "bssid", "ssid", "rssi_dbm", "frequency_mhz", "channel", "channel_band",
                "building", "floor", "room"
            ])
            empty_features = pd.DataFrame(columns=[
                "sweep_id", "session_id", "normalized_epoch_ms", "normalized_iso",
                "num_visible_aps", "rssi_mean", "rssi_max", "rssi_min", "rssi_std",
                "strongest_bssid", "strongest_rssi", "building", "floor", "room", "is_mock"
            ])
            return empty_cleaned, empty_features, stats, {}

        df = raw_df.copy()

        # 1. Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # 2. Filter missing or invalid BSSIDs
        if "bssid" not in df.columns:
            stats.records_removed_missing_bssid = len(df)
            return pd.DataFrame(), pd.DataFrame(), stats, {}

        def is_valid_bssid(val: Any) -> bool:
            if not isinstance(val, str):
                return False
            s = val.strip().lower()
            return bool(MAC_REGEX.match(s))

        bssid_valid_mask = df["bssid"].apply(is_valid_bssid)
        stats.records_removed_missing_bssid = int((~bssid_valid_mask).sum())
        df = df[bssid_valid_mask].copy()
        df["bssid"] = df["bssid"].str.strip().str.lower()

        # 3. Filter invalid timestamps
        if "normalized_epoch_ms" in df.columns:
            df["normalized_epoch_ms"] = pd.to_numeric(df["normalized_epoch_ms"], errors="coerce")
            valid_time_mask = df["normalized_epoch_ms"].notna() & (df["normalized_epoch_ms"] > 0)
            stats.records_removed_invalid_time = int((~valid_time_mask).sum())
            df = df[valid_time_mask].copy()
            df["normalized_epoch_ms"] = df["normalized_epoch_ms"].astype("int64")
        else:
            stats.records_removed_invalid_time = len(df)
            return pd.DataFrame(), pd.DataFrame(), stats, {}

        # 4. Filter RSSI values
        df["rssi_dbm"] = pd.to_numeric(df.get("rssi_dbm", df.get("rssi")), errors="coerce")
        rssi_valid_mask = (
            df["rssi_dbm"].notna() &
            (df["rssi_dbm"] >= self.config.min_rssi_dbm) &
            (df["rssi_dbm"] <= self.config.max_rssi_dbm)
        )
        stats.records_removed_invalid_rssi = int((~rssi_valid_mask).sum())
        df = df[rssi_valid_mask].copy()

        # 5. Filter Frequencies
        if "frequency_mhz" in df.columns:
            df["frequency_mhz"] = pd.to_numeric(df["frequency_mhz"], errors="coerce").fillna(2412.0)
            freq_valid_mask = (
                (df["frequency_mhz"] >= self.config.min_frequency_mhz) &
                (df["frequency_mhz"] <= self.config.max_frequency_mhz)
            )
            stats.records_removed_invalid_freq = int((~freq_valid_mask).sum())
            df = df[freq_valid_mask].copy()
        else:
            df["frequency_mhz"] = 2412.0

        # 6. Filter stale scans if configured
        if self.config.require_fresh_scans_only and "is_fresh" in df.columns:
            fresh_mask = df["is_fresh"].astype(str).str.lower().isin(["true", "1"])
            stats.records_removed_stale = int((~fresh_mask).sum())
            df = df[fresh_mask].copy()

        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), stats, {}

        # 7. Identify Sweeps:
        # A sweep groups AP records that were collected in the same hardware scan batch.
        # Typically they share normalized_epoch_ms (within a 100ms cluster) or observation prefix.
        # If timestamps have jitter < 100ms, group by 100ms cluster or discrete timestamp.
        df = df.sort_values("normalized_epoch_ms").reset_index(drop=True)
        time_diffs = df["normalized_epoch_ms"].diff().fillna(0)
        # New sweep if time gap between adjacent records is > 200 ms
        sweep_indices = (time_diffs > 200).cumsum()
        df["sweep_id"] = [f"SWEEP-{idx+1:05d}" for idx in sweep_indices]

        # 8. Deduplicate BSSID within the same sweep
        before_dedup = len(df)
        if self.config.deduplicate_wifi_ap == "keep_strongest":
            # Sort by RSSI descending so strongest is first
            df = df.sort_values(by=["sweep_id", "bssid", "rssi_dbm"], ascending=[True, True, False])
            df = df.drop_duplicates(subset=["sweep_id", "bssid"], keep="first")
        elif self.config.deduplicate_wifi_ap == "keep_first":
            df = df.drop_duplicates(subset=["sweep_id", "bssid"], keep="first")
        elif self.config.deduplicate_wifi_ap == "reject":
            dups = df.duplicated(subset=["sweep_id", "bssid"], keep=False)
            df = df[~dups]

        stats.records_removed_duplicates = before_dedup - len(df)
        stats.records_retained = len(df)
        stats.unique_bssids_found = int(df["bssid"].nunique())
        stats.total_sweeps_formed = int(df["sweep_id"].nunique())

        # Ensure standard columns in cleaned DataFrame
        cleaned_columns = [
            "observation_id", "sweep_id", "session_id", "normalized_epoch_ms", "normalized_iso",
            "time_source", "sync_method", "uncertainty_ms", "is_fresh", "is_mock",
            "bssid", "ssid", "rssi_dbm", "frequency_mhz", "channel", "channel_band",
            "building", "floor", "room"
        ]
        for col in cleaned_columns:
            if col not in df.columns:
                df[col] = "UNKNOWN" if col in ("building", "floor", "room", "ssid", "channel_band") else None

        cleaned_df = df[cleaned_columns].copy()

        # 9. Extract per-sweep feature aggregations
        feature_rows: List[Dict[str, Any]] = []
        for sweep_id, group in df.groupby("sweep_id"):
            rssi_vals = group["rssi_dbm"].values
            strongest_idx = int(np.argmax(rssi_vals))
            strongest_bssid = group["bssid"].iloc[strongest_idx]
            strongest_rssi = float(rssi_vals[strongest_idx])

            feat = {
                "sweep_id": sweep_id,
                "session_id": group["session_id"].iloc[0] if "session_id" in group else "UNKNOWN",
                "normalized_epoch_ms": int(group["normalized_epoch_ms"].median()),
                "normalized_iso": group["normalized_iso"].iloc[0] if "normalized_iso" in group else "",
                "num_visible_aps": int(len(group)),
                "rssi_mean": float(np.mean(rssi_vals)),
                "rssi_max": float(np.max(rssi_vals)),
                "rssi_min": float(np.min(rssi_vals)),
                "rssi_std": float(np.std(rssi_vals)) if len(rssi_vals) > 1 else 0.0,
                "strongest_bssid": strongest_bssid,
                "strongest_rssi": strongest_rssi,
                "building": group["building"].iloc[0] if "building" in group else "UNKNOWN",
                "floor": group["floor"].iloc[0] if "floor" in group else "UNKNOWN",
                "room": group["room"].iloc[0] if "room" in group else "UNKNOWN",
                "is_mock": bool(group["is_mock"].iloc[0]) if "is_mock" in group else False,
            }
            feature_rows.append(feat)

        features_df = pd.DataFrame(feature_rows)

        # 10. Compute per-BSSID session statistics
        bssid_stats: Dict[str, Any] = {}
        total_sweeps = max(1, stats.total_sweeps_formed)
        for bssid, group in df.groupby("bssid"):
            b_rssi = group["rssi_dbm"].values
            bssid_stats[bssid] = {
                "total_detections": int(len(group)),
                "mean_rssi": float(np.mean(b_rssi)),
                "std_rssi": float(np.std(b_rssi)) if len(b_rssi) > 1 else 0.0,
                "min_rssi": float(np.min(b_rssi)),
                "max_rssi": float(np.max(b_rssi)),
                "observation_frequency": float(len(group) / total_sweeps),
                "ssid": str(group["ssid"].iloc[0]) if "ssid" in group else ""
            }

        logger.info(
            "Wi-Fi Preprocessing: %d input -> %d retained across %d sweeps and %d unique BSSIDs.",
            stats.records_input, stats.records_retained, stats.total_sweeps_formed, stats.unique_bssids_found
        )
        return cleaned_df, features_df, stats, bssid_stats
