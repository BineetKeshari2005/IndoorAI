"""Timestamp normalization, clock uncertainty preservation, and configurable temporal window alignment (Task 20)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.core.logging import setup_logger
from src.multimodal_preprocessing.config import MultimodalPreprocessingConfig

logger = setup_logger("MultimodalTimestampAlignment")


@dataclass
class AlignmentStats:
    """Summary metrics of cross-modal temporal alignment."""

    session_start_ms: int = 0
    session_end_ms: int = 0
    total_session_duration_ms: int = 0
    window_size_ms: float = 500.0
    total_windows_created: int = 0
    windows_with_all_modalities: int = 0
    windows_multimodal_concurrent: int = 0
    windows_wifi_only: int = 0
    windows_sensors_only: int = 0
    windows_photos_only: int = 0
    windows_empty: int = 0
    total_ambiguous_timing_observations: int = 0
    mean_clock_uncertainty_ms: float = 0.0


class TimestampAligner:
    """Discretizes multimodal session observations into non-interpolated temporal windows with boundary uncertainty tracking."""

    def __init__(self, config: Optional[MultimodalPreprocessingConfig] = None):
        self.config = config or MultimodalPreprocessingConfig()

    def align(
        self,
        wifi_df: Optional[pd.DataFrame] = None,
        sensors_df: Optional[pd.DataFrame] = None,
        photos_df: Optional[pd.DataFrame] = None,
        annotations_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, AlignmentStats]:
        """Aligns multimodal observations into temporal windows without inventing missing data.

        Returns:
            Tuple of:
            - windows_df: DataFrame where each row represents a temporal alignment window.
            - stats: Summary alignment statistics.
        """
        stats = AlignmentStats(window_size_ms=self.config.temporal_window_size_ms)

        # 1. Collect all timestamps to find global temporal span
        all_timestamps: List[int] = []
        all_uncertainties: List[float] = []

        dfs = {
            "wifi": wifi_df,
            "sensors": sensors_df,
            "photos": photos_df,
            "annotations": annotations_df,
        }

        for mod_name, df in dfs.items():
            if df is not None and not df.empty and "normalized_epoch_ms" in df.columns:
                valid_ts = df["normalized_epoch_ms"].dropna().astype("int64")
                all_timestamps.extend(valid_ts.tolist())
                if "uncertainty_ms" in df.columns:
                    unc = pd.to_numeric(df["uncertainty_ms"], errors="coerce").dropna()
                    all_uncertainties.extend(unc.tolist())

        if not all_timestamps:
            logger.info("No valid timestamps found across any modality; returning empty alignment.")
            empty_windows = pd.DataFrame(columns=[
                "window_id", "window_start_ms", "window_end_ms", "window_duration_ms",
                "has_wifi", "has_sensors", "has_photos", "has_annotations",
                "is_multimodal_concurrent", "wifi_obs_count", "sensor_obs_count",
                "photo_obs_count", "annotation_obs_count", "wifi_obs_ids",
                "sensor_obs_ids", "photo_obs_ids", "annotation_obs_ids",
                "is_timing_ambiguous", "ambiguous_obs_count"
            ])
            return empty_windows, stats

        t_min = min(all_timestamps)
        t_max = max(all_timestamps)
        stats.session_start_ms = t_min
        stats.session_end_ms = t_max
        stats.total_session_duration_ms = max(0, t_max - t_min)
        stats.mean_clock_uncertainty_ms = float(np.mean(all_uncertainties)) if all_uncertainties else 0.0

        # 2. Build temporal window grid
        window_size = int(self.config.temporal_window_size_ms)
        stride = int(self.config.temporal_window_stride_ms)
        if window_size <= 0:
            window_size = 500
        if stride <= 0:
            stride = window_size

        window_starts = list(range(t_min, t_max + 1, stride))
        if not window_starts:
            window_starts = [t_min]

        # Pre-index modality DataFrames by normalized_epoch_ms for high performance
        mod_records: Dict[str, List[Dict[str, Any]]] = {}
        for mod_name, df in dfs.items():
            if df is not None and not df.empty and "normalized_epoch_ms" in df.columns:
                cols = ["observation_id", "normalized_epoch_ms", "uncertainty_ms"]
                available_cols = [c for c in cols if c in df.columns]
                mod_records[mod_name] = df[available_cols].to_dict(orient="records")
            else:
                mod_records[mod_name] = []

        window_rows: List[Dict[str, Any]] = []
        total_ambiguous_obs = 0

        for idx, w_start in enumerate(window_starts):
            w_end = w_start + window_size
            w_id = f"WIN-{idx+1:06d}"

            matched_ids: Dict[str, List[str]] = {
                "wifi": [],
                "sensors": [],
                "photos": [],
                "annotations": [],
            }
            ambiguous_in_window = 0

            # Find matching observations for each modality
            for mod_name, records in mod_records.items():
                for rec in records:
                    ts = rec.get("normalized_epoch_ms")
                    if ts is not None and w_start <= ts < w_end:
                        obs_id = str(rec.get("observation_id", "OBS"))
                        matched_ids[mod_name].append(obs_id)

                        # Check if observation timing is ambiguous near boundary
                        if self.config.flag_ambiguous_timing_boundary:
                            unc = float(rec.get("uncertainty_ms") or 0.0)
                            if unc > 0.0:
                                # If interval [ts - unc, ts + unc] crosses window start or end
                                if (ts - unc < w_start) or (ts + unc >= w_end):
                                    ambiguous_in_window += 1

            has_wifi = len(matched_ids["wifi"]) > 0
            has_sensors = len(matched_ids["sensors"]) > 0
            has_photos = len(matched_ids["photos"]) > 0
            has_annotations = len(matched_ids["annotations"]) > 0

            mod_count = sum([has_wifi, has_sensors, has_photos, has_annotations])
            is_concurrent = mod_count >= 2
            is_all_modalities = mod_count == 4

            if mod_count == 0:
                stats.windows_empty += 1
            elif is_all_modalities:
                stats.windows_with_all_modalities += 1
            if is_concurrent:
                stats.windows_multimodal_concurrent += 1
            if has_wifi and mod_count == 1:
                stats.windows_wifi_only += 1
            if has_sensors and mod_count == 1:
                stats.windows_sensors_only += 1
            if has_photos and mod_count == 1:
                stats.windows_photos_only += 1

            total_ambiguous_obs += ambiguous_in_window

            window_rows.append({
                "window_id": w_id,
                "window_start_ms": w_start,
                "window_end_ms": w_end,
                "window_duration_ms": window_size,
                "has_wifi": has_wifi,
                "has_sensors": has_sensors,
                "has_photos": has_photos,
                "has_annotations": has_annotations,
                "is_multimodal_concurrent": is_concurrent,
                "wifi_obs_count": len(matched_ids["wifi"]),
                "sensor_obs_count": len(matched_ids["sensors"]),
                "photo_obs_count": len(matched_ids["photos"]),
                "annotation_obs_count": len(matched_ids["annotations"]),
                "wifi_obs_ids": json.dumps(matched_ids["wifi"]),
                "sensor_obs_ids": json.dumps(matched_ids["sensors"]),
                "photo_obs_ids": json.dumps(matched_ids["photos"]),
                "annotation_obs_ids": json.dumps(matched_ids["annotations"]),
                "is_timing_ambiguous": ambiguous_in_window > 0,
                "ambiguous_obs_count": ambiguous_in_window,
            })

        stats.total_windows_created = len(window_rows)
        stats.total_ambiguous_timing_observations = total_ambiguous_obs

        windows_df = pd.DataFrame(window_rows)
        logger.info(
            "Temporal Alignment: Created %d windows (size=%d ms) across %d ms span. Concurrent windows: %d.",
            stats.total_windows_created, window_size, stats.total_session_duration_ms, stats.windows_multimodal_concurrent
        )
        return windows_df, stats
