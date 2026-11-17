"""Session segmentation into discrete temporal or topological analytical segments (Task 20)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from src.core.logging import setup_logger
from src.multimodal_preprocessing.config import MultimodalPreprocessingConfig

logger = setup_logger("MultimodalSegmentation")


@dataclass
class SegmentRecord:
    """Analytical segment of a continuous multimodal survey session."""

    segment_id: str
    parent_session_id: str
    segment_index: int
    start_timestamp_ms: int
    end_timestamp_ms: int
    duration_ms: int
    available_modalities: List[str]
    observation_counts: Dict[str, int]
    observation_references: Dict[str, List[str]]
    data_quality_flags: Dict[str, Any]


class SessionSegmenter:
    """Partitions continuous multimodal sessions into manageable analytical segments."""

    def __init__(self, config: Optional[MultimodalPreprocessingConfig] = None):
        self.config = config or MultimodalPreprocessingConfig()

    def segment(
        self,
        session_id: str,
        wifi_df: Optional[pd.DataFrame] = None,
        sensors_df: Optional[pd.DataFrame] = None,
        photos_df: Optional[pd.DataFrame] = None,
        annotations_df: Optional[pd.DataFrame] = None
    ) -> List[SegmentRecord]:
        """Segments session observations based on configured strategy."""
        # 1. Collect all observations into a unified chronological stream
        records: List[Dict[str, Any]] = []

        dfs = {
            "wifi": wifi_df,
            "sensors": sensors_df,
            "photos": photos_df,
            "annotations": annotations_df,
        }

        for mod_name, df in dfs.items():
            if df is not None and not df.empty and "normalized_epoch_ms" in df.columns:
                for row in df.to_dict(orient="records"):
                    ts = row.get("normalized_epoch_ms")
                    if ts is not None and not pd.isna(ts):
                        records.append({
                            "modality": mod_name,
                            "observation_id": str(row.get("observation_id", row.get("photo_id", row.get("annotation_id", "OBS")))),
                            "timestamp_ms": int(ts),
                            "uncertainty_ms": float(row.get("uncertainty_ms") or 0.0),
                            "is_gap": bool(row.get("is_gap", False)),
                            "type_code": str(row.get("type_code", "")),
                        })

        if not records:
            logger.info("No records available to segment for session %s.", session_id)
            return []

        records.sort(key=lambda r: r["timestamp_ms"])
        t_start_global = records[0]["timestamp_ms"]
        t_end_global = records[-1]["timestamp_ms"]

        strategy = self.config.segmentation_strategy.lower()
        boundary_timestamps: List[int] = [t_start_global]

        if strategy == "fixed_duration":
            dur_ms = int(self.config.segmentation_duration_seconds * 1000)
            curr = t_start_global + dur_ms
            while curr < t_end_global:
                boundary_timestamps.append(curr)
                curr += dur_ms
            boundary_timestamps.append(t_end_global + 1)

        elif strategy == "annotation_boundaries":
            # New segment boundary at each annotation event
            ann_times = sorted([r["timestamp_ms"] for r in records if r["modality"] == "annotations"])
            for t_ann in ann_times:
                if t_ann > boundary_timestamps[-1]:
                    boundary_timestamps.append(t_ann)
            boundary_timestamps.append(t_end_global + 1)

        elif strategy == "temporal_gaps":
            gap_threshold_ms = int(self.config.segmentation_gap_threshold_seconds * 1000)
            for i in range(1, len(records)):
                dt = records[i]["timestamp_ms"] - records[i-1]["timestamp_ms"]
                if dt >= gap_threshold_ms or records[i].get("is_gap"):
                    boundary_timestamps.append(records[i]["timestamp_ms"])
            boundary_timestamps.append(t_end_global + 1)

        elif strategy == "observation_count":
            count_thresh = max(10, self.config.segmentation_observation_count)
            for i in range(count_thresh, len(records), count_thresh):
                boundary_timestamps.append(records[i]["timestamp_ms"])
            boundary_timestamps.append(t_end_global + 1)

        else:
            # Fallback to single segment
            boundary_timestamps.append(t_end_global + 1)

        # Remove duplicate boundary timestamps and sort
        boundary_timestamps = sorted(list(set(boundary_timestamps)))
        if len(boundary_timestamps) == 1:
            boundary_timestamps.append(t_end_global + 1)

        # 2. Construct segments
        segments: List[SegmentRecord] = []
        seg_idx = 1

        for b_idx in range(len(boundary_timestamps) - 1):
            s_start = boundary_timestamps[b_idx]
            s_end = boundary_timestamps[b_idx + 1]

            # Collect observations strictly inside [s_start, s_end)
            seg_obs: List[Dict[str, Any]] = [
                r for r in records if s_start <= r["timestamp_ms"] < s_end
            ]

            if not seg_obs:
                # Do not emit empty segments
                continue

            obs_counts: Dict[str, int] = {"wifi": 0, "sensors": 0, "photos": 0, "annotations": 0}
            obs_refs: Dict[str, List[str]] = {"wifi": [], "sensors": [], "photos": [], "annotations": []}
            has_sensor_gap = False
            ambiguous_timing_count = 0

            for r in seg_obs:
                mod = r["modality"]
                obs_counts[mod] += 1
                obs_refs[mod].append(r["observation_id"])
                if r.get("is_gap"):
                    has_sensor_gap = True
                unc = r.get("uncertainty_ms", 0.0)
                if unc > 0.0:
                    if (r["timestamp_ms"] - unc < s_start) or (r["timestamp_ms"] + unc >= s_end):
                        ambiguous_timing_count += 1

            avail_mods = [m for m, count in obs_counts.items() if count > 0]
            is_concurrent = len(avail_mods) >= 2

            seg_record = SegmentRecord(
                segment_id=f"SEG-{seg_idx:05d}",
                parent_session_id=session_id,
                segment_index=seg_idx,
                start_timestamp_ms=s_start,
                end_timestamp_ms=s_end,
                duration_ms=max(0, s_end - s_start),
                available_modalities=avail_mods,
                observation_counts=obs_counts,
                observation_references=obs_refs,
                data_quality_flags={
                    "is_multimodal_concurrent": is_concurrent,
                    "has_sensor_gap": has_sensor_gap,
                    "ambiguous_boundary_observations": ambiguous_timing_count,
                    "is_boundary_ambiguous": ambiguous_timing_count > 0,
                }
            )
            segments.append(seg_record)
            seg_idx += 1

        logger.info(
            "Session Segmentation: Partitioned session %s into %d segments (strategy=%s).",
            session_id, len(segments), strategy
        )
        return segments
