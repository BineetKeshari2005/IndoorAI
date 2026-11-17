"""Heuristic and signal-processing step detection with refractory duplicate rejection (Task 21)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from src.core.logging import setup_logger
from src.pdr.config import PDRConfig
from src.pdr.filtering import butterworth_bandpass, butterworth_lowpass

logger = setup_logger("PDRStepDetection")


@dataclass
class StepDetectionStats:
    """Summary metrics of step detection algorithm."""

    total_candidates_detected: int = 0
    total_steps_accepted: int = 0
    total_rejected_prominence: int = 0
    total_rejected_refractory: int = 0
    mean_cadence_spm: float = 0.0
    mean_step_interval_ms: float = 0.0


class StepDetector:
    """Detects human stepping events from vertical/magnitude acceleration signals."""

    def __init__(self, config: Optional[PDRConfig] = None):
        self.config = config or PDRConfig()

    def process(
        self,
        gravity_df: pd.DataFrame,
        session_id: str = "UNKNOWN"
    ) -> Tuple[pd.DataFrame, StepDetectionStats]:
        """Detects steps, enforces refractory periods, and computes step intervals."""
        stats = StepDetectionStats()
        if gravity_df.empty or len(gravity_df) < 5:
            logger.warning("Insufficient acceleration data for step detection.")
            empty_df = pd.DataFrame(columns=[
                "step_id", "session_id", "timestamp_ms", "peak_accel_mps2",
                "valley_accel_mps2", "accel_range_mps2", "step_interval_ms",
                "step_frequency_hz", "confidence", "validation_status", "source_obs_id"
            ])
            return empty_df, stats

        df = gravity_df.sort_values("normalized_epoch_ms").reset_index(drop=True).copy()
        n = len(df)
        t_ms = df["normalized_epoch_ms"].values

        # 1. Select signal for step detection
        if self.config.step_detection_signal == "vertical" and "vertical_accel" in df.columns:
            raw_sig = df["vertical_accel"].values
        elif "magnitude" in df.columns:
            raw_sig = df["magnitude"].values - self.config.nominal_gravity_mps2
        else:
            raw_sig = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2) - self.config.nominal_gravity_mps2

        # 2. Estimate sampling rate
        dt_vals = np.diff(t_ms)
        valid_dt = dt_vals[(dt_vals > 0) & (dt_vals < self.config.sensor_gap_threshold_ms)]
        fs_hz = float(1000.0 / np.median(valid_dt)) if len(valid_dt) > 0 else 50.0

        # 3. Apply Band-pass filter (0.5 to 3.5 Hz)
        filtered_sig = butterworth_bandpass(
            raw_sig,
            low_cutoff_hz=self.config.accel_band_pass_low_hz,
            high_cutoff_hz=self.config.accel_band_pass_high_hz,
            fs_hz=fs_hz,
            order=2
        )

        # 4. Find candidate peaks with prominence
        # Use a short initial window (e.g. 150ms) so our algorithm explicitly enforces refractory period
        initial_min_dist = max(2, int(0.15 * fs_hz))
        prominence_thresh = self.config.min_peak_prominence_mps2
        peak_thresh = self.config.peak_threshold_mps2

        # Use scipy.signal.find_peaks
        peaks, properties = signal.find_peaks(
            filtered_sig,
            height=peak_thresh * 0.5,
            prominence=prominence_thresh,
            distance=initial_min_dist
        )
        stats.total_candidates_detected = len(peaks)

        if len(peaks) == 0:
            logger.info("No walking steps detected in session %s.", session_id)
            empty_df = pd.DataFrame(columns=[
                "step_id", "session_id", "timestamp_ms", "peak_accel_mps2",
                "valley_accel_mps2", "accel_range_mps2", "step_interval_ms",
                "step_frequency_hz", "confidence", "validation_status", "source_obs_id"
            ])
            return empty_df, stats

        # 5. Enforce Refractory period and filter candidate steps
        min_interval_ms = int(self.config.min_step_interval_s * 1000)
        accepted_peaks: List[int] = []

        for p in peaks:
            if not accepted_peaks:
                accepted_peaks.append(p)
                continue

            last_p = accepted_peaks[-1]
            dt_step = t_ms[p] - t_ms[last_p]

            if dt_step >= min_interval_ms:
                accepted_peaks.append(p)
            else:
                stats.total_rejected_refractory += 1
                # If current peak is significantly larger than previous, replace it
                if filtered_sig[p] > filtered_sig[last_p] * 1.2:
                    accepted_peaks[-1] = p

        stats.total_steps_accepted = len(accepted_peaks)

        # 6. Extract step features (peak, valley, interval, frequency, confidence)
        step_rows: List[Dict[str, Any]] = []
        intervals: List[float] = []

        for idx, p in enumerate(accepted_peaks):
            p_time = int(t_ms[p])
            p_val = float(filtered_sig[p])

            # Find local valley in pre-peak window (up to 0.5s prior)
            w_start = max(0, p - int(0.5 * fs_hz))
            valley_val = float(np.min(filtered_sig[w_start:p+1])) if p > w_start else 0.0
            accel_range = max(0.1, p_val - valley_val)

            if idx > 0:
                interval_ms = float(t_ms[p] - t_ms[accepted_peaks[idx - 1]])
                freq_hz = float(1000.0 / interval_ms) if interval_ms > 0 else 0.0
                intervals.append(interval_ms)
            else:
                interval_ms = 600.0  # Nominal default for first step
                freq_hz = 1.67

            # Confidence heuristic (prominence + cadence regularity)
            is_cadence_normal = (0.33 <= (interval_ms / 1000.0) <= 1.5)
            conf = 0.9 if (p_val >= peak_thresh and is_cadence_normal) else 0.65
            status = "VALID" if conf >= 0.7 else "LOW_CONFIDENCE"

            obs_id = str(df["observation_id"].iloc[p]) if "observation_id" in df else f"ACC-{p_time}"

            step_rows.append({
                "step_id": f"STEP-{idx+1:05d}",
                "session_id": session_id,
                "timestamp_ms": p_time,
                "peak_accel_mps2": round(p_val, 3),
                "valley_accel_mps2": round(valley_val, 3),
                "accel_range_mps2": round(accel_range, 3),
                "step_interval_ms": round(interval_ms, 1),
                "step_frequency_hz": round(freq_hz, 2),
                "confidence": conf,
                "validation_status": status,
                "source_obs_id": obs_id,
            })

        steps_df = pd.DataFrame(step_rows)

        if intervals:
            mean_int = float(np.mean(intervals))
            stats.mean_step_interval_ms = round(mean_int, 1)
            stats.mean_cadence_spm = round(60000.0 / mean_int, 1) if mean_int > 0 else 0.0

        logger.info(
            "Step Detection for %s: %d candidates -> %d accepted steps (Mean Cadence: %.1f SPM).",
            session_id, stats.total_candidates_detected, stats.total_steps_accepted, stats.mean_cadence_spm
        )
        return steps_df, stats
