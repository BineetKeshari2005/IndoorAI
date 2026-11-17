"""Digital signal filtering utilities for IMU kinematics and movement analysis (Task 21)."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from scipy import signal

from src.core.logging import setup_logger

logger = setup_logger("PDRFiltering")


def butterworth_lowpass(
    data: np.ndarray,
    cutoff_hz: float,
    fs_hz: float,
    order: int = 2
) -> np.ndarray:
    """Applies a zero-phase forward-backward Butterworth low-pass filter."""
    if len(data) < order * 3 or fs_hz <= 0:
        return data.copy()

    nyq = 0.5 * fs_hz
    normal_cutoff = min(cutoff_hz / nyq, 0.99)
    if normal_cutoff <= 0:
        return data.copy()

    try:
        b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
        padlen = min(len(data) - 1, 3 * max(len(a), len(b)))
        if padlen <= 0:
            return data.copy()
        filtered = signal.filtfilt(b, a, data, padlen=padlen)
        return filtered
    except Exception as e:
        logger.warning("Butterworth lowpass failed; returning raw signal: %s", e)
        return data.copy()


def butterworth_bandpass(
    data: np.ndarray,
    low_cutoff_hz: float,
    high_cutoff_hz: float,
    fs_hz: float,
    order: int = 2
) -> np.ndarray:
    """Applies a zero-phase Butterworth band-pass filter for step detection."""
    if len(data) < order * 3 or fs_hz <= 0:
        return data.copy()

    nyq = 0.5 * fs_hz
    low = max(0.01, low_cutoff_hz / nyq)
    high = min(0.99, high_cutoff_hz / nyq)
    if low >= high:
        return data.copy()

    try:
        b, a = signal.butter(order, [low, high], btype="band", analog=False)
        padlen = min(len(data) - 1, 3 * max(len(a), len(b)))
        if padlen <= 0:
            return data.copy()
        filtered = signal.filtfilt(b, a, data, padlen=padlen)
        return filtered
    except Exception as e:
        logger.warning("Butterworth bandpass failed; returning raw signal: %s", e)
        return data.copy()


def exponential_smoothing_online(
    values: np.ndarray,
    timestamps_ms: np.ndarray,
    cutoff_hz: float,
    gap_threshold_ms: float = 500.0,
    initial_value: Optional[float] = None
) -> np.ndarray:
    """Computes variable-sampling rate IIR exponential low-pass filter, resetting across gaps."""
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.float64)

    tau = 1.0 / (2.0 * math.pi * cutoff_hz) if cutoff_hz > 0 else 0.318
    out = np.zeros(n, dtype=np.float64)

    current_val = float(values[0]) if initial_value is None else float(initial_value)
    out[0] = current_val

    for i in range(1, n):
        dt = (timestamps_ms[i] - timestamps_ms[i - 1]) / 1000.0  # seconds
        # Reset if sampling gap
        if (timestamps_ms[i] - timestamps_ms[i - 1]) > gap_threshold_ms or dt <= 0:
            current_val = float(values[i])
        else:
            alpha = tau / (tau + dt)
            current_val = alpha * current_val + (1.0 - alpha) * float(values[i])
        out[i] = current_val

    return out
