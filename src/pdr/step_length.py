"""Calibrated heuristic step-length estimation using Weinberg acceleration dynamics (Task 21)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from src.core.logging import setup_logger
from src.pdr.config import PDRConfig

logger = setup_logger("PDRStepLength")


class StepLengthEstimator:
    """Estimates human step length from vertical acceleration bounce dynamics."""

    def __init__(self, config: Optional[PDRConfig] = None):
        self.config = config or PDRConfig()

    def process(self, steps_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates estimated step length (m) and uncertainty for each detected step."""
        if steps_df.empty:
            return steps_df.copy()

        df = steps_df.copy()
        model = self.config.step_length_model.lower()

        lengths = []
        uncertainties = []

        k = self.config.weinberg_k
        min_l = self.config.min_step_length_m
        max_l = self.config.max_step_length_m

        for i in range(len(df)):
            if model == "weinberg":
                accel_range = float(df["accel_range_mps2"].iloc[i])
                # Weinberg formulation: L = K * (a_max - a_min)^(1/4)
                raw_len = k * (max(0.01, accel_range) ** 0.25)
                # Clamp within physiological bounds
                length = float(np.clip(raw_len, min_l, max_l))
                # Step length uncertainty (~12% relative standard deviation)
                sigma_l = float(max(0.05, length * 0.12))
            else:
                # Fixed step length
                length = float(self.config.fixed_step_length_m)
                sigma_l = 0.15

            lengths.append(round(length, 3))
            uncertainties.append(round(sigma_l, 3))

        df["step_length_m"] = lengths
        df["step_length_uncertainty_m"] = uncertainties
        df["step_length_model"] = model.upper()

        logger.info(
            "Step Length Estimation: Mean length %.2f m (Min: %.2f m, Max: %.2f m, Model: %s).",
            float(np.mean(lengths)), float(np.min(lengths)), float(np.max(lengths)), model.upper()
        )
        return df
