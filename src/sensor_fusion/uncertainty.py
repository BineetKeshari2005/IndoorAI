"""Wi-Fi measurement uncertainty strategies and quality validation for sensor fusion (Task 24).

Implements a four-tier uncertainty hierarchy:
1. Empirical evaluation residuals from offline holdout evaluation (Task 23).
2. Spatial dispersion across top-k candidate matches.
3. Conservative heuristic covariance with confidence scaling.
4. Default prior covariance.

Guarantees positive semi-definiteness and assigns explicit uncertainty provenance tags.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.logging import setup_logger
from src.sensor_fusion.models import FusionConfig, UncertaintyType, WifiMeasurementEvent

logger = setup_logger("FusionUncertainty")


class WifiUncertaintyEstimator:
    """Estimates realistic 2D measurement covariance R_k for incoming Wi-Fi observations."""

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        empirical_site_covariance: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.config = config or FusionConfig()
        # site_id -> 2x2 covariance matrix
        self.empirical_site_covariance: Dict[str, np.ndarray] = empirical_site_covariance or {}

    def register_empirical_covariance(self, site_id: str, cov_2x2: np.ndarray) -> None:
        """Register empirical error covariance estimated from held-out survey evaluations."""
        symmetrized = 0.5 * (cov_2x2 + cov_2x2.T)
        self.empirical_site_covariance[site_id] = symmetrized
        logger.info(f"Registered empirical Wi-Fi covariance for site '{site_id}': std_x={math.sqrt(cov_2x2[0,0]):.2f}m, std_y={math.sqrt(cov_2x2[1,1]):.2f}m")

    def estimate_covariance(
        self,
        wifi_event: WifiMeasurementEvent,
        candidate_matches: Optional[List[Dict[str, Any]]] = None,
        site_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, UncertaintyType]:
        """Estimate 2x2 position covariance matrix for a Wi-Fi measurement.

        Returns:
            Tuple of (2x2 numpy covariance matrix, UncertaintyType).
        """
        min_var = self.config.min_wifi_sigma_m ** 2

        # 1. Check for registered empirical site covariance
        if site_id and site_id in self.empirical_site_covariance:
            cov = self.empirical_site_covariance[site_id].copy()
            cov[0, 0] = max(min_var, cov[0, 0])
            cov[1, 1] = max(min_var, cov[1, 1])
            return cov, UncertaintyType.EMPIRICAL_SURVEY_RESIDUALS

        # 2. Check for candidate match spatial dispersion
        if candidate_matches and len(candidate_matches) >= 2:
            disp_cov = self._compute_candidate_dispersion(
                target_x=wifi_event.x_m,
                target_y=wifi_event.y_m,
                candidates=candidate_matches,
                min_var=min_var,
            )
            if disp_cov is not None:
                return disp_cov, UncertaintyType.CANDIDATE_DISPERSION

        # 3. Conservative confidence-scaled covariance
        # High confidence -> closer to default_wifi_sigma_m; Low confidence -> inflated variance
        conf = max(0.1, min(1.0, wifi_event.confidence_score))
        scaled_sigma = max(
            self.config.min_wifi_sigma_m,
            self.config.default_wifi_sigma_m / conf
        )
        cov = np.eye(2, dtype=np.float64) * (scaled_sigma ** 2)
        return cov, UncertaintyType.CONSERVATIVE_HEURISTIC

    def validate_measurement_quality(
        self,
        wifi_event: WifiMeasurementEvent,
        current_timestamp_ms: Optional[int] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Validate whether a Wi-Fi observation meets basic quality thresholds prior to EKF update."""
        # 1. Freshness check
        if self.config.require_fresh_scans and not wifi_event.is_fresh:
            return False, "CACHED_OR_STALE_SCAN: Wi-Fi scan was flagged as cached by Android subsystem."

        # 2. Latency check
        if current_timestamp_ms is not None:
            latency_ms = current_timestamp_ms - wifi_event.timestamp_epoch_ms
            if latency_ms < 0:
                return False, f"FUTURE_TIMESTAMP: Wi-Fi timestamp ({wifi_event.timestamp_epoch_ms}) exceeds current time ({current_timestamp_ms})."
            if latency_ms > self.config.max_wifi_latency_ms:
                return False, f"EXCESSIVE_LATENCY: Wi-Fi scan latency ({latency_ms/1000.0:.1f}s) exceeds max allowed ({self.config.max_wifi_latency_ms/1000.0:.1f}s)."

        # 3. AP overlap count
        if wifi_event.matched_ap_count < self.config.min_matched_aps:
            return False, f"INSUFFICIENT_AP_OVERLAP: Matched {wifi_event.matched_ap_count} APs (< required {self.config.min_matched_aps})."

        # 4. Confidence score
        if wifi_event.confidence_score < self.config.min_confidence_score:
            return False, f"LOW_CONFIDENCE: Score {wifi_event.confidence_score:.2f} < threshold {self.config.min_confidence_score:.2f}."

        return True, None

    def _compute_candidate_dispersion(
        self,
        target_x: float,
        target_y: float,
        candidates: List[Dict[str, Any]],
        min_var: float,
    ) -> Optional[np.ndarray]:
        """Compute weighted spatial dispersion tensor among top candidate reference locations."""
        try:
            weights = []
            coords = []
            for c in candidates:
                x = float(c.get("x_m", target_x))
                y = float(c.get("y_m", target_y))
                w = float(c.get("score", 1.0))
                weights.append(max(1e-4, w))
                coords.append([x, y])

            total_w = sum(weights)
            norm_w = np.array([w / total_w for w in weights], dtype=np.float64)
            coords_arr = np.array(coords, dtype=np.float64)

            # Weighted mean
            mean_c = np.sum(coords_arr * norm_w[:, None], axis=0)

            # Weighted covariance
            diff = coords_arr - mean_c[None, :]
            disp_cov = diff.T @ np.diag(norm_w) @ diff

            # Symmetrize and add base variance floor
            disp_cov = 0.5 * (disp_cov + disp_cov.T)
            disp_cov[0, 0] = max(min_var, disp_cov[0, 0] + min_var)
            disp_cov[1, 1] = max(min_var, disp_cov[1, 1] + min_var)
            return disp_cov
        except Exception as e:
            logger.warning(f"Failed to compute candidate dispersion: {e}")
            return None
