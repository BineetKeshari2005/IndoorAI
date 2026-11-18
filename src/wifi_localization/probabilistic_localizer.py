"""Probabilistic Gaussian Wi-Fi Fingerprint Localizer.

Computes log-likelihood matching under per-AP Gaussian signal distributions,
incorporating standard deviation floors, missing AP penalties, softmax posterior
weighting, and spatial interpolation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.logging import setup_logger
from src.wifi_localization.models import (
    CandidateMatch,
    LocalizationPrediction,
    LocationRadioProfile,
    RadioMap,
)
from src.wifi_localization.normalization import normalize_bssid

logger = setup_logger("ProbabilisticLocalizer")


class ProbabilisticLocalizer:
    """Probabilistic Gaussian log-likelihood Wi-Fi fingerprint localizer."""

    def __init__(
        self,
        radio_map: RadioMap,
        k: int = 3,
        missing_penalty_rssi: Optional[float] = None,
        min_matched_aps: int = 2,
        std_rssi_floor: float = 2.5,
        missing_ap_log_penalty: float = -15.0,
        low_confidence_threshold: float = 0.35,
    ) -> None:
        """Initialize the probabilistic localizer.

        Args:
            radio_map: Compiled site RadioMap.
            k: Top candidates to consider for spatial coordinate interpolation.
            missing_penalty_rssi: Floor RSSI for unobserved APs.
            min_matched_aps: Minimum overlapping APs threshold.
            std_rssi_floor: Minimum standard deviation floor for Gaussian likelihood.
            missing_ap_log_penalty: Constant log-likelihood penalty for missing/unobserved APs.
            low_confidence_threshold: Threshold below which is_low_confidence is flagged.
        """
        self.radio_map = radio_map
        self.k = max(1, k)
        self.missing_penalty_rssi = (
            missing_penalty_rssi
            if missing_penalty_rssi is not None
            else radio_map.metadata.missing_penalty_rssi
        )
        self.min_matched_aps = max(1, min_matched_aps)
        self.std_rssi_floor = max(0.5, std_rssi_floor)
        self.missing_ap_log_penalty = missing_ap_log_penalty
        self.low_confidence_threshold = low_confidence_threshold

    def localize(
        self,
        query_rssi: Dict[str, float],
        top_k: Optional[int] = None,
    ) -> LocalizationPrediction:
        """Estimate location using Gaussian log-likelihood matching.

        Note: Relative softmax scores represent uncalibrated posterior likelihood weights,
        not absolute calibrated probabilities.
        """
        k_val = top_k or self.k
        warnings: List[str] = []

        # 1. Clean and normalize query BSSIDs
        normalized_query: Dict[str, float] = {}
        for b_raw, rssi_val in query_rssi.items():
            norm_b = normalize_bssid(b_raw)
            if norm_b and rssi_val is not None:
                try:
                    normalized_query[norm_b] = float(rssi_val)
                except (ValueError, TypeError):
                    continue

        n_query_aps = len(normalized_query)
        if n_query_aps == 0:
            return LocalizationPrediction(
                predicted_building_id="UNKNOWN",
                predicted_floor_id=0,
                predicted_x_m=0.0,
                predicted_y_m=0.0,
                confidence=0.0,
                algorithm="probabilistic",
                matched_ap_count=0,
                total_query_aps=0,
                candidate_matches=[],
                warnings=["Query contains no valid Wi-Fi observations."],
                is_low_confidence=True,
            )

        if not self.radio_map.profiles:
            return LocalizationPrediction(
                predicted_building_id="UNKNOWN",
                predicted_floor_id=0,
                predicted_x_m=0.0,
                predicted_y_m=0.0,
                confidence=0.0,
                algorithm="probabilistic",
                matched_ap_count=0,
                total_query_aps=n_query_aps,
                candidate_matches=[],
                warnings=["Radio map contains no reference locations."],
                is_low_confidence=True,
            )

        # 2. Compute log-likelihood for each reference location
        candidates_raw: List[Tuple[str, float, int]] = []
        max_matched_ap_count = 0

        for loc_id, profile in self.radio_map.profiles.items():
            log_lik, matched_count = self._compute_log_likelihood(normalized_query, profile)
            if matched_count > max_matched_ap_count:
                max_matched_ap_count = matched_count
            candidates_raw.append((loc_id, log_lik, matched_count))

        # Check overlap
        if max_matched_ap_count == 0:
            warnings.append(
                f"Zero overlapping APs between query ({n_query_aps} APs) and site radio map."
            )
        if max_matched_ap_count < self.min_matched_aps:
            warnings.append(
                f"Low AP overlap: matched at most {max_matched_ap_count} APs "
                f"(recommended minimum: {self.min_matched_aps})."
            )

        # 3. Sort candidates descending by log-likelihood
        candidates_raw.sort(key=lambda item: item[1], reverse=True)

        # 4. Softmax normalization over top candidates for interpolation weights
        effective_k = min(k_val, len(candidates_raw))
        top_k_raw = candidates_raw[:effective_k]

        max_log_lik = top_k_raw[0][1]
        exp_vals = [math.exp(max(-50.0, item[1] - max_log_lik)) for item in top_k_raw]
        sum_exp = sum(exp_vals)
        softmax_weights = [v / sum_exp for v in exp_vals] if sum_exp > 0 else [1.0 / effective_k] * effective_k

        candidate_matches: List[CandidateMatch] = []
        for (loc_id, log_lik, matched_count), weight in zip(top_k_raw, softmax_weights):
            profile = self.radio_map.profiles[loc_id]
            candidate_matches.append(
                CandidateMatch(
                    location_id=loc_id,
                    building_id=profile.building_id,
                    floor_id=profile.floor_id,
                    room_id=profile.room_id,
                    x_m=profile.x_m,
                    y_m=profile.y_m,
                    distance_metric=round(-log_lik, 2),  # Negative log-likelihood as distance
                    score=round(weight, 4),
                    matched_ap_count=matched_count,
                    total_query_aps=n_query_aps,
                )
            )

        # 5. Spatial coordinate interpolation
        pred_x = float(sum(w * cand.x_m for w, cand in zip(softmax_weights, candidate_matches)))
        pred_y = float(sum(w * cand.y_m for w, cand in zip(softmax_weights, candidate_matches)))

        # 6. Building and floor estimation
        building_votes: Dict[str, float] = {}
        floor_votes: Dict[int, float] = {}
        room_votes: Dict[str, float] = {}

        for cand, w in zip(candidate_matches, softmax_weights):
            building_votes[cand.building_id] = building_votes.get(cand.building_id, 0.0) + w
            floor_votes[cand.floor_id] = floor_votes.get(cand.floor_id, 0.0) + w
            if cand.room_id:
                room_votes[cand.room_id] = room_votes.get(cand.room_id, 0.0) + w

        pred_building = max(building_votes.items(), key=lambda item: item[1])[0]
        pred_floor = max(floor_votes.items(), key=lambda item: item[1])[0]
        pred_room = max(room_votes.items(), key=lambda item: item[1])[0] if room_votes else None

        # 7. Confidence estimation
        # Top candidate weight ratio and overlap ratio
        top_cand = candidate_matches[0]
        overlap_factor = min(1.0, top_cand.matched_ap_count / float(max(1, self.min_matched_aps)))
        # Confidence reflects how dominant top candidate is and its overlap
        confidence = float(np.clip(top_cand.score * overlap_factor, 0.0, 1.0))

        is_low_conf = confidence < self.low_confidence_threshold or top_cand.matched_ap_count == 0
        if is_low_conf:
            warnings.append(
                f"Confidence {confidence:.2f} is below threshold {self.low_confidence_threshold:.2f}."
            )

        return LocalizationPrediction(
            predicted_building_id=pred_building,
            predicted_floor_id=pred_floor,
            predicted_x_m=round(pred_x, 2),
            predicted_y_m=round(pred_y, 2),
            predicted_room_id=pred_room,
            confidence=round(confidence, 3),
            algorithm="probabilistic",
            matched_ap_count=top_cand.matched_ap_count,
            total_query_aps=n_query_aps,
            candidate_matches=candidate_matches,
            warnings=warnings,
            is_low_confidence=is_low_conf,
        )

    def _compute_log_likelihood(
        self,
        query: Dict[str, float],
        profile: LocationRadioProfile,
    ) -> Tuple[float, int]:
        """Compute Gaussian log-likelihood of query observation given location profile."""
        log_lik_total = 0.0
        matched_count = 0

        query_bssids = set(query.keys())
        profile_bssids = set(profile.ap_stats.keys())
        common_bssids = query_bssids & profile_bssids

        for b in common_bssids:
            stat = profile.ap_stats[b]
            z = query[b]
            mu = stat.mean_rssi
            sigma = max(stat.std_rssi, self.std_rssi_floor)

            # Gaussian log-likelihood: -0.5*ln(2*pi*sigma^2) - (z - mu)^2 / (2*sigma^2)
            log_norm = -0.5 * math.log(2.0 * math.pi * sigma * sigma)
            log_exp = -((z - mu) ** 2) / (2.0 * sigma * sigma)
            log_lik_total += (log_norm + log_exp)
            matched_count += 1

        # Penalty for query APs absent in profile
        query_only = query_bssids - profile_bssids
        log_lik_total += len(query_only) * self.missing_ap_log_penalty

        # Penalty for highly visible profile APs (>50% visibility) absent in query
        for b in profile_bssids - query_bssids:
            stat = profile.ap_stats[b]
            if stat.visibility_rate >= 0.5:
                # Penalty proportional to visibility
                log_lik_total += stat.visibility_rate * (self.missing_ap_log_penalty * 0.5)

        return log_lik_total, matched_count
