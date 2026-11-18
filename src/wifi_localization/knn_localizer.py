"""Site-Specific Weighted k-NN Wi-Fi Fingerprint Localizer.

Computes distance-weighted nearest-neighbor spatial interpolation over site radio maps,
incorporating missing AP penalties, building/floor voting, and confidence estimation.
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
from src.wifi_localization.normalization import clean_ap_observation, normalize_bssid

logger = setup_logger("KnnLocalizer")


class KnnLocalizer:
    """Weighted k-Nearest Neighbors Wi-Fi fingerprint localizer."""

    def __init__(
        self,
        radio_map: RadioMap,
        k: int = 3,
        distance_metric: str = "euclidean",
        missing_penalty_rssi: Optional[float] = None,
        min_matched_aps: int = 2,
        confidence_distance_scale: float = 18.0,
        low_confidence_threshold: float = 0.35,
    ) -> None:
        """Initialize the k-NN localizer.

        Args:
            radio_map: Target site compiled RadioMap.
            k: Number of nearest neighbors for interpolation and voting.
            distance_metric: 'euclidean' or 'manhattan'.
            missing_penalty_rssi: Floor RSSI for unobserved APs (defaults to radio_map metadata).
            min_matched_aps: Minimum overlapping APs required before issuing warnings.
            confidence_distance_scale: Characteristic distance (dBm) for exponential confidence mapping.
            low_confidence_threshold: Confidence score below which is_low_confidence is flagged.
        """
        self.radio_map = radio_map
        self.k = max(1, k)
        self.distance_metric = distance_metric.lower()
        self.missing_penalty_rssi = (
            missing_penalty_rssi
            if missing_penalty_rssi is not None
            else radio_map.metadata.missing_penalty_rssi
        )
        self.min_matched_aps = max(1, min_matched_aps)
        self.confidence_distance_scale = max(1.0, confidence_distance_scale)
        self.low_confidence_threshold = low_confidence_threshold

    def localize(
        self,
        query_rssi: Dict[str, float],
        top_k: Optional[int] = None,
    ) -> LocalizationPrediction:
        """Estimate location for an input Wi-Fi query fingerprint.

        Args:
            query_rssi: Dictionary mapping BSSID -> observed RSSI (dBm).
            top_k: Override for number of nearest neighbors to consider.

        Returns:
            LocalizationPrediction containing estimated building, floor, (x, y), confidence, and candidate matches.
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
                algorithm="weighted_knn",
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
                algorithm="weighted_knn",
                matched_ap_count=0,
                total_query_aps=n_query_aps,
                candidate_matches=[],
                warnings=["Radio map contains no reference locations."],
                is_low_confidence=True,
            )

        # 2. Score query against every reference location profile
        candidates: List[CandidateMatch] = []
        max_matched_ap_count = 0

        for loc_id, profile in self.radio_map.profiles.items():
            dist, matched_count = self._compute_distance(normalized_query, profile.mean_rssi_vector)
            if matched_count > max_matched_ap_count:
                max_matched_ap_count = matched_count

            candidates.append(
                CandidateMatch(
                    location_id=loc_id,
                    building_id=profile.building_id,
                    floor_id=profile.floor_id,
                    room_id=profile.room_id,
                    x_m=profile.x_m,
                    y_m=profile.y_m,
                    distance_metric=round(dist, 3),
                    score=0.0,  # Computed below
                    matched_ap_count=matched_count,
                    total_query_aps=n_query_aps,
                )
            )

        # Sort candidates ascending by distance metric
        candidates.sort(key=lambda c: c.distance_metric)

        # 3. Handle edge case: zero overlapping APs anywhere
        if max_matched_ap_count == 0:
            warnings.append(
                f"Zero overlapping APs between query ({n_query_aps} APs) and site radio map."
            )

        if max_matched_ap_count < self.min_matched_aps:
            warnings.append(
                f"Low AP overlap: matched at most {max_matched_ap_count} APs "
                f"(recommended minimum: {self.min_matched_aps})."
            )

        # 4. Select top-k candidates and compute distance weights
        effective_k = min(k_val, len(candidates))
        top_candidates = candidates[:effective_k]

        weights = []
        for cand in top_candidates:
            # Inverse distance weight with epsilon
            w = 1.0 / (cand.distance_metric + 1e-3)
            weights.append(w)

        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights] if total_weight > 0 else [1.0 / effective_k] * effective_k

        for cand, nw in zip(top_candidates, norm_weights):
            cand.score = round(nw, 4)

        # 5. Weighted spatial coordinate interpolation
        pred_x = float(sum(nw * cand.x_m for nw, cand in zip(norm_weights, top_candidates)))
        pred_y = float(sum(nw * cand.y_m for nw, cand in zip(norm_weights, top_candidates)))

        # 6. Weighted voting for building and floor
        building_votes: Dict[str, float] = {}
        floor_votes: Dict[int, float] = {}
        room_votes: Dict[str, float] = {}

        for cand, nw in zip(top_candidates, norm_weights):
            building_votes[cand.building_id] = building_votes.get(cand.building_id, 0.0) + nw
            floor_votes[cand.floor_id] = floor_votes.get(cand.floor_id, 0.0) + nw
            if cand.room_id:
                room_votes[cand.room_id] = room_votes.get(cand.room_id, 0.0) + nw

        pred_building = max(building_votes.items(), key=lambda item: item[1])[0]
        pred_floor = max(floor_votes.items(), key=lambda item: item[1])[0]
        pred_room = max(room_votes.items(), key=lambda item: item[1])[0] if room_votes else None

        # 7. Confidence estimation
        # Based on distance to best candidate and overlap fraction
        best_cand = top_candidates[0]
        dist_factor = math.exp(-best_cand.distance_metric / self.confidence_distance_scale)
        overlap_factor = min(1.0, best_cand.matched_ap_count / float(max(1, self.min_matched_aps)))
        confidence = float(np.clip(dist_factor * overlap_factor, 0.0, 1.0))

        is_low_conf = confidence < self.low_confidence_threshold or best_cand.matched_ap_count == 0
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
            algorithm="weighted_knn",
            matched_ap_count=best_cand.matched_ap_count,
            total_query_aps=n_query_aps,
            candidate_matches=top_candidates,
            warnings=warnings,
            is_low_confidence=is_low_conf,
        )

    def _compute_distance(
        self,
        query: Dict[str, float],
        ref: Dict[str, float],
    ) -> Tuple[float, int]:
        """Compute RSSI distance metric between query and reference profile vector.

        Uses union-based distance with missing penalty RSSI for unobserved APs.
        """
        all_bssids = set(query.keys()) | set(ref.keys())
        if not all_bssids:
            return float("inf"), 0

        matched_count = len(set(query.keys()) & set(ref.keys()))
        diff_sq_sum = 0.0
        diff_abs_sum = 0.0

        for b in all_bssids:
            q_rssi = query.get(b, self.missing_penalty_rssi)
            r_rssi = ref.get(b, self.missing_penalty_rssi)
            diff = q_rssi - r_rssi
            if self.distance_metric == "manhattan":
                diff_abs_sum += abs(diff)
            else:
                diff_sq_sum += diff * diff

        if self.distance_metric == "manhattan":
            # Normalized mean absolute error
            dist = diff_abs_sum / len(all_bssids)
        else:
            # Root mean square error over union
            dist = math.sqrt(diff_sq_sum / len(all_bssids))

        return dist, matched_count
