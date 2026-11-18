"""Site-Specific Wi-Fi Radio Map Builder.

Aggregates multiple survey sweeps into clean, versioned location radio profiles
with signal distribution statistics (mean, median, standard deviation, visibility rate).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.logging import setup_logger
from src.wifi_localization.models import (
    ApObservation,
    ApStatistics,
    LocationRadioProfile,
    RadioMap,
    RadioMapMetadata,
    ReferenceLocation,
    SurveySweep,
)
from src.wifi_localization.normalization import (
    clean_ap_observation,
    deduplicate_sweep_observations,
    normalize_bssid,
)

logger = setup_logger("RadioMapBuilder")


class RadioMapBuilder:
    """Constructs site radio maps by aggregating multi-sweep Wi-Fi survey data."""

    def __init__(
        self,
        site_id: str,
        version: str = "v1",
        missing_penalty_rssi: float = -105.0,
        min_sweeps_per_location: int = 1,
        min_visibility_rate: float = 0.0,
        std_rssi_floor: float = 2.0,
    ) -> None:
        """Initialize the RadioMapBuilder.

        Args:
            site_id: Identifier of physical site (e.g., 'SITE-MAIN-CAMPUS').
            version: Radio map version tag (e.g., 'v1', 'v2').
            missing_penalty_rssi: Default RSSI assigned to unobserved APs (-105.0 dBm).
            min_sweeps_per_location: Minimum sweeps required to retain a reference location.
            min_visibility_rate: Minimum visibility fraction (0.0 to 1.0) to keep an AP in profile.
            std_rssi_floor: Minimum standard deviation floor for Gaussian likelihood modeling.
        """
        self.site_id = site_id
        self.version = version
        self.missing_penalty_rssi = missing_penalty_rssi
        self.min_sweeps_per_location = max(1, min_sweeps_per_location)
        self.min_visibility_rate = max(0.0, min(1.0, min_visibility_rate))
        self.std_rssi_floor = max(0.5, std_rssi_floor)

    def build_from_sweeps(
        self,
        sweeps: List[SurveySweep],
        locations: Dict[str, ReferenceLocation],
        description: Optional[str] = None,
    ) -> RadioMap:
        """Build radio map from a collection of survey sweeps and reference locations.

        Args:
            sweeps: List of survey sweeps collected during one or more survey sessions.
            locations: Mapping from location_id to ReferenceLocation metadata.
            description: Optional human-readable description for metadata.

        Returns:
            Compiled RadioMap instance.
        """
        logger.info(
            f"Building radio map '{self.version}' for site '{self.site_id}' from "
            f"{len(sweeps)} sweeps across {len(locations)} locations..."
        )

        # 1. Group sweeps by location_id
        sweeps_by_loc: Dict[str, List[SurveySweep]] = defaultdict(list)
        all_session_ids: Set[str] = set()

        for sweep in sweeps:
            if sweep.location_id not in locations:
                logger.warning(
                    f"Sweep {sweep.sweep_id} refers to unregistered location '{sweep.location_id}'; skipping."
                )
                continue
            sweeps_by_loc[sweep.location_id].append(sweep)
            if sweep.session_id:
                all_session_ids.add(sweep.session_id)

        # 2. Build location profiles
        profiles: Dict[str, LocationRadioProfile] = {}
        all_bssids_set: Set[str] = set()
        retained_locations: Dict[str, ReferenceLocation] = {}

        for loc_id, loc_sweeps in sweeps_by_loc.items():
            if len(loc_sweeps) < self.min_sweeps_per_location:
                logger.warning(
                    f"Location '{loc_id}' has {len(loc_sweeps)} sweeps "
                    f"(< {self.min_sweeps_per_location}); skipping."
                )
                continue

            ref_loc = locations[loc_id]
            retained_locations[loc_id] = ref_loc
            n_sweeps = len(loc_sweeps)

            # Accumulate observations per BSSID
            # bssid -> list of rssi values
            rssi_by_bssid: Dict[str, List[float]] = defaultdict(list)
            meta_by_bssid: Dict[str, Dict[str, Any]] = {}

            for sweep in loc_sweeps:
                deduped = deduplicate_sweep_observations(sweep.observations)
                for obs in deduped:
                    rssi_by_bssid[obs.bssid].append(obs.rssi_dbm)
                    if obs.bssid not in meta_by_bssid:
                        meta_by_bssid[obs.bssid] = {
                            "ssid": obs.ssid,
                            "frequency_mhz": obs.frequency_mhz,
                            "channel_band": obs.channel_band,
                        }

            # Compute statistics for each BSSID
            ap_stats_dict: Dict[str, ApStatistics] = {}
            mean_rssi_vec: Dict[str, float] = {}

            for bssid, rssi_list in rssi_by_bssid.items():
                k = len(rssi_list)
                visibility = k / float(n_sweeps)
                if visibility < self.min_visibility_rate:
                    continue

                arr = np.array(rssi_list, dtype=np.float64)
                mean_val = float(np.mean(arr))
                median_val = float(np.median(arr))
                # Sample std with Bessel correction if k > 1
                if k > 1:
                    raw_std = float(np.std(arr, ddof=1))
                else:
                    raw_std = self.std_rssi_floor
                std_val = max(raw_std, self.std_rssi_floor)
                min_val = float(np.min(arr))
                max_val = float(np.max(arr))

                b_meta = meta_by_bssid.get(bssid, {})
                stat = ApStatistics(
                    bssid=bssid,
                    observation_count=k,
                    total_sweeps_at_location=n_sweeps,
                    visibility_rate=round(visibility, 4),
                    mean_rssi=round(mean_val, 2),
                    median_rssi=round(median_val, 2),
                    std_rssi=round(std_val, 2),
                    min_rssi=round(min_val, 2),
                    max_rssi=round(max_val, 2),
                    ssid=b_meta.get("ssid"),
                    frequency_mhz=b_meta.get("frequency_mhz"),
                    channel_band=b_meta.get("channel_band"),
                )
                ap_stats_dict[bssid] = stat
                mean_rssi_vec[bssid] = round(mean_val, 2)
                all_bssids_set.add(bssid)

            profile = LocationRadioProfile(
                location_id=loc_id,
                site_id=self.site_id,
                building_id=ref_loc.building_id,
                floor_id=ref_loc.floor_id,
                room_id=ref_loc.room_id,
                x_m=ref_loc.x_m,
                y_m=ref_loc.y_m,
                total_sweeps=n_sweeps,
                ap_stats=ap_stats_dict,
                mean_rssi_vector=mean_rssi_vec,
            )
            profiles[loc_id] = profile

        # Build metadata
        now_iso = datetime.now(timezone.utc).isoformat()
        metadata = RadioMapMetadata(
            site_id=self.site_id,
            version=self.version,
            created_at_iso=now_iso,
            total_locations=len(profiles),
            total_sweeps=sum(p.total_sweeps for p in profiles.values()),
            unique_bssids=len(all_bssids_set),
            missing_penalty_rssi=self.missing_penalty_rssi,
            session_ids=sorted(list(all_session_ids)),
            description=description or f"Site radio map compiled at {now_iso}",
        )

        logger.info(
            f"Successfully compiled radio map '{self.version}' for site '{self.site_id}': "
            f"{len(profiles)} locations, {len(all_bssids_set)} unique BSSIDs, "
            f"{metadata.total_sweeps} total sweeps."
        )

        return RadioMap(
            metadata=metadata,
            locations=retained_locations,
            profiles=profiles,
            all_bssids=sorted(list(all_bssids_set)),
        )

    def group_raw_records_into_sweeps(
        self,
        records: List[Dict[str, Any]],
        sweep_gap_ms: int = 500,
    ) -> List[SurveySweep]:
        """Group flat raw Wi-Fi scan records into distinct temporal survey sweeps.

        Records should contain:
        - 'bssid' (str)
        - 'rssi_dbm' or 'rssi' (float)
        - 'timestamp_epoch_ms' or 'normalized_epoch_ms' (int)
        - 'location_id' (str)
        - 'session_id' (str)

        Args:
            records: List of Wi-Fi scan observation dictionaries.
            sweep_gap_ms: Timestamp gap threshold separating two scan sweeps.

        Returns:
            List of structured SurveySweep objects.
        """
        if not records:
            return []

        # Sort records by (session_id, location_id, timestamp)
        def _get_ts(r: Dict[str, Any]) -> int:
            return int(r.get("timestamp_epoch_ms") or r.get("normalized_epoch_ms") or 0)

        sorted_records = sorted(
            records,
            key=lambda r: (str(r.get("session_id", "")), str(r.get("location_id", "")), _get_ts(r)),
        )

        sweeps: List[SurveySweep] = []
        current_sweep_obs: List[ApObservation] = []
        current_key: Optional[Tuple[str, str]] = None
        last_ts: int = 0
        sweep_idx: int = 1

        for r in sorted_records:
            cleaned_obs = clean_ap_observation(r)
            if not cleaned_obs:
                continue

            session_id = str(r.get("session_id", "DEFAULT_SESSION"))
            loc_id = str(r.get("location_id", "UNKNOWN_LOC"))
            ts = _get_ts(r)
            key = (session_id, loc_id)

            # Check if this record belongs to a new sweep
            is_new_sweep = (
                current_key is None
                or key != current_key
                or (ts - last_ts) > sweep_gap_ms
            )

            if is_new_sweep and current_sweep_obs:
                sweeps.append(
                    SurveySweep(
                        sweep_id=f"SWEEP-{current_key[0]}-{sweep_idx:04d}",
                        session_id=current_key[0],
                        location_id=current_key[1],
                        timestamp_epoch_ms=last_ts,
                        observations=current_sweep_obs,
                    )
                )
                sweep_idx += 1
                current_sweep_obs = []

            current_key = key
            last_ts = ts
            current_sweep_obs.append(cleaned_obs)

        if current_sweep_obs and current_key:
            sweeps.append(
                SurveySweep(
                    sweep_id=f"SWEEP-{current_key[0]}-{sweep_idx:04d}",
                    session_id=current_key[0],
                    location_id=current_key[1],
                    timestamp_epoch_ms=last_ts,
                    observations=current_sweep_obs,
                )
            )

        return sweeps
