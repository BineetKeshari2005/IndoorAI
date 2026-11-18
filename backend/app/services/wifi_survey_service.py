"""Service layer for site-specific Wi-Fi survey radio maps, localization, and evaluation (Task 23)."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.core.dataset_config import DatasetConfig, default_dataset_config
from backend.app.schemas.wifi_survey import (
    RadioMapBuildRequest,
    ReferenceLocationCreate,
    WifiEvaluationRequest,
    WifiLocalizationQueryRequest,
)
from backend.app.services.dataset_registry import DatasetRegistry
from src.core.logging import setup_logger
from src.wifi_localization.evaluator import WifiLocalizationEvaluator
from src.wifi_localization.knn_localizer import KnnLocalizer
from src.wifi_localization.models import (
    EvaluationMetrics,
    LocalizationPrediction,
    RadioMap,
    ReferenceLocation,
    SurveySweep,
)
from src.wifi_localization.probabilistic_localizer import ProbabilisticLocalizer
from src.wifi_localization.radio_map_builder import RadioMapBuilder
from src.wifi_localization.storage import (
    get_radio_map_dir,
    list_radio_maps,
    load_radio_map,
    load_reference_locations,
    save_radio_map,
    save_reference_locations,
)

logger = setup_logger("WifiSurveyService")


class WifiSurveyService:
    """Orchestrates reference location management, radio map compilation, online localization, and evaluation."""

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        registry: Optional[DatasetRegistry] = None,
        storage_base_dir: Optional[Path] = None,
    ) -> None:
        self.config = config or default_dataset_config
        self.registry = registry or DatasetRegistry(self.config)
        self.storage_base_dir = storage_base_dir or Path("data/radio_maps")
        self.storage_base_dir.mkdir(parents=True, exist_ok=True)
        self.processed_base_dir = self.config.base_dir.parent / "processed" / "multimodal"

    # -------------------------------------------------------------------------
    # 1. Reference Location Management
    # -------------------------------------------------------------------------

    def register_location(
        self, site_id: str, loc_data: ReferenceLocationCreate
    ) -> ReferenceLocation:
        """Register or update a reference location for a physical site."""
        current_locations = load_reference_locations(site_id, base_dir=self.storage_base_dir)
        now_iso = datetime.now(timezone.utc).isoformat()

        ref_loc = ReferenceLocation(
            location_id=loc_data.location_id,
            site_id=site_id,
            building_id=loc_data.building_id,
            floor_id=loc_data.floor_id,
            x_m=loc_data.x_m,
            y_m=loc_data.y_m,
            room_id=loc_data.room_id,
            description=loc_data.description,
            is_verified=loc_data.is_verified,
            created_at_iso=now_iso,
            metadata=loc_data.metadata,
        )
        current_locations[ref_loc.location_id] = ref_loc
        save_reference_locations(site_id, current_locations, base_dir=self.storage_base_dir)
        return ref_loc

    def get_locations(self, site_id: str) -> List[ReferenceLocation]:
        """Get all registered reference locations for a site."""
        loc_dict = load_reference_locations(site_id, base_dir=self.storage_base_dir)
        return list(loc_dict.values())

    def get_location(self, site_id: str, location_id: str) -> Optional[ReferenceLocation]:
        """Get a specific reference location by ID."""
        loc_dict = load_reference_locations(site_id, base_dir=self.storage_base_dir)
        return loc_dict.get(location_id)

    # -------------------------------------------------------------------------
    # 2. Radio Map Compilation from Dataset
    # -------------------------------------------------------------------------

    def build_radio_map(
        self,
        dataset_id: str,
        request: RadioMapBuildRequest,
    ) -> RadioMap:
        """Compile a site radio map from survey dataset observations."""
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_id}' not found in registry.")

        site_id = request.site_id
        current_locations = load_reference_locations(site_id, base_dir=self.storage_base_dir)

        # 1. Read Wi-Fi observations and annotations
        wifi_records, annotations = self._load_survey_records(dataset_id)
        if not wifi_records:
            raise ValueError(f"No Wi-Fi observation records found for dataset '{dataset_id}'.")

        # 2. Auto-register reference locations from annotations if requested
        if request.auto_register_locations:
            current_locations = self._auto_register_locations_from_survey(
                site_id=site_id,
                wifi_records=wifi_records,
                annotations=annotations,
                existing_locations=current_locations,
            )

        if not current_locations:
            raise ValueError(
                f"No reference locations registered for site '{site_id}', and none could be deduced."
            )

        # 3. Associate observations with location IDs
        attributed_records = self._attribute_location_ids(wifi_records, annotations, current_locations)

        # 4. Group observations into sweeps and compile radio map
        builder = RadioMapBuilder(
            site_id=site_id,
            version=request.version,
            missing_penalty_rssi=request.missing_penalty_rssi,
            min_sweeps_per_location=request.min_sweeps_per_location,
            min_visibility_rate=request.min_visibility_rate,
            std_rssi_floor=request.std_rssi_floor,
        )

        sweeps = builder.group_raw_records_into_sweeps(attributed_records)
        if not sweeps:
            raise ValueError("Could not extract any valid Wi-Fi sweeps from survey observations.")

        radio_map = builder.build_from_sweeps(
            sweeps=sweeps,
            locations=current_locations,
            description=request.description or f"Radio map built from dataset {dataset_id}",
        )

        # 5. Persist radio map
        save_radio_map(radio_map, base_dir=self.storage_base_dir)
        return radio_map

    # -------------------------------------------------------------------------
    # 3. Online Fingerprint Localization
    # -------------------------------------------------------------------------

    def localize(
        self,
        request: WifiLocalizationQueryRequest,
    ) -> LocalizationPrediction:
        """Estimate indoor location from a live or test query fingerprint."""
        radio_map = load_radio_map(
            site_id=request.site_id,
            version=request.version,
            base_dir=self.storage_base_dir,
        )
        if not radio_map:
            raise ValueError(
                f"Radio map for site '{request.site_id}' version '{request.version}' not found on disk."
            )

        # Parse query observations into {bssid: rssi}
        query_rssi: Dict[str, float] = {}
        if isinstance(request.observations, dict):
            for k, v in request.observations.items():
                try:
                    query_rssi[str(k)] = float(v)
                except (ValueError, TypeError):
                    continue
        elif isinstance(request.observations, list):
            for item in request.observations:
                b = item.get("bssid")
                r = item.get("rssi_dbm") if item.get("rssi_dbm") is not None else item.get("rssi")
                if b and r is not None:
                    try:
                        query_rssi[str(b)] = float(r)
                    except (ValueError, TypeError):
                        continue

        algo = request.algorithm.lower()
        if algo == "probabilistic":
            localizer = ProbabilisticLocalizer(
                radio_map=radio_map,
                k=request.k,
            )
        else:
            localizer = KnnLocalizer(
                radio_map=radio_map,
                k=request.k,
                distance_metric=request.distance_metric,
            )

        return localizer.localize(query_rssi)

    # -------------------------------------------------------------------------
    # 4. Session-Partitioned Offline Evaluation
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        dataset_id: str,
        request: WifiEvaluationRequest,
    ) -> EvaluationMetrics:
        """Run session-partitioned offline evaluation on survey dataset."""
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_id}' not found in registry.")

        site_id = request.site_id
        current_locations = load_reference_locations(site_id, base_dir=self.storage_base_dir)
        wifi_records, annotations = self._load_survey_records(dataset_id)

        if not current_locations:
            current_locations = self._auto_register_locations_from_survey(
                site_id=site_id,
                wifi_records=wifi_records,
                annotations=annotations,
                existing_locations={},
            )

        attributed = self._attribute_location_ids(wifi_records, annotations, current_locations)
        builder = RadioMapBuilder(site_id=site_id)
        sweeps = builder.group_raw_records_into_sweeps(attributed)

        if not sweeps:
            raise ValueError("No sweeps available for evaluation.")

        # Identify all distinct session IDs
        distinct_sessions = sorted(list({s.session_id for s in sweeps if s.session_id}))
        if len(distinct_sessions) < 2 and (not request.train_session_ids or not request.test_session_ids):
            # If only 1 session, partition sweeps into virtual blocks to prevent stationary scan leakage
            logger.warning(
                "Only 1 survey session found. Partitioning sweeps sequentially into holdout partitions."
            )
            n_sweeps = len(sweeps)
            n_test = max(1, int(n_sweeps * request.test_ratio))
            n_train = n_sweeps - n_test

            for idx, sw in enumerate(sweeps):
                sw.session_id = f"{sw.session_id}-TRAIN" if idx < n_train else f"{sw.session_id}-TEST"

            train_sessions = [f"{distinct_sessions[0]}-TRAIN"]
            test_sessions = [f"{distinct_sessions[0]}-TEST"]
        else:
            if request.train_session_ids and request.test_session_ids:
                train_sessions = request.train_session_ids
                test_sessions = request.test_session_ids
            else:
                # Split distinct sessions
                split_idx = max(1, int(len(distinct_sessions) * (1.0 - request.test_ratio)))
                train_sessions = distinct_sessions[:split_idx]
                test_sessions = distinct_sessions[split_idx:]

        evaluator = WifiLocalizationEvaluator(
            algorithm=request.algorithm,
            k=request.k,
            distance_metric=request.distance_metric,
        )

        return evaluator.evaluate_session_split(
            sweeps=sweeps,
            locations=current_locations,
            train_session_ids=train_sessions,
            test_session_ids=test_sessions,
            site_id=site_id,
        )

    # -------------------------------------------------------------------------
    # 5. Radio Map Indexing & Retrieval
    # -------------------------------------------------------------------------

    def list_maps(self, site_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List metadata for all compiled radio maps."""
        return list_radio_maps(site_id=site_id, base_dir=self.storage_base_dir)

    def get_radio_map(self, site_id: str, version: str = "v1") -> Optional[RadioMap]:
        """Load full radio map for a site and version."""
        return load_radio_map(site_id=site_id, version=version, base_dir=self.storage_base_dir)

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _load_survey_records(
        self, dataset_id: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load raw or preprocessed Wi-Fi records and annotations from dataset storage."""
        wifi_records: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []

        extracted_dir = self.config.extracted_path / dataset_id
        processed_dir = self.processed_base_dir / dataset_id

        # 1. Check for Task 20 processed Wi-Fi parquet
        wifi_parquet = processed_dir / "wifi_cleaned.parquet"
        if wifi_parquet.is_file():
            try:
                df = pd.read_parquet(wifi_parquet)
                wifi_records = df.to_dict(orient="records")
                logger.info(f"Loaded {len(wifi_records)} Wi-Fi records from {wifi_parquet.name}")
            except Exception as e:
                logger.warning(f"Failed to read {wifi_parquet}: {e}")

        # Fallback to extracted wifi_observations.csv
        if not wifi_records:
            wifi_csv = extracted_dir / "wifi_observations.csv"
            if wifi_csv.is_file():
                try:
                    df = pd.read_csv(wifi_csv)
                    wifi_records = df.to_dict(orient="records")
                    logger.info(f"Loaded {len(wifi_records)} Wi-Fi records from {wifi_csv.name}")
                except Exception as e:
                    logger.warning(f"Failed to read {wifi_csv}: {e}")

        # 2. Check for annotations.csv
        ann_csv = extracted_dir / "annotations.csv"
        if ann_csv.is_file():
            try:
                df_ann = pd.read_csv(ann_csv)
                annotations = df_ann.to_dict(orient="records")
            except Exception as e:
                logger.warning(f"Failed to read {ann_csv}: {e}")

        return wifi_records, annotations

    def _auto_register_locations_from_survey(
        self,
        site_id: str,
        wifi_records: List[Dict[str, Any]],
        annotations: List[Dict[str, Any]],
        existing_locations: Dict[str, ReferenceLocation],
    ) -> Dict[str, ReferenceLocation]:
        """Infer and register reference locations from survey metadata if not pre-registered."""
        locations = dict(existing_locations)
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. From annotations
        for ann in annotations:
            loc_id = str(ann.get("label") or ann.get("annotation_id") or f"LOC-ANN-{ann.get('timestamp_epoch_ms')}")
            if loc_id not in locations:
                bldg = str(ann.get("building") or "BLDG-1")
                try:
                    floor = int(ann.get("floor", 0))
                except (ValueError, TypeError):
                    floor = 0
                x = float(ann.get("x_m", 0.0)) if ann.get("x_m") is not None else 0.0
                y = float(ann.get("y_m", 0.0)) if ann.get("y_m") is not None else 0.0

                locations[loc_id] = ReferenceLocation(
                    location_id=loc_id,
                    site_id=site_id,
                    building_id=bldg,
                    floor_id=floor,
                    room_id=str(ann.get("room")) if ann.get("room") else None,
                    x_m=x,
                    y_m=y,
                    description=str(ann.get("notes")) if ann.get("notes") else f"Inferred from annotation {loc_id}",
                    is_verified=True,
                    created_at_iso=now_iso,
                )

        # 2. If still empty, infer distinct (building, floor, room) combinations from Wi-Fi records
        if not locations:
            seen_combos = set()
            for r in wifi_records:
                bldg = str(r.get("building") or "BLDG-1")
                try:
                    flr = int(r.get("floor", 0))
                except (ValueError, TypeError):
                    flr = 0
                rm = str(r.get("room") or "")
                combo = (bldg, flr, rm)
                if combo not in seen_combos:
                    seen_combos.add(combo)
                    loc_id = f"LOC-{bldg}-F{flr}-{rm}" if rm else f"LOC-{bldg}-F{flr}"
                    # Assign a grid coordinate based on hash or index
                    idx = len(locations)
                    locations[loc_id] = ReferenceLocation(
                        location_id=loc_id,
                        site_id=site_id,
                        building_id=bldg,
                        floor_id=flr,
                        room_id=rm if rm else None,
                        x_m=float((idx % 5) * 10.0),
                        y_m=float((idx // 5) * 10.0),
                        description=f"Auto-generated location for {bldg} F{flr} {rm}",
                        is_verified=False,
                        created_at_iso=now_iso,
                    )

        if locations != existing_locations:
            save_reference_locations(site_id, locations, base_dir=self.storage_base_dir)

        return locations

    def _attribute_location_ids(
        self,
        wifi_records: List[Dict[str, Any]],
        annotations: List[Dict[str, Any]],
        locations: Dict[str, ReferenceLocation],
    ) -> List[Dict[str, Any]]:
        """Ensure every Wi-Fi record has a valid location_id matching a registered ReferenceLocation."""
        attributed = []
        loc_keys = list(locations.keys())
        first_loc_id = loc_keys[0] if loc_keys else "LOC-DEFAULT"

        # Build mapping from (building, floor, room) to location_id
        combo_to_loc: Dict[Tuple[str, int, Optional[str]], str] = {}
        for loc_id, loc in locations.items():
            combo_to_loc[(loc.building_id, loc.floor_id, loc.room_id)] = loc_id
            if (loc.building_id, loc.floor_id, None) not in combo_to_loc:
                combo_to_loc[(loc.building_id, loc.floor_id, None)] = loc_id

        for r in wifi_records:
            rec = dict(r)
            loc_id = rec.get("location_id")
            if not loc_id or loc_id not in locations:
                # Try to resolve via (building, floor, room)
                bldg = str(rec.get("building") or "BLDG-1")
                try:
                    flr = int(rec.get("floor", 0))
                except (ValueError, TypeError):
                    flr = 0
                rm = str(rec.get("room")) if rec.get("room") else None

                matched_loc = combo_to_loc.get((bldg, flr, rm)) or combo_to_loc.get((bldg, flr, None))
                rec["location_id"] = matched_loc or first_loc_id

            attributed.append(rec)

        return attributed
