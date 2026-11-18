"""Storage layer and file-backed repository for PDR ground-truth trials and calibration profiles (Task 22)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.logging import setup_logger
from src.pdr.calibration.models import (
    CalibrationProfile,
    CalibrationStatus,
    GroundTruthMeasurement,
    PhonePlacement,
    ProfileType,
    WalkingTrial,
)

logger = setup_logger("CalibrationStorage")


def find_project_root() -> Path:
    """Locates the project root by searching upward for pyproject.toml."""
    curr = Path(__file__).resolve().parent
    for p in [curr] + list(curr.parents):
        if (p / "pyproject.toml").is_file():
            return p
    return Path.cwd()


class CalibrationStorage:
    """Manages filesystem persistence for walking trials and calibration profiles under data/calibration/pdr/."""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            self.base_dir = find_project_root() / "data" / "calibration" / "pdr"
        else:
            self.base_dir = Path(base_dir)

        self.trials_dir = self.base_dir / "trials"
        self.results_dir = self.base_dir / "calibration_results"
        self.registry_file = self.base_dir / "calibration_registry.json"

        self.ensure_directories()
        self._ensure_default_profile()

    def ensure_directories(self) -> None:
        """Creates necessary directory hierarchy if missing."""
        self.trials_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_default_profile(self) -> None:
        """Ensures the baseline uncalibrated profile exists in the registry."""
        default_prof = self.get_profile("PROF-DEFAULT")
        if default_prof is None:
            now_iso = datetime.now(timezone.utc).isoformat()
            default_prof = CalibrationProfile(
                profile_id="PROF-DEFAULT",
                profile_name="Default Uncalibrated Weinberg Profile",
                profile_type=ProfileType.DEFAULT_UNCALIBRATED,
                calibration_method="DEFAULT_UNIFORM",
                weinberg_k=0.42,
                step_length_bounds=(0.30, 1.20),
                step_detection_config={
                    "step_detection_signal": "vertical",
                    "min_peak_prominence_mps2": 0.4,
                    "peak_threshold_mps2": 0.8,
                    "min_step_interval_s": 0.33,
                },
                heading_config={
                    "heading_source": "auto",
                },
                source_trial_ids=[],
                calibration_date_iso=now_iso,
                calibration_status=CalibrationStatus.UNCALIBRATED,
                notes="Standard uncalibrated Weinberg model with nominal coefficient K=0.42."
            )
            self.save_profile(default_prof)

    # -------------------------------------------------------------------------
    # Trial Operations
    # -------------------------------------------------------------------------

    def save_trial(self, trial: WalkingTrial) -> None:
        """Saves a ground-truth walking trial partitioned into metadata and ground_truth JSON files."""
        trial_folder = self.trials_dir / trial.trial_id
        trial_folder.mkdir(parents=True, exist_ok=True)

        trial_dict = trial.to_dict()
        gt_dict = trial_dict.pop("ground_truth")

        metadata_path = trial_folder / "trial_metadata.json"
        gt_path = trial_folder / "ground_truth.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(trial_dict, f, indent=2)

        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_dict, f, indent=2)

        logger.info("Saved walking trial '%s' under %s", trial.trial_id, trial_folder)

    def get_trial(self, trial_id: str) -> Optional[WalkingTrial]:
        """Loads a walking trial by trial ID."""
        trial_folder = self.trials_dir / trial_id
        metadata_path = trial_folder / "trial_metadata.json"
        gt_path = trial_folder / "ground_truth.json"

        if not metadata_path.is_file() or not gt_path.is_file():
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            with open(gt_path, "r", encoding="utf-8") as f:
                gt = json.load(f)

            meta["ground_truth"] = gt
            return WalkingTrial.from_dict(meta)
        except Exception as e:
            logger.error("Failed to load trial '%s': %s", trial_id, e)
            return None

    def list_trials(self, dataset_id: Optional[str] = None) -> List[WalkingTrial]:
        """Lists all registered trials, optionally filtered by dataset ID."""
        trials: List[WalkingTrial] = []
        if not self.trials_dir.is_dir():
            return trials

        for sub in sorted(self.trials_dir.iterdir()):
            if sub.is_dir():
                trial = self.get_trial(sub.name)
                if trial:
                    if dataset_id is None or trial.dataset_id == dataset_id:
                        trials.append(trial)
        return trials

    # -------------------------------------------------------------------------
    # Calibration Profile Operations
    # -------------------------------------------------------------------------

    def save_profile(self, profile: CalibrationProfile) -> None:
        """Saves or updates a calibration profile in calibration_registry.json."""
        registry = self._load_registry()
        registry[profile.profile_id] = profile.to_dict()
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        logger.info("Saved calibration profile '%s' (K=%.4f, status=%s)", profile.profile_id, profile.weinberg_k, profile.calibration_status)

    def get_profile(self, profile_id: str) -> Optional[CalibrationProfile]:
        """Retrieves a profile by profile ID."""
        registry = self._load_registry()
        data = registry.get(profile_id)
        if data:
            return CalibrationProfile.from_dict(data)
        return None

    def list_profiles(self) -> List[CalibrationProfile]:
        """Returns all registered calibration profiles."""
        registry = self._load_registry()
        return [CalibrationProfile.from_dict(v) for v in registry.values()]

    def resolve_best_profile(
        self,
        participant_id: Optional[str] = None,
        phone_placement: Optional[PhonePlacement] = None,
        device_model: Optional[str] = None
    ) -> CalibrationProfile:
        """Hierarchically selects the most specific calibrated profile available, falling back to default."""
        profiles = self.list_profiles()
        # Filter for valid calibrated/validated profiles only
        calibrated_profs = [
            p for p in profiles
            if p.calibration_status in (CalibrationStatus.CALIBRATED, CalibrationStatus.VALIDATED)
        ]

        # 1. Exact match: Participant + Placement + Device
        for p in calibrated_profs:
            if (p.participant_id and p.participant_id == participant_id and
                p.phone_placement and p.phone_placement == phone_placement and
                p.device_model and p.device_model == device_model):
                return p

        # 2. Participant + Placement
        for p in calibrated_profs:
            if (p.participant_id and p.participant_id == participant_id and
                p.phone_placement and p.phone_placement == phone_placement):
                return p

        # 3. Placement + Device
        for p in calibrated_profs:
            if (p.phone_placement and p.phone_placement == phone_placement and
                p.device_model and p.device_model == device_model):
                return p

        # 4. Placement only
        for p in calibrated_profs:
            if p.phone_placement and p.phone_placement == phone_placement:
                return p

        # 5. Device only
        for p in calibrated_profs:
            if p.device_model and p.device_model == device_model:
                return p

        # Fallback to default
        default_prof = self.get_profile("PROF-DEFAULT")
        if default_prof is None:
            self._ensure_default_profile()
            default_prof = self.get_profile("PROF-DEFAULT")
        return default_prof  # type: ignore

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Loads the registry JSON dictionary."""
        if not self.registry_file.is_file():
            return {}
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Corrupt or empty calibration registry, initializing fresh: %s", e)
            return {}
