"""Data models and enums for PDR ground-truth calibration, validation, and drift analysis (Task 22)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TrialType(str, Enum):
    """Categorization of walking trial path geometry."""
    STRAIGHT_LINE = "STRAIGHT_LINE"
    FORWARD_RETURN = "FORWARD_RETURN"
    L_SHAPED = "L_SHAPED"
    MULTIPLE_TURNS = "MULTIPLE_TURNS"
    SHORT_DISTANCE = "SHORT_DISTANCE"
    LONG_DISTANCE = "LONG_DISTANCE"
    STATIONARY = "STATIONARY"


class PhonePlacement(str, Enum):
    """Physiological placement / carriage mode of the smartphone during trial."""
    HAND_HELD = "HAND_HELD"
    POCKET = "POCKET"
    BAG = "BAG"
    TEXTING = "TEXTING"
    EAR = "EAR"
    UNKNOWN = "UNKNOWN"


class WalkingCondition(str, Enum):
    """Pace and environment condition of the walking trial."""
    NORMAL_PACE = "NORMAL_PACE"
    SLOW_WALK = "SLOW_WALK"
    BRISK_WALK = "BRISK_WALK"
    STAIRS_EXCLUDED = "STAIRS_EXCLUDED"
    SHUFFLE = "SHUFFLE"
    IRREGULAR = "IRREGULAR"


class CalibrationStatus(str, Enum):
    """Empirical calibration and validation status of a profile or trial."""
    UNCALIBRATED = "UNCALIBRATED"
    CALIBRATED = "CALIBRATED"
    VALIDATED = "VALIDATED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ProfileType(str, Enum):
    """Scope of application for a calibration profile."""
    DEFAULT_UNCALIBRATED = "DEFAULT_UNCALIBRATED"
    PARTICIPANT_SPECIFIC = "PARTICIPANT_SPECIFIC"
    DEVICE_SPECIFIC = "DEVICE_SPECIFIC"
    PHONE_PLACEMENT_SPECIFIC = "PHONE_PLACEMENT_SPECIFIC"


@dataclass
class ReferenceWaypoint:
    """Manually measured 2D spatial waypoint along a ground-truth trial path."""
    waypoint_id: str
    x_m: float
    y_m: float
    timestamp_ms: Optional[int] = None
    uncertainty_m: float = 0.05
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReferenceWaypoint:
        return cls(**data)


@dataclass
class GroundTruthMeasurement:
    """Rigorous, manually measured ground-truth spatial and kinematic parameters."""
    measured_distance_m: float
    measurement_method: str  # e.g., "LASER_RANGEFINDER", "SURVEY_TAPE", "BLUEPRINT_DIMENSIONS"
    measurement_uncertainty_m: float
    measured_start_heading_deg: Optional[float] = None
    measured_end_heading_deg: Optional[float] = None
    heading_measurement_uncertainty_deg: float = 2.0
    ground_truth_step_count: Optional[int] = None
    step_count_uncertainty: int = 0
    num_turns: int = 0
    waypoints: List[ReferenceWaypoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["waypoints"] = [wp.to_dict() if isinstance(wp, ReferenceWaypoint) else wp for wp in self.waypoints]
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroundTruthMeasurement:
        d = data.copy()
        raw_wps = d.get("waypoints", [])
        d["waypoints"] = [
            ReferenceWaypoint.from_dict(wp) if isinstance(wp, dict) else wp
            for wp in raw_wps
        ]
        return cls(**d)


@dataclass
class WalkingTrial:
    """Complete record of a ground-truth walking trial linked to an IndoorAI survey dataset."""
    trial_id: str
    dataset_id: str
    session_id: str
    participant_id: str  # Pseudonymous, e.g. "PARTICIPANT-01"
    device_model: str
    phone_placement: PhonePlacement
    walking_condition: WalkingCondition
    trial_type: TrialType
    start_timestamp_ms: int
    end_timestamp_ms: int
    ground_truth: GroundTruthMeasurement
    trial_notes: str = ""
    is_synthetic: bool = False  # Strictly distinguishes synthetic test fixtures from measured trials
    created_at_iso: str = ""

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["phone_placement"] = self.phone_placement.value if isinstance(self.phone_placement, PhonePlacement) else self.phone_placement
        res["walking_condition"] = self.walking_condition.value if isinstance(self.walking_condition, WalkingCondition) else self.walking_condition
        res["trial_type"] = self.trial_type.value if isinstance(self.trial_type, TrialType) else self.trial_type
        res["ground_truth"] = self.ground_truth.to_dict()
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WalkingTrial:
        d = data.copy()
        d["phone_placement"] = PhonePlacement(d["phone_placement"])
        d["walking_condition"] = WalkingCondition(d["walking_condition"])
        d["trial_type"] = TrialType(d["trial_type"])
        d["ground_truth"] = GroundTruthMeasurement.from_dict(d["ground_truth"])
        return cls(**d)


@dataclass
class StepDetectionMetrics:
    """Statistical evaluation metrics for step detection against ground-truth counts."""
    total_detected_steps: int
    ground_truth_steps: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    count_error: int
    relative_count_error: float
    timing_mean_error_ms: Optional[float] = None
    is_statistically_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HeadingMetrics:
    """Evaluation metrics for estimated direction against reference headings with wraparound handling."""
    mean_absolute_error_deg: float
    median_absolute_error_deg: float
    p95_absolute_error_deg: float
    circular_mean_error_deg: float
    circular_std_deg: float
    drift_rate_deg_per_sec: float
    drift_rate_deg_per_min: float
    num_heading_samples: int
    magnetic_disturbance_sample_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryDriftMetrics:
    """Quantified dead-reckoning drift and positional error against reference geometry."""
    estimated_distance_m: float
    ground_truth_distance_m: float
    absolute_distance_error_m: float
    relative_distance_error: float  # |d_est - d_gt| / d_gt
    endpoint_error_m: float          # Euclidean distance from estimated endpoint to ground-truth endpoint
    relative_endpoint_error: float   # endpoint_error / d_gt
    net_displacement_error_m: float
    mean_cross_track_error_m: Optional[float] = None  # Deviation from reference path line segments
    drift_per_meter: float = 0.0     # endpoint_error_m / ground_truth_distance_m (m/m)
    drift_per_minute_m: float = 0.0  # endpoint_error_m / duration_minutes (m/min)
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConfidenceEllipse:
    """Statistically derived 2D Gaussian confidence ellipse parameters from spatial covariance."""
    semi_major_axis_m: float   # sqrt(chi2 * lambda_1)
    semi_minor_axis_m: float   # sqrt(chi2 * lambda_2)
    orientation_deg: float     # Angle of semi-major axis relative to X-axis in degrees
    chi2_quantile: float = 5.991  # 95% quantile for 2 DOF: chi2(2, 0.95)
    conservative_radius_m: float = 0.0  # Equal to semi_major_axis_m

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UncertaintyEvaluationMetrics:
    """Audit of spatial covariance propagation against reference truth."""
    final_covariance_trace: float
    confidence_ellipse: ConfidenceEllipse
    ground_truth_within_95_ellipse: bool
    empirical_coverage_ratio: Optional[float] = None  # Percentage of reference waypoints inside estimated ellipse

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["confidence_ellipse"] = self.confidence_ellipse.to_dict()
        return res


@dataclass
class CalibrationProfile:
    """A calibrated parameter profile for the PDR engine."""
    profile_id: str
    profile_name: str
    profile_type: ProfileType
    calibration_method: str
    weinberg_k: float
    step_length_bounds: Tuple[float, float] = (0.30, 1.20)
    step_detection_config: Dict[str, Any] = field(default_factory=dict)
    heading_config: Dict[str, Any] = field(default_factory=dict)
    device_model: Optional[str] = None
    phone_placement: Optional[PhonePlacement] = None
    participant_id: Optional[str] = None
    source_trial_ids: List[str] = field(default_factory=list)
    calibration_date_iso: str = ""
    calibration_status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
    training_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["profile_type"] = self.profile_type.value if isinstance(self.profile_type, ProfileType) else self.profile_type
        res["calibration_status"] = self.calibration_status.value if isinstance(self.calibration_status, CalibrationStatus) else self.calibration_status
        if self.phone_placement:
            res["phone_placement"] = self.phone_placement.value if isinstance(self.phone_placement, PhonePlacement) else self.phone_placement
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CalibrationProfile:
        d = data.copy()
        d["profile_type"] = ProfileType(d["profile_type"])
        d["calibration_status"] = CalibrationStatus(d["calibration_status"])
        if d.get("phone_placement"):
            d["phone_placement"] = PhonePlacement(d["phone_placement"])
        if isinstance(d.get("step_length_bounds"), list):
            d["step_length_bounds"] = tuple(d["step_length_bounds"])
        return cls(**d)


@dataclass
class PDREvaluationReport:
    """Comprehensive evaluation report comparing PDR estimates against ground-truth trials."""
    report_id: str
    dataset_id: str
    session_id: str
    generated_at_iso: str
    is_synthetic_fixture: bool
    calibration_profile_id: str
    trial_ids_evaluated: List[str]
    step_metrics: StepDetectionMetrics
    trajectory_metrics: TrajectoryDriftMetrics
    heading_metrics: Optional[HeadingMetrics] = None
    uncertainty_metrics: Optional[UncertaintyEvaluationMetrics] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["step_metrics"] = self.step_metrics.to_dict()
        res["trajectory_metrics"] = self.trajectory_metrics.to_dict()
        if self.heading_metrics:
            res["heading_metrics"] = self.heading_metrics.to_dict()
        if self.uncertainty_metrics:
            res["uncertainty_metrics"] = self.uncertainty_metrics.to_dict()
        return res
