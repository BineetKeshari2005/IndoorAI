"""Pydantic schemas for PDR ground-truth trial registration, Weinberg calibration, and evaluation APIs (Task 22)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WaypointSchema(BaseModel):
    """Reference spatial waypoint along walking path."""
    waypoint_id: str
    x_m: float
    y_m: float
    timestamp_ms: Optional[int] = None
    uncertainty_m: float = Field(0.05, description="Measurement precision in meters")
    description: str = ""


class GroundTruthSchema(BaseModel):
    """Manually measured ground truth data for a walking trial."""
    measured_distance_m: float = Field(..., gt=0.0, description="Measured distance in meters")
    measurement_method: str = Field(..., description="e.g. LASER_RANGEFINDER, SURVEY_TAPE")
    measurement_uncertainty_m: float = Field(0.05, ge=0.0, description="Measurement error bound in meters")
    measured_start_heading_deg: Optional[float] = Field(None, ge=0.0, le=360.0, description="Start heading azimuth")
    measured_end_heading_deg: Optional[float] = Field(None, ge=0.0, le=360.0, description="End heading azimuth")
    heading_measurement_uncertainty_deg: float = Field(2.0, ge=0.0)
    ground_truth_step_count: Optional[int] = Field(None, ge=0, description="Manually counted steps")
    step_count_uncertainty: int = Field(0, ge=0)
    num_turns: int = Field(0, ge=0, description="Number of directional turns")
    waypoints: List[WaypointSchema] = Field(default_factory=list)


class TrialRegisterRequest(BaseModel):
    """Request payload to register a measured walking trial."""
    trial_id: Optional[str] = Field(None, description="Unique trial ID (auto-generated if omitted)")
    session_id: str = Field(..., description="Survey session ID in the dataset")
    participant_id: str = Field("PARTICIPANT-ANON", description="Pseudonymous participant ID")
    device_model: str = Field("Smartphone", description="Device hardware model")
    phone_placement: str = Field("HAND_HELD", description="HAND_HELD, POCKET, BAG, TEXTING, EAR")
    walking_condition: str = Field("NORMAL_PACE", description="NORMAL_PACE, SLOW_WALK, BRISK_WALK, etc.")
    trial_type: str = Field("STRAIGHT_LINE", description="STRAIGHT_LINE, L_SHAPED, FORWARD_RETURN, etc.")
    start_timestamp_ms: int = Field(..., description="Start epoch millisecond")
    end_timestamp_ms: int = Field(..., description="End epoch millisecond")
    ground_truth: GroundTruthSchema
    trial_notes: str = ""
    is_synthetic: bool = Field(False, description="Whether this is a synthetic test fixture")


class TrialSummaryResponse(BaseModel):
    """Summary of a registered walking trial."""
    trial_id: str
    dataset_id: str
    session_id: str
    participant_id: str
    device_model: str
    phone_placement: str
    walking_condition: str
    trial_type: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    measured_distance_m: float
    measurement_method: str
    ground_truth_step_count: Optional[int] = None
    is_synthetic: bool
    created_at_iso: str


class TrialListResponse(BaseModel):
    """List of registered trials for a dataset."""
    dataset_id: str
    total_trials: int
    trials: List[TrialSummaryResponse]


class CalibrateRequest(BaseModel):
    """Request payload to trigger Weinberg step-length parameter estimation."""
    training_trial_ids: List[str] = Field(..., min_items=1, description="Trial IDs used for fitting K")
    held_out_trial_ids: Optional[List[str]] = Field(None, description="Optional held-out trial IDs for validation")
    profile_name: str = Field("Calibrated Weinberg Profile", description="Human-readable profile name")
    profile_type: str = Field("PHONE_PLACEMENT_SPECIFIC", description="Scope of the calibrated profile")
    phone_placement: Optional[str] = Field(None, description="Target phone placement")
    device_model: Optional[str] = Field(None, description="Target device model")
    participant_id: Optional[str] = Field(None, description="Target participant ID")


class CalibrationProfileResponse(BaseModel):
    """Calibrated PDR parameter profile."""
    profile_id: str
    profile_name: str
    profile_type: str
    calibration_method: str
    weinberg_k: float
    step_length_bounds: List[float]
    device_model: Optional[str] = None
    phone_placement: Optional[str] = None
    participant_id: Optional[str] = None
    source_trial_ids: List[str]
    calibration_date_iso: str
    calibration_status: str
    training_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    notes: str


class CalibrationProfilesListResponse(BaseModel):
    """All registered calibration profiles."""
    total_profiles: int
    profiles: List[CalibrationProfileResponse]


class EvaluateRequest(BaseModel):
    """Request payload to evaluate PDR against ground-truth walking trials."""
    trial_ids: List[str] = Field(..., min_items=1, description="Trial IDs to evaluate")
    calibration_profile_id: Optional[str] = Field(None, description="Profile ID to use for evaluation (default: PROF-DEFAULT)")


class EvaluationReportSummaryResponse(BaseModel):
    """High-level summary of an evaluation report."""
    report_id: str
    dataset_id: str
    session_id: str
    generated_at_iso: str
    is_synthetic_fixture: bool
    calibration_profile_id: str
    trial_ids_evaluated: List[str]
    step_precision: float
    step_recall: float
    step_f1_score: float
    distance_error_m: float
    relative_distance_error_pct: float
    endpoint_error_m: float
    drift_per_meter: float
    drift_per_minute_m: float
    ground_truth_within_95_ellipse: Optional[bool] = None
