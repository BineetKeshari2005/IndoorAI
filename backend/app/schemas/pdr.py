"""Pydantic schemas for Pedestrian Dead Reckoning (PDR) API endpoints (Task 21)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PDRRunRequest(BaseModel):
    """Optional parameters passed when triggering PDR trajectory reconstruction."""

    step_detection_signal: str = Field("vertical", description="Signal to use: 'vertical' or 'magnitude'")
    weinberg_k: float = Field(0.42, description="Weinberg step length calibration coefficient")
    min_step_interval_s: float = Field(0.33, description="Minimum refractory period between steps (s)")
    min_peak_prominence_mps2: float = Field(0.8, description="Minimum peak prominence (m/s^2)")
    initial_heading_rad: float = Field(0.0, description="Initial heading angle in radians")
    origin_x_m: float = Field(0.0, description="Local origin X coordinate (m)")
    origin_y_m: float = Field(0.0, description="Local origin Y coordinate (m)")


class PDRRunResponse(BaseModel):
    """Result of PDR pipeline execution."""

    dataset_id: str
    status: str
    message: str
    report: Dict[str, Any]


class PDRStatusResponse(BaseModel):
    """Status and report of previously executed PDR."""

    dataset_id: str
    is_computed: bool
    report: Optional[Dict[str, Any]] = None
    manifest: Optional[Dict[str, Any]] = None


class TrajectoryPoint(BaseModel):
    """2D trajectory coordinate point in the local relative navigation frame."""

    point_id: str
    step_id: str
    timestamp_ms: int
    rel_x_m: float
    rel_y_m: float
    heading_deg: float
    step_length_m: float
    cum_distance_m: float
    pos_uncertainty_radius_m: float
    confidence: float
    is_reliable: bool


class TrajectoryResponse(BaseModel):
    """Sequential 2D relative trajectory reconstructed from PDR."""

    dataset_id: str
    total_points: int
    points: List[TrajectoryPoint]


class StepRecord(BaseModel):
    """Individual human step event detected from accelerometer signal."""

    step_id: str
    timestamp_ms: int
    peak_accel_mps2: float
    valley_accel_mps2: float
    step_interval_ms: float
    step_frequency_hz: float
    step_length_m: float
    confidence: float
    validation_status: str


class StepsResponse(BaseModel):
    """List of detected step events with kinematics and intervals."""

    dataset_id: str
    total_steps: int
    steps: List[StepRecord]
