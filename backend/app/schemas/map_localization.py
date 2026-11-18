"""FastAPI Pydantic schemas for Map-Constrained Multimodal Particle Filter Localization (Task 27)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ParticleFilterRunRequest(BaseModel):
    """Payload for executing map-constrained particle filter localization on a dataset."""
    building_id: str = Field("BLDG-1", description="Building identifier")
    floor_id: int = Field(1, description="Floor index")
    particle_count: int = Field(500, ge=50, le=5000, description="Number of Monte Carlo particles")
    resample_threshold_ratio: float = Field(0.50, ge=0.1, le=0.9, description="Resample when Neff < N * ratio")
    sigma_step_length_m: float = Field(0.10, ge=0.01, le=1.0, description="Step length propagation noise std dev (m)")
    sigma_heading_rad: float = Field(0.087, ge=0.01, le=0.5, description="Heading propagation noise std dev (rad)")
    wall_crossing_penalty_factor: float = Field(0.0, ge=0.0, le=1.0, description="Wall crossing weight factor (0.0=hard wall rejection)")
    allow_unknown_areas: bool = Field(True, description="Do not eliminate particles in unmapped / unverified regions")
    random_seed: Optional[int] = Field(42, description="Random seed for repeatable particle generation")
    initial_x_m: Optional[float] = Field(None, description="Starting X coordinate (defaults to first Wi-Fi or map center)")
    initial_y_m: Optional[float] = Field(None, description="Starting Y coordinate (defaults to first Wi-Fi or map center)")
    initial_heading_deg: Optional[float] = Field(0.0, description="Starting heading in degrees (0 = East)")
    building_map_id: Optional[str] = Field(None, description="Specific building map ID to constrain against")


class ParticleFilterStateSchema(BaseModel):
    """Single discrete filtered state in a map-constrained trajectory."""
    timestamp_epoch_ms: int
    building_id: str
    floor_id: int
    x_m: float
    y_m: float
    theta_rad: float
    theta_deg: float
    status: str
    particle_count: int
    effective_sample_size: float
    particle_spread_m: float
    position_uncertainty_m: float
    wall_crossings_rejected_count: int = 0
    resampled_this_step: bool = False
    observation_type: str = "PDR_STEP"
    map_version: str = "1.0.0"
    ekf_reference_x_m: Optional[float] = None
    ekf_reference_y_m: Optional[float] = None
    innovation_x_m: Optional[float] = None
    innovation_y_m: Optional[float] = None
    metadata: Union[Dict[str, Any], str] = Field(default_factory=dict)


class ParticleFilterTrajectoryResponse(BaseModel):
    """Full trajectory output and operational summary from a particle filter run."""
    dataset_id: str
    total_points: int
    summary: Dict[str, Any]
    trajectory: List[ParticleFilterStateSchema]


class ParticleFilterStatusResponse(BaseModel):
    """Availability status and cached summary of particle filter artifacts."""
    dataset_id: str
    is_available: bool
    summary: Optional[Dict[str, Any]] = None
    artifact_paths: Dict[str, str] = Field(default_factory=dict)


class ParticleFilterEvaluateRequest(BaseModel):
    """Request payload for running comparative evaluation against ground truth or EKF baseline."""
    building_map_id: Optional[str] = Field(None, description="Optional map ID")
    use_synthetic_fixture: bool = Field(False, description="Run on multi-floor synthetic benchmark trajectory")
    particle_count: int = Field(500, ge=50, le=2000, description="Particle count")
    allow_unknown_areas: bool = Field(True, description="Permit traversal in unmapped zones")


class ParticleFilterEvaluationResponse(BaseModel):
    """Comparative trajectory evaluation metrics and wall crossing diagnostics."""
    dataset_id: str
    report: Dict[str, Any]
