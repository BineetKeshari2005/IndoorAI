"""FastAPI Pydantic schemas for Wi-Fi + PDR Extended Kalman Filter Sensor Fusion (Task 24)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FusionRunRequest(BaseModel):
    """Configuration payload for running EKF trajectory fusion."""
    site_id: str = Field(..., description="Target site identifier (e.g. 'SITE-MAIN')")
    building_id: str = Field("BLDG-1", description="Building identifier")
    floor_id: int = Field(1, description="Floor integer index")
    initial_x_m: Optional[float] = Field(None, description="Starting X coordinate (defaults to first Wi-Fi or 0)")
    initial_y_m: Optional[float] = Field(None, description="Starting Y coordinate (defaults to first Wi-Fi or 0)")
    initial_heading_deg: Optional[float] = Field(0.0, description="Initial heading in degrees (0 = East, 90 = North)")
    chi2_threshold: float = Field(5.99146, description="2-DOF 95% Chi-Square threshold for outlier gating")
    enable_outlier_gating: bool = Field(True, description="Whether to reject statistical Wi-Fi outliers")
    default_wifi_sigma_m: float = Field(4.0, ge=0.5, description="Default 1-sigma Wi-Fi measurement uncertainty (m)")
    radio_map_version: str = Field("v1", description="Site radio map version tag")


class FusedStateResponse(BaseModel):
    """Single point in a fused trajectory."""
    timestamp_epoch_ms: int
    building_id: str
    floor_id: int
    x_m: float
    y_m: float
    theta_deg: float
    sigma_x_m: float
    sigma_y_m: float
    status: str
    semi_major_axis_m: float
    semi_minor_axis_m: float
    ellipse_angle_deg: float
    conservative_95_radius_m: float
    mahalanobis_dist_sq: Optional[float] = None
    rejection_reason: Optional[str] = None


class FusionTrajectoryResponse(BaseModel):
    """Response containing full fused trajectory points and summary."""
    dataset_id: str
    total_points: int
    summary: Dict[str, Any]
    trajectory: List[FusedStateResponse]


class FusionStatusResponse(BaseModel):
    """Status of fusion artifacts for a dataset."""
    dataset_id: str
    is_available: bool
    summary: Optional[Dict[str, Any]] = None
    artifact_paths: Dict[str, str] = Field(default_factory=dict)


class FusionEvaluationRequest(BaseModel):
    """Request payload for running comparative trajectory evaluation."""
    site_id: str = Field(..., description="Site identifier")
    chi2_threshold: float = Field(5.99146, description="Outlier gating threshold")
    use_synthetic_fixture: bool = Field(False, description="Use mathematically controlled synthetic fixture")


class FusionEvaluationResponse(BaseModel):
    """Comparative evaluation response across PDR-only, Wi-Fi-only, and Fused modes."""
    dataset_id: str
    report: Dict[str, Any]
