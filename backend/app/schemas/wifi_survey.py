"""FastAPI Pydantic schemas for site-specific Wi-Fi survey and localization (Task 23)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ReferenceLocationCreate(BaseModel):
    """Schema for registering or updating a reference location."""
    location_id: str = Field(..., description="Unique reference location identifier (e.g. 'LOC-01')")
    site_id: str = Field(..., description="Physical site identifier (e.g. 'SITE-ENG-BLDG')")
    building_id: str = Field(..., description="Building identifier (e.g. 'BLDG-01')")
    floor_id: int = Field(..., description="Floor integer index (e.g. 0, 1, 2)")
    x_m: float = Field(..., description="Local Cartesian X coordinate in meters")
    y_m: float = Field(..., description="Local Cartesian Y coordinate in meters")
    room_id: Optional[str] = Field(None, description="Optional room or zone identifier")
    description: Optional[str] = Field(None, description="Human readable description of landmark")
    is_verified: bool = Field(True, description="Whether location coordinates were surveyed and verified")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom survey metadata")


class ReferenceLocationResponse(BaseModel):
    """Schema for returning reference location details."""
    location_id: str
    site_id: str
    building_id: str
    floor_id: int
    x_m: float
    y_m: float
    room_id: Optional[str] = None
    description: Optional[str] = None
    survey_session_ids: List[str] = Field(default_factory=list)
    is_verified: bool = True
    created_at_iso: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReferenceLocationsListResponse(BaseModel):
    """Schema for listing registered reference locations for a site."""
    site_id: str
    total_locations: int
    locations: List[ReferenceLocationResponse]


class RadioMapBuildRequest(BaseModel):
    """Schema for requesting radio map compilation from survey datasets."""
    site_id: str = Field(..., description="Target site identifier")
    version: str = Field("v1", description="Radio map version tag (e.g. 'v1', 'v2')")
    missing_penalty_rssi: float = Field(-105.0, description="Floor RSSI for unobserved APs (dBm)")
    min_sweeps_per_location: int = Field(1, ge=1, description="Minimum sweeps required per location")
    min_visibility_rate: float = Field(0.0, ge=0.0, le=1.0, description="Minimum AP visibility threshold")
    std_rssi_floor: float = Field(2.0, ge=0.5, description="Gaussian standard deviation floor (dBm)")
    description: Optional[str] = Field(None, description="Optional radio map description")
    auto_register_locations: bool = Field(
        True,
        description="If True, automatically creates reference locations from survey annotations/metadata if missing"
    )


class RadioMapMetadataResponse(BaseModel):
    """Metadata response describing a compiled radio map."""
    site_id: str
    version: str
    created_at_iso: str
    total_locations: int
    total_sweeps: int
    unique_bssids: int
    missing_penalty_rssi: float
    session_ids: List[str]
    description: Optional[str] = None


class RadioMapResponse(BaseModel):
    """Full radio map details including location profiles."""
    metadata: RadioMapMetadataResponse
    locations: Dict[str, ReferenceLocationResponse]
    profiles: Dict[str, Any]
    all_bssids: List[str]


class RadioMapListResponse(BaseModel):
    """List of compiled radio maps."""
    total_radio_maps: int
    radio_maps: List[RadioMapMetadataResponse]


class WifiLocalizationQueryRequest(BaseModel):
    """Schema for querying live or test Wi-Fi fingerprint localization."""
    site_id: str = Field(..., description="Site identifier to query against")
    version: str = Field("v1", description="Radio map version")
    observations: Union[Dict[str, float], List[Dict[str, Any]]] = Field(
        ...,
        description="Wi-Fi observations as {bssid: rssi} or list of {bssid, rssi_dbm} objects"
    )
    algorithm: str = Field("weighted_knn", description="'weighted_knn' or 'probabilistic'")
    k: int = Field(3, ge=1, le=20, description="Number of nearest neighbors to consider")
    distance_metric: str = Field("euclidean", description="Distance metric ('euclidean' or 'manhattan')")


class CandidateMatchResponse(BaseModel):
    """Scored candidate reference location."""
    location_id: str
    building_id: str
    floor_id: int
    x_m: float
    y_m: float
    room_id: Optional[str] = None
    distance_metric: float
    score: float
    matched_ap_count: int
    total_query_aps: int


class LocalizationPredictionResponse(BaseModel):
    """Fingerprint localization result."""
    predicted_building_id: str
    predicted_floor_id: int
    predicted_x_m: float
    predicted_y_m: float
    predicted_room_id: Optional[str] = None
    confidence: float
    algorithm: str
    matched_ap_count: int
    total_query_aps: int
    candidate_matches: List[CandidateMatchResponse]
    warnings: List[str] = Field(default_factory=list)
    is_low_confidence: bool = False


class WifiEvaluationRequest(BaseModel):
    """Schema for running session-partitioned offline evaluation."""
    site_id: str = Field(..., description="Site identifier")
    algorithm: str = Field("weighted_knn", description="'weighted_knn' or 'probabilistic'")
    k: int = Field(3, ge=1, le=20, description="Top-k nearest neighbors")
    distance_metric: str = Field("euclidean", description="'euclidean' or 'manhattan'")
    train_session_ids: Optional[List[str]] = Field(None, description="Explicit training survey sessions")
    test_session_ids: Optional[List[str]] = Field(None, description="Explicit test survey sessions")
    test_ratio: float = Field(0.3, ge=0.1, le=0.9, description="Holdout ratio if sessions not specified")


class WifiEvaluationResponse(BaseModel):
    """Schema for Wi-Fi localization evaluation results."""
    site_id: str
    algorithm: str
    metrics: Dict[str, Any]
