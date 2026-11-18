"""Site-Specific Wi-Fi Localization Domain Models.

Defines schemas and dataclasses for reference locations, survey observations,
aggregated radio map profiles, candidate matches, localization predictions,
and session-level evaluation metrics.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReferenceLocation:
    """Manually surveyed or designated reference location within a physical site."""
    location_id: str
    site_id: str
    building_id: str
    floor_id: int
    x_m: float
    y_m: float
    room_id: Optional[str] = None
    description: Optional[str] = None
    survey_session_ids: List[str] = field(default_factory=list)
    is_verified: bool = True
    created_at_iso: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReferenceLocation:
        return cls(
            location_id=str(data["location_id"]),
            site_id=str(data["site_id"]),
            building_id=str(data["building_id"]),
            floor_id=int(data["floor_id"]),
            x_m=float(data["x_m"]),
            y_m=float(data["y_m"]),
            room_id=data.get("room_id"),
            description=data.get("description"),
            survey_session_ids=list(data.get("survey_session_ids", [])),
            is_verified=bool(data.get("is_verified", True)),
            created_at_iso=str(data.get("created_at_iso", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ApObservation:
    """Observation of a single Wi-Fi access point in a scan."""
    bssid: str
    rssi_dbm: float
    ssid: Optional[str] = None
    frequency_mhz: Optional[int] = None
    channel: Optional[int] = None
    channel_band: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ApObservation:
        return cls(
            bssid=str(data["bssid"]).lower(),
            rssi_dbm=float(data["rssi_dbm"]),
            ssid=data.get("ssid"),
            frequency_mhz=int(data["frequency_mhz"]) if data.get("frequency_mhz") is not None else None,
            channel=int(data["channel"]) if data.get("channel") is not None else None,
            channel_band=data.get("channel_band"),
        )


@dataclass
class SurveySweep:
    """A distinct Wi-Fi sweep/burst collected during a survey session."""
    sweep_id: str
    session_id: str
    location_id: str
    timestamp_epoch_ms: int
    observations: List[ApObservation] = field(default_factory=list)

    def to_rssi_dict(self) -> Dict[str, float]:
        """Convert sweep observations to a {bssid: rssi} dictionary."""
        return {obs.bssid: obs.rssi_dbm for obs in self.observations}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "session_id": self.session_id,
            "location_id": self.location_id,
            "timestamp_epoch_ms": self.timestamp_epoch_ms,
            "observations": [obs.to_dict() for obs in self.observations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SurveySweep:
        return cls(
            sweep_id=str(data["sweep_id"]),
            session_id=str(data["session_id"]),
            location_id=str(data["location_id"]),
            timestamp_epoch_ms=int(data["timestamp_epoch_ms"]),
            observations=[ApObservation.from_dict(o) for o in data.get("observations", [])],
        )


@dataclass
class ApStatistics:
    """Aggregated radio frequency statistics for a specific BSSID at a reference location."""
    bssid: str
    observation_count: int
    total_sweeps_at_location: int
    visibility_rate: float
    mean_rssi: float
    median_rssi: float
    std_rssi: float
    min_rssi: float
    max_rssi: float
    ssid: Optional[str] = None
    frequency_mhz: Optional[int] = None
    channel_band: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ApStatistics:
        return cls(
            bssid=str(data["bssid"]).lower(),
            observation_count=int(data["observation_count"]),
            total_sweeps_at_location=int(data["total_sweeps_at_location"]),
            visibility_rate=float(data["visibility_rate"]),
            mean_rssi=float(data["mean_rssi"]),
            median_rssi=float(data["median_rssi"]),
            std_rssi=float(data["std_rssi"]),
            min_rssi=float(data["min_rssi"]),
            max_rssi=float(data["max_rssi"]),
            ssid=data.get("ssid"),
            frequency_mhz=int(data["frequency_mhz"]) if data.get("frequency_mhz") is not None else None,
            channel_band=data.get("channel_band"),
        )


@dataclass
class LocationRadioProfile:
    """Aggregated multi-sweep radio profile for a reference location."""
    location_id: str
    site_id: str
    building_id: str
    floor_id: int
    x_m: float
    y_m: float
    room_id: Optional[str] = None
    total_sweeps: int = 0
    ap_stats: Dict[str, ApStatistics] = field(default_factory=dict)
    mean_rssi_vector: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location_id": self.location_id,
            "site_id": self.site_id,
            "building_id": self.building_id,
            "floor_id": self.floor_id,
            "room_id": self.room_id,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "total_sweeps": self.total_sweeps,
            "ap_stats": {b: s.to_dict() for b, s in self.ap_stats.items()},
            "mean_rssi_vector": self.mean_rssi_vector,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LocationRadioProfile:
        ap_stats = {
            b: ApStatistics.from_dict(s)
            for b, s in data.get("ap_stats", {}).items()
        }
        return cls(
            location_id=str(data["location_id"]),
            site_id=str(data["site_id"]),
            building_id=str(data["building_id"]),
            floor_id=int(data["floor_id"]),
            room_id=data.get("room_id"),
            x_m=float(data["x_m"]),
            y_m=float(data["y_m"]),
            total_sweeps=int(data.get("total_sweeps", 0)),
            ap_stats=ap_stats,
            mean_rssi_vector=dict(data.get("mean_rssi_vector", {})),
        )


@dataclass
class RadioMapMetadata:
    """Metadata describing a compiled site radio map."""
    site_id: str
    version: str
    created_at_iso: str
    total_locations: int
    total_sweeps: int
    unique_bssids: int
    missing_penalty_rssi: float = -105.0
    session_ids: List[str] = field(default_factory=list)
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RadioMapMetadata:
        return cls(
            site_id=str(data["site_id"]),
            version=str(data["version"]),
            created_at_iso=str(data["created_at_iso"]),
            total_locations=int(data["total_locations"]),
            total_sweeps=int(data["total_sweeps"]),
            unique_bssids=int(data["unique_bssids"]),
            missing_penalty_rssi=float(data.get("missing_penalty_rssi", -105.0)),
            session_ids=list(data.get("session_ids", [])),
            description=data.get("description"),
        )


@dataclass
class RadioMap:
    """Compiled fingerprint database for a specific site version."""
    metadata: RadioMapMetadata
    locations: Dict[str, ReferenceLocation] = field(default_factory=dict)
    profiles: Dict[str, LocationRadioProfile] = field(default_factory=dict)
    all_bssids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "locations": {k: v.to_dict() for k, v in self.locations.items()},
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
            "all_bssids": self.all_bssids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RadioMap:
        meta = RadioMapMetadata.from_dict(data["metadata"])
        locations = {
            k: ReferenceLocation.from_dict(v)
            for k, v in data.get("locations", {}).items()
        }
        profiles = {
            k: LocationRadioProfile.from_dict(v)
            for k, v in data.get("profiles", {}).items()
        }
        return cls(
            metadata=meta,
            locations=locations,
            profiles=profiles,
            all_bssids=list(data.get("all_bssids", [])),
        )


@dataclass
class CandidateMatch:
    """A scored candidate reference location returned during fingerprint matching."""
    location_id: str
    building_id: str
    floor_id: int
    x_m: float
    y_m: float
    room_id: Optional[str] = None
    distance_metric: float = 0.0
    score: float = 0.0
    matched_ap_count: int = 0
    total_query_aps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class LocalizationPrediction:
    """Estimated location from an input Wi-Fi query fingerprint."""
    predicted_building_id: str
    predicted_floor_id: int
    predicted_x_m: float
    predicted_y_m: float
    predicted_room_id: Optional[str] = None
    confidence: float = 0.0
    algorithm: str = "weighted_knn"
    matched_ap_count: int = 0
    total_query_aps: int = 0
    candidate_matches: List[CandidateMatch] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_low_confidence: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_building_id": self.predicted_building_id,
            "predicted_floor_id": self.predicted_floor_id,
            "predicted_room_id": self.predicted_room_id,
            "predicted_x_m": self.predicted_x_m,
            "predicted_y_m": self.predicted_y_m,
            "confidence": self.confidence,
            "algorithm": self.algorithm,
            "matched_ap_count": self.matched_ap_count,
            "total_query_aps": self.total_query_aps,
            "candidate_matches": [c.to_dict() for c in self.candidate_matches],
            "warnings": self.warnings,
            "is_low_confidence": self.is_low_confidence,
        }


@dataclass
class EvaluationMetrics:
    """Metrics from session-partitioned offline evaluation on survey test sessions."""
    total_test_samples: int
    building_accuracy: float
    floor_accuracy: float
    building_floor_accuracy: float
    location_id_accuracy: float
    top_3_location_accuracy: float
    top_5_location_accuracy: float
    mean_coordinate_error_m: float
    median_coordinate_error_m: float
    p95_coordinate_error_m: float
    max_coordinate_error_m: float
    low_confidence_rate: float
    train_session_ids: List[str] = field(default_factory=list)
    test_session_ids: List[str] = field(default_factory=list)
    confusion_matrix_floors: Dict[str, Dict[str, int]] = field(default_factory=dict)
    per_location_errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
