"""Domain models, configurations, and state definitions for Map-Constrained Particle Filter Localization (Task 27)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FilterStatus(str, Enum):
    """Operational state of the particle filter."""
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    MAP_CONSTRAINED = "MAP_CONSTRAINED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_MAP = "INVALID_MAP"
    LOST = "LOST"


class ObservationType(str, Enum):
    """Categorical source of observation updates."""
    INITIALIZATION = "INITIALIZATION"
    PDR_STEP = "PDR_STEP"
    WIFI_FIX = "WIFI_FIX"
    VISUAL_LANDMARK = "VISUAL_LANDMARK"
    MAP_PROXIMITY = "MAP_PROXIMITY"
    STATIONARY = "STATIONARY"


class ResampleAlgorithm(str, Enum):
    """Resampling algorithm strategy."""
    SYSTEMATIC = "SYSTEMATIC"
    STRATIFIED = "STRATIFIED"
    MULTINOMIAL = "MULTINOMIAL"


@dataclass
class ParticleFilterConfig:
    """Configuration parameters for the map-constrained particle filter."""
    particle_count: int = 500
    resample_threshold_ratio: float = 0.50  # Resample when Neff < particle_count * ratio
    sigma_step_length_m: float = 0.10       # Standard deviation of step length noise
    sigma_heading_rad: float = 0.087        # ~5 degrees heading noise during steps
    stationary_sigma_heading_rad: float = 0.01
    wifi_position_std_fallback: float = 4.0 # Fallback 1-sigma uncertainty for Wi-Fi in meters
    landmark_visibility_radius_m: float = 6.0 # Proximity radius around landmark
    wall_crossing_penalty_factor: float = 0.0 # 0.0 = hard wall crossing rejection
    out_of_boundary_penalty_factor: float = 0.0
    floor_mismatch_log_penalty: float = -20.0
    max_vertical_transition_radius_m: float = 3.5 # Proximity to stairs/elevator to allow floor transition
    allow_unknown_areas: bool = True        # If area is unmapped/partially surveyed, do NOT reject particles
    random_seed: Optional[int] = 42
    resample_algorithm: ResampleAlgorithm = ResampleAlgorithm.SYSTEMATIC

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["resample_algorithm"] = self.resample_algorithm.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParticleFilterConfig":
        clean = dict(data)
        if "resample_algorithm" in clean and isinstance(clean["resample_algorithm"], str):
            clean["resample_algorithm"] = ResampleAlgorithm(clean["resample_algorithm"])
        return cls(**clean)


@dataclass
class ParticleFilterState:
    """Estimated positioning state and uncertainty at a discrete point in time."""
    timestamp_epoch_ms: int
    building_id: str
    floor_id: int
    x_m: float
    y_m: float
    theta_rad: float
    status: FilterStatus
    particle_count: int
    effective_sample_size: float
    particle_spread_m: float            # Weighted std dev of particle cloud positions (empirical RMS spread)
    position_uncertainty_m: float        # Conservative circular 95% confidence radius (2.447 * spread)
    var_x: float = 0.0                  # Spatial variance along X axis (m^2)
    var_y: float = 0.0                  # Spatial variance along Y axis (m^2)
    cov_xy: float = 0.0                 # Spatial cross-covariance between X and Y (m^2)
    wall_crossings_rejected_count: int = 0
    resampled_this_step: bool = False
    observation_type: ObservationType = ObservationType.PDR_STEP
    map_version: str = "1.0.0"
    ekf_reference_x_m: Optional[float] = None
    ekf_reference_y_m: Optional[float] = None
    innovation_x_m: Optional[float] = None
    innovation_y_m: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def theta_deg(self) -> float:
        deg = math.degrees(self.theta_rad)
        return ((deg + 180.0) % 360.0) - 180.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_epoch_ms": self.timestamp_epoch_ms,
            "building_id": self.building_id,
            "floor_id": self.floor_id,
            "x_m": round(self.x_m, 3),
            "y_m": round(self.y_m, 3),
            "theta_rad": round(self.theta_rad, 4),
            "theta_deg": round(self.theta_deg, 2),
            "status": self.status.value,
            "particle_count": self.particle_count,
            "effective_sample_size": round(self.effective_sample_size, 1),
            "particle_spread_m": round(self.particle_spread_m, 3),
            "position_uncertainty_m": round(self.position_uncertainty_m, 3),
            "var_x": round(self.var_x, 4),
            "var_y": round(self.var_y, 4),
            "cov_xy": round(self.cov_xy, 4),
            "wall_crossings_rejected_count": self.wall_crossings_rejected_count,
            "resampled_this_step": self.resampled_this_step,
            "observation_type": self.observation_type.value,
            "map_version": self.map_version,
            "ekf_reference_x_m": round(self.ekf_reference_x_m, 3) if self.ekf_reference_x_m is not None else None,
            "ekf_reference_y_m": round(self.ekf_reference_y_m, 3) if self.ekf_reference_y_m is not None else None,
            "innovation_x_m": round(self.innovation_x_m, 3) if self.innovation_x_m is not None else None,
            "innovation_y_m": round(self.innovation_y_m, 3) if self.innovation_y_m is not None else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParticleFilterState":
        return cls(
            timestamp_epoch_ms=int(data["timestamp_epoch_ms"]),
            building_id=data["building_id"],
            floor_id=int(data["floor_id"]),
            x_m=float(data["x_m"]),
            y_m=float(data["y_m"]),
            theta_rad=float(data["theta_rad"]),
            status=FilterStatus(data["status"]),
            particle_count=int(data["particle_count"]),
            effective_sample_size=float(data["effective_sample_size"]),
            particle_spread_m=float(data["particle_spread_m"]),
            position_uncertainty_m=float(data["position_uncertainty_m"]),
            var_x=float(data.get("var_x", 0.0)),
            var_y=float(data.get("var_y", 0.0)),
            cov_xy=float(data.get("cov_xy", 0.0)),
            wall_crossings_rejected_count=int(data.get("wall_crossings_rejected_count", 0)),
            resampled_this_step=bool(data.get("resampled_this_step", False)),
            observation_type=ObservationType(data.get("observation_type", "PDR_STEP")),
            map_version=data.get("map_version", "1.0.0"),
            ekf_reference_x_m=data.get("ekf_reference_x_m"),
            ekf_reference_y_m=data.get("ekf_reference_y_m"),
            innovation_x_m=data.get("innovation_x_m"),
            innovation_y_m=data.get("innovation_y_m"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ParticleFilterSummary:
    """Consolidated summary metrics from a completed particle filter run."""
    total_states: int = 0
    pdr_steps_count: int = 0
    wifi_updates_count: int = 0
    landmark_updates_count: int = 0
    total_wall_crossings_rejected: int = 0
    resampling_events_count: int = 0
    mean_effective_sample_size: float = 0.0
    mean_particle_spread_m: float = 0.0
    mean_position_uncertainty_m: float = 0.0
    total_distance_traveled_m: float = 0.0
    mean_deviation_from_ekf_m: Optional[float] = None
    max_deviation_from_ekf_m: Optional[float] = None
    duration_seconds: float = 0.0
    status: FilterStatus = FilterStatus.TRACKING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_states": self.total_states,
            "pdr_steps_count": self.pdr_steps_count,
            "wifi_updates_count": self.wifi_updates_count,
            "landmark_updates_count": self.landmark_updates_count,
            "total_wall_crossings_rejected": self.total_wall_crossings_rejected,
            "resampling_events_count": self.resampling_events_count,
            "mean_effective_sample_size": round(self.mean_effective_sample_size, 2),
            "mean_particle_spread_m": round(self.mean_particle_spread_m, 3),
            "mean_position_uncertainty_m": round(self.mean_position_uncertainty_m, 3),
            "total_distance_traveled_m": round(self.total_distance_traveled_m, 3),
            "mean_deviation_from_ekf_m": round(self.mean_deviation_from_ekf_m, 3) if self.mean_deviation_from_ekf_m is not None else None,
            "max_deviation_from_ekf_m": round(self.max_deviation_from_ekf_m, 3) if self.max_deviation_from_ekf_m is not None else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "status": self.status.value,
        }

