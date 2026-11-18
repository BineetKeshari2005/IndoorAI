"""Domain schemas and dataclasses for Wi-Fi + PDR Extended Kalman Filter Sensor Fusion (Task 24)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.sensor_fusion.coordinates import wrap_angle_deg, wrap_angle_rad

# Chi-Square 95% Quantile for 2 Degrees of Freedom: -2 * ln(1 - 0.95) ≈ 5.99146
CHI2_2DOF_95 = 5.9914645471


class FusionStatus(str, Enum):
    """Operational state of the fusion engine for a sequential update."""
    INITIALIZING = "INITIALIZING"
    PDR_ONLY = "PDR_ONLY"
    WIFI_CORRECTED = "WIFI_CORRECTED"
    WIFI_REJECTED = "WIFI_REJECTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_COORDINATE_FRAME = "INVALID_COORDINATE_FRAME"


class UncertaintyType(str, Enum):
    """Classification of measurement uncertainty provenance."""
    EMPIRICAL_SURVEY_RESIDUALS = "EMPIRICAL_SURVEY_RESIDUALS"
    CANDIDATE_DISPERSION = "CANDIDATE_DISPERSION"
    CONSERVATIVE_HEURISTIC = "CONSERVATIVE_HEURISTIC"
    DEFAULT_PRIOR = "DEFAULT_PRIOR"


@dataclass
class ConfidenceEllipse2D:
    """Rigorous 2D Gaussian 95% Confidence Ellipse parameters."""
    semi_major_axis_m: float
    semi_minor_axis_m: float
    ellipse_angle_deg: float
    conservative_95_radius_m: float

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)

    @classmethod
    def from_covariance(cls, cov_2x2: np.ndarray, chi2_val: float = CHI2_2DOF_95) -> ConfidenceEllipse2D:
        """Compute exact 2D confidence ellipse from 2x2 position covariance."""
        var_x = float(cov_2x2[0, 0])
        var_y = float(cov_2x2[1, 1])
        var_xy = float(cov_2x2[0, 1])

        mean_diag = (var_x + var_y) / 2.0
        half_diff = (var_x - var_y) / 2.0
        delta = math.sqrt(max(0.0, half_diff**2 + var_xy**2))
        lambda_1 = max(0.0, mean_diag + delta)
        lambda_2 = max(0.0, mean_diag - delta)

        semi_major = math.sqrt(chi2_val * lambda_1)
        semi_minor = math.sqrt(chi2_val * lambda_2)
        angle_deg = math.degrees(0.5 * math.atan2(2.0 * var_xy, var_x - var_y))

        return cls(
            semi_major_axis_m=round(semi_major, 3),
            semi_minor_axis_m=round(semi_minor, 3),
            ellipse_angle_deg=round(angle_deg, 2),
            conservative_95_radius_m=round(semi_major, 3),
        )


@dataclass
class PdrStepEvent:
    """Discrete walking step detected by PDR kinematics."""
    step_index: int
    timestamp_epoch_ms: int
    step_length_m: float
    heading_rad: float
    heading_delta_rad: float = 0.0
    sigma_length_m: float = 0.08
    sigma_heading_rad: float = 0.05
    has_sensor_gap: bool = False
    is_magnetic_disturbance: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class WifiMeasurementEvent:
    """Absolute position observation derived from Wi-Fi fingerprinting."""
    scan_index: int
    timestamp_epoch_ms: int
    x_m: float
    y_m: float
    building_id: str
    floor_id: int
    covariance: Optional[np.ndarray] = None  # 2x2
    uncertainty_type: UncertaintyType = UncertaintyType.CONSERVATIVE_HEURISTIC
    confidence_score: float = 0.0
    matched_ap_count: int = 0
    total_query_aps: int = 0
    is_fresh: bool = True
    raw_prediction_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "scan_index": self.scan_index,
            "timestamp_epoch_ms": self.timestamp_epoch_ms,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "building_id": self.building_id,
            "floor_id": self.floor_id,
            "uncertainty_type": self.uncertainty_type.value,
            "confidence_score": self.confidence_score,
            "matched_ap_count": self.matched_ap_count,
            "total_query_aps": self.total_query_aps,
            "is_fresh": self.is_fresh,
            "raw_prediction_id": self.raw_prediction_id,
        }
        if self.covariance is not None:
            d["covariance"] = self.covariance.tolist()
        return d


@dataclass
class FusedState:
    """Unified state estimated at a discrete point in time."""
    timestamp_epoch_ms: int
    building_id: str
    floor_id: int
    x_m: float
    y_m: float
    theta_rad: float
    state_covariance: np.ndarray  # 3x3
    status: FusionStatus
    ellipse: ConfidenceEllipse2D
    step_event: Optional[Dict[str, Any]] = None
    wifi_event: Optional[Dict[str, Any]] = None
    innovation: Optional[List[float]] = None  # [dx, dy]
    mahalanobis_dist_sq: Optional[float] = None
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def theta_deg(self) -> float:
        return wrap_angle_deg(float(np.degrees(self.theta_rad)))

    @property
    def pos_covariance_2x2(self) -> np.ndarray:
        return self.state_covariance[0:2, 0:2].copy()

    @property
    def sigma_x_m(self) -> float:
        return math.sqrt(max(0.0, float(self.state_covariance[0, 0])))

    @property
    def sigma_y_m(self) -> float:
        return math.sqrt(max(0.0, float(self.state_covariance[1, 1])))

    @property
    def sigma_theta_rad(self) -> float:
        return math.sqrt(max(0.0, float(self.state_covariance[2, 2])))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_epoch_ms": self.timestamp_epoch_ms,
            "building_id": self.building_id,
            "floor_id": self.floor_id,
            "x_m": round(self.x_m, 3),
            "y_m": round(self.y_m, 3),
            "theta_rad": round(self.theta_rad, 4),
            "theta_deg": round(self.theta_deg, 2),
            "sigma_x_m": round(self.sigma_x_m, 3),
            "sigma_y_m": round(self.sigma_y_m, 3),
            "sigma_theta_deg": round(float(np.degrees(self.sigma_theta_rad)), 2),
            "state_covariance": self.state_covariance.tolist(),
            "status": self.status.value,
            "ellipse": self.ellipse.to_dict(),
            "step_event": self.step_event,
            "wifi_event": self.wifi_event,
            "innovation": [round(v, 3) for v in self.innovation] if self.innovation else None,
            "mahalanobis_dist_sq": round(self.mahalanobis_dist_sq, 3) if self.mahalanobis_dist_sq is not None else None,
            "rejection_reason": self.rejection_reason,
            "metadata": self.metadata,
        }


@dataclass
class FusionConfig:
    """Tunable hyperparameters and gating thresholds for Extended Kalman Filter fusion."""
    chi2_threshold: float = CHI2_2DOF_95  # 5.99146 (95% Chi-Square for 2-DOF)
    initial_pos_uncertainty_m: float = 2.0
    initial_heading_uncertainty_rad: float = 0.174  # ~10 degrees
    default_wifi_sigma_m: float = 4.0
    min_wifi_sigma_m: float = 1.0
    min_matched_aps: int = 2
    min_confidence_score: float = 0.25
    max_wifi_latency_ms: int = 30000  # 30 seconds
    enable_outlier_gating: bool = True
    require_fresh_scans: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class FusionTrajectorySummary:
    """Summary metrics of a fused sensor-fusion trajectory."""
    total_points: int = 0
    pdr_steps_count: int = 0
    wifi_scans_total: int = 0
    wifi_scans_accepted: int = 0
    wifi_scans_rejected: int = 0
    acceptance_rate: float = 0.0
    total_distance_m: float = 0.0
    duration_s: float = 0.0
    mean_uncertainty_radius_m: float = 0.0
    final_x_m: float = 0.0
    final_y_m: float = 0.0
    rejection_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
