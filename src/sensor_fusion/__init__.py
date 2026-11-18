"""IndoorAI Sensor Fusion Module: Extended Kalman Filter (EKF) combining Wi-Fi & PDR (Task 24)."""

from src.sensor_fusion.coordinates import (
    CoordinateFrameMetadata,
    Pose2D,
    compass_azimuth_to_local_heading_rad,
    local_heading_rad_to_compass_azimuth,
    wrap_angle_deg,
    wrap_angle_rad,
)
from src.sensor_fusion.ekf import IndoorEKF, enforce_psd_covariance
from src.sensor_fusion.evaluation import (
    FusionComparisonReport,
    FusionEvaluator,
    TrajectoryEvaluationMetrics,
)
from src.sensor_fusion.models import (
    CHI2_2DOF_95,
    ConfidenceEllipse2D,
    FusedState,
    FusionConfig,
    FusionStatus,
    FusionTrajectorySummary,
    PdrStepEvent,
    UncertaintyType,
    WifiMeasurementEvent,
)
from src.sensor_fusion.pipeline import FusionPipeline
from src.sensor_fusion.uncertainty import WifiUncertaintyEstimator

__all__ = [
    "CoordinateFrameMetadata",
    "Pose2D",
    "wrap_angle_rad",
    "wrap_angle_deg",
    "compass_azimuth_to_local_heading_rad",
    "local_heading_rad_to_compass_azimuth",
    "CHI2_2DOF_95",
    "IndoorEKF",
    "enforce_psd_covariance",
    "ConfidenceEllipse2D",
    "FusedState",
    "FusionConfig",
    "FusionStatus",
    "FusionTrajectorySummary",
    "PdrStepEvent",
    "WifiMeasurementEvent",
    "UncertaintyType",
    "WifiUncertaintyEstimator",
    "FusionPipeline",
    "TrajectoryEvaluationMetrics",
    "FusionComparisonReport",
    "FusionEvaluator",
]
