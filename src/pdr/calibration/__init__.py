"""IndoorAI PDR Calibration, Ground-Truth Validation & Drift Analysis Package (Task 22)."""

from src.pdr.calibration.evaluator import PDREvaluator
from src.pdr.calibration.heading_evaluator import HeadingEvaluator, circular_difference_deg
from src.pdr.calibration.models import (
    CalibrationProfile,
    CalibrationStatus,
    ConfidenceEllipse,
    GroundTruthMeasurement,
    HeadingMetrics,
    PDREvaluationReport,
    PhonePlacement,
    ProfileType,
    ReferenceWaypoint,
    StepDetectionMetrics,
    TrajectoryDriftMetrics,
    TrialType,
    UncertaintyEvaluationMetrics,
    WalkingCondition,
    WalkingTrial,
)
from src.pdr.calibration.step_evaluator import StepDetectionEvaluator
from src.pdr.calibration.step_length_calibrator import WeinbergCalibrator
from src.pdr.calibration.storage import CalibrationStorage
from src.pdr.calibration.trajectory_evaluator import TrajectoryDriftEvaluator
from src.pdr.calibration.uncertainty_evaluator import (
    CHI2_2DOF_95,
    UncertaintyAuditor,
    compute_confidence_ellipse_2d,
    is_point_inside_ellipse,
)

__all__ = [
    "CalibrationProfile",
    "CalibrationStatus",
    "CalibrationStorage",
    "ConfidenceEllipse",
    "GroundTruthMeasurement",
    "HeadingEvaluator",
    "HeadingMetrics",
    "PDREvaluationReport",
    "PDREvaluator",
    "PhonePlacement",
    "ProfileType",
    "ReferenceWaypoint",
    "StepDetectionEvaluator",
    "StepDetectionMetrics",
    "TrajectoryDriftEvaluator",
    "TrajectoryDriftMetrics",
    "TrialType",
    "UncertaintyAuditor",
    "UncertaintyEvaluationMetrics",
    "WalkingCondition",
    "WalkingTrial",
    "WeinbergCalibrator",
    "circular_difference_deg",
    "compute_confidence_ellipse_2d",
    "is_point_inside_ellipse",
    "CHI2_2DOF_95",
]
