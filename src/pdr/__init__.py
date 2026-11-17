"""Pedestrian Dead Reckoning (PDR) kinematics, gravity compensation, and trajectory reconstruction (Task 21)."""

from src.pdr.config import PDRConfig
from src.pdr.coordinate_frames import (
    project_to_horizontal_plane,
    quaternion_from_android_rotation_vector,
    quaternion_to_euler_angles,
    quaternion_to_rotation_matrix,
    rotate_body_to_world,
    rotate_world_to_body,
)
from src.pdr.filtering import (
    butterworth_bandpass,
    butterworth_lowpass,
    exponential_smoothing_online,
)
from src.pdr.gravity import GravityCompResult, GravityCompensator
from src.pdr.heading import HeadingEstimator, StepHeading
from src.pdr.orientation import OrientationEstimator, OrientationResult
from src.pdr.pipeline import PDRPipeline
from src.pdr.reporting import PDRManifest, PDRReport, calculate_sha256
from src.pdr.step_detection import StepDetectionStats, StepDetector
from src.pdr.step_length import StepLengthEstimator
from src.pdr.trajectory import TrajectoryReconstructor, TrajectorySummary
from src.pdr.uncertainty import PDRUncertaintyPropagator, StepCovariance

__all__ = [
    "PDRConfig",
    "PDRPipeline",
    "GravityCompensator",
    "GravityCompResult",
    "OrientationEstimator",
    "OrientationResult",
    "HeadingEstimator",
    "StepHeading",
    "StepDetector",
    "StepDetectionStats",
    "StepLengthEstimator",
    "TrajectoryReconstructor",
    "TrajectorySummary",
    "PDRUncertaintyPropagator",
    "StepCovariance",
    "PDRReport",
    "PDRManifest",
    "quaternion_from_android_rotation_vector",
    "quaternion_to_rotation_matrix",
    "quaternion_to_euler_angles",
    "rotate_body_to_world",
    "rotate_world_to_body",
    "project_to_horizontal_plane",
    "butterworth_lowpass",
    "butterworth_bandpass",
    "exponential_smoothing_online",
    "calculate_sha256",
]
