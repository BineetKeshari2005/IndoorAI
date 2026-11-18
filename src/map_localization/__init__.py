"""Map-Constrained Multimodal Localization Module (Task 27).

Combines PDR step propagation, building boundaries, solid wall collision checks,
Wi-Fi position likelihoods, and visual landmark observations using a sequential Monte Carlo
particle filter to prevent physically implausible indoor trajectories.
"""

from src.map_localization.evaluation import (
    MapLocalizationEvaluationReport,
    MapLocalizationEvaluator,
    TrajectoryComparisonMetrics,
    generate_controlled_benchmark_trajectory,
)
from src.map_localization.initialization import ParticleInitializer
from src.map_localization.landmark_observation import LandmarkMeasurementModel
from src.map_localization.map_constraints import MapConstraintEvaluator
from src.map_localization.measurement_model import WifiMeasurementModel
from src.map_localization.models import (
    FilterStatus,
    ObservationType,
    ParticleFilterConfig,
    ParticleFilterState,
    ParticleFilterSummary,
    ResampleAlgorithm,
)
from src.map_localization.motion_model import PdrMotionModel
from src.map_localization.particle import Particle, wrap_heading_rad
from src.map_localization.particle_filter import MapConstrainedParticleFilter
from src.map_localization.pipeline import MapLocalizationPipeline
from src.map_localization.resampling import Resampler

__all__ = [
    "FilterStatus",
    "ObservationType",
    "ResampleAlgorithm",
    "ParticleFilterConfig",
    "ParticleFilterState",
    "ParticleFilterSummary",
    "Particle",
    "wrap_heading_rad",
    "ParticleInitializer",
    "PdrMotionModel",
    "MapConstraintEvaluator",
    "WifiMeasurementModel",
    "LandmarkMeasurementModel",
    "Resampler",
    "MapConstrainedParticleFilter",
    "MapLocalizationEvaluator",
    "TrajectoryComparisonMetrics",
    "MapLocalizationEvaluationReport",
    "generate_controlled_benchmark_trajectory",
    "MapLocalizationPipeline",
]
