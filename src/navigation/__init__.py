"""Indoor Navigation & Intelligent Route Planning Module (Task 28).

Provides graph-based shortest path calculation (Dijkstra and A*), multi-floor transitions,
accessibility constraint enforcement, turn-by-turn instruction compilation,
live progress tracking, and automatic dynamic rerouting.
"""

from src.navigation.accessibility import AccessibilityAuditor, AccessibilityAuditStatus
from src.navigation.destination import Destination, DestinationCategory, DestinationManager
from src.navigation.evaluation import (
    NavigationEvaluationReport,
    NavigationEvaluator,
    NavigationSimulationMetrics,
    RouteEvaluationMetrics,
)
from src.navigation.instruction_generator import InstructionGenerator
from src.navigation.models import (
    DestinationCategory,
    InstructionType,
    NavigationInstruction,
    NavigationProgress,
    NavigationSession,
    NavigationStatus,
    NavigationWaypoint,
    PlannedRoute,
    RerouteEvent,
    RouteCostWeights,
    RoutePreference,
    RoutingAlgorithm,
)
from src.navigation.navigation_session import NavigationSessionManager
from src.navigation.path_geometry import PathGeometryBuilder
from src.navigation.pipeline import NavigationPipeline
from src.navigation.progress_tracker import ProgressTracker
from src.navigation.rerouting import ReroutingEngine
from src.navigation.route_graph import RouteGraph
from src.navigation.route_planner import RoutePlanner
from src.navigation.route_validator import RouteValidationResult, RouteValidator
from src.navigation.turn_detection import TurnDetectionThresholds, TurnDetector

__all__ = [
    "AccessibilityAuditor",
    "AccessibilityAuditStatus",
    "Destination",
    "DestinationCategory",
    "DestinationManager",
    "InstructionGenerator",
    "InstructionType",
    "NavigationEvaluationReport",
    "NavigationEvaluator",
    "NavigationInstruction",
    "NavigationPipeline",
    "NavigationProgress",
    "NavigationSession",
    "NavigationSessionManager",
    "NavigationSimulationMetrics",
    "NavigationStatus",
    "NavigationWaypoint",
    "PathGeometryBuilder",
    "PlannedRoute",
    "ProgressTracker",
    "RerouteEvent",
    "ReroutingEngine",
    "RouteCostWeights",
    "RouteEvaluationMetrics",
    "RouteGraph",
    "RoutePlanner",
    "RoutePreference",
    "RouteValidationResult",
    "RouteValidator",
    "RoutingAlgorithm",
    "TurnDetectionThresholds",
    "TurnDetector",
]
