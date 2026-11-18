"""Indoor spatial mapping: 2D geometry, domain models, topological graphs, validation, and storage."""

from src.mapping.evaluation import (
    MapEvaluationMetrics,
    MapEvaluator,
    generate_controlled_synthetic_building_map,
)
from src.mapping.geometry import (
    BoundingBox2D,
    LineSegment2D,
    Point2D,
    Polygon2D,
)
from src.mapping.map_builder import MapBuilder
from src.mapping.map_export import MapExporter
from src.mapping.map_storage import MapStorage
from src.mapping.map_validator import (
    MapValidationError,
    MapValidationReport,
    MapValidator,
    ValidationSeverity,
)
from src.mapping.models import (
    AccessibilityType,
    BuildingMap,
    ConnectionType,
    Corridor,
    Door,
    DoorType,
    FloorMap,
    MapVerificationStatus,
    Room,
    RoomCategory,
    SpatialConnection,
    VerticalTransition,
)
from src.mapping.pipeline import MappingPipeline
from src.mapping.topology import (
    IndoorTopologyGraph,
    TopologyEdge,
    TopologyNode,
    TopologyNodeType,
)

__all__ = [
    # Geometry
    "Point2D",
    "LineSegment2D",
    "BoundingBox2D",
    "Polygon2D",
    # Models & Enums
    "MapVerificationStatus",
    "RoomCategory",
    "ConnectionType",
    "AccessibilityType",
    "DoorType",
    "Room",
    "Corridor",
    "Door",
    "VerticalTransition",
    "SpatialConnection",
    "FloorMap",
    "BuildingMap",
    # Topology
    "TopologyNodeType",
    "TopologyNode",
    "TopologyEdge",
    "IndoorTopologyGraph",
    # Validation
    "ValidationSeverity",
    "MapValidationError",
    "MapValidationReport",
    "MapValidator",
    # Storage & Export
    "MapStorage",
    "MapExporter",
    # Builder & Pipeline
    "MapBuilder",
    "MappingPipeline",
    # Evaluation & Fixtures
    "MapEvaluationMetrics",
    "MapEvaluator",
    "generate_controlled_synthetic_building_map",
]
