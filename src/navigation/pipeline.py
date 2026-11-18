"""End-to-end execution pipeline for indoor destination search, routing, and session management (Task 28)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import setup_logger
from src.mapping.models import BuildingMap
from src.navigation.destination import Destination, DestinationManager
from src.navigation.evaluation import NavigationEvaluationReport, NavigationEvaluator
from src.navigation.models import (
    DestinationCategory,
    NavigationProgress,
    NavigationSession,
    PlannedRoute,
    RouteCostWeights,
    RoutePreference,
    RoutingAlgorithm,
)
from src.navigation.navigation_session import NavigationSessionManager
from src.navigation.route_planner import RoutePlanner

logger = setup_logger("NavigationPipeline")


class NavigationPipeline:
    """Unified orchestration pipeline for indoor navigation operations."""

    def __init__(
        self,
        building_map: BuildingMap,
        weights: Optional[RouteCostWeights] = None,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self.building_map: BuildingMap = building_map
        self.weights: RouteCostWeights = weights or RouteCostWeights()
        self.destinations: DestinationManager = DestinationManager(building_map)
        self.planner: RoutePlanner = RoutePlanner(
            building_map=building_map,
            weights=self.weights,
            destination_manager=self.destinations,
        )
        self.sessions: NavigationSessionManager = NavigationSessionManager(
            planner=self.planner,
            storage_dir=storage_dir,
        )

    def set_building_map(self, building_map: BuildingMap) -> None:
        """Updates active building map and re-indexes destinations and routing graphs."""
        self.building_map = building_map
        self.destinations.load_from_building_map(building_map)
        self.planner.set_building_map(building_map)

    def search_destinations(
        self,
        query: str = "",
        floor_id: Optional[int] = None,
        category: Optional[DestinationCategory] = None,
        limit: int = 15,
    ) -> List[Destination]:
        """Searches verified navigation targets across the active building."""
        return self.destinations.search_destinations(
            query=query,
            floor_id=floor_id,
            category=category,
            limit=limit,
        )

    def plan_route(
        self,
        origin_node_id: Optional[str] = None,
        origin_coords: Optional[Tuple[float, float, int]] = None,
        destination_id: Optional[str] = None,
        destination_coords: Optional[Tuple[float, float, int]] = None,
        algorithm: RoutingAlgorithm = RoutingAlgorithm.ASTAR,
        preference: RoutePreference = RoutePreference.SHORTEST,
    ) -> Tuple[Optional[PlannedRoute], Optional[str]]:
        """Plans and validates an indoor walking route."""
        return self.planner.plan_route(
            origin_node_id=origin_node_id,
            origin_coords=origin_coords,
            destination_id=destination_id,
            destination_coords=destination_coords,
            algorithm=algorithm,
            preference=preference,
        )

    def start_session(
        self,
        route: PlannedRoute,
        dataset_id: str = "default",
    ) -> NavigationSession:
        """Starts an active navigation session for a planned route."""
        return self.sessions.create_session(route=route, dataset_id=dataset_id)

    def update_progress(
        self,
        session_id: str,
        x_m: float,
        y_m: float,
        floor_id: int,
        localization_status: str = "TRACKING",
        timestamp_epoch_ms: Optional[int] = None,
    ) -> Tuple[Optional[NavigationProgress], Optional[PlannedRoute], Optional[str]]:
        """Updates pedestrian location and handles automatic progress / reroutes."""
        return self.sessions.update_progress(
            session_id=session_id,
            x_m=x_m,
            y_m=y_m,
            floor_id=floor_id,
            localization_status=localization_status,
            timestamp_epoch_ms=timestamp_epoch_ms,
        )

    def evaluate(self, evaluation_id: str = "EVAL-NAV") -> NavigationEvaluationReport:
        """Runs comparative benchmark evaluation."""
        return NavigationEvaluator.run_navigation_evaluation(
            building_map=self.building_map,
            evaluation_id=evaluation_id,
        )
