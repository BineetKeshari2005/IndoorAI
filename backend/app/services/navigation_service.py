"""Service layer for Indoor Navigation & Intelligent Route Planning (Task 28)."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.app.core.dataset_config import DatasetConfig, default_dataset_config
from backend.app.schemas.navigation import (
    CreateSessionRequest,
    DestinationListResponse,
    DestinationResponse,
    NavigationEvaluateRequest,
    NavigationEvaluateResponse,
    NavigationProgressRequest,
    NavigationProgressResponse,
    NavigationSessionResponse,
    PlannedRouteResponse,
    RoutePlanRequest,
)
from backend.app.services.dataset_registry import DatasetRegistry
from backend.app.services.mapping_service import MappingService
from src.core.logging import setup_logger
from src.mapping.evaluation import generate_controlled_synthetic_building_map
from src.mapping.models import BuildingMap
from src.navigation.destination import Destination, DestinationCategory
from src.navigation.evaluation import NavigationEvaluationReport, NavigationEvaluator
from src.navigation.models import (
    NavigationProgress,
    NavigationSession,
    PlannedRoute,
    RoutePreference,
    RoutingAlgorithm,
)
from src.navigation.pipeline import NavigationPipeline

logger = setup_logger("NavigationService")


class NavigationService:
    """Manages indoor routing operations, destination search, live sessions, and evaluation."""

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        registry: Optional[DatasetRegistry] = None,
        mapping_service: Optional[MappingService] = None,
    ) -> None:
        self.config = config or default_dataset_config
        self.registry = registry or DatasetRegistry(self.config)
        self.mapping_service = mapping_service or MappingService(self.config)

        self.processed_base_dir = self.config.base_dir.parent / "processed" / "multimodal"
        self.processed_base_dir.mkdir(parents=True, exist_ok=True)

        # Cache of active pipelines by building_id
        self._pipelines: Dict[str, NavigationPipeline] = {}
        # Cache of planned routes by route_id
        self._route_cache: Dict[str, PlannedRoute] = {}

    def get_pipeline(self, building_id: str, dataset_id: str = "default") -> NavigationPipeline:
        """Retrieves or initializes a NavigationPipeline for a building map."""
        if building_id in self._pipelines:
            return self._pipelines[building_id]

        # Retrieve building map from mapping service
        bmap = self.mapping_service.get_building_map(building_id)
        if not bmap:
            logger.info(f"Building map '{building_id}' not found. Initializing synthetic benchmark building map.")
            bmap = generate_controlled_synthetic_building_map(building_id=building_id)
            try:
                self.mapping_service.storage.save_building_map(bmap)
            except Exception as e:
                logger.debug(f"Could not persist synthetic building map: {e}")

        sess_storage = self.processed_base_dir / dataset_id / "navigation"
        sess_storage.mkdir(parents=True, exist_ok=True)

        pipeline = NavigationPipeline(
            building_map=bmap,
            storage_dir=sess_storage,
        )
        self._pipelines[building_id] = pipeline
        return pipeline

    def list_destinations(
        self,
        dataset_id: str,
        building_id: str = "SYNTH_BLDG_01",
        floor_id: Optional[int] = None,
        category: Optional[str] = None,
        query: str = "",
        limit: int = 20,
    ) -> DestinationListResponse:
        """Searches or lists verified destinations within a building."""
        pipeline = self.get_pipeline(building_id=building_id, dataset_id=dataset_id)
        cat_enum = DestinationCategory.from_string(category) if category else None

        results = pipeline.search_destinations(
            query=query,
            floor_id=floor_id,
            category=cat_enum,
            limit=limit,
        )

        resp_items = [DestinationResponse(**d.to_dict()) for d in results]
        return DestinationListResponse(
            total_destinations=len(resp_items),
            building_id=building_id,
            destinations=resp_items,
        )

    def plan_route(
        self,
        dataset_id: str,
        req: RoutePlanRequest,
    ) -> PlannedRoute:
        """Plans and validates an optimal route based on origin, destination, and preferences."""
        pipeline = self.get_pipeline(building_id=req.building_id, dataset_id=dataset_id)

        # Coordinate origin
        origin_coords = None
        if req.origin_x_m is not None and req.origin_y_m is not None:
            origin_coords = (req.origin_x_m, req.origin_y_m, req.origin_floor_id or 0)

        # Coordinate destination
        dest_coords = None
        if req.destination_x_m is not None and req.destination_y_m is not None:
            dest_coords = (req.destination_x_m, req.destination_y_m, req.destination_floor_id or 0)

        alg = RoutingAlgorithm.DIJKSTRA if req.algorithm.upper() == "DIJKSTRA" else RoutingAlgorithm.ASTAR
        pref = RoutePreference(req.preference) if req.preference in [p.value for p in RoutePreference] else RoutePreference.SHORTEST

        route, err = pipeline.plan_route(
            origin_node_id=req.origin_node_id,
            origin_coords=origin_coords,
            destination_id=req.destination_id,
            destination_coords=dest_coords,
            algorithm=alg,
            preference=pref,
        )

        if not route:
            raise ValueError(err or "NO_VALID_ROUTE: Could not generate valid indoor route.")

        self._route_cache[route.route_id] = route
        return route

    def get_route(self, dataset_id: str, route_id: str) -> Optional[PlannedRoute]:
        """Retrieves a cached planned route."""
        return self._route_cache.get(route_id)

    def create_session(
        self,
        dataset_id: str,
        req: CreateSessionRequest,
    ) -> NavigationSession:
        """Instantiates an active navigation session."""
        route: Optional[PlannedRoute] = None
        if req.route_id:
            route = self.get_route(dataset_id=dataset_id, route_id=req.route_id)

        if not route:
            # Plan route inline
            bldg = req.building_id or "SYNTH_BLDG_01"
            dest = req.destination_id or "DEST-RM-room_102"
            orig = req.origin_node_id or "NODE-RM-room_101"
            plan_req = RoutePlanRequest(
                building_id=bldg,
                origin_node_id=orig,
                destination_id=dest,
                preference=req.preference,
            )
            route = self.plan_route(dataset_id=dataset_id, req=plan_req)

        pipeline = self.get_pipeline(building_id=route.building_id, dataset_id=dataset_id)
        session = pipeline.start_session(route=route, dataset_id=dataset_id)
        return session

    def get_session(self, dataset_id: str, session_id: str) -> Optional[NavigationSession]:
        """Retrieves navigation session by ID across active pipelines."""
        for pipeline in self._pipelines.values():
            sess = pipeline.sessions.get_session(session_id)
            if sess:
                return sess
        return None

    def update_progress(
        self,
        dataset_id: str,
        session_id: str,
        req: NavigationProgressRequest,
    ) -> Tuple[NavigationProgress, Optional[PlannedRoute], str]:
        """Feeds a localization position update to the active navigation session."""
        session = self.get_session(dataset_id=dataset_id, session_id=session_id)
        if not session:
            raise ValueError(f"Navigation session '{session_id}' not found.")

        pipeline = self.get_pipeline(building_id=session.building_id, dataset_id=dataset_id)
        prog, new_r, msg = pipeline.update_progress(
            session_id=session_id,
            x_m=req.x_m,
            y_m=req.y_m,
            floor_id=req.floor_id,
            localization_status=req.localization_status,
            timestamp_epoch_ms=req.timestamp_epoch_ms,
        )

        if not prog:
            raise ValueError(msg or "Failed to update navigation progress.")

        if new_r:
            self._route_cache[new_r.route_id] = new_r

        return prog, new_r, msg or "OK"

    def complete_session(self, dataset_id: str, session_id: str) -> NavigationSession:
        """Marks navigation session as completed."""
        session = self.get_session(dataset_id=dataset_id, session_id=session_id)
        if not session:
            raise ValueError(f"Navigation session '{session_id}' not found.")

        pipeline = self.get_pipeline(building_id=session.building_id, dataset_id=dataset_id)
        res = pipeline.sessions.complete_session(session_id)
        return res or session

    def evaluate(
        self,
        dataset_id: str,
        req: NavigationEvaluateRequest,
    ) -> NavigationEvaluationReport:
        """Executes comparative routing benchmark."""
        pipeline = self.get_pipeline(building_id=req.building_id, dataset_id=dataset_id)
        report = pipeline.evaluate(evaluation_id=f"NAV-EVAL-{dataset_id[:8]}")
        return report
