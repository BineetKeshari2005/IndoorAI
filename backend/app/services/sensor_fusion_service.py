"""Service layer for Wi-Fi + PDR Extended Kalman Filter Sensor Fusion (Task 24)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.app.core.dataset_config import DatasetConfig, default_dataset_config
from backend.app.schemas.sensor_fusion import (
    FusionEvaluationRequest,
    FusionRunRequest,
    FusionStatusResponse,
)
from backend.app.services.dataset_registry import DatasetRegistry
from backend.app.services.pdr_service import PDRService
from backend.app.services.wifi_survey_service import WifiSurveyService
from src.core.logging import setup_logger
from src.sensor_fusion.coordinates import Pose2D
from src.sensor_fusion.evaluation import FusionComparisonReport, FusionEvaluator
from src.sensor_fusion.models import (
    FusedState,
    FusionConfig,
    FusionTrajectorySummary,
    PdrStepEvent,
    WifiMeasurementEvent,
)
from src.sensor_fusion.pipeline import FusionPipeline
from src.sensor_fusion.uncertainty import WifiUncertaintyEstimator

logger = setup_logger("SensorFusionService")


class SensorFusionService:
    """Manages Wi-Fi + PDR sensor fusion execution, artifact persistence, and comparative evaluation."""

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        registry: Optional[DatasetRegistry] = None,
        pdr_service: Optional[PDRService] = None,
        wifi_service: Optional[WifiSurveyService] = None,
    ) -> None:
        self.config = config or default_dataset_config
        self.registry = registry or DatasetRegistry(self.config)
        self.pdr_service = pdr_service or PDRService(self.config, self.registry)
        self.wifi_service = wifi_service or WifiSurveyService(self.config, self.registry)
        self.processed_base_dir = self.config.base_dir.parent / "processed" / "multimodal"
        self.processed_base_dir.mkdir(parents=True, exist_ok=True)

    def run_fusion(
        self,
        dataset_id: str,
        request: FusionRunRequest,
    ) -> Tuple[List[FusedState], FusionTrajectorySummary]:
        """Execute EKF fusion combining PDR step events and Wi-Fi measurements for a dataset."""
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_id}' not found in catalog.")

        # 1. Retrieve or run PDR steps
        pdr_steps = self._load_or_run_pdr_steps(dataset_id)
        if not pdr_steps:
            raise ValueError(f"No PDR steps available or detected for dataset '{dataset_id}'.")

        # 2. Retrieve and localize Wi-Fi observations
        wifi_events, candidate_map = self._load_and_localize_wifi_scans(
            dataset_id=dataset_id,
            site_id=request.site_id,
            version=request.radio_map_version,
            default_building_id=request.building_id,
            default_floor_id=request.floor_id,
        )

        # 3. Determine initial pose
        init_x = request.initial_x_m
        init_y = request.initial_y_m
        if init_x is None or init_y is None:
            if wifi_events:
                init_x = wifi_events[0].x_m
                init_y = wifi_events[0].y_m
            else:
                init_x = 0.0
                init_y = 0.0

        init_heading_rad = math.radians(request.initial_heading_deg or 0.0)
        initial_pose = Pose2D(
            x_m=float(init_x),
            y_m=float(init_y),
            theta_rad=init_heading_rad,
            building_id=request.building_id,
            floor_id=request.floor_id,
        )

        # 4. Instantiate pipeline with request config
        fusion_cfg = FusionConfig(
            chi2_threshold=request.chi2_threshold,
            enable_outlier_gating=request.enable_outlier_gating,
            default_wifi_sigma_m=request.default_wifi_sigma_m,
        )
        pipeline = FusionPipeline(config=fusion_cfg)

        fused_states, summary = pipeline.run(
            pdr_steps=pdr_steps,
            wifi_measurements=wifi_events,
            initial_pose=initial_pose,
            candidate_matches_by_scan=candidate_map,
            site_id=request.site_id,
        )

        # 5. Persist artifacts
        out_dir = self.processed_base_dir / dataset_id / "fusion"
        pipeline.save_artifacts(fused_states, summary, out_dir)

        return fused_states, summary

    def get_status(self, dataset_id: str) -> FusionStatusResponse:
        """Retrieve current fusion artifact status and summary for a dataset."""
        fusion_dir = self.processed_base_dir / dataset_id / "fusion"
        traj_file = fusion_dir / "fused_trajectory.parquet"
        sum_file = fusion_dir / "fusion_summary.json"

        if not traj_file.is_file() or not sum_file.is_file():
            return FusionStatusResponse(
                dataset_id=dataset_id,
                is_available=False,
                summary=None,
                artifact_paths={},
            )

        summary_data = None
        try:
            with open(sum_file, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read summary JSON: {e}")

        return FusionStatusResponse(
            dataset_id=dataset_id,
            is_available=True,
            summary=summary_data,
            artifact_paths={
                "trajectory_parquet": str(traj_file),
                "summary_json": str(sum_file),
            },
        )

    def get_trajectory(
        self, dataset_id: str, limit: int = 5000
    ) -> List[Dict[str, Any]]:
        """Retrieve sequential fused trajectory points from disk."""
        fusion_dir = self.processed_base_dir / dataset_id / "fusion"
        states_file = fusion_dir / "fusion_states.json"

        if states_file.is_file():
            try:
                with open(states_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data[:limit]
            except Exception as e:
                logger.warning(f"Failed to load fusion states: {e}")

        # Fallback to parquet
        traj_file = fusion_dir / "fused_trajectory.parquet"
        if traj_file.is_file():
            df = pd.read_parquet(traj_file)
            return df.head(limit).to_dict(orient="records")

        return []

    def evaluate(
        self,
        dataset_id: str,
        request: FusionEvaluationRequest,
    ) -> FusionComparisonReport:
        """Run comparative evaluation of PDR-only vs Wi-Fi-only vs Fused trajectory."""
        evaluator = FusionEvaluator(config=FusionConfig(chi2_threshold=request.chi2_threshold))

        if request.use_synthetic_fixture:
            # Generate strictly labeled mathematical synthetic scenario
            gt_points, pdr_steps, wifi_events = evaluator.generate_controlled_synthetic_trajectory()

            pipeline = FusionPipeline(config=FusionConfig(chi2_threshold=request.chi2_threshold))
            init_pose = Pose2D(x_m=0.0, y_m=0.0, theta_rad=0.0, building_id="BLDG-1", floor_id=1)
            fused_states, _ = pipeline.run(pdr_steps, wifi_events, initial_pose=init_pose)

            # Reconstruct PDR-only coordinates
            pdr_coords = []
            curr_x, curr_y = 0.0, 0.0
            for st in pdr_steps:
                curr_x += st.step_length_m * math.cos(st.heading_rad)
                curr_y += st.step_length_m * math.sin(st.heading_rad)
                pdr_coords.append((st.timestamp_epoch_ms, curr_x, curr_y))

            report = evaluator.evaluate_against_ground_truth(
                fused_states=fused_states,
                ground_truth=gt_points,
                pdr_states=pdr_coords,
                wifi_measurements=wifi_events,
                evaluation_id=f"SYNTHETIC-FIXTURE-{dataset_id[:8]}",
                is_synthetic_fixture=True,
                notes="Mathematically controlled synthetic fixture testing PDR drift correction and Wi-Fi outlier rejection.",
            )
            return report

        # Real survey dataset evaluation
        # Try to find existing fused states or run fusion
        status = self.get_status(dataset_id)
        if not status.is_available:
            self.run_fusion(
                dataset_id=dataset_id,
                request=FusionRunRequest(
                    site_id=request.site_id,
                    chi2_threshold=request.chi2_threshold,
                ),
            )

        traj_points = self.get_trajectory(dataset_id)
        if not traj_points:
            raise ValueError(f"No fused trajectory points found for dataset '{dataset_id}'.")

        # Convert back to FusedState objects
        from src.sensor_fusion.models import ConfidenceEllipse2D, FusionStatus
        fused_objs: List[FusedState] = []
        for p in traj_points:
            cov = np.array(p.get("state_covariance", np.eye(3)), dtype=np.float64)
            ellipse_d = p.get("ellipse", {})
            ellipse = ConfidenceEllipse2D(
                semi_major_axis_m=ellipse_d.get("semi_major_axis_m", 1.0),
                semi_minor_axis_m=ellipse_d.get("semi_minor_axis_m", 1.0),
                ellipse_angle_deg=ellipse_d.get("ellipse_angle_deg", 0.0),
                conservative_95_radius_m=ellipse_d.get("conservative_95_radius_m", 1.0),
            )
            fused_objs.append(
                FusedState(
                    timestamp_epoch_ms=int(p["timestamp_epoch_ms"]),
                    building_id=p.get("building_id", "BLDG-1"),
                    floor_id=int(p.get("floor_id", 1)),
                    x_m=float(p["x_m"]),
                    y_m=float(p["y_m"]),
                    theta_rad=float(p.get("theta_rad", 0.0)),
                    state_covariance=cov,
                    status=FusionStatus(p.get("status", "PDR_ONLY")),
                    ellipse=ellipse,
                    innovation=p.get("innovation"),
                    mahalanobis_dist_sq=p.get("mahalanobis_dist_sq"),
                    rejection_reason=p.get("rejection_reason"),
                )
            )

        # Check for reference annotations as ground truth
        annotations_csv = self.config.extracted_path / dataset_id / "annotations.csv"
        gt_points = []
        if annotations_csv.is_file():
            try:
                df_ann = pd.read_csv(annotations_csv)
                for _, r in df_ann.iterrows():
                    ts = int(r.get("timestamp_epoch_ms") or 0)
                    gx = float(r.get("x_m") or 0.0)
                    gy = float(r.get("y_m") or 0.0)
                    gt_points.append((ts, gx, gy))
            except Exception as e:
                logger.warning(f"Error reading annotations: {e}")

        if not gt_points or len(gt_points) < 2:
            # If no ground truth annotations exist, use synthetic benchmark fixture
            logger.info("No reference ground truth annotations found. Falling back to synthetic fixture evaluation.")
            return self.evaluate(
                dataset_id=dataset_id,
                request=FusionEvaluationRequest(
                    site_id=request.site_id,
                    chi2_threshold=request.chi2_threshold,
                    use_synthetic_fixture=True,
                ),
            )

        return evaluator.evaluate_against_ground_truth(
            fused_states=fused_objs,
            ground_truth=gt_points,
            evaluation_id=f"REAL-SURVEY-{dataset_id[:8]}",
            is_synthetic_fixture=False,
            notes=f"Evaluated against {len(gt_points)} verified survey landmark annotations.",
        )

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _load_or_run_pdr_steps(self, dataset_id: str) -> List[PdrStepEvent]:
        """Load PDR steps from parquet or run PDR pipeline if not yet computed."""
        steps_file = self.processed_base_dir / dataset_id / "pdr" / "pdr_steps.parquet"
        if not steps_file.is_file():
            logger.info(f"PDR steps not found for '{dataset_id}'. Running PDR kinematics pipeline...")
            try:
                self.pdr_service.run_pdr(dataset_id)
            except Exception as e:
                logger.error(f"Failed to run PDR kinematics for {dataset_id}: {e}")
                return []

        if not steps_file.is_file():
            return []

        try:
            df_steps = pd.read_parquet(steps_file)
            pdr_steps: List[PdrStepEvent] = []
            last_heading = 0.0

            for idx, r in df_steps.iterrows():
                ts = int(r.get("timestamp_ms") or 0)
                length = float(r.get("step_length_m") or 0.7)
                heading = float(r.get("heading_rad") or 0.0)
                delta_h = heading - last_heading
                last_heading = heading

                pdr_steps.append(
                    PdrStepEvent(
                        step_index=idx + 1,
                        timestamp_epoch_ms=ts,
                        step_length_m=length,
                        heading_rad=heading,
                        heading_delta_rad=delta_h,
                        sigma_length_m=float(r.get("sigma_step_length_m") or 0.08),
                        sigma_heading_rad=float(r.get("sigma_heading_rad") or 0.05),
                    )
                )
            return pdr_steps
        except Exception as e:
            logger.error(f"Failed to read PDR steps from {steps_file}: {e}")
            return []

    def _load_and_localize_wifi_scans(
        self,
        dataset_id: str,
        site_id: str,
        version: str = "v1",
        default_building_id: str = "BLDG-1",
        default_floor_id: int = 1,
    ) -> Tuple[List[WifiMeasurementEvent], Dict[int, List[Dict[str, Any]]]]:
        """Extract Wi-Fi sweeps from dataset and localize against site radio map."""
        wifi_records, _ = self.wifi_service._load_survey_records(dataset_id)
        if not wifi_records:
            return [], {}

        # Load radio map
        radio_map = self.wifi_service.get_radio_map(site_id, version)
        if not radio_map:
            logger.warning(
                f"Site radio map '{site_id}/{version}' not found on disk. Compiling default radio map..."
            )
            from backend.app.schemas.wifi_survey import RadioMapBuildRequest
            try:
                radio_map = self.wifi_service.build_radio_map(
                    dataset_id=dataset_id,
                    request=RadioMapBuildRequest(site_id=site_id, version=version),
                )
            except Exception as e:
                logger.error(f"Failed to auto-compile radio map for fusion: {e}")
                return [], {}

        from src.wifi_localization.knn_localizer import KnnLocalizer
        from src.wifi_localization.radio_map_builder import RadioMapBuilder

        builder = RadioMapBuilder(site_id=site_id)
        sweeps = builder.group_raw_records_into_sweeps(wifi_records)
        localizer = KnnLocalizer(radio_map=radio_map, k=3)

        wifi_events: List[WifiMeasurementEvent] = []
        candidate_map: Dict[int, List[Dict[str, Any]]] = {}

        for scan_idx, sweep in enumerate(sweeps, start=1):
            query = sweep.to_rssi_dict()
            pred = localizer.localize(query)

            cands_dict = [c.to_dict() for c in pred.candidate_matches]
            candidate_map[scan_idx] = cands_dict

            # Determine building and floor
            bldg = pred.predicted_building_id if pred.predicted_building_id != "UNKNOWN" else default_building_id
            floor = pred.predicted_floor_id

            wifi_events.append(
                WifiMeasurementEvent(
                    scan_index=scan_idx,
                    timestamp_epoch_ms=sweep.timestamp_epoch_ms,
                    x_m=pred.predicted_x_m,
                    y_m=pred.predicted_y_m,
                    building_id=bldg,
                    floor_id=floor,
                    confidence_score=pred.confidence,
                    matched_ap_count=pred.matched_ap_count,
                    total_query_aps=pred.total_query_aps,
                    is_fresh=True,
                    raw_prediction_id=f"PRED-{scan_idx:04d}",
                )
            )

        return wifi_events, candidate_map
