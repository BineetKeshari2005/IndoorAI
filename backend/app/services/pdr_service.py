"""Service layer managing Pedestrian Dead Reckoning (PDR) execution and queries (Task 21)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.core.dataset_config import DatasetConfig, default_dataset_config
from backend.app.schemas.pdr import PDRRunRequest, StepRecord, TrajectoryPoint
from backend.app.services.dataset_registry import DatasetRegistry
from src.core.logging import setup_logger
from src.pdr.config import PDRConfig
from src.pdr.pipeline import PDRPipeline
from src.pdr.reporting import PDRManifest, PDRReport

logger = setup_logger("PDRService")


class PDRService:
    """Manages PDR trajectory computation, step retrieval, and artifact status for survey datasets."""

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        registry: Optional[DatasetRegistry] = None
    ):
        self.config = config or default_dataset_config
        self.registry = registry or DatasetRegistry(self.config)
        self.output_base_dir = self.config.base_dir.parent / "processed" / "multimodal"
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def run_pdr(
        self,
        dataset_id: str,
        params: Optional[PDRRunRequest] = None
    ) -> Tuple[PDRReport, PDRManifest]:
        """Executes the PDR pipeline on a preprocessed survey dataset."""
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_id}' not found in registry.")

        dataset_processed_dir = self.output_base_dir / dataset_id
        if not dataset_processed_dir.is_dir():
            raise FileNotFoundError(
                f"Processed directory for '{dataset_id}' not found. Please run Task 20 preprocessing first."
            )

        # Build pipeline configuration
        pipeline_config = PDRConfig(
            base_output_dir=self.output_base_dir
        )
        if params is not None:
            pipeline_config.step_detection_signal = params.step_detection_signal
            pipeline_config.weinberg_k = params.weinberg_k
            pipeline_config.min_step_interval_s = params.min_step_interval_s
            pipeline_config.min_peak_prominence_mps2 = params.min_peak_prominence_mps2
            pipeline_config.initial_heading_rad = params.initial_heading_rad
            pipeline_config.origin_x_m = params.origin_x_m
            pipeline_config.origin_y_m = params.origin_y_m

        pipeline = PDRPipeline(pipeline_config)
        report, manifest = pipeline.run(dataset_id=dataset_id, processed_dir=dataset_processed_dir)
        return report, manifest

    def get_status(self, dataset_id: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Retrieves latest PDR status, report, and manifest for a dataset."""
        pdr_dir = self.output_base_dir / dataset_id / "pdr"
        report_file = pdr_dir / "pdr_report.json"
        manifest_file = pdr_dir / "pdr_manifest.json"

        if not report_file.is_file():
            return False, None, None

        report_data = None
        manifest_data = None
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        except Exception as e:
            logger.error("Failed to read PDR report for %s: %s", dataset_id, e)

        if manifest_file.is_file():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception as e:
                logger.error("Failed to read PDR manifest for %s: %s", dataset_id, e)

        return True, report_data, manifest_data

    def get_trajectory(self, dataset_id: str) -> List[TrajectoryPoint]:
        """Loads and returns 2D relative trajectory points."""
        traj_file = self.output_base_dir / dataset_id / "pdr" / "relative_trajectory.parquet"
        if not traj_file.is_file():
            return []

        try:
            df = pd.read_parquet(traj_file)
            points: List[TrajectoryPoint] = []
            for row in df.to_dict(orient="records"):
                points.append(
                    TrajectoryPoint(
                        point_id=str(row.get("point_id", "")),
                        step_id=str(row.get("step_id", "")),
                        timestamp_ms=int(row.get("timestamp_ms", 0)),
                        rel_x_m=float(row.get("rel_x_m", 0.0)),
                        rel_y_m=float(row.get("rel_y_m", 0.0)),
                        heading_deg=float(row.get("heading_deg", 0.0)),
                        step_length_m=float(row.get("step_length_m", 0.0)),
                        cum_distance_m=float(row.get("cum_distance_m", 0.0)),
                        pos_uncertainty_radius_m=float(row.get("pos_uncertainty_radius_m", 0.0)),
                        confidence=float(row.get("confidence", 0.0)),
                        is_reliable=bool(row.get("is_reliable", True)),
                    )
                )
            return points
        except Exception as e:
            logger.error("Failed to read trajectory Parquet for %s: %s", dataset_id, e)
            return []

    def get_steps(self, dataset_id: str) -> List[StepRecord]:
        """Loads and returns detected human step records."""
        steps_file = self.output_base_dir / dataset_id / "pdr" / "detected_steps.parquet"
        if not steps_file.is_file():
            return []

        try:
            df = pd.read_parquet(steps_file)
            steps: List[StepRecord] = []
            for row in df.to_dict(orient="records"):
                steps.append(
                    StepRecord(
                        step_id=str(row.get("step_id", "")),
                        timestamp_ms=int(row.get("timestamp_ms", 0)),
                        peak_accel_mps2=float(row.get("peak_accel_mps2", 0.0)),
                        valley_accel_mps2=float(row.get("valley_accel_mps2", 0.0)),
                        step_interval_ms=float(row.get("step_interval_ms", 0.0)),
                        step_frequency_hz=float(row.get("step_frequency_hz", 0.0)),
                        step_length_m=float(row.get("step_length_m", 0.0)),
                        confidence=float(row.get("confidence", 0.0)),
                        validation_status=str(row.get("validation_status", "VALID")),
                    )
                )
            return steps
        except Exception as e:
            logger.error("Failed to read steps Parquet for %s: %s", dataset_id, e)
            return []
