"""Pedestrian Dead Reckoning (PDR) pipeline orchestrator (Task 21)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.core.logging import setup_logger
from src.pdr.config import PDRConfig
from src.pdr.gravity import GravityCompResult, GravityCompensator
from src.pdr.orientation import OrientationEstimator, OrientationResult
from src.pdr.reporting import PDRManifest, PDRReport, calculate_sha256
from src.pdr.step_detection import StepDetectionStats, StepDetector
from src.pdr.step_length import StepLengthEstimator
from src.pdr.trajectory import TrajectoryReconstructor, TrajectorySummary

logger = setup_logger("PDRPipeline")


class PDRPipeline:
    """Orchestrates gravity separation, orientation estimation, step detection, and 2D relative trajectory reconstruction."""

    def __init__(self, config: Optional[PDRConfig] = None):
        self.config = config or PDRConfig()

    def run(self, dataset_id: str, processed_dir: Optional[Path] = None) -> Tuple[PDRReport, PDRManifest]:
        """Executes PDR movement analysis on a preprocessed survey dataset.

        Args:
            dataset_id: Identifier of the dataset.
            processed_dir: Path to directory containing Task 20 outputs (default: data/processed/multimodal/<dataset_id>).

        Returns:
            Tuple of (PDRReport, PDRManifest).
        """
        start_time = time.perf_counter()
        target_processed_dir = (processed_dir or (self.config.base_output_dir / dataset_id)).resolve()
        output_pdr_dir = target_processed_dir / "pdr"
        output_pdr_dir.mkdir(parents=True, exist_ok=True)

        warnings: List[str] = []
        errors: List[str] = []

        # 1. Load sensor data from Task 20 outputs
        sensor_parquet = target_processed_dir / "sensors" / "sensor_cleaned.parquet"
        sensor_df = pd.DataFrame()

        input_checksums: Dict[str, str] = {}

        if sensor_parquet.is_file():
            try:
                sensor_df = pd.read_parquet(sensor_parquet)
                input_checksums["sensors/sensor_cleaned.parquet"] = calculate_sha256(sensor_parquet)
            except Exception as e:
                errors.append(f"Failed to read sensor Parquet file: {e}")
        else:
            errors.append(f"Sensor cleaned Parquet not found at {sensor_parquet}. Dataset must be preprocessed (Task 20) first.")

        if errors or sensor_df.empty:
            logger.error("PDR Pipeline aborted for %s: %s", dataset_id, errors)
            report = PDRReport(
                dataset_id=dataset_id,
                session_id=dataset_id,
                status="FAILED",
                warnings=warnings,
                errors=errors,
                processing_time_ms=round((time.perf_counter() - start_time) * 1000, 2)
            )
            manifest = PDRManifest(
                source_dataset_id=dataset_id,
                session_id=dataset_id,
                pdr_version=self.config.version,
                config=self.config.to_dict(),
                creation_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                processing_warnings=warnings,
                processing_errors=errors
            )
            return report, manifest

        session_id = str(sensor_df["session_id"].iloc[0]) if "session_id" in sensor_df.columns else dataset_id

        # 2. Segregate sensor modalities
        sensor_df["sensor_type_upper"] = sensor_df["sensor_type"].astype(str).str.strip().str.upper()
        accel_df = sensor_df[sensor_df["sensor_type_upper"] == "ACCELEROMETER"].copy()
        rot_df = sensor_df[sensor_df["sensor_type_upper"] == "ROTATION_VECTOR"].copy()
        gyro_df = sensor_df[sensor_df["sensor_type_upper"] == "GYROSCOPE"].copy()
        mag_df = sensor_df[sensor_df["sensor_type_upper"] == "MAGNETIC_FIELD"].copy()

        if accel_df.empty:
            errors.append("No accelerometer observations found; cannot execute PDR.")
            report = PDRReport(
                dataset_id=dataset_id,
                session_id=session_id,
                status="FAILED",
                warnings=warnings,
                errors=errors,
                processing_time_ms=round((time.perf_counter() - start_time) * 1000, 2)
            )
            manifest = PDRManifest(
                source_dataset_id=dataset_id,
                session_id=session_id,
                pdr_version=self.config.version,
                config=self.config.to_dict(),
                creation_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                processing_warnings=warnings,
                processing_errors=errors
            )
            return report, manifest

        # 3. Gravity Separation
        grav_comp = GravityCompensator(self.config)
        grav_res: GravityCompResult = grav_comp.process(accel_df, rot_df)
        gravity_df = grav_res.df

        # 4. Device Orientation Estimation
        orient_est = OrientationEstimator(self.config)
        orient_res: OrientationResult = orient_est.process(
            rot_df=rot_df,
            accel_df=accel_df,
            gyro_df=gyro_df,
            mag_df=mag_df
        )
        orientation_df = orient_res.df

        # 5. Step Detection
        step_detector = StepDetector(self.config)
        steps_df, step_stats = step_detector.process(gravity_df, session_id=session_id)

        # 6. Step Length Estimation
        step_len_est = StepLengthEstimator(self.config)
        steps_with_len_df = step_len_est.process(steps_df)

        # 7. 2D Relative Trajectory Reconstruction
        traj_reconstructor = TrajectoryReconstructor(self.config)
        traj_df, traj_summary = traj_reconstructor.process(
            steps_df=steps_with_len_df,
            orientation_df=orientation_df,
            session_id=session_id
        )

        # 8. Save PDR Artifacts to Parquet & JSON
        output_checksums: Dict[str, str] = {}
        output_files_info: Dict[str, Any] = {}

        def save_parquet(df: pd.DataFrame, filename: str):
            if df is not None:
                full_path = output_pdr_dir / filename
                df.to_parquet(full_path, index=False, engine="pyarrow")
                sha = calculate_sha256(full_path)
                output_checksums[filename] = sha
                output_files_info[filename] = {
                    "records": len(df),
                    "file_size_bytes": full_path.stat().st_size,
                    "sha256": sha
                }

        save_parquet(gravity_df, "gravity_compensated.parquet")
        save_parquet(orientation_df, "orientation_estimates.parquet")
        save_parquet(steps_with_len_df, "detected_steps.parquet")
        save_parquet(steps_with_len_df, "step_features.parquet")
        save_parquet(traj_df, "relative_trajectory.parquet")

        # 9. Build Report & Manifest
        status = "COMPLETED"
        if grav_res.mean_compensation_quality < 0.7:
            warnings.append(f"Gravity compensation heuristic used (quality: {grav_res.mean_compensation_quality:.2f}).")
        if orient_res.magnetic_disturbance_count > 0:
            warnings.append(f"Detected {orient_res.magnetic_disturbance_count} magnetic field anomalies.")
        if len(steps_df) == 0:
            warnings.append("Zero walking steps detected in survey accelerometer stream.")

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        report = PDRReport(
            dataset_id=dataset_id,
            session_id=session_id,
            status=status if not warnings else "WARNINGS",
            movement_summary=asdict(traj_summary),
            gravity_summary={
                "method_used": grav_res.method_used,
                "mean_compensation_quality": round(grav_res.mean_compensation_quality, 3),
                "rotation_vector_coverage_ratio": round(grav_res.rotation_vector_coverage_ratio, 3),
            },
            orientation_summary={
                "primary_source": orient_res.primary_source,
                "mean_confidence": round(orient_res.mean_confidence, 3),
                "magnetic_disturbance_count": orient_res.magnetic_disturbance_count,
            },
            step_summary=asdict(step_stats),
            output_summary=output_files_info,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms
        )

        manifest = PDRManifest(
            source_dataset_id=dataset_id,
            session_id=session_id,
            pdr_version=self.config.version,
            config=self.config.to_dict(),
            creation_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            input_file_checksums=input_checksums,
            output_file_checksums=output_checksums,
            number_of_detected_steps=len(steps_df),
            trajectory_point_count=len(traj_df),
            processing_warnings=warnings,
            processing_errors=errors
        )

        # Write report and manifest to disk
        report_path = output_pdr_dir / "pdr_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        manifest_path = output_pdr_dir / "pdr_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        logger.info(
            "PDR Pipeline for %s finished in %.2f ms (Steps: %d, Distance: %.2f m, Status: %s).",
            dataset_id, elapsed_ms, len(steps_df), traj_summary.total_distance_m, report.status
        )
        return report, manifest
