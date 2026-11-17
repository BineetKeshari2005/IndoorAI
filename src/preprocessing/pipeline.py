"""End-to-End Data Preprocessing Pipeline for IndoorAI.

Coordinates validation, duplicate handling, Wi-Fi signal remapping (-105 dBm floor),
visibility mask generation, training-only feature scaling, target encoding,
leakage verification, and Parquet dataset persistence.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is in sys.path
_current_dir = Path(__file__).resolve().parent
_repo_root = _current_dir.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import yaml

from src.core.config import get_settings
from src.core.logging import setup_logger
from src.preprocessing.cleaning import audit_ap_feature_quality, select_features
from src.preprocessing.duplicate_handler import handle_duplicates
from src.preprocessing.scaling import fit_and_transform_features
from src.preprocessing.target_encoding import (
    extract_targets_and_metadata,
    fit_target_encoders,
    verify_no_leakage,
)
from src.preprocessing.validation import validate_datasets
from src.preprocessing.wifi_preprocessing import extract_wap_columns, generate_wifi_features

logger = setup_logger("PreprocessingPipeline")


@dataclass
class PreprocessingPipelineSummary:
    """Summary record of preprocessing execution and outputs."""
    dataset_name: str
    execution_epoch_ms: int
    duplicate_strategy: str
    raw_train_rows: int
    raw_val_rows: int
    processed_train_rows: int
    processed_val_rows: int
    input_feature_count: int
    feature_representation: str
    rssi_floor_dbm: float
    scaling_method: str
    num_constant_training_waps: int
    output_files: Dict[str, str]
    artifacts: Dict[str, str]
    leakage_checks_passed: bool


def load_preprocessing_config() -> Dict[str, Any]:
    """Load configuration from configs/preprocessing_config.yaml with fallback defaults."""
    settings = get_settings()
    config_file = settings.paths.configs_dir / "preprocessing_config.yaml"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    return cfg
        except Exception as e:
            logger.warning("Failed to load preprocessing YAML config (%s), using defaults.", e)

    return {
        "pipeline": {
            "dataset_name": "UJIIndoorLoc",
            "raw_data_dir": "data/raw/ujiindoorloc",
            "processed_data_dir": "data/processed/ujiindoorloc",
            "experiments_dir": "experiments/preprocessing",
            "artifacts_dir": "experiments/preprocessing/artifacts",
        },
        "duplicate_handling": {"strategy": "keep_all"},
        "wifi_signal": {
            "unavailable_sentinel": 100,
            "rssi_floor_dbm": -105.0,
            "feature_representation": "rssi_with_visibility",
        },
        "ap_filtering": {
            "drop_zero_variance": False,
            "low_visibility_threshold_pct": 0.1,
        },
        "scaling": {
            "method": "minmax",
            "feature_range": [0.0, 1.0],
        },
        "leakage_prevention": {
            "strict_checks": True,
        },
    }


def run_pipeline(
    config_override: Optional[Dict[str, Any]] = None
) -> PreprocessingPipelineSummary:
    """Execute the complete data preprocessing pipeline.

    Args:
        config_override: Optional dictionary of configuration overrides.

    Returns:
        PreprocessingPipelineSummary dataclass.
    """
    settings = get_settings()
    config = load_preprocessing_config()
    if config_override:
        config.update(config_override)

    # 1. Resolve paths
    raw_dir = settings.paths.root_dir / config["pipeline"].get("raw_data_dir", "data/raw/ujiindoorloc")
    processed_dir = settings.paths.root_dir / config["pipeline"].get("processed_data_dir", "data/processed/ujiindoorloc")
    exp_dir = settings.paths.root_dir / config["pipeline"].get("experiments_dir", "experiments/preprocessing")
    artifacts_dir = settings.paths.root_dir / config["pipeline"].get("artifacts_dir", "experiments/preprocessing/artifacts")

    processed_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_dir / "trainingData.csv"
    val_path = raw_dir / "validationData.csv"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Raw dataset files missing in {raw_dir}. Run download module first.")

    # 2. Load Raw Datasets
    logger.info("Step 1: Loading raw datasets from %s ...", raw_dir)
    df_train_raw = pd.read_csv(train_path)
    df_val_raw = pd.read_csv(val_path)
    raw_train_rows = len(df_train_raw)
    raw_val_rows = len(df_val_raw)

    # 3. Data Validation
    logger.info("Step 2: Validating dataset schemas and physical constraints...")
    val_report_path = exp_dir / "validation_report.json"
    val_report = validate_datasets(df_train_raw, df_val_raw, save_report_path=val_report_path)
    if not val_report.overall_passed:
        logger.warning("Validation completed with notices; review %s", val_report_path)

    # 4. Duplicate Handling (Applied to training data only)
    dup_strategy = config["duplicate_handling"].get("strategy", "keep_all")
    logger.info("Step 3: Handling duplicate records using strategy '%s'...", dup_strategy)
    dup_report_path = exp_dir / "duplicate_handling_report.json"
    df_train_dedup, dup_report = handle_duplicates(
        df_train_raw,
        strategy=dup_strategy,
        save_report_path=dup_report_path
    )

    # Validation data duplicates are never dropped or altered
    df_val_work = df_val_raw.copy()
    df_val_work["original_row_index"] = df_val_raw.index.values
    df_val_work["sample_weight"] = 1.0

    # 5. Extract WAP Features & Audit AP Feature Quality
    logger.info("Step 4: Auditing feature quality and zero-variance WAPs on training split...")
    quality_csv_path = exp_dir / "ap_feature_quality.csv"
    ap_quality_df = audit_ap_feature_quality(
        df_train_dedup,
        df_val=df_val_work,
        low_visibility_pct=config["ap_filtering"].get("low_visibility_threshold_pct", 0.1),
        save_path=quality_csv_path
    )
    num_constant_waps = int(ap_quality_df["is_constant_train"].sum())

    # 6. Wi-Fi Representation Engineering (RSSI floor replacement & visibility masks)
    wifi_cfg = config["wifi_signal"]
    feature_rep = wifi_cfg.get("feature_representation", "rssi_with_visibility")
    rssi_floor = float(wifi_cfg.get("rssi_floor_dbm", -105.0))
    sentinel = int(wifi_cfg.get("unavailable_sentinel", 100))

    logger.info("Step 5: Generating '%s' Wi-Fi features (floor=%.1f dBm)...", feature_rep, rssi_floor)
    X_train_raw = generate_wifi_features(
        df_train_dedup,
        representation=feature_rep,
        sentinel=sentinel,
        rssi_floor=rssi_floor
    )
    X_val_raw = generate_wifi_features(
        df_val_work,
        representation=feature_rep,
        sentinel=sentinel,
        rssi_floor=rssi_floor
    )

    # 7. Configurable Feature Filtering (defaults to drop_zero_variance=False)
    drop_zv = config["ap_filtering"].get("drop_zero_variance", False)
    X_train_clean, X_val_clean, selected_cols = select_features(
        X_train_raw,
        X_val_raw,
        drop_zero_variance=drop_zv,
        ap_quality_df=ap_quality_df
    )

    # 8. Feature Scaling (Fitted STRICTLY on training split)
    scale_cfg = config["scaling"]
    scale_method = scale_cfg.get("method", "minmax")
    scale_range = tuple(scale_cfg.get("feature_range", [0.0, 1.0]))
    scaler_artifact_path = artifacts_dir / "scaler.joblib"

    logger.info("Step 6: Fitting '%s' scaler strictly on training split...", scale_method)
    X_train_scaled, X_val_scaled, fitted_scaler = fit_and_transform_features(
        X_train_clean,
        X_val=X_val_clean,
        method=scale_method,
        feature_range=scale_range,
        save_artifact_path=scaler_artifact_path
    )

    # 9. Target Encoding & Metadata Separation (Fitted STRICTLY on training split)
    logger.info("Step 7: Encoding localization targets and isolating metadata...")
    encoders_json_path = artifacts_dir / "target_encoders.json"
    target_encoder_meta = fit_target_encoders(df_train_dedup)

    y_train, meta_train = extract_targets_and_metadata(
        df_train_dedup,
        target_encoder_meta,
        save_encoders_path=encoders_json_path
    )
    y_val, meta_val = extract_targets_and_metadata(
        df_val_work,
        target_encoder_meta,
        save_encoders_path=None  # Already saved from training
    )

    # 10. Data Leakage Verification
    logger.info("Step 8: Running programmatic data leakage verification checks...")
    verify_no_leakage(X_train_scaled, y_train, meta_train, split_name="training")
    verify_no_leakage(X_val_scaled, y_val, meta_val, split_name="validation")

    # Assert train and validation schemas are identical in columns, count, and order
    if list(X_train_scaled.columns) != list(X_val_scaled.columns):
        raise AssertionError("Feature columns mismatch between X_train and X_val!")

    # 11. Save Clean Machine-Learning-Ready Datasets as Parquet
    logger.info("Step 9: Persisting processed datasets to %s ...", processed_dir)
    out_files = {
        "X_train": str(processed_dir / "X_train.parquet"),
        "X_validation": str(processed_dir / "X_validation.parquet"),
        "y_train": str(processed_dir / "y_train.parquet"),
        "y_validation": str(processed_dir / "y_validation.parquet"),
        "metadata_train": str(processed_dir / "metadata_train.parquet"),
        "metadata_validation": str(processed_dir / "metadata_validation.parquet"),
    }

    X_train_scaled.to_parquet(out_files["X_train"], index=False)
    X_val_scaled.to_parquet(out_files["X_validation"], index=False)
    y_train.to_parquet(out_files["y_train"], index=False)
    y_val.to_parquet(out_files["y_validation"], index=False)
    meta_train.to_parquet(out_files["metadata_train"], index=False)
    meta_val.to_parquet(out_files["metadata_validation"], index=False)

    for k, p in out_files.items():
        logger.info("Saved %s -> %s (%d bytes)", k, Path(p).name, Path(p).stat().st_size)

    # 12. Persist Processing Metadata and Summary
    artifacts_map = {
        "scaler": str(scaler_artifact_path),
        "target_encoders": str(encoders_json_path),
        "ap_feature_quality": str(quality_csv_path),
        "duplicate_report": str(dup_report_path),
        "validation_report": str(val_report_path),
    }

    summary = PreprocessingPipelineSummary(
        dataset_name=config["pipeline"].get("dataset_name", "UJIIndoorLoc"),
        execution_epoch_ms=int(time.time() * 1000),
        duplicate_strategy=dup_strategy,
        raw_train_rows=raw_train_rows,
        raw_val_rows=raw_val_rows,
        processed_train_rows=len(X_train_scaled),
        processed_val_rows=len(X_val_scaled),
        input_feature_count=len(X_train_scaled.columns),
        feature_representation=feature_rep,
        rssi_floor_dbm=rssi_floor,
        scaling_method=scale_method,
        num_constant_training_waps=num_constant_waps,
        output_files=out_files,
        artifacts=artifacts_map,
        leakage_checks_passed=True,
    )

    summary_path = exp_dir / "preprocessing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    logger.info("Saved preprocessing summary to %s", summary_path)

    meta_artifact_path = artifacts_dir / "preprocessing_metadata.json"
    with open(meta_artifact_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    logger.info("Preprocessing Pipeline Completed Successfully!")
    return summary


if __name__ == "__main__":
    try:
        summary = run_pipeline()
        print("\n=======================================================")
        print("IndoorAI Preprocessing Pipeline Succeeded!")
        print("=======================================================")
        print(f"Duplicate Strategy:        {summary.duplicate_strategy}")
        print(f"Training Set Dimensions:   {summary.processed_train_rows:,} rows x {summary.input_feature_count} features")
        print(f"Validation Set Dimensions: {summary.processed_val_rows:,} rows x {summary.input_feature_count} features")
        print(f"Wi-Fi Representation:      {summary.feature_representation} (floor: {summary.rssi_floor_dbm} dBm)")
        print(f"Scaling Method:            {summary.scaling_method} (fitted strictly on training split)")
        print(f"Zero-Variance APs Tracked: {summary.num_constant_training_waps}")
        print(f"Data Leakage Assertions:   PASSED (Zero target or metadata in X)")
        print("All Parquet datasets saved to data/processed/ujiindoorloc/")
        print("All metadata & reports saved to experiments/preprocessing/")
    except Exception as exc:
        logger.exception("Pipeline execution failed: %s", exc)
        sys.exit(1)

