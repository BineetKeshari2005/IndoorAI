"""Preprocessing package for IndoorAI.

Includes data validation, duplicate handling, Wi-Fi signal remapping,
feature quality auditing, scaling with leakage prevention, target encoding,
and end-to-end reproducible pipeline execution.
"""

from src.preprocessing.cleaning import audit_ap_feature_quality, select_features
from src.preprocessing.duplicate_handler import DuplicateStrategy, handle_duplicates
from src.preprocessing.pipeline import run_pipeline
from src.preprocessing.scaling import fit_and_transform_features, get_scaler_instance
from src.preprocessing.target_encoding import (
    extract_targets_and_metadata,
    fit_target_encoders,
    verify_no_leakage,
)
from src.preprocessing.validation import validate_datasets, validate_partition
from src.preprocessing.wifi_preprocessing import (
    create_visibility_mask,
    extract_wap_columns,
    generate_wifi_features,
    process_rssi_matrix,
)

__all__ = [
    "validate_datasets",
    "validate_partition",
    "handle_duplicates",
    "DuplicateStrategy",
    "extract_wap_columns",
    "create_visibility_mask",
    "process_rssi_matrix",
    "generate_wifi_features",
    "audit_ap_feature_quality",
    "select_features",
    "fit_and_transform_features",
    "get_scaler_instance",
    "fit_target_encoders",
    "extract_targets_and_metadata",
    "verify_no_leakage",
    "run_pipeline",
]
