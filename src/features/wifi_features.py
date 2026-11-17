"""Unified Wi-Fi feature engineering aggregator and transformer module.

Combines baseline preprocessed features (520 continuous RSSI + 520 AP visibility flags)
with newly engineered signal summary statistics, access point descriptors, and optional
relative differential signal features.

Enforces strict zero-leakage training-only fitting for scalers and feature selectors,
and validates feature matrices against target/metadata contamination and NaN/inf values.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.core.logging import setup_logger
from src.features.signal_statistics import (
    SignalStatsConfig,
    extract_signal_statistics_from_dataframe,
    SUMMARY_FEATURE_NAMES,
)
from src.features.ap_features import (
    compute_training_ap_statistics,
    compute_relative_signal_features,
)
from src.features.feature_selection import (
    evaluate_feature_selection_thresholds,
    apply_feature_selection,
)

logger = setup_logger("WiFiFeatures")

FORBIDDEN_METADATA_COLUMNS = {
    "BUILDINGID",
    "FLOOR",
    "LONGITUDE",
    "LATITUDE",
    "SPACEID",
    "RELATIVEPOSITION",
    "USERID",
    "PHONEID",
    "TIMESTAMP",
    "BUILDING_FLOOR",
    "SAMPLE_WEIGHT",
    "ORIGINAL_ROW_INDEX",
}


@dataclass
class WiFiFeaturePipelineConfig:
    """Configuration container for end-to-end feature engineering."""
    dataset_name: str = "UJIIndoorLoc"
    raw_data_dir: str = "data/raw/ujiindoorloc"
    processed_data_dir: str = "data/processed/ujiindoorloc"
    feature_engineered_dir: str = "data/processed/ujiindoorloc/feature_engineered"
    experiments_dir: str = "experiments/feature_engineering"
    plots_dir: str = "experiments/feature_engineering/plots"

    # Signal statistics
    enable_signal_statistics: bool = True
    unavailable_sentinel: int = 100
    default_rssi_floor_dbm: float = -105.0
    strong_threshold_dbm: float = -60.0
    medium_threshold_dbm: float = -80.0

    # Relative features
    enable_relative_features: bool = False
    diff_to_max: bool = True
    diff_to_mean: bool = False
    default_unavailable_diff: float = 0.0

    # Feature selection
    selection_strategy: str = "all"  # 'all', 'remove_constant', 'min_detection_count', 'variance_threshold'
    min_detection_count: int = 10
    min_detection_pct: float = 0.05
    variance_threshold: float = 0.001

    # Scaling
    scale_engineered_features: bool = False
    scaling_method: str = "minmax"  # 'minmax' or 'standard'
    feature_range: Tuple[float, float] = (0.0, 1.0)

    # Leakage checks
    strict_leakage_checks: bool = True


class WiFiFeatureTransformer:
    """Stateful feature engineering transformer fitted exclusively on training data.

    Preserves fitted scalers, training AP statistics, and selected feature masks
    to ensure zero information leakage during validation or production inference.
    """

    def __init__(self, config: Optional[WiFiFeaturePipelineConfig] = None):
        self.config = config or WiFiFeaturePipelineConfig()
        self.ap_statistics_: Optional[pd.DataFrame] = None
        self.scaler_: Optional[Union[MinMaxScaler, StandardScaler]] = None
        self.engineered_feature_names_: List[str] = []
        self.selected_feature_names_: List[str] = []
        self.all_feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        df_train_raw: pd.DataFrame,
        X_train_baseline: pd.DataFrame
    ) -> "WiFiFeatureTransformer":
        """Fit feature engineering components strictly on the training partition.

        Args:
            df_train_raw: Original raw training DataFrame (with WAP001..WAP520 in raw dBm/sentinel).
            X_train_baseline: Preprocessed baseline features (1,040 columns: 520 WAPs + 520 VISIBLE).

        Returns:
            Self (fitted transformer).
        """
        logger.info("Fitting WiFiFeatureTransformer strictly on training partition (%d rows)...", len(df_train_raw))

        # 1. Compute training-only Access Point statistics
        self.ap_statistics_ = compute_training_ap_statistics(
            df_train=df_train_raw,
            wap_prefix="WAP",
            sentinel=self.config.unavailable_sentinel
        )

        # 2. Extract engineered summary statistics on training data
        stats_cfg = SignalStatsConfig(
            unavailable_sentinel=self.config.unavailable_sentinel,
            rssi_floor_dbm=self.config.default_rssi_floor_dbm,
            strong_threshold_dbm=self.config.strong_threshold_dbm,
            medium_threshold_dbm=self.config.medium_threshold_dbm
        )

        engineered_dfs = []
        if self.config.enable_signal_statistics:
            train_stats_df = extract_signal_statistics_from_dataframe(df_train_raw, config=stats_cfg)
            engineered_dfs.append(train_stats_df)

        if self.config.enable_relative_features:
            train_rel_df = compute_relative_signal_features(
                df=df_train_raw,
                wap_prefix="WAP",
                sentinel=self.config.unavailable_sentinel,
                diff_to_max=self.config.diff_to_max,
                diff_to_mean=self.config.diff_to_mean,
                default_unavailable_diff=self.config.default_unavailable_diff
            )
            engineered_dfs.append(train_rel_df)

        if engineered_dfs:
            X_engineered_raw = pd.concat(engineered_dfs, axis=1)
            self.engineered_feature_names_ = list(X_engineered_raw.columns)

            # 3. Fit scaler strictly on training engineered features
            if self.config.scale_engineered_features:
                if self.config.scaling_method == "minmax":
                    self.scaler_ = MinMaxScaler(feature_range=self.config.feature_range)
                elif self.config.scaling_method == "standard":
                    self.scaler_ = StandardScaler()
                else:
                    raise ValueError(f"Unknown scaling method: {self.config.scaling_method}")

                self.scaler_.fit(X_engineered_raw.values)
                logger.info(
                    "Fitted %s scaler on %d engineered features.",
                    self.config.scaling_method,
                    len(self.engineered_feature_names_)
                )
        else:
            self.engineered_feature_names_ = []
            self.scaler_ = None

        # 4. Construct candidate combined feature set for selection
        X_combined_train = self._assemble_features(
            X_baseline=X_train_baseline,
            df_raw=df_train_raw,
            is_training=True
        )
        self.all_feature_names_ = list(X_combined_train.columns)

        # 5. Evaluate and determine feature selection mask
        if self.config.selection_strategy == "all":
            self.selected_feature_names_ = list(self.all_feature_names_)
            logger.info("Feature selection strategy 'all': retaining all %d features.", len(self.selected_feature_names_))
        else:
            selection_eval = evaluate_feature_selection_thresholds(
                X_train=X_combined_train,
                ap_stats_df=self.ap_statistics_
            )
            strategy_info = selection_eval["strategies"].get(self.config.selection_strategy)
            if strategy_info is None:
                raise ValueError(f"Unknown selection strategy: {self.config.selection_strategy}")
            self.selected_feature_names_ = strategy_info["retained_features"]
            logger.info(
                "Feature selection strategy '%s': retained %d of %d features (dropped %d).",
                self.config.selection_strategy,
                len(self.selected_feature_names_),
                len(self.all_feature_names_),
                strategy_info["dropped_feature_count"]
            )

        self.is_fitted_ = True
        return self

    def _assemble_features(
        self,
        X_baseline: pd.DataFrame,
        df_raw: pd.DataFrame,
        is_training: bool = False
    ) -> pd.DataFrame:
        """Internal helper to compute, scale, and assemble all candidate features."""
        parts = [X_baseline.reset_index(drop=True)]

        stats_cfg = SignalStatsConfig(
            unavailable_sentinel=self.config.unavailable_sentinel,
            rssi_floor_dbm=self.config.default_rssi_floor_dbm,
            strong_threshold_dbm=self.config.strong_threshold_dbm,
            medium_threshold_dbm=self.config.medium_threshold_dbm
        )

        engineered_dfs = []
        if self.config.enable_signal_statistics:
            stats_df = extract_signal_statistics_from_dataframe(df_raw, config=stats_cfg)
            engineered_dfs.append(stats_df)

        if self.config.enable_relative_features:
            rel_df = compute_relative_signal_features(
                df=df_raw,
                wap_prefix="WAP",
                sentinel=self.config.unavailable_sentinel,
                diff_to_max=self.config.diff_to_max,
                diff_to_mean=self.config.diff_to_mean,
                default_unavailable_diff=self.config.default_unavailable_diff
            )
            engineered_dfs.append(rel_df)

        if engineered_dfs:
            X_engineered_raw = pd.concat(engineered_dfs, axis=1)

            if self.scaler_ is not None:
                scaled_values = self.scaler_.transform(X_engineered_raw.values)
                X_engineered_scaled = pd.DataFrame(
                    scaled_values,
                    columns=self.engineered_feature_names_,
                    index=X_engineered_raw.index
                )
                parts.append(X_engineered_scaled.reset_index(drop=True))
            else:
                parts.append(X_engineered_raw.reset_index(drop=True))

        combined_df = pd.concat(parts, axis=1)
        return combined_df

    def transform(
        self,
        df_raw: pd.DataFrame,
        X_baseline: pd.DataFrame
    ) -> pd.DataFrame:
        """Transform a dataset using already-fitted training parameters.

        Args:
            df_raw: Raw DataFrame with WAP columns in original dBm/sentinel values.
            X_baseline: Baseline preprocessed features (1,040 columns).

        Returns:
            Engineered feature DataFrame.
        """
        if not self.is_fitted_:
            raise RuntimeError("WiFiFeatureTransformer must be fitted on training data before transform().")

        logger.info("Transforming %d samples using fitted feature engineering pipeline...", len(df_raw))

        # Assemble candidate features using fitted scaler
        X_combined = self._assemble_features(
            X_baseline=X_baseline,
            df_raw=df_raw,
            is_training=False
        )

        # Apply fitted feature selection mask
        X_final = X_combined[self.selected_feature_names_].copy()

        # Perform strict validation checks
        if self.config.strict_leakage_checks:
            self._validate_output_matrix(X_final)

        logger.info(
            "Feature transformation complete: produced matrix of shape %s with %d features.",
            X_final.shape,
            len(self.selected_feature_names_)
        )
        return X_final

    def fit_transform(
        self,
        df_train_raw: pd.DataFrame,
        X_train_baseline: pd.DataFrame
    ) -> pd.DataFrame:
        """Fit on training data and transform in a single call."""
        return self.fit(df_train_raw, X_train_baseline).transform(df_train_raw, X_train_baseline)

    def _validate_output_matrix(self, X: pd.DataFrame) -> None:
        """Strict leakage and data integrity validation."""
        # Check 1: Forbidden target or metadata column presence
        leaked_cols = set(X.columns).intersection(FORBIDDEN_METADATA_COLUMNS)
        if leaked_cols:
            raise ValueError(
                f"DATA LEAKAGE DETECTED! Forbidden target/metadata columns found in feature matrix: {leaked_cols}"
            )

        # Check 2: Missing or NaN values
        nan_counts = X.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        if not cols_with_nan.empty:
            raise ValueError(
                f"DATA INTEGRITY ERROR! Features contain NaN values: {cols_with_nan.to_dict()}"
            )

        # Check 3: Infinite values
        inf_mask = np.isinf(X.values)
        if np.any(inf_mask):
            inf_cols = [col for col in X.columns if np.any(np.isinf(X[col].values))]
            raise ValueError(
                f"DATA INTEGRITY ERROR! Features contain infinite values in columns: {inf_cols}"
            )


def load_feature_config(config_path: Union[str, Path]) -> WiFiFeaturePipelineConfig:
    """Load feature configuration from YAML file or return defaults."""
    import yaml

    path = Path(config_path)
    if not path.is_file():
        logger.warning("Feature config %s not found. Using default configurations.", path)
        return WiFiFeaturePipelineConfig()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    p_cfg = data.get("pipeline", {})
    s_cfg = data.get("signal_statistics", {})
    r_cfg = data.get("relative_features", {})
    fs_cfg = data.get("feature_selection", {})
    sc_cfg = data.get("scaling", {})
    l_cfg = data.get("leakage_prevention", {})

    return WiFiFeaturePipelineConfig(
        dataset_name=p_cfg.get("dataset_name", "UJIIndoorLoc"),
        raw_data_dir=p_cfg.get("raw_data_dir", "data/raw/ujiindoorloc"),
        processed_data_dir=p_cfg.get("processed_data_dir", "data/processed/ujiindoorloc"),
        feature_engineered_dir=p_cfg.get("feature_engineered_dir", "data/processed/ujiindoorloc/feature_engineered"),
        experiments_dir=p_cfg.get("experiments_dir", "experiments/feature_engineering"),
        plots_dir=p_cfg.get("plots_dir", "experiments/feature_engineering/plots"),
        enable_signal_statistics=s_cfg.get("enabled", True),
        unavailable_sentinel=s_cfg.get("unavailable_sentinel", 100),
        default_rssi_floor_dbm=s_cfg.get("default_rssi_floor_dbm", -105.0),
        strong_threshold_dbm=s_cfg.get("strong_threshold_dbm", -60.0),
        medium_threshold_dbm=s_cfg.get("medium_threshold_dbm", -80.0),
        enable_relative_features=r_cfg.get("enabled", False),
        diff_to_max=r_cfg.get("diff_to_max", True),
        diff_to_mean=r_cfg.get("diff_to_mean", False),
        default_unavailable_diff=r_cfg.get("default_unavailable_diff", 0.0),
        selection_strategy=fs_cfg.get("strategy", "all"),
        min_detection_count=fs_cfg.get("min_detection_count", 10),
        min_detection_pct=fs_cfg.get("min_detection_pct", 0.05),
        variance_threshold=fs_cfg.get("variance_threshold", 0.001),
        scale_engineered_features=sc_cfg.get("enabled", True),
        scaling_method=sc_cfg.get("method", "minmax"),
        feature_range=tuple(sc_cfg.get("feature_range", [0.0, 1.0])),
        strict_leakage_checks=l_cfg.get("strict_checks", True),
    )

