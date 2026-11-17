"""k-Nearest Neighbors (k-NN) baseline models for indoor localization.

Provides:
1. Multi-head k-NN orchestrator (`IndoorAIKNN`) wrapping 5 prediction targets:
   - Model A: Building Classification (`BUILDINGID`)
   - Model B: Floor Classification (`FLOOR`)
   - Model C: Building-Floor Classification (`BUILDING_FLOOR_INDEX`)
   - Model D: Longitude Regression (`LONGITUDE`)
   - Model E: Latitude Regression (`LATITUDE`)
2. Feature-scaling transformer (`SummaryFeatureScaler`) to ensure the 18 engineered
   summary features in Variant B are scaled strictly on training data to [0, 1]
   without dominating distance metrics over binary visibility and normalized RSSI.
3. Strict zero-leakage verification preventing metadata and targets from entering X.
4. Model serialization and deserialization via joblib.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.core.logging import setup_logger

logger = setup_logger("KNNModel")

# Non-feature columns strictly prohibited from entering model input
FORBIDDEN_COLUMNS: Set[str] = {
    "BUILDINGID",
    "FLOOR",
    "BUILDING_FLOOR_INDEX",
    "LONGITUDE",
    "LATITUDE",
    "SPACEID",
    "RELATIVEPOSITION",
    "USERID",
    "PHONEID",
    "TIMESTAMP",
    "original_row_index",
    "sample_weight",
}


@dataclass
class KNNHyperparameters:
    """Hyperparameters for k-Nearest Neighbors models."""
    n_neighbors: int = 5
    metric: str = "cosine"  # "cosine", "manhattan", "euclidean"
    weights: str = "distance"  # "distance", "uniform"
    p: int = 2  # Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)
    algorithm: str = "auto"
    n_jobs: int = -1
    scale_summary: bool = True
    scaler_type: str = "minmax"  # "minmax" or "standard"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SummaryFeatureScaler:
    """Scaler applied strictly to the 18 engineered summary features in Variant B.
    
    Preserves binary visibility flags {0, 1} and normalized RSSI [0, 1], while
    scaling the summary statistics to [0, 1] so high-magnitude counts do not
    overwhelm nearest-neighbor distance computations.
    """

    def __init__(self, scaler_type: str = "minmax"):
        self.scaler_type = scaler_type
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0)) if scaler_type == "minmax" else StandardScaler()
        self.summary_columns: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, base_feature_count: int = 1040) -> "SummaryFeatureScaler":
        """Identify and fit on summary columns exceeding base feature count."""
        cols = list(X.columns)
        if len(cols) > base_feature_count:
            # Summary features are columns beyond the 1,040 basic RSSI + visibility features
            # Or columns not starting with WAP
            self.summary_columns = [c for c in cols if not c.startswith("WAP")]
            if self.summary_columns:
                self.scaler.fit(X[self.summary_columns].values)
                logger.info(
                    "Fitted %s scaler on %d summary features: %s",
                    self.scaler_type,
                    len(self.summary_columns),
                    self.summary_columns[:5],
                )
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform summary features in a copy of DataFrame."""
        if not self.is_fitted or not self.summary_columns:
            return X
        X_out = X.copy()
        matching_cols = [c for c in self.summary_columns if c in X_out.columns]
        if matching_cols:
            X_out[matching_cols] = self.scaler.transform(X_out[matching_cols].values)
        return X_out

    def fit_transform(self, X: pd.DataFrame, base_feature_count: int = 1040) -> pd.DataFrame:
        return self.fit(X, base_feature_count).transform(X)


class IndoorAIKNN:
    """Unified k-Nearest Neighbors estimator suite for indoor localization.
    
    Manages five distinct model heads:
    - Model A: Building Classifier (`BUILDINGID`, 3 classes)
    - Model B: Floor Classifier (`FLOOR`, 5 classes)
    - Model C: Combined Building-Floor Classifier (`BUILDING_FLOOR_INDEX`, 13 classes)
    - Model D: Longitude Coordinate Regressor (`LONGITUDE`, 1D continuous)
    - Model E: Latitude Coordinate Regressor (`LATITUDE`, 1D continuous)
    """

    def __init__(
        self,
        params: Optional[KNNHyperparameters] = None,
        feature_variant: str = "variant_b_engineered",
    ):
        self.params = params or KNNHyperparameters()
        self.feature_variant = feature_variant
        self.feature_names_in_: Optional[List[str]] = None
        self.summary_scaler = SummaryFeatureScaler(scaler_type=self.params.scaler_type) if self.params.scale_summary else None
        self.is_fitted_ = False

        # Model heads
        self.model_building: Optional[KNeighborsClassifier] = None
        self.model_floor: Optional[KNeighborsClassifier] = None
        self.model_building_floor: Optional[KNeighborsClassifier] = None
        self.model_longitude: Optional[KNeighborsRegressor] = None
        self.model_latitude: Optional[KNeighborsRegressor] = None

        # Build sub-models
        self._init_models()

    def _init_models(self) -> None:
        """Initialize the five scikit-learn nearest neighbor estimators."""
        metric_kwargs = {}
        if self.params.metric == "minkowski":
            metric_kwargs["p"] = self.params.p
        elif self.params.metric == "manhattan":
            metric = "minkowski"
            metric_kwargs["p"] = 1
        elif self.params.metric == "euclidean":
            metric = "minkowski"
            metric_kwargs["p"] = 2
        else:
            metric = self.params.metric

        clf_kwargs = {
            "n_neighbors": self.params.n_neighbors,
            "weights": self.params.weights,
            "metric": metric,
            "algorithm": self.params.algorithm,
            "n_jobs": self.params.n_jobs,
            **metric_kwargs,
        }
        reg_kwargs = {
            "n_neighbors": self.params.n_neighbors,
            "weights": self.params.weights,
            "metric": metric,
            "algorithm": self.params.algorithm,
            "n_jobs": self.params.n_jobs,
            **metric_kwargs,
        }

        self.model_building = KNeighborsClassifier(**clf_kwargs)
        self.model_floor = KNeighborsClassifier(**clf_kwargs)
        self.model_building_floor = KNeighborsClassifier(**clf_kwargs)
        self.model_longitude = KNeighborsRegressor(**reg_kwargs)
        self.model_latitude = KNeighborsRegressor(**reg_kwargs)

    def _verify_features(self, X: pd.DataFrame) -> None:
        """Enforce strict zero data leakage by checking for forbidden columns."""
        leaked = FORBIDDEN_COLUMNS.intersection(set(X.columns))
        if leaked:
            raise ValueError(
                f"DATA LEAKAGE DETECTED in k-NN input! "
                f"Found prohibited columns: {sorted(list(leaked))}. "
                f"Targets and metadata must never enter feature matrix."
            )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "IndoorAIKNN":
        """Fit all 5 nearest-neighbor heads strictly on training data."""
        self._verify_features(X)
        self.feature_names_in_ = list(X.columns)

        # Scale summary features strictly on training data if applicable
        if self.summary_scaler is not None and len(self.feature_names_in_) > 1040:
            X_proc = self.summary_scaler.fit_transform(X, base_feature_count=1040)
        else:
            X_proc = X

        X_mat = X_proc.values if isinstance(X_proc, pd.DataFrame) else X_proc

        logger.info(
            "Fitting IndoorAIKNN on %d samples (%d features, metric=%s, k=%d, weights=%s)...",
            len(X),
            X_mat.shape[1],
            self.params.metric,
            self.params.n_neighbors,
            self.params.weights,
        )

        # Model A: Building
        self.model_building.fit(X_mat, y["BUILDINGID"].values)
        # Model B: Floor
        self.model_floor.fit(X_mat, y["FLOOR"].values)
        # Model C: Building-Floor
        self.model_building_floor.fit(X_mat, y["BUILDING_FLOOR_INDEX"].values)
        # Model D: Longitude
        self.model_longitude.fit(X_mat, y["LONGITUDE"].values)
        # Model E: Latitude
        self.model_latitude.fit(X_mat, y["LATITUDE"].values)

        self.is_fitted_ = True
        logger.info("All 5 k-NN models successfully fitted.")
        return self

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate predictions from all 5 model heads."""
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")
        self._verify_features(X)

        if self.summary_scaler is not None and len(X.columns) > 1040:
            X_proc = self.summary_scaler.transform(X)
        else:
            X_proc = X

        X_mat = X_proc.values if isinstance(X_proc, pd.DataFrame) else X_proc

        pred_building = self.model_building.predict(X_mat)
        pred_floor = self.model_floor.predict(X_mat)
        pred_bf = self.model_building_floor.predict(X_mat)
        pred_lon = self.model_longitude.predict(X_mat)
        pred_lat = self.model_latitude.predict(X_mat)

        return {
            "pred_building": pred_building,
            "pred_floor": pred_floor,
            "pred_building_floor": pred_bf,
            "pred_longitude": pred_lon,
            "pred_latitude": pred_lat,
        }

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate predicted class probabilities for classification heads."""
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict_proba().")
        self._verify_features(X)

        if self.summary_scaler is not None and len(X.columns) > 1040:
            X_proc = self.summary_scaler.transform(X)
        else:
            X_proc = X

        X_mat = X_proc.values if isinstance(X_proc, pd.DataFrame) else X_proc

        return {
            "proba_building": self.model_building.predict_proba(X_mat),
            "proba_floor": self.model_floor.predict_proba(X_mat),
            "proba_building_floor": self.model_building_floor.predict_proba(X_mat),
        }

    def save(self, save_dir: Union[str, Path]) -> None:
        """Serialize all 5 k-NN model heads and metadata to disk using joblib."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted model.")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model_building, save_path / "model_building.joblib", compress=3)
        joblib.dump(self.model_floor, save_path / "model_floor.joblib", compress=3)
        joblib.dump(self.model_building_floor, save_path / "model_building_floor.joblib", compress=3)
        joblib.dump(self.model_longitude, save_path / "model_longitude.joblib", compress=3)
        joblib.dump(self.model_latitude, save_path / "model_latitude.joblib", compress=3)

        if self.summary_scaler is not None:
            joblib.dump(self.summary_scaler, save_path / "summary_scaler.joblib", compress=3)

        metadata = {
            "params": self.params.to_dict(),
            "feature_variant": self.feature_variant,
            "feature_names_in": self.feature_names_in_,
            "feature_count": len(self.feature_names_in_),
            "is_fitted": self.is_fitted_,
        }
        joblib.dump(metadata, save_path / "knn_metadata.joblib", compress=3)
        with open(save_path / "knn_metadata.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in metadata.items() if k != "feature_names_in"}, f, indent=2)

        logger.info("Saved 5 k-NN models and metadata to %s", save_path)

    @classmethod
    def load(cls, load_dir: Union[str, Path]) -> "IndoorAIKNN":
        """Deserialize trained k-NN suite from disk."""
        load_path = Path(load_dir)
        meta_path = load_path / "knn_metadata.joblib"
        if not meta_path.exists():
            raise FileNotFoundError(f"k-NN metadata not found at {meta_path}")

        metadata = joblib.load(meta_path)
        params = KNNHyperparameters(**metadata["params"])
        instance = cls(params=params, feature_variant=metadata.get("feature_variant", "variant_b_engineered"))
        instance.feature_names_in_ = metadata["feature_names_in"]
        instance.is_fitted_ = metadata["is_fitted"]

        instance.model_building = joblib.load(load_path / "model_building.joblib")
        instance.model_floor = joblib.load(load_path / "model_floor.joblib")
        instance.model_building_floor = joblib.load(load_path / "model_building_floor.joblib")
        instance.model_longitude = joblib.load(load_path / "model_longitude.joblib")
        instance.model_latitude = joblib.load(load_path / "model_latitude.joblib")

        scaler_path = load_path / "summary_scaler.joblib"
        if scaler_path.exists():
            instance.summary_scaler = joblib.load(scaler_path)
        else:
            instance.summary_scaler = None

        logger.info("Loaded 5 k-NN models from %s (%d features)", load_path, len(instance.feature_names_in_))
        return instance

