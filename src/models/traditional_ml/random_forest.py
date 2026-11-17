"""Random Forest baseline models for indoor localization and spatial mapping.

Provides Random Forest estimators for:
1. Model A: Building classification (BUILDINGID)
2. Model B: Floor classification (FLOOR)
3. Model C: Combined building-floor classification (BUILDING_FLOOR_INDEX)
4. Model D: 2D Longitude coordinate regression (LONGITUDE)
5. Model E: 2D Latitude coordinate regression (LATITUDE)

Supports full serialization with joblib and feature importance extraction.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.core.logging import setup_logger

logger = setup_logger("RandomForestModel")

FORBIDDEN_COLUMNS = {
    "BUILDINGID",
    "FLOOR",
    "LONGITUDE",
    "LATITUDE",
    "SPACEID",
    "RELATIVEPOSITION",
    "USERID",
    "PHONEID",
    "TIMESTAMP",
    "BUILDING_INDEX",
    "FLOOR_INDEX",
    "BUILDING_FLOOR_INDEX",
    "original_row_index",
    "sample_weight",
}


@dataclass
class RFHyperparameters:
    """Hyperparameters for Random Forest estimators."""
    n_estimators: int = 200
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Union[str, float, int] = "sqrt"
    random_state: int = 42
    n_jobs: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


def get_random_forest_classifier(
    params: Optional[RFHyperparameters] = None,
    **kwargs
) -> RandomForestClassifier:
    """Instantiate a configured RandomForestClassifier."""
    p = params or RFHyperparameters()
    d = p.to_dict()
    d.update(kwargs)
    return RandomForestClassifier(**d)


def get_random_forest_regressor(
    params: Optional[RFHyperparameters] = None,
    **kwargs
) -> RandomForestRegressor:
    """Instantiate a configured RandomForestRegressor."""
    p = params or RFHyperparameters()
    d = p.to_dict()
    d.update(kwargs)
    return RandomForestRegressor(**d)


class IndoorAIRandomForest:
    """Unified baseline suite of Random Forest models for IndoorAI.

    Coordinates 3 classification heads and 2 regression heads:
    - Building ID classification
    - Floor ID classification
    - Combined Building-Floor classification
    - Longitude coordinate regression
    - Latitude coordinate regression
    """

    def __init__(
        self,
        classifier_params: Optional[RFHyperparameters] = None,
        regressor_params: Optional[RFHyperparameters] = None,
        random_state: int = 42
    ):
        self.random_state = random_state
        clf_p = classifier_params or RFHyperparameters(random_state=random_state)
        reg_p = regressor_params or RFHyperparameters(random_state=random_state)

        self.model_building = get_random_forest_classifier(clf_p)
        self.model_floor = get_random_forest_classifier(clf_p)
        self.model_building_floor = get_random_forest_classifier(clf_p)
        self.model_longitude = get_random_forest_regressor(reg_p)
        self.model_latitude = get_random_forest_regressor(reg_p)

        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def _verify_features(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """Strict leakage check ensuring no target/metadata columns are in X."""
        if isinstance(X, pd.DataFrame):
            leaks = set(X.columns).intersection(FORBIDDEN_COLUMNS)
            if leaks:
                raise ValueError(
                    f"DATA LEAKAGE DETECTED! Forbidden target/metadata columns found in input features: {sorted(leaks)}"
                )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: pd.DataFrame,
        sample_weight: Optional[np.ndarray] = None
    ) -> "IndoorAIRandomForest":
        """Fit all 5 sub-models on the training dataset.

        Args:
            X: Input feature matrix (DataFrame or ndarray).
            y: Target DataFrame containing BUILDINGID, FLOOR, BUILDING_FLOOR_INDEX,
               LONGITUDE, and LATITUDE.
            sample_weight: Optional sample weights for duplicate mitigation.

        Returns:
            self
        """
        self._verify_features(X)

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        logger.info(
            "Fitting IndoorAIRandomForest on %d samples (%d features)...",
            X.shape[0],
            len(self.feature_names_)
        )

        logger.info("Training Model A: Building Classifier (BUILDINGID)...")
        self.model_building.fit(X, y["BUILDINGID"], sample_weight=sample_weight)

        logger.info("Training Model B: Floor Classifier (FLOOR)...")
        self.model_floor.fit(X, y["FLOOR"], sample_weight=sample_weight)

        logger.info("Training Model C: Building-Floor Classifier (BUILDING_FLOOR_INDEX)...")
        self.model_building_floor.fit(X, y["BUILDING_FLOOR_INDEX"], sample_weight=sample_weight)

        logger.info("Training Model D: Longitude Regressor (LONGITUDE)...")
        self.model_longitude.fit(X, y["LONGITUDE"], sample_weight=sample_weight)

        logger.info("Training Model E: Latitude Regressor (LATITUDE)...")
        self.model_latitude.fit(X, y["LATITUDE"], sample_weight=sample_weight)

        self.is_fitted_ = True
        logger.info("All 5 Random Forest models successfully fitted.")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate predictions from all 5 sub-models.

        Args:
            X: Input feature matrix.

        Returns:
            Dict containing:
            - pred_building: Predicted building IDs.
            - pred_floor: Predicted floor IDs.
            - pred_building_floor: Predicted combined building-floor indices.
            - pred_longitude: Predicted longitudes.
            - pred_latitude: Predicted latitudes.
        """
        if not self.is_fitted_:
            raise RuntimeError("IndoorAIRandomForest must be fitted before predict().")

        self._verify_features(X)

        return {
            "pred_building": self.model_building.predict(X),
            "pred_floor": self.model_floor.predict(X),
            "pred_building_floor": self.model_building_floor.predict(X),
            "pred_longitude": self.model_longitude.predict(X),
            "pred_latitude": self.model_latitude.predict(X),
        }

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """Predict class probabilities for the classification heads."""
        if not self.is_fitted_:
            raise RuntimeError("IndoorAIRandomForest must be fitted before predict_proba().")

        self._verify_features(X)

        return {
            "prob_building": self.model_building.predict_proba(X),
            "prob_floor": self.model_floor.predict_proba(X),
            "prob_building_floor": self.model_building_floor.predict_proba(X),
        }

    def get_feature_importances(self) -> Dict[str, pd.DataFrame]:
        """Extract feature importances across all 5 models."""
        if not self.is_fitted_:
            raise RuntimeError("IndoorAIRandomForest must be fitted before extracting importances.")

        names = self.feature_names_
        importances = {}

        for model_name, model in [
            ("building", self.model_building),
            ("floor", self.model_floor),
            ("building_floor", self.model_building_floor),
            ("longitude", self.model_longitude),
            ("latitude", self.model_latitude),
        ]:
            imp = model.feature_importances_
            df_imp = pd.DataFrame({
                "feature": names,
                "importance": imp
            }).sort_values(by="importance", ascending=False).reset_index(drop=True)

            # Categorize feature type
            df_imp["feature_type"] = df_imp["feature"].apply(
                lambda f: "visibility" if f.endswith("_VISIBLE") else (
                    "summary" if not f.startswith("WAP") else "rssi"
                )
            )
            importances[model_name] = df_imp

        return importances

    def save(self, models_dir: Union[str, Path]) -> None:
        """Persist all 5 trained models and metadata using joblib."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted model suite.")

        out_dir = Path(models_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model_building, out_dir / "model_building.joblib")
        joblib.dump(self.model_floor, out_dir / "model_floor.joblib")
        joblib.dump(self.model_building_floor, out_dir / "model_building_floor.joblib")
        joblib.dump(self.model_longitude, out_dir / "model_longitude.joblib")
        joblib.dump(self.model_latitude, out_dir / "model_latitude.joblib")

        metadata = {
            "feature_names": self.feature_names_,
            "random_state": self.random_state,
            "feature_count": len(self.feature_names_),
        }
        joblib.dump(metadata, out_dir / "model_metadata.joblib")
        logger.info("Saved 5 Random Forest models and metadata to %s", out_dir)

    @classmethod
    def load(cls, models_dir: Union[str, Path]) -> "IndoorAIRandomForest":
        """Load trained models from directory."""
        in_dir = Path(models_dir)
        metadata = joblib.load(in_dir / "model_metadata.joblib")

        instance = cls(random_state=metadata["random_state"])
        instance.model_building = joblib.load(in_dir / "model_building.joblib")
        instance.model_floor = joblib.load(in_dir / "model_floor.joblib")
        instance.model_building_floor = joblib.load(in_dir / "model_building_floor.joblib")
        instance.model_longitude = joblib.load(in_dir / "model_longitude.joblib")
        instance.model_latitude = joblib.load(in_dir / "model_latitude.joblib")

        instance.feature_names_ = metadata["feature_names"]
        instance.is_fitted_ = True
        logger.info("Loaded 5 Random Forest models from %s (%d features)", in_dir, len(instance.feature_names_))
        return instance

