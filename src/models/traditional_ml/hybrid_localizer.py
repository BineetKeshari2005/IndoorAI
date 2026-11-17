"""
Hybrid Indoor Localizer Module for IndoorAI (Task 10).

This module implements a Two-Stage Hierarchical Hybrid Architecture combining:
- Stage 1: Random Forest for Macro Structural Classification (Building ID and Floor ID).
- Stage 2: k-Nearest Neighbors for Micro Metric Coordinate Regression (Longitude, Latitude).

Architectural Principles:
1. Training Integrity: The local k-NN models are trained strictly on training partitions
   belonging to specific building-floor combinations.
2. Robust Fallback: If a predicted building-floor subset has insufficient training samples
   (or if Stage 1 RF confidence falls below a calibrated threshold tau), the model gracefully
   falls back to the global k-NN model.
3. Coordinate Uncertainty Estimation: Computes the spatial standard deviation (dispersion)
   of retrieved nearest neighbors as a continuous measure of spatial uncertainty.
"""

from __future__ import annotations

import json
import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.neighbors import KNeighborsRegressor

from src.core.logging import setup_logger
from src.models.traditional_ml.random_forest import IndoorAIRandomForest
from src.models.traditional_ml.knn import IndoorAIKNN, SummaryFeatureScaler

logger = setup_logger("HybridLocalizer")


class HybridLocalizer:
    """
    Two-Stage Hierarchical Hybrid Localizer.
    
    Stage 1: Random Forest classifies Building and Floor.
    Stage 2: Distance-weighted k-NN interpolates continuous coordinates from
             training fingerprints within the predicted Building-Floor subset,
             with fallback to global k-NN when confidence is low or subset is small.
    """

    def __init__(
        self,
        rf_model: Optional[IndoorAIRandomForest] = None,
        knn_global_model: Optional[IndoorAIKNN] = None,
        confidence_threshold: float = 0.70,
        min_subset_samples: int = 10,
        local_k: int = 7,
    ):
        """
        Initialize the HybridLocalizer.
        
        Args:
            rf_model: Pre-trained IndoorAIRandomForest model instance.
            knn_global_model: Pre-trained IndoorAIKNN model instance (with fitted SummaryFeatureScaler).
            confidence_threshold: Confidence threshold tau in [0.0, 1.0]. If RF confidence
                                  is below tau, falls back to global k-NN.
            min_subset_samples: Minimum training samples required in a (Building, Floor) slice.
            local_k: Number of nearest neighbors to query in the local k-NN stage.
        """
        self.rf_model = rf_model
        self.knn_global_model = knn_global_model
        self.confidence_threshold = float(confidence_threshold)
        self.min_subset_samples = int(min_subset_samples)
        self.local_k = int(local_k)

        # Dictionary of local k-NN models indexed by building_floor integer index
        self.local_knn_models: Dict[int, Dict[str, Any]] = {}
        self.subset_sample_counts: Dict[int, int] = {}
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
    ) -> "HybridLocalizer":
        """
        Fit the local k-NN models on the training dataset partitioned by building-floor.
        
        Args:
            X_train: Training feature matrix (1,058 engineered features).
            y_train: Training target DataFrame containing 'LONGITUDE', 'LATITUDE',
                     and 'BUILDING_FLOOR_INDEX'.
        """
        if self.knn_global_model is None or self.knn_global_model.summary_scaler is None:
            raise ValueError("knn_global_model with fitted summary_scaler must be provided before fitting HybridLocalizer.")

        logger.info("Fitting HybridLocalizer local k-NN models on %d training fingerprints...", len(X_train))

        # Scale summary features using the pre-fitted summary scaler
        scaler = self.knn_global_model.summary_scaler
        X_train_scaled = scaler.transform(X_train)

        bf_indices = y_train["BUILDING_FLOOR_INDEX"].values
        lon_coords = y_train["LONGITUDE"].values
        lat_coords = y_train["LATITUDE"].values

        unique_bfs = np.unique(bf_indices)
        logger.info("Found %d distinct building-floor combinations in training data.", len(unique_bfs))

        for bf_idx in sorted(unique_bfs):
            mask = (bf_indices == bf_idx)
            sub_count = int(np.sum(mask))
            self.subset_sample_counts[int(bf_idx)] = sub_count

            if sub_count < self.min_subset_samples:
                logger.warning(
                    "Building-floor index %d has only %d samples (< min %d); will fallback to global k-NN.",
                    bf_idx, sub_count, self.min_subset_samples
                )
                continue

            sub_X = X_train_scaled[mask]
            sub_lon = lon_coords[mask]
            sub_lat = lat_coords[mask]

            k = min(self.local_k, sub_count)

            knn_lon = KNeighborsRegressor(
                n_neighbors=k,
                metric="cosine",
                weights="distance",
                algorithm="brute",
            )
            knn_lat = KNeighborsRegressor(
                n_neighbors=k,
                metric="cosine",
                weights="distance",
                algorithm="brute",
            )

            knn_lon.fit(sub_X, sub_lon)
            knn_lat.fit(sub_X, sub_lat)

            self.local_knn_models[int(bf_idx)] = {
                "knn_lon": knn_lon,
                "knn_lat": knn_lat,
                "sample_count": sub_count,
                "k": k,
                "sub_lon": sub_lon,
                "sub_lat": sub_lat,
            }

        self.is_fitted = True
        logger.info("Successfully fitted %d local k-NN building-floor models.", len(self.local_knn_models))
        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """
        Execute Two-Stage Hybrid Prediction.
        
        Stage 1: RF predicts Building, Floor, Building-Floor, and class probabilities.
        Stage 2: Gated k-NN predicts Coordinates.
        
        Args:
            X: Input feature DataFrame (1,058 features).
            
        Returns:
            Dictionary containing:
                pred_building: 1D array of predicted building IDs
                pred_floor: 1D array of predicted floor IDs
                pred_building_floor: 1D array of predicted building-floor indices
                pred_longitude: 1D array of predicted longitude coordinates
                pred_latitude: 1D array of predicted latitude coordinates
                rf_confidence: 1D array of Stage 1 RF maximum class probabilities
                coord_uncertainty: 1D array of neighbor spatial standard deviations
                used_local_knn: 1D boolean array indicating whether local k-NN was used
        """
        if not self.is_fitted:
            raise RuntimeError("HybridLocalizer must be fitted before calling predict.")

        n_samples = len(X)

        # Stage 1: Random Forest predictions and probabilities
        rf_preds = self.rf_model.predict(X)
        rf_probas = self.rf_model.predict_proba(X)

        pred_b = rf_preds["pred_building"]
        pred_f = rf_preds["pred_floor"]
        pred_bf = rf_preds["pred_building_floor"]

        prob_bf = rf_probas["prob_building_floor"]
        rf_confidence = np.max(prob_bf, axis=1)

        # Scale features for k-NN
        scaler = self.knn_global_model.summary_scaler
        X_scaled_df = scaler.transform(X)
        X_scaled_arr = X_scaled_df.values if isinstance(X_scaled_df, pd.DataFrame) else X_scaled_df

        # Global k-NN predictions for fallback and comparison
        global_knn_preds = self.knn_global_model.predict(X)
        global_lon = global_knn_preds["pred_longitude"]
        global_lat = global_knn_preds["pred_latitude"]

        # Global neighbor distances and dispersion (global k-NN fitted on array without feature names)
        global_dists, global_indices = self.knn_global_model.model_longitude.kneighbors(
            X_scaled_arr, n_neighbors=self.local_k, return_distance=True
        )

        pred_lon = np.zeros(n_samples, dtype=np.float64)
        pred_lat = np.zeros(n_samples, dtype=np.float64)
        coord_uncertainty = np.zeros(n_samples, dtype=np.float64)
        used_local_knn = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            chosen_bf = int(pred_bf[i])
            conf_val = float(rf_confidence[i])

            # Gating condition: confidence >= threshold and valid local model available
            if conf_val >= self.confidence_threshold and chosen_bf in self.local_knn_models:
                local_entry = self.local_knn_models[chosen_bf]
                knn_lon = local_entry["knn_lon"]
                knn_lat = local_entry["knn_lat"]

                sample_x = (
                    X_scaled_df.iloc[i : i + 1]
                    if isinstance(X_scaled_df, pd.DataFrame)
                    else X_scaled_arr[i : i + 1]
                )
                pred_lon[i] = knn_lon.predict(sample_x)[0]
                pred_lat[i] = knn_lat.predict(sample_x)[0]

                # Compute local neighbor spatial dispersion
                _, local_indices = knn_lon.kneighbors(sample_x, return_distance=True)
                neighbor_lons = local_entry["sub_lon"][local_indices[0]]
                neighbor_lats = local_entry["sub_lat"][local_indices[0]]
                coord_uncertainty[i] = np.sqrt(np.std(neighbor_lons) ** 2 + np.std(neighbor_lats) ** 2)
                used_local_knn[i] = True
            else:
                # Fallback to global k-NN
                pred_lon[i] = global_lon[i]
                pred_lat[i] = global_lat[i]

                # Compute global neighbor spatial dispersion
                g_idx = global_indices[i]
                train_lons = self.knn_global_model.model_longitude._y[g_idx]
                train_lats = self.knn_global_model.model_latitude._y[g_idx]
                coord_uncertainty[i] = np.sqrt(np.std(train_lons) ** 2 + np.std(train_lats) ** 2)
                used_local_knn[i] = False

        return {
            "pred_building": pred_b,
            "pred_floor": pred_f,
            "pred_building_floor": pred_bf,
            "pred_longitude": pred_lon,
            "pred_latitude": pred_lat,
            "rf_confidence": rf_confidence,
            "coord_uncertainty": coord_uncertainty,
            "used_local_knn": used_local_knn,
        }

    def save(self, output_dir: str) -> None:
        """Serialize hybrid model components to disk."""
        os.makedirs(output_dir, exist_ok=True)
        meta = {
            "confidence_threshold": self.confidence_threshold,
            "min_subset_samples": self.min_subset_samples,
            "local_k": self.local_k,
            "subset_sample_counts": self.subset_sample_counts,
            "fitted_subsets": list(self.local_knn_models.keys()),
        }
        with open(os.path.join(output_dir, "hybrid_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        joblib.dump(self.local_knn_models, os.path.join(output_dir, "local_knn_models.joblib"))
        logger.info("Saved HybridLocalizer artifacts to %s", output_dir)

    @classmethod
    def load(
        cls,
        model_dir: str,
        rf_model: IndoorAIRandomForest,
        knn_global_model: IndoorAIKNN,
    ) -> "HybridLocalizer":
        """Load hybrid model components from disk."""
        with open(os.path.join(model_dir, "hybrid_metadata.json")) as f:
            meta = json.load(f)

        local_knn_models = joblib.load(os.path.join(model_dir, "local_knn_models.joblib"))

        instance = cls(
            rf_model=rf_model,
            knn_global_model=knn_global_model,
            confidence_threshold=meta["confidence_threshold"],
            min_subset_samples=meta["min_subset_samples"],
            local_k=meta["local_k"],
        )
        instance.local_knn_models = local_knn_models
        instance.subset_sample_counts = meta["subset_sample_counts"]
        instance.is_fitted = True
        logger.info("Loaded HybridLocalizer from %s (%d local subsets)", model_dir, len(local_knn_models))
        return instance

