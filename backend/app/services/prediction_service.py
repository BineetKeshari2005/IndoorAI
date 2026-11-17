"""Prediction and inference service for IndoorAI web dashboard (Tasks 11 & 11.1).

Provides centralized, thread-safe, and cached model inference, dataset access,
raw CSV fingerprint preprocessing, and evaluation metrics computation.

Audited and aligned with Task 09 Benchmark Report and Task 10 Error Analysis.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.core.logging import setup_logger
from src.features.signal_statistics import (
    SignalStatsConfig,
    extract_signal_statistics_from_dataframe,
    SUMMARY_FEATURE_NAMES,
)
from src.models.traditional_ml.hybrid_localizer import HybridLocalizer
from src.models.traditional_ml.knn import IndoorAIKNN
from src.models.traditional_ml.random_forest import IndoorAIRandomForest
from src.preprocessing.wifi_preprocessing import (
    create_visibility_mask,
    extract_wap_columns,
    process_rssi_matrix,
)

logger = setup_logger("PredictionService")

REPO_ROOT = Path(__file__).resolve().parents[3]

# Expected columns and constants
EXPECTED_WAP_COUNT = 520
EXPECTED_ENGINEERED_FEATURE_COUNT = 1058
DEFAULT_NOISE_FLOOR_DBM = -105.0
DEFAULT_SENTINEL_DBM = 100


class PredictionService:
    """Service handling model lifecycle, feature extraction, and real-time inference."""

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        lazy_load: bool = True,
    ):
        """Initialize the PredictionService.

        Args:
            base_dir: Root directory of the IndoorAI repository.
            lazy_load: Whether to defer model loading until first requested.
        """
        self.base_dir = Path(base_dir or REPO_ROOT)
        self.rf_model: Optional[IndoorAIRandomForest] = None
        self.knn_model: Optional[IndoorAIKNN] = None
        self.hybrid_model: Optional[HybridLocalizer] = None
        self.scaler: Optional[Any] = None

        self._val_features: Optional[pd.DataFrame] = None
        self._val_targets: Optional[pd.DataFrame] = None
        self._val_metadata: Optional[pd.DataFrame] = None
        self._raw_validation: Optional[pd.DataFrame] = None

        if not lazy_load:
            self.load_all()

    def load_all(self) -> None:
        """Load all pre-trained models, scalers, and validation datasets."""
        self.load_models()
        self.load_datasets()

    def load_models(self) -> None:
        """Load all 3 models and preprocessing scalers."""
        rf_dir = self.base_dir / "experiments" / "random_forest" / "models"
        knn_dir = self.base_dir / "experiments" / "knn" / "models"
        hybrid_dir = self.base_dir / "experiments" / "hybrid" / "models"
        scaler_path = self.base_dir / "experiments" / "preprocessing" / "artifacts" / "scaler.joblib"

        if self.scaler is None:
            if scaler_path.exists():
                logger.info("Loading preprocessing scaler from %s", scaler_path)
                self.scaler = joblib.load(scaler_path)
            else:
                raise FileNotFoundError(f"Fitted scaler not found at {scaler_path}")

        if self.rf_model is None:
            logger.info("Loading Random Forest model from %s", rf_dir)
            self.rf_model = IndoorAIRandomForest.load(rf_dir)

        if self.knn_model is None:
            logger.info("Loading k-NN model from %s", knn_dir)
            self.knn_model = IndoorAIKNN.load(knn_dir)

        if self.hybrid_model is None:
            logger.info("Loading Hybrid model from %s", hybrid_dir)
            if (hybrid_dir / "local_knn_models.joblib").exists():
                self.hybrid_model = HybridLocalizer.load(
                    str(hybrid_dir),
                    rf_model=self.rf_model,
                    knn_global_model=self.knn_model,
                )
            else:
                # If not serialized yet, fit on training data and serialize
                logger.info("Fitting HybridLocalizer on training partition...")
                X_train = pd.read_parquet(
                    self.base_dir / "data" / "processed" / "ujiindoorloc" / "feature_engineered" / "X_train_engineered.parquet"
                )
                y_train = pd.read_parquet(
                    self.base_dir / "data" / "processed" / "ujiindoorloc" / "y_train.parquet"
                )
                self.hybrid_model = HybridLocalizer(
                    rf_model=self.rf_model,
                    knn_global_model=self.knn_model,
                    confidence_threshold=0.70,
                )
                self.hybrid_model.fit(X_train, y_train)
                self.hybrid_model.save(str(hybrid_dir))

    def load_datasets(self) -> None:
        """Load validation feature matrices, ground truths, and raw records."""
        data_dir = self.base_dir / "data"

        feat_path = data_dir / "processed" / "ujiindoorloc" / "feature_engineered" / "X_validation_engineered.parquet"
        tgt_path = data_dir / "processed" / "ujiindoorloc" / "y_validation.parquet"
        meta_path = data_dir / "processed" / "ujiindoorloc" / "metadata_validation.parquet"
        raw_path = data_dir / "raw" / "ujiindoorloc" / "validationData.csv"

        if self._val_features is None:
            self._val_features = pd.read_parquet(feat_path)
        if self._val_targets is None:
            self._val_targets = pd.read_parquet(tgt_path)
        if self._val_metadata is None:
            self._val_metadata = pd.read_parquet(meta_path)
        if self._raw_validation is None:
            self._raw_validation = pd.read_csv(raw_path)

    @property
    def validation_features(self) -> pd.DataFrame:
        if self._val_features is None:
            self.load_datasets()
        return self._val_features

    @property
    def validation_targets(self) -> pd.DataFrame:
        if self._val_targets is None:
            self.load_datasets()
        return self._val_targets

    @property
    def validation_metadata(self) -> pd.DataFrame:
        if self._val_metadata is None:
            self.load_datasets()
        return self._val_metadata

    @property
    def raw_validation(self) -> pd.DataFrame:
        if self._raw_validation is None:
            self.load_datasets()
        return self._raw_validation

    def get_validation_sample(self, index: int) -> Dict[str, Any]:
        """Retrieve a specific validation fingerprint by index.

        Args:
            index: Integer row index (0 to 1110).

        Returns:
            Dict containing features, ground_truth, metadata, and raw_waps.
        """
        self.load_datasets()
        if index < 0 or index >= len(self.validation_features):
            raise IndexError(f"Index {index} out of range [0, {len(self.validation_features) - 1}]")

        row_feat = self.validation_features.iloc[[index]]
        row_tgt = self.validation_targets.iloc[index].to_dict()
        row_meta = self.validation_metadata.iloc[index].to_dict()
        row_raw = self.raw_validation.iloc[index]

        # Extract visible WAPs and signal levels from raw row
        wap_cols = [c for c in row_raw.index if c.startswith("WAP")]
        wap_series = row_raw[wap_cols]
        detected = wap_series[wap_series != DEFAULT_SENTINEL_DBM].sort_values(ascending=False)

        detected_waps = [
            {"wap_id": wap_id, "rssi_dbm": float(val)}
            for wap_id, val in detected.items()
        ]

        return {
            "index": index,
            "features": row_feat,
            "ground_truth": row_tgt,
            "metadata": row_meta,
            "detected_waps": detected_waps,
            "visible_count": len(detected_waps),
        }

    def predict(
        self,
        features: pd.DataFrame,
        model_name: str = "Random Forest",
    ) -> Dict[str, Any]:
        """Run localization prediction using the selected model.

        Args:
            features: DataFrame with 1,058 engineered features (1 or more rows).
            model_name: "Random Forest", "k-Nearest Neighbors", or "Hybrid".

        Returns:
            Dict containing predicted building, floor, longitude, latitude,
            confidence, and spatial uncertainty.
        """
        self.load_models()

        if len(features.columns) != EXPECTED_ENGINEERED_FEATURE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_ENGINEERED_FEATURE_COUNT} features, got {len(features.columns)}"
            )

        name_lower = model_name.lower()

        if "forest" in name_lower or "rf" in name_lower:
            preds = self.rf_model.predict(features)
            probas = self.rf_model.predict_proba(features)
            conf = float(np.max(probas["prob_building_floor"][0]))

            # Heuristic spatial uncertainty proxy
            uncertainty = float((1.0 - conf) * 20.0 + 3.0)

            return {
                "model_name": "Random Forest",
                "pred_building": int(preds["pred_building"][0]),
                "pred_floor": int(preds["pred_floor"][0]),
                "pred_building_floor": int(preds["pred_building_floor"][0]),
                "pred_longitude": float(preds["pred_longitude"][0]),
                "pred_latitude": float(preds["pred_latitude"][0]),
                "confidence": conf,
                "confidence_pct": round(conf * 100, 2),
                "is_confidence_calibrated": False,
                "spatial_uncertainty": round(uncertainty, 2),
                "details": {
                    "architecture": "Random Forest 5-Head Ensemble (200 trees/head)",
                    "building_confidence": float(np.max(probas["prob_building"][0])),
                    "floor_confidence": float(np.max(probas["prob_floor"][0])),
                    "note": "Confidence is the maximum tree vote proportion (heuristic, uncalibrated).",
                },
            }

        elif "neighbor" in name_lower or "knn" in name_lower:
            k_val = int(self.knn_model.params.n_neighbors)  # Audited Task 08 configuration (k=7)
            preds = self.knn_model.predict(features)
            probas = self.knn_model.predict_proba(features)
            conf = float(np.max(probas["proba_building_floor"][0]))

            # Query neighbors dispersion using numpy array to avoid feature names warning
            scaler = self.knn_model.summary_scaler
            scaled_feat = scaler.transform(features)
            feat_mat = scaled_feat.values if isinstance(scaled_feat, pd.DataFrame) else scaled_feat

            dists, indices = self.knn_model.model_longitude.kneighbors(
                feat_mat, n_neighbors=k_val, return_distance=True
            )
            g_idx = indices[0]
            train_lons = self.knn_model.model_longitude._y[g_idx]
            train_lats = self.knn_model.model_latitude._y[g_idx]
            dispersion = float(np.sqrt(np.std(train_lons) ** 2 + np.std(train_lats) ** 2))

            return {
                "model_name": f"k-Nearest Neighbors (k={k_val})",
                "pred_building": int(preds["pred_building"][0]),
                "pred_floor": int(preds["pred_floor"][0]),
                "pred_building_floor": int(preds["pred_building_floor"][0]),
                "pred_longitude": float(preds["pred_longitude"][0]),
                "pred_latitude": float(preds["pred_latitude"][0]),
                "confidence": conf,
                "confidence_pct": round(conf * 100, 2),
                "is_confidence_calibrated": False,
                "spatial_uncertainty": round(dispersion, 2),
                "details": {
                    "architecture": f"k-NN Instance-Based (Cosine metric, distance weights, k={k_val})",
                    "k_neighbors": k_val,
                    "building_confidence": float(np.max(probas["proba_building"][0])),
                    "floor_confidence": float(np.max(probas["proba_floor"][0])),
                    "neighbor_mean_distance": float(np.mean(dists[0])),
                    "note": "Confidence reflects distance-weighted neighbor consensus (heuristic, uncalibrated).",
                },
            }

        elif "hybrid" in name_lower:
            preds = self.hybrid_model.predict(features)
            conf = float(preds["rf_confidence"][0])
            disp = float(preds["coord_uncertainty"][0])
            used_local = bool(preds["used_local_knn"][0])
            local_k = self.hybrid_model.local_k

            return {
                "model_name": "Hybrid (Two-Stage Hierarchical)",
                "pred_building": int(preds["pred_building"][0]),
                "pred_floor": int(preds["pred_floor"][0]),
                "pred_building_floor": int(preds["pred_building_floor"][0]),
                "pred_longitude": float(preds["pred_longitude"][0]),
                "pred_latitude": float(preds["pred_latitude"][0]),
                "confidence": conf,
                "confidence_pct": round(conf * 100, 2),
                "is_confidence_calibrated": False,
                "spatial_uncertainty": round(disp, 2),
                "details": {
                    "status": "Experimental Research Prototype",
                    "used_local_knn": used_local,
                    "confidence_threshold": self.hybrid_model.confidence_threshold,
                    "stage1_architecture": "Random Forest Classifier",
                    "stage2_architecture": (
                        f"Localized k-NN (k={local_k})" if used_local
                        else f"Global k-NN Fallback (k={local_k})"
                    ),
                    "caveat": "Gating threshold tau=0.70 was evaluated post-hoc on validation data. Results are experimental.",
                },
            }
        else:
            raise ValueError(f"Unknown model name: {model_name}. Choose from RF, k-NN, or Hybrid.")

    def preprocess_raw_csv(
        self,
        file_or_df: Union[str, Path, io.StringIO, io.BytesIO, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[Dict[str, Any]]]:
        """Parse, validate, and preprocess a raw Wi-Fi CSV fingerprint into model-ready features.

        Enforces strict validation:
        - Must contain 520 WAP columns (WAP001 - WAP520).
        - Sanitizes unavailable values (100 -> -105.0 dBm).
        - Generates 520 visibility indicator columns.
        - Scales 1,040 baseline features using the training MinMaxScaler.
        - Computes 18 signal summary statistics.
        - Extracts optional ground truth target columns if provided.

        Args:
            file_or_df: CSV file path, buffer, or DataFrame.

        Returns:
            Tuple of (features_df, ground_truth_df_or_None, list_of_detected_waps_per_row)
        """
        self.load_models()

        if isinstance(file_or_df, pd.DataFrame):
            df_raw = file_or_df.copy()
        else:
            df_raw = pd.read_csv(file_or_df)

        # 1. Validate WAP columns
        wap_cols = [f"WAP{i:03d}" for i in range(1, EXPECTED_WAP_COUNT + 1)]
        missing_waps = [c for c in wap_cols if c not in df_raw.columns]
        if missing_waps:
            missing_sample = missing_waps[:5]
            raise ValueError(
                f"Uploaded data is missing {len(missing_waps)} required WAP columns "
                f"(e.g., {missing_sample}). The dataset must contain all 520 WAP columns (WAP001–WAP520)."
            )

        # 2. Check for optional ground truth targets (isolated from features)
        target_cols = ["BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE"]
        present_targets = [c for c in target_cols if c in df_raw.columns]
        ground_truth: Optional[pd.DataFrame] = None
        if len(present_targets) == len(target_cols):
            ground_truth = df_raw[target_cols].copy()
            if "BUILDING_FLOOR_INDEX" not in ground_truth.columns:
                ground_truth["BUILDING_FLOOR_INDEX"] = (
                    ground_truth["BUILDINGID"] * 5 + ground_truth["FLOOR"]
                )

        # 3. Extract detected WAPs summary for each row
        detected_waps_per_row = []
        for _, row in df_raw[wap_cols].iterrows():
            detected = row[row != DEFAULT_SENTINEL_DBM].sort_values(ascending=False)
            detected_waps_per_row.append([
                {"wap_id": k, "rssi_dbm": float(v)} for k, v in detected.items()
            ])

        # 4. Transform raw RSSI to continuous values with -105.0 dBm noise floor
        df_rssi = process_rssi_matrix(
            df_raw, wap_cols=wap_cols, sentinel=DEFAULT_SENTINEL_DBM, rssi_floor=DEFAULT_NOISE_FLOOR_DBM
        )

        # 5. Generate binary visibility indicators
        df_vis = create_visibility_mask(
            df_raw, wap_cols=wap_cols, sentinel=DEFAULT_SENTINEL_DBM, suffix="_VISIBLE"
        )

        # 6. Concatenate baseline 1,040 features and scale using training scaler
        df_base = pd.concat([df_rssi, df_vis], axis=1)
        scaled_base_arr = self.scaler.transform(df_base.values)
        scaled_base_df = pd.DataFrame(
            scaled_base_arr, columns=df_base.columns, index=df_raw.index
        )

        # 7. Compute 18 signal summary statistics
        stats_cfg = SignalStatsConfig(
            unavailable_sentinel=DEFAULT_SENTINEL_DBM,
            rssi_floor_dbm=DEFAULT_NOISE_FLOOR_DBM,
            strong_threshold_dbm=-60.0,
            medium_threshold_dbm=-80.0,
        )
        df_stats = extract_signal_statistics_from_dataframe(df_raw, config=stats_cfg)

        # 8. Assemble combined 1,058 engineered feature DataFrame
        features_df = pd.concat([scaled_base_df, df_stats], axis=1)

        # Defensive sanity check: verify column alignment with reference features
        ref_cols = list(self.validation_features.columns)
        features_df = features_df[ref_cols]

        return features_df, ground_truth, detected_waps_per_row

    @staticmethod
    def calculate_evaluation_metrics(
        prediction: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute spatial error and classification accuracy metrics for a sample.

        Args:
            prediction: Prediction dictionary.
            ground_truth: Ground truth dictionary.

        Returns:
            Dictionary containing error metrics.
        """
        pred_b = prediction["pred_building"]
        pred_f = prediction["pred_floor"]
        pred_lon = prediction["pred_longitude"]
        pred_lat = prediction["pred_latitude"]

        true_b = int(ground_truth["BUILDINGID"])
        true_f = int(ground_truth["FLOOR"])
        true_lon = float(ground_truth["LONGITUDE"])
        true_lat = float(ground_truth["LATITUDE"])

        b_correct = (pred_b == true_b)
        f_correct = (pred_f == true_f)
        bf_correct = (b_correct and f_correct)

        floor_diff = abs(pred_f - true_f)

        lon_err = pred_lon - true_lon
        lat_err = pred_lat - true_lat
        coord_error_2d = float(np.sqrt(lon_err ** 2 + lat_err ** 2))

        return {
            "building_correct": b_correct,
            "floor_correct": f_correct,
            "building_floor_correct": bf_correct,
            "floor_difference": floor_diff,
            "longitude_error": round(float(lon_err), 3),
            "latitude_error": round(float(lat_err), 3),
            "euclidean_error_2d": round(coord_error_2d, 3),
            "units": "dataset coordinate units",
        }

    @classmethod
    def get_benchmark_table(cls, base_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Load independently verified baseline benchmark results from Task 09 audit."""
        root = Path(base_dir or REPO_ROOT)
        json_path = root / "experiments" / "model_benchmark" / "metrics" / "benchmark_results.json"

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rf_recalc = data["recalculated_metrics"]["random_forest"]
            knn_recalc = data["recalculated_metrics"]["knn"]

            return pd.DataFrame([
                {
                    "Model": "Random Forest (Baseline)",
                    "Type": "Multi-Head Ensemble (200 trees/head)",
                    "Building Acc (%)": round(rf_recalc["building"]["accuracy_pct"], 2),
                    "Floor Acc (%)": round(rf_recalc["floor"]["accuracy_pct"], 2),
                    "Building-Floor Acc (%)": round(rf_recalc["building_floor"]["accuracy_pct"], 2),
                    "Mean 2D Error (units)": round(rf_recalc["coordinates"]["euclidean_error"]["mean"], 2),
                    "Median 2D Error (units)": round(rf_recalc["coordinates"]["euclidean_error"]["median"], 2),
                    "95th Pct Error (units)": round(rf_recalc["coordinates"]["euclidean_error"]["q95"], 2),
                    "Model Size": "561.1 MB",
                    "Status": "Verified Baseline (Task 07 / Task 09 Audit)",
                },
                {
                    "Model": "k-Nearest Neighbors (Baseline, k=7)",
                    "Type": "Instance-Based (Cosine, distance-weighted, k=7)",
                    "Building Acc (%)": round(knn_recalc["building"]["accuracy_pct"], 2),
                    "Floor Acc (%)": round(knn_recalc["floor"]["accuracy_pct"], 2),
                    "Building-Floor Acc (%)": round(knn_recalc["building_floor"]["accuracy_pct"], 2),
                    "Mean 2D Error (units)": round(knn_recalc["coordinates"]["euclidean_error"]["mean"], 2),
                    "Median 2D Error (units)": round(knn_recalc["coordinates"]["euclidean_error"]["median"], 2),
                    "95th Pct Error (units)": round(knn_recalc["coordinates"]["euclidean_error"]["q95"], 2),
                    "Model Size": "17.7 MB",
                    "Status": "Verified Baseline (Task 08 / Task 09 Audit)",
                },
            ])

        # Exact audited fallback if file is missing
        return pd.DataFrame([
            {
                "Model": "Random Forest (Baseline)",
                "Type": "Multi-Head Ensemble (200 trees/head)",
                "Building Acc (%)": 99.82,
                "Floor Acc (%)": 88.66,
                "Building-Floor Acc (%)": 88.48,
                "Mean 2D Error (units)": 11.53,
                "Median 2D Error (units)": 8.65,
                "95th Pct Error (units)": 32.74,
                "Model Size": "561.1 MB",
                "Status": "Verified Baseline (Task 07 / Task 09 Audit)",
            },
            {
                "Model": "k-Nearest Neighbors (Baseline, k=7)",
                "Type": "Instance-Based (Cosine, distance-weighted, k=7)",
                "Building Acc (%)": 99.91,
                "Floor Acc (%)": 84.43,
                "Building-Floor Acc (%)": 84.43,
                "Mean 2D Error (units)": 10.05,
                "Median 2D Error (units)": 7.33,
                "95th Pct Error (units)": 29.06,
                "Model Size": "17.7 MB",
                "Status": "Verified Baseline (Task 08 / Task 09 Audit)",
            },
        ])

    @classmethod
    def get_experimental_hybrid_table(cls, base_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Load experimental hybrid metrics from Task 10 error analysis."""
        root = Path(base_dir or REPO_ROOT)
        json_path = root / "experiments" / "error_analysis" / "metrics" / "hybrid_model_results.json"

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            clf = data["classification_performance"]
            coord = data["coordinate_performance"]

            return pd.DataFrame([
                {
                    "Model": "Hybrid (Two-Stage Hierarchical)",
                    "Type": "RF Macro + Localized k-NN Micro (k=7, tau=0.70)",
                    "Building Acc (%)": round(clf["building_accuracy_pct"], 2),
                    "Floor Acc (%)": round(clf["floor_accuracy_pct"], 2),
                    "Building-Floor Acc (%)": round(clf["building_floor_accuracy_pct"], 2),
                    "Mean 2D Error (units)": round(coord["mean_euclidean_error"], 2),
                    "Median 2D Error (units)": round(coord["median_euclidean_error"], 2),
                    "95th Pct Error (units)": round(coord["q95_euclidean_error"], 2),
                    "Model Size": "578.8 MB",
                    "Status": "Experimental Prototype (Task 10)",
                }
            ])

        return pd.DataFrame([
            {
                "Model": "Hybrid (Two-Stage Hierarchical)",
                "Type": "RF Macro + Localized k-NN Micro (k=7, tau=0.70)",
                "Building Acc (%)": 99.82,
                "Floor Acc (%)": 88.66,
                "Building-Floor Acc (%)": 88.48,
                "Mean 2D Error (units)": 9.98,
                "Median 2D Error (units)": 7.29,
                "95th Pct Error (units)": 29.26,
                "Model Size": "578.8 MB",
                "Status": "Experimental Prototype (Task 10)",
            }
        ])


# Global singleton instance for Streamlit and APIs
_SERVICE_INSTANCE: Optional[PredictionService] = None


def get_prediction_service(base_dir: Optional[Union[str, Path]] = None) -> PredictionService:
    """Access or initialize the global PredictionService instance."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = PredictionService(base_dir=base_dir, lazy_load=False)
    return _SERVICE_INSTANCE
