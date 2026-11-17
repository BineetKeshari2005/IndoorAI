"""Feature scaling module for IndoorAI with strict data leakage prevention.

Supports MinMaxScaler, StandardScaler, RobustScaler, and identity passthrough.
Ensures that scalers are fitted exclusively on the training partition and persists
fitted artifacts for production inference.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.core.logging import setup_logger

logger = setup_logger("FeatureScaler")


class IdentityScaler(BaseEstimator, TransformerMixin):
    """Passthrough scaler when scaling_method is 'none'."""

    def fit(self, X: Any, y: Any = None) -> "IdentityScaler":
        return self

    def transform(self, X: Any) -> Any:
        return X


def get_scaler_instance(
    method: str = "minmax",
    feature_range: Tuple[float, float] = (0.0, 1.0)
) -> TransformerMixin:
    """Instantiate the requested scikit-learn scaler.

    Args:
        method: 'minmax', 'standard', 'robust', or 'none'.
        feature_range: Range for MinMaxScaler (default (0.0, 1.0)).

    Returns:
        Configured unfitted scaler instance.
    """
    resolved_method = method.lower().strip()
    if resolved_method == "minmax":
        return MinMaxScaler(feature_range=feature_range)
    elif resolved_method == "standard":
        return StandardScaler()
    elif resolved_method == "robust":
        return RobustScaler()
    elif resolved_method in ("none", "identity", "passthrough"):
        return IdentityScaler()
    else:
        raise ValueError(
            f"Unsupported scaling method '{method}'. Valid options: ['minmax', 'standard', 'robust', 'none']"
        )


def fit_and_transform_features(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    method: str = "minmax",
    feature_range: Tuple[float, float] = (0.0, 1.0),
    save_artifact_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], TransformerMixin]:
    """Fit scaler strictly on X_train, transform X_train and X_val, and handle constant features.

    LEAKAGE INVARIANT: X_val is NEVER passed to scaler.fit(). It is transformed
    purely using the already fitted X_train parameters.

    Args:
        X_train: Training feature DataFrame.
        X_val: Optional validation feature DataFrame.
        method: 'minmax', 'standard', 'robust', or 'none'.
        feature_range: Feature bounds if method='minmax'.
        save_artifact_path: Optional path to save scaler.joblib.

    Returns:
        Tuple of (scaled_X_train, scaled_X_val, fitted_scaler).
    """
    scaler = get_scaler_instance(method, feature_range)
    col_names = list(X_train.columns)

    logger.info("Fitting '%s' scaler strictly on %d training samples (%d features)...", method, len(X_train), len(col_names))
    X_train_scaled_arr = scaler.fit_transform(X_train.values)

    # Defensively verify no NaNs were generated (e.g., from zero-variance columns)
    if np.isnan(X_train_scaled_arr).any() or np.isinf(X_train_scaled_arr).any():
        logger.warning("Detected NaN or Inf in scaled training data. Sanitizing to 0.0.")
        X_train_scaled_arr = np.nan_to_num(X_train_scaled_arr, nan=0.0, posinf=1.0, neginf=0.0)

    X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=col_names, index=X_train.index)

    X_val_scaled: Optional[pd.DataFrame] = None
    if X_val is not None:
        logger.info("Transforming %d validation samples using fitted training scaler...", len(X_val))
        # Ensure validation features match training column order exactly
        X_val_aligned = X_val[col_names]
        X_val_scaled_arr = scaler.transform(X_val_aligned.values)

        if np.isnan(X_val_scaled_arr).any() or np.isinf(X_val_scaled_arr).any():
            logger.warning("Detected NaN or Inf in scaled validation data. Sanitizing to 0.0.")
            X_val_scaled_arr = np.nan_to_num(X_val_scaled_arr, nan=0.0, posinf=1.0, neginf=0.0)

        X_val_scaled = pd.DataFrame(X_val_scaled_arr, columns=col_names, index=X_val.index)

    if save_artifact_path:
        save_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_artifact_path)
        logger.info("Saved fitted scaler artifact to %s", save_artifact_path)

    return X_train_scaled, X_val_scaled, scaler

