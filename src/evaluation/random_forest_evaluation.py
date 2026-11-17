"""Evaluation metrics, diagnostic analysis, and publication plotting for Random Forest models.

Computes:
1. Classification metrics: Accuracy, Precision, Recall, F1 (macro & weighted), confusion matrices.
2. Regression metrics: MAE, RMSE, MedAE, R², and 2D Euclidean coordinate errors.
3. Hierarchical breakdowns by Building and Floor.
4. Publication-quality diagnostic plots (confusion matrices, feature importance, coordinate scatter).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from src.core.logging import setup_logger

logger = setup_logger("RFEvaluation")


def compute_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    target_name: str,
    class_labels: Optional[List[Any]] = None,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute comprehensive classification performance metrics.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        target_name: Identifier for target (e.g. 'building', 'floor', 'building_floor').
        class_labels: Explicit class labels for confusion matrix and report.
        target_names: Human-readable class names.

    Returns:
        Structured dictionary of classification performance metrics.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if class_labels is None:
        class_labels = sorted(list(np.unique(y_t)))

    acc = float(accuracy_score(y_t, y_p))
    prec_macro = float(precision_score(y_t, y_p, labels=class_labels, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_t, y_p, labels=class_labels, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_t, y_p, labels=class_labels, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_t, y_p, labels=class_labels, average="weighted", zero_division=0))
    f1_mac = float(f1_score(y_t, y_p, labels=class_labels, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_t, y_p, labels=class_labels, average="weighted", zero_division=0))

    cm = confusion_matrix(y_t, y_p, labels=class_labels)
    clf_rep = classification_report(
        y_t,
        y_p,
        labels=class_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    return {
        "target_name": target_name,
        "sample_count": len(y_t),
        "accuracy": round(acc, 6),
        "accuracy_pct": round(acc * 100.0, 2),
        "precision_macro": round(prec_macro, 6),
        "precision_weighted": round(prec_weighted, 6),
        "recall_macro": round(rec_macro, 6),
        "recall_weighted": round(rec_weighted, 6),
        "f1_macro": round(f1_mac, 6),
        "f1_weighted": round(f1_wt, 6),
        "class_labels": [str(c) for c in class_labels],
        "confusion_matrix": cm.tolist(),
        "per_class_report": clf_rep,
    }


def compute_regression_metrics(
    y_true_lon: Union[pd.Series, np.ndarray],
    y_pred_lon: Union[pd.Series, np.ndarray],
    y_true_lat: Union[pd.Series, np.ndarray],
    y_pred_lat: Union[pd.Series, np.ndarray]
) -> Dict[str, Any]:
    """Compute 2D coordinate regression metrics and Euclidean spatial errors.

    Note: Euclidean coordinate error is reported strictly in dataset coordinate units
    because CRS planar metric units are unverified.

    Args:
        y_true_lon: Ground truth longitudes.
        y_pred_lon: Predicted longitudes.
        y_true_lat: Ground truth latitudes.
        y_pred_lat: Predicted latitudes.

    Returns:
        Structured dictionary of regression metrics.
    """
    t_lon = np.asarray(y_true_lon, dtype=np.float64)
    p_lon = np.asarray(y_pred_lon, dtype=np.float64)
    t_lat = np.asarray(y_true_lat, dtype=np.float64)
    p_lat = np.asarray(y_pred_lat, dtype=np.float64)

    lon_mae = float(mean_absolute_error(t_lon, p_lon))
    lon_rmse = float(root_mean_squared_error(t_lon, p_lon))
    lon_medae = float(median_absolute_error(t_lon, p_lon))
    lon_r2 = float(r2_score(t_lon, p_lon))

    lat_mae = float(mean_absolute_error(t_lat, p_lat))
    lat_rmse = float(root_mean_squared_error(t_lat, p_lat))
    lat_medae = float(median_absolute_error(t_lat, p_lat))
    lat_r2 = float(r2_score(t_lat, p_lat))

    # 2D Euclidean distance in dataset coordinate units
    euclidean_errors = np.sqrt((p_lon - t_lon) ** 2 + (p_lat - t_lat) ** 2)

    eucl_mean = float(np.mean(euclidean_errors))
    eucl_median = float(np.median(euclidean_errors))
    eucl_std = float(np.std(euclidean_errors))
    eucl_min = float(np.min(euclidean_errors))
    eucl_max = float(np.max(euclidean_errors))
    eucl_q25 = float(np.percentile(euclidean_errors, 25))
    eucl_q75 = float(np.percentile(euclidean_errors, 75))
    eucl_q90 = float(np.percentile(euclidean_errors, 90))
    eucl_q95 = float(np.percentile(euclidean_errors, 95))

    return {
        "sample_count": len(t_lon),
        "longitude": {
            "mae": round(lon_mae, 4),
            "rmse": round(lon_rmse, 4),
            "medae": round(lon_medae, 4),
            "r2": round(lon_r2, 4),
        },
        "latitude": {
            "mae": round(lat_mae, 4),
            "rmse": round(lat_rmse, 4),
            "medae": round(lat_medae, 4),
            "r2": round(lat_r2, 4),
        },
        "euclidean_error": {
            "unit": "dataset_coordinate_units",
            "mean": round(eucl_mean, 4),
            "median": round(eucl_median, 4),
            "std": round(eucl_std, 4),
            "min": round(eucl_min, 4),
            "max": round(eucl_max, 4),
            "q25": round(eucl_q25, 4),
            "q75": round(eucl_q75, 4),
            "q90": round(eucl_q90, 4),
            "q95": round(eucl_q95, 4),
        },
    }


def compute_hierarchical_breakdown(
    y_true: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    idx_to_bf_map: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Compute performance breakdown stratified by building and floor.

    Args:
        y_true: DataFrame with true BUILDINGID, FLOOR, BUILDING_FLOOR_INDEX, LONGITUDE, LATITUDE.
        predictions: Dict containing 'pred_building', 'pred_floor', 'pred_building_floor',
                     'pred_longitude', 'pred_latitude'.
        idx_to_bf_map: Mapping from building_floor index string to 'B_F' string.

    Returns:
        Structured dictionary of hierarchical metrics.
    """
    df = y_true.copy()
    for k, v in predictions.items():
        df[k] = v

    df["building_correct"] = (df["BUILDINGID"] == df["pred_building"]).astype(int)
    df["floor_correct"] = (df["FLOOR"] == df["pred_floor"]).astype(int)
    df["building_floor_correct"] = (df["BUILDING_FLOOR_INDEX"] == df["pred_building_floor"]).astype(int)
    df["euclidean_error"] = np.sqrt(
        (df["pred_longitude"] - df["LONGITUDE"]) ** 2 +
        (df["pred_latitude"] - df["LATITUDE"]) ** 2
    )

    # Breakdown by Building
    by_building = {}
    for b_id in sorted(df["BUILDINGID"].unique()):
        sub = df[df["BUILDINGID"] == b_id]
        by_building[f"building_{b_id}"] = {
            "sample_count": len(sub),
            "sample_pct": round((len(sub) / len(df)) * 100.0, 2),
            "building_accuracy": round(float(sub["building_correct"].mean()) * 100.0, 2),
            "floor_accuracy": round(float(sub["floor_correct"].mean()) * 100.0, 2),
            "building_floor_accuracy": round(float(sub["building_floor_correct"].mean()) * 100.0, 2),
            "euclidean_mean": round(float(sub["euclidean_error"].mean()), 2),
            "euclidean_median": round(float(sub["euclidean_error"].median()), 2),
            "euclidean_q95": round(float(np.percentile(sub["euclidean_error"], 95)), 2),
        }

    # Breakdown by Floor
    by_floor = {}
    for f_id in sorted(df["FLOOR"].unique()):
        sub = df[df["FLOOR"] == f_id]
        by_floor[f"floor_{f_id}"] = {
            "sample_count": len(sub),
            "sample_pct": round((len(sub) / len(df)) * 100.0, 2),
            "floor_accuracy": round(float(sub["floor_correct"].mean()) * 100.0, 2),
            "building_floor_accuracy": round(float(sub["building_floor_correct"].mean()) * 100.0, 2),
            "euclidean_mean": round(float(sub["euclidean_error"].mean()), 2),
            "euclidean_median": round(float(sub["euclidean_error"].median()), 2),
        }

    # Breakdown by Building-Floor Pair
    by_bf_pair = {}
    for bf_idx in sorted(df["BUILDING_FLOOR_INDEX"].unique()):
        sub = df[df["BUILDING_FLOOR_INDEX"] == bf_idx]
        b_label = f"B{sub['BUILDINGID'].iloc[0]}_F{sub['FLOOR'].iloc[0]}"
        by_bf_pair[b_label] = {
            "class_index": int(bf_idx),
            "sample_count": len(sub),
            "accuracy": round(float(sub["building_floor_correct"].mean()) * 100.0, 2),
            "euclidean_mean": round(float(sub["euclidean_error"].mean()), 2),
            "euclidean_median": round(float(sub["euclidean_error"].median()), 2),
        }

    return {
        "by_building": by_building,
        "by_floor": by_floor,
        "by_building_floor_pair": by_bf_pair,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path,
    normalize: bool = True
) -> None:
    """Generate high-resolution heatmap for classification confusion matrix."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", font_scale=0.9)

    cm_data = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] if normalize else cm
    cm_data = np.nan_to_num(cm_data)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.7), max(5, len(class_names) * 0.6)), dpi=300)
    sns.heatmap(
        cm_data,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Normalized Recall" if normalize else "Count"},
        ax=ax
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("True Ground Truth Label", fontsize=10)
    ax.set_xlabel("Random Forest Prediction", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", save_path)


def plot_feature_importance(
    df_importance: pd.DataFrame,
    top_n: int = 20,
    title: str = "Top 20 Most Important Features (Random Forest)",
    save_path: Optional[Path] = None
) -> None:
    """Plot bar chart of top N feature importances categorized by type."""
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=0.9)
    top_df = df_importance.head(top_n).copy()

    # Color palette by feature type
    type_colors = {
        "rssi": "#1f77b4",       # Blue
        "visibility": "#2ca02c", # Green
        "summary": "#ff7f0e",    # Orange
    }
    top_df["color"] = top_df["feature_type"].map(type_colors).fillna("#7f7f7f")

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    bars = ax.barh(top_df["feature"][::-1], top_df["importance"][::-1], color=top_df["color"][::-1])

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Gini Feature Importance", fontsize=10)
    ax.set_ylabel("Feature Name", fontsize=10)

    # Custom legend for feature types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=type_colors["rssi"], label="Continuous RSSI"),
        Patch(facecolor=type_colors["visibility"], label="Binary Visibility"),
        Patch(facecolor=type_colors["summary"], label="Summary Statistic"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        logger.info("Saved feature importance plot to %s", save_path)
    plt.close(fig)


def plot_coordinate_errors(
    y_true_lon: np.ndarray,
    y_true_lat: np.ndarray,
    y_pred_lon: np.ndarray,
    y_pred_lat: np.ndarray,
    building_ids: np.ndarray,
    scatter_path: Path,
    hist_path: Path
) -> None:
    """Plot 2D actual vs predicted coordinate scatter and Euclidean error histogram."""
    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)

    # 1. 2D Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    palette = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

    for b_id in sorted(np.unique(building_ids)):
        mask = (building_ids == b_id)
        ax.scatter(
            y_true_lon[mask],
            y_true_lat[mask],
            c=palette.get(b_id, "gray"),
            alpha=0.5,
            s=25,
            label=f"Building {b_id} Actual"
        )
        ax.scatter(
            y_pred_lon[mask],
            y_pred_lat[mask],
            c="black",
            marker="x",
            alpha=0.4,
            s=20,
            label=f"Building {b_id} Pred" if b_id == 0 else None
        )

    ax.set_title("2D Coordinate Predictions vs Ground Truth (Random Forest Baseline)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude (dataset coordinate units)", fontsize=10)
    ax.set_ylabel("Latitude (dataset coordinate units)", fontsize=10)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig.savefig(scatter_path)
    plt.close(fig)
    logger.info("Saved coordinate scatter plot to %s", scatter_path)

    # 2. Euclidean Error Distribution Histogram
    eucl_errors = np.sqrt((y_pred_lon - y_true_lon) ** 2 + (y_pred_lat - y_true_lat) ** 2)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.histplot(eucl_errors, bins=40, kde=True, color="#9467bd", ax=ax)
    mean_err = np.mean(eucl_errors)
    med_err = np.median(eucl_errors)
    q95_err = np.percentile(eucl_errors, 95)

    ax.axvline(mean_err, color="red", linestyle="--", label=f"Mean: {mean_err:.2f}")
    ax.axvline(med_err, color="green", linestyle="-.", label=f"Median: {med_err:.2f}")
    ax.axvline(q95_err, color="orange", linestyle=":", label=f"95th %: {q95_err:.2f}")

    ax.set_title("2D Euclidean Coordinate Error Distribution (Validation Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Euclidean Error (dataset coordinate units)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(hist_path)
    plt.close(fig)
    logger.info("Saved coordinate error distribution plot to %s", hist_path)

