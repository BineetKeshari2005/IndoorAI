"""Evaluation metrics, diagnostic visualizations, and benchmark plotting for Multitask MLP.

Reuses core evaluation logic from `src.evaluation.random_forest_evaluation`:
- Classification metrics (Accuracy, Macro Precision, Recall, F1, Confusion Matrix)
- Regression metrics (MAE, RMSE, MedAE, R², 2D Euclidean Error)
- Hierarchical breakdowns by Building and Floor

Provides dedicated deep learning diagnostic plots:
- Loss curves (total + task component breakdown)
- Accuracy curves (building and floor across training epochs)
- Confusion matrices for Building and Floor
- Coordinate regression scatter (actual vs predicted with ideal line)
- 2D Euclidean spatial error distribution and CDF
- Three-way model comparison benchmark (Random Forest vs k-NN vs Multitask MLP)
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

from src.core.logging import setup_logger
from src.evaluation.random_forest_evaluation import (
    compute_classification_metrics,
    compute_hierarchical_breakdown,
    compute_regression_metrics,
)

logger = setup_logger("MLPEvaluation")


def plot_loss_curves(
    history: Dict[str, List[float]],
    save_path: Union[str, Path],
) -> None:
    """Plot training and validation loss curves (total and task components)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Total Loss
    ax1 = axes[0]
    ax1.plot(epochs, history["train_loss"], label="Train Total Loss", color="#1f77b4", linewidth=2.2)
    if "val_loss" in history and len(history["val_loss"]) > 0:
        ax1.plot(epochs[:len(history["val_loss"])], history["val_loss"], label="Val Total Loss", color="#d62728", linewidth=2.2, linestyle="--")
        best_epoch = int(np.argmin(history["val_loss"])) + 1
        best_val = min(history["val_loss"])
        ax1.scatter([best_epoch], [best_val], color="#2ca02c", s=80, zorder=5, label=f"Best Val: {best_val:.3f} (ep {best_epoch})")

    ax1.set_title("Total Multitask Loss Curve", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Weighted Loss", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Task Component Losses (Validation if available, else Train)
    ax2 = axes[1]
    if "val_loss_bldg" in history and len(history["val_loss_bldg"]) > 0:
        v_ep = epochs[:len(history["val_loss_bldg"])]
        ax2.plot(v_ep, history["val_loss_bldg"], label="Val Building Loss (CE)", color="#2b5c8f", linewidth=1.8)
        ax2.plot(v_ep, history["val_loss_floor"], label="Val Floor Loss (CE)", color="#e07a5f", linewidth=1.8)
        ax2.plot(v_ep, history["val_loss_lon"], label="Val Longitude Loss", color="#3d405b", linewidth=1.8, linestyle="--")
        ax2.plot(v_ep, history["val_loss_lat"], label="Val Latitude Loss", color="#81b29a", linewidth=1.8, linestyle="--")
        ax2.set_title("Validation Task-Specific Loss Breakdown", fontsize=13, fontweight="bold", pad=12)
    else:
        ax2.plot(epochs, history["train_loss_bldg"], label="Train Building (CE)", color="#2b5c8f", linewidth=1.8)
        ax2.plot(epochs, history["train_loss_floor"], label="Train Floor (CE)", color="#e07a5f", linewidth=1.8)
        ax2.plot(epochs, history["train_loss_lon"], label="Train Longitude", color="#3d405b", linewidth=1.8, linestyle="--")
        ax2.plot(epochs, history["train_loss_lat"], label="Train Latitude", color="#81b29a", linewidth=1.8, linestyle="--")
        ax2.set_title("Training Task-Specific Loss Breakdown", fontsize=13, fontweight="bold", pad=12)

    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Loss Component", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved loss curves to %s", save_path)


def plot_accuracy_curves(
    history: Dict[str, List[float]],
    save_path: Union[str, Path],
) -> None:
    """Plot validation Building and Floor accuracy curves across epochs."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if "val_acc_bldg" not in history or len(history["val_acc_bldg"]) == 0:
        logger.warning("No validation accuracy history available for plotting.")
        return

    epochs = range(1, len(history["val_acc_bldg"]) + 1)
    bldg_pct = [v * 100.0 for v in history["val_acc_bldg"]]
    floor_pct = [v * 100.0 for v in history["val_acc_floor"]]

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epochs, bldg_pct, label="Building Accuracy (%)", color="#2b5c8f", linewidth=2.2)
    plt.plot(epochs, floor_pct, label="Floor Accuracy (%)", color="#e07a5f", linewidth=2.2)

    # Highlight best floor epoch
    best_floor_idx = int(np.argmax(floor_pct))
    best_floor_acc = floor_pct[best_floor_idx]
    plt.scatter([best_floor_idx + 1], [best_floor_acc], color="#2ca02c", s=90, zorder=5,
                label=f"Peak Floor Acc: {best_floor_acc:.2f}% (ep {best_floor_idx + 1})")

    # Highlight final values
    plt.annotate(f"Final Bldg: {bldg_pct[-1]:.2f}%", xy=(len(epochs), bldg_pct[-1]),
                 xytext=(-80, 8), textcoords="offset points", fontsize=9, fontweight="bold",
                 color="#2b5c8f", arrowprops=dict(arrowstyle="->", color="#2b5c8f"))
    plt.annotate(f"Final Floor: {floor_pct[-1]:.2f}%", xy=(len(epochs), floor_pct[-1]),
                 xytext=(-80, -18), textcoords="offset points", fontsize=9, fontweight="bold",
                 color="#e07a5f", arrowprops=dict(arrowstyle="->", color="#e07a5f"))

    plt.title("Multitask MLP Classification Accuracy across Epochs", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.ylim(min(floor_pct) * 0.95, 101.5)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved accuracy curves to %s", save_path)


def plot_mlp_confusion_matrices(
    cm_bldg: np.ndarray,
    cm_floor: np.ndarray,
    save_path: Union[str, Path],
) -> None:
    """Plot Building and Floor confusion matrix heatmaps side-by-side."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Building Confusion Matrix (3x3)
    ax1 = axes[0]
    sns.heatmap(
        cm_bldg,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        ax=ax1,
        xticklabels=[f"Bldg {i}" for i in range(cm_bldg.shape[1])],
        yticklabels=[f"Bldg {i}" for i in range(cm_bldg.shape[0])],
    )
    bldg_acc = np.trace(cm_bldg) / np.sum(cm_bldg) * 100.0
    ax1.set_title(f"Building Classification (Accuracy: {bldg_acc:.2f}%)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Predicted Building", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Building", fontsize=11, fontweight="bold")

    # Floor Confusion Matrix (5x5)
    ax2 = axes[1]
    sns.heatmap(
        cm_floor,
        annot=True,
        fmt="d",
        cmap="Oranges",
        cbar=True,
        ax=ax2,
        xticklabels=[f"Floor {i}" for i in range(cm_floor.shape[1])],
        yticklabels=[f"Floor {i}" for i in range(cm_floor.shape[0])],
    )
    floor_acc = np.trace(cm_floor) / np.sum(cm_floor) * 100.0
    ax2.set_title(f"Floor Classification (Accuracy: {floor_acc:.2f}%)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Predicted Floor", fontsize=11, fontweight="bold")
    ax2.set_ylabel("True Floor", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrices to %s", save_path)


def plot_mlp_coordinate_predictions(
    y_true_lon: np.ndarray,
    y_pred_lon: np.ndarray,
    y_true_lat: np.ndarray,
    y_pred_lat: np.ndarray,
    save_path: Union[str, Path],
) -> None:
    """Plot actual vs predicted scatter plots for Longitude and Latitude."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import mean_absolute_error, r2_score

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Longitude
    ax1 = axes[0]
    ax1.scatter(y_true_lon, y_pred_lon, alpha=0.35, color="#1f77b4", edgecolors="none", s=25)
    min_lon = min(y_true_lon.min(), y_pred_lon.min())
    max_lon = max(y_true_lon.max(), y_pred_lon.max())
    ax1.plot([min_lon, max_lon], [min_lon, max_lon], "r--", linewidth=1.8, label="Ideal (y = x)")
    r2_lon = r2_score(y_true_lon, y_pred_lon)
    mae_lon = mean_absolute_error(y_true_lon, y_pred_lon)
    ax1.set_title(f"Longitude Regression (R² = {r2_lon:.4f}, MAE = {mae_lon:.2f})", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Actual Longitude", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predicted Longitude", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Latitude
    ax2 = axes[1]
    ax2.scatter(y_true_lat, y_pred_lat, alpha=0.35, color="#e07a5f", edgecolors="none", s=25)
    min_lat = min(y_true_lat.min(), y_pred_lat.min())
    max_lat = max(y_true_lat.max(), y_pred_lat.max())
    ax2.plot([min_lat, max_lat], [min_lat, max_lat], "r--", linewidth=1.8, label="Ideal (y = x)")
    r2_lat = r2_score(y_true_lat, y_pred_lat)
    mae_lat = mean_absolute_error(y_true_lat, y_pred_lat)
    ax2.set_title(f"Latitude Regression (R² = {r2_lat:.4f}, MAE = {mae_lat:.2f})", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Actual Latitude", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Predicted Latitude", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved coordinate prediction scatter plots to %s", save_path)


def plot_mlp_error_distribution(
    euclidean_errors: np.ndarray,
    save_path: Union[str, Path],
) -> None:
    """Plot 2D Euclidean coordinate error histogram, KDE, and Cumulative Distribution Function."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    mean_err = float(np.mean(euclidean_errors))
    median_err = float(np.median(euclidean_errors))
    p75 = float(np.percentile(euclidean_errors, 75))
    p90 = float(np.percentile(euclidean_errors, 90))
    p95 = float(np.percentile(euclidean_errors, 95))

    # 1. Error Distribution Histogram + KDE
    ax1 = axes[0]
    sns.histplot(euclidean_errors, bins=40, kde=True, color="#3d405b", ax=ax1, stat="density", alpha=0.55)
    ax1.axvline(mean_err, color="#e07a5f", linestyle="-", linewidth=2.0, label=f"Mean Error: {mean_err:.2f}")
    ax1.axvline(median_err, color="#2ca02c", linestyle="--", linewidth=2.0, label=f"Median Error: {median_err:.2f}")
    ax1.axvline(p95, color="#d62728", linestyle=":", linewidth=2.0, label=f"95th Percentile: {p95:.2f}")

    ax1.set_title("2D Euclidean Coordinate Error Distribution", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("2D Spatial Error (dataset coordinate units)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Cumulative Distribution Function (CDF)
    ax2 = axes[1]
    sorted_errs = np.sort(euclidean_errors)
    cdf = np.arange(1, len(sorted_errs) + 1) / len(sorted_errs) * 100.0

    ax2.plot(sorted_errs, cdf, color="#1f77b4", linewidth=2.2, label="Empirical CDF")
    ax2.axhline(50, color="#2ca02c", linestyle="--", alpha=0.7, label=f"50% ≤ {median_err:.2f}")
    ax2.axhline(75, color="#ff7f0e", linestyle="--", alpha=0.7, label=f"75% ≤ {p75:.2f}")
    ax2.axhline(90, color="#9467bd", linestyle="--", alpha=0.7, label=f"90% ≤ {p90:.2f}")
    ax2.axhline(95, color="#d62728", linestyle="--", alpha=0.7, label=f"95% ≤ {p95:.2f}")

    ax2.set_title("Cumulative Error Distribution (CDF)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("2D Spatial Error (dataset coordinate units)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Percentage (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 102)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved error distribution and CDF plots to %s", save_path)


def plot_three_way_benchmark(
    rf_metrics: Dict[str, Any],
    knn_metrics: Dict[str, Any],
    mlp_metrics: Dict[str, Any],
    save_path: Union[str, Path],
) -> None:
    """Grouped bar chart comparing Random Forest, k-NN (k=7), and Multitask MLP."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # 1. Classification Metrics Comparison
    clf_categories = ["Building Acc", "Floor Acc", "Building-Floor Acc"]
    rf_clf = [
        rf_metrics.get("building_acc", 99.82),
        rf_metrics.get("floor_acc", 88.66),
        rf_metrics.get("building_floor_acc", 88.48),
    ]
    knn_clf = [
        knn_metrics.get("building_acc", 99.91),
        knn_metrics.get("floor_acc", 84.43),
        knn_metrics.get("building_floor_acc", 84.43),
    ]
    mlp_clf = [
        mlp_metrics.get("building_acc", 0.0),
        mlp_metrics.get("floor_acc", 0.0),
        mlp_metrics.get("building_floor_acc", 0.0),
    ]

    x = np.arange(len(clf_categories))
    width = 0.26

    ax1 = axes[0]
    r1 = ax1.bar(x - width, rf_clf, width, label="Random Forest (Task 07)", color="#2b5c8f", alpha=0.9)
    r2 = ax1.bar(x, knn_clf, width, label="k-NN k=7 (Task 08)", color="#e07a5f", alpha=0.9)
    r3 = ax1.bar(x + width, mlp_clf, width, label="Multitask MLP (Task 12)", color="#81b29a", alpha=0.9)

    ax1.set_ylabel("Validation Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Classification Accuracy: RF vs. k-NN vs. Multitask MLP", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(clf_categories, fontsize=10, fontweight="bold")
    ax1.set_ylim(70, 106)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", frameon=True)

    for rect in list(r1) + list(r2) + list(r3):
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # 2. Coordinate Spatial Error Comparison
    err_categories = ["Longitude MAE", "Latitude MAE", "Mean 2D Error", "Median 2D Error"]
    rf_err = [
        rf_metrics.get("lon_mae", 7.82),
        rf_metrics.get("lat_mae", 6.09),
        rf_metrics.get("eucl_mean", 11.53),
        rf_metrics.get("eucl_median", 8.65),
    ]
    knn_err = [
        knn_metrics.get("lon_mae", 6.84),
        knn_metrics.get("lat_mae", 5.23),
        knn_metrics.get("eucl_mean", 10.05),
        knn_metrics.get("eucl_median", 7.33),
    ]
    mlp_err = [
        mlp_metrics.get("lon_mae", 0.0),
        mlp_metrics.get("lat_mae", 0.0),
        mlp_metrics.get("eucl_mean", 0.0),
        mlp_metrics.get("eucl_median", 0.0),
    ]

    x2 = np.arange(len(err_categories))
    ax2 = axes[1]
    r4 = ax2.bar(x2 - width, rf_err, width, label="Random Forest (Task 07)", color="#2b5c8f", alpha=0.9)
    r5 = ax2.bar(x2, knn_err, width, label="k-NN k=7 (Task 08)", color="#e07a5f", alpha=0.9)
    r6 = ax2.bar(x2 + width, mlp_err, width, label="Multitask MLP (Task 12)", color="#81b29a", alpha=0.9)

    ax2.set_ylabel("Coordinate Error (dataset units)", fontsize=11, fontweight="bold")
    ax2.set_title("Coordinate Regression Error: RF vs. k-NN vs. Multitask MLP", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(err_categories, fontsize=10, fontweight="bold")
    max_val = max(max(rf_err), max(knn_err), max(mlp_err))
    ax2.set_ylim(0, max_val * 1.3)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    for rect in list(r4) + list(r5) + list(r6):
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved three-way benchmark comparison plot to %s", save_path)

