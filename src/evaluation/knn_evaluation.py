"""Evaluation metrics, diagnostic analysis, and publication plotting for k-NN models.

Reuses core evaluation logic from `src.evaluation.random_forest_evaluation`:
- Classification metrics (Accuracy, Macro Precision, Recall, F1, Confusion Matrix)
- Regression metrics (MAE, RMSE, MedAE, R², 2D Euclidean Error)
- Hierarchical breakdowns by Building and Floor

Adds k-NN specific visualizations:
- Performance across different k values (accuracy and Euclidean error curves)
- Grouped model comparison bar charts (Random Forest vs k-NN)
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
    plot_confusion_matrix,
    plot_coordinate_errors,
)

logger = setup_logger("KNNEvaluation")


def plot_k_comparison(
    cv_summary_df: pd.DataFrame,
    save_path: Union[str, Path],
) -> None:
    """Plot performance metrics across different neighbor counts (k).

    Args:
        cv_summary_df: DataFrame with columns:
            ['metric', 'weights', 'k', 'mean_bf_acc', 'mean_euclidean_error']
        save_path: File path to save output PNG plot.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Palette and markers
    metrics = cv_summary_df["metric"].unique()
    weightings = cv_summary_df["weights"].unique()
    markers = {"distance": "o", "uniform": "s"}
    colors = {"cosine": "#1f77b4", "manhattan": "#ff7f0e", "euclidean": "#2ca02c"}

    # Left plot: Building-Floor Accuracy vs k
    ax1 = axes[0]
    for metric in metrics:
        for weight in weightings:
            sub = cv_summary_df[(cv_summary_df["metric"] == metric) & (cv_summary_df["weights"] == weight)]
            if not sub.empty:
                sub_sorted = sub.sort_values("k")
                linestyle = "-" if weight == "distance" else "--"
                label = f"{metric.capitalize()} ({weight})"
                ax1.plot(
                    sub_sorted["k"],
                    sub_sorted["mean_bf_acc"],
                    marker=markers.get(weight, "o"),
                    linestyle=linestyle,
                    color=colors.get(metric, "#333333"),
                    linewidth=2,
                    markersize=6,
                    label=label,
                )

    ax1.set_title("Building-Floor Classification Accuracy vs. k (5-Fold CV)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Number of Neighbors (k)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks(sorted(cv_summary_df["k"].unique()))
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    # Right plot: Mean 2D Euclidean Error vs k
    ax2 = axes[1]
    for metric in metrics:
        for weight in weightings:
            sub = cv_summary_df[(cv_summary_df["metric"] == metric) & (cv_summary_df["weights"] == weight)]
            if not sub.empty:
                sub_sorted = sub.sort_values("k")
                linestyle = "-" if weight == "distance" else "--"
                label = f"{metric.capitalize()} ({weight})"
                ax2.plot(
                    sub_sorted["k"],
                    sub_sorted["mean_euclidean_error"],
                    marker=markers.get(weight, "o"),
                    linestyle=linestyle,
                    color=colors.get(metric, "#333333"),
                    linewidth=2,
                    markersize=6,
                    label=label,
                )

    ax2.set_title("Mean 2D Euclidean Error vs. k (5-Fold CV)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Number of Neighbors (k)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Error (dataset coordinate units)", fontsize=11, fontweight="bold")
    ax2.set_xticks(sorted(cv_summary_df["k"].unique()))
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved k-NN parameter sensitivity plot to %s", save_path)


def plot_rf_vs_knn_comparison(
    rf_results: Dict[str, Any],
    knn_results: Dict[str, Any],
    save_path: Union[str, Path],
) -> None:
    """Generate side-by-side comparison charts between Random Forest and k-NN.

    Args:
        rf_results: Structured validation results dictionary for Random Forest.
        knn_results: Structured validation results dictionary for k-NN.
        save_path: Path to save comparison PNG plot.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. Classification Metrics Comparison
    clf_categories = ["Building Acc", "Floor Acc", "Building-Floor Acc"]
    rf_clf_vals = [
        rf_results["validation_performance"]["variant_b"]["building"]["accuracy_pct"],
        rf_results["validation_performance"]["variant_b"]["floor"]["accuracy_pct"],
        rf_results["validation_performance"]["variant_b"]["building_floor"]["accuracy_pct"],
    ]
    knn_clf_vals = [
        knn_results["validation_performance"]["building"]["accuracy_pct"],
        knn_results["validation_performance"]["floor"]["accuracy_pct"],
        knn_results["validation_performance"]["building_floor"]["accuracy_pct"],
    ]

    x = np.arange(len(clf_categories))
    width = 0.35

    ax1 = axes[0]
    rects1 = ax1.bar(x - width/2, rf_clf_vals, width, label="Random Forest (Task 07)", color="#2b5c8f", alpha=0.9)
    rects2 = ax1.bar(x + width/2, knn_clf_vals, width, label="k-Nearest Neighbors (Task 08)", color="#e07a5f", alpha=0.9)

    ax1.set_ylabel("Validation Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Classification Accuracy: Random Forest vs. k-NN", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(clf_categories, fontsize=10, fontweight="bold")
    ax1.set_ylim(70, 105)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", frameon=True)

    # Annotate bar values
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Coordinate Positioning Error Comparison
    err_categories = ["Longitude MAE", "Latitude MAE", "Mean 2D Error", "Median 2D Error"]
    rf_err_vals = [
        rf_results["validation_performance"]["variant_b"]["coordinates"]["longitude"]["mae"],
        rf_results["validation_performance"]["variant_b"]["coordinates"]["latitude"]["mae"],
        rf_results["validation_performance"]["variant_b"]["coordinates"]["euclidean_error"]["mean"],
        rf_results["validation_performance"]["variant_b"]["coordinates"]["euclidean_error"]["median"],
    ]
    knn_err_vals = [
        knn_results["validation_performance"]["coordinates"]["longitude"]["mae"],
        knn_results["validation_performance"]["coordinates"]["latitude"]["mae"],
        knn_results["validation_performance"]["coordinates"]["euclidean_error"]["mean"],
        knn_results["validation_performance"]["coordinates"]["euclidean_error"]["median"],
    ]

    x2 = np.arange(len(err_categories))
    ax2 = axes[1]
    rects3 = ax2.bar(x2 - width/2, rf_err_vals, width, label="Random Forest (Task 07)", color="#2b5c8f", alpha=0.9)
    rects4 = ax2.bar(x2 + width/2, knn_err_vals, width, label="k-Nearest Neighbors (Task 08)", color="#e07a5f", alpha=0.9)

    ax2.set_ylabel("Coordinate Error (dataset units)", fontsize=11, fontweight="bold")
    ax2.set_title("Coordinate Regression Error: Random Forest vs. k-NN", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(err_categories, fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(max(rf_err_vals), max(knn_err_vals)) * 1.3)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved RF vs k-NN comparison plot to %s", save_path)

