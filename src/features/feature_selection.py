"""Feature selection analysis and training-only feature filter module.

Performs training-only variance and detection threshold analysis, evaluates
different feature retention policies, generates 'feature_selection_report.md',
and applies deterministic feature filtering to both training and validation splits.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.logging import setup_logger

logger = setup_logger("FeatureSelection")


@dataclass
class SelectionThresholdEvaluation:
    """Evaluation metrics for a specific feature filtering threshold."""
    strategy_name: str
    retained_feature_count: int
    dropped_feature_count: int
    retained_percentage: float
    description: str
    retained_features: List[str] = field(default_factory=list)
    dropped_features: List[str] = field(default_factory=list)


def evaluate_feature_selection_thresholds(
    X_train: pd.DataFrame,
    ap_stats_df: pd.DataFrame,
    save_json_path: Optional[Path] = None,
    save_md_report_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Evaluate multiple feature selection thresholds strictly on training features.

    Compares:
    1. Baseline (all features retained).
    2. Zero-variance removal (drops constant features).
    3. Minimum detection count (>= 5, >= 10, >= 20, >= 50 samples).
    4. Minimum detection percentage (>= 0.05%, >= 0.1%, >= 0.5%).
    5. Variance threshold (> 0.0, > 0.001, > 0.01).

    Args:
        X_train: Training feature DataFrame.
        ap_stats_df: Precomputed AP statistics DataFrame.
        save_json_path: Optional path for feature_selection_summary.json.
        save_md_report_path: Optional path for feature_selection_report.md.

    Returns:
        Dictionary of threshold evaluations.
    """
    total_features = len(X_train.columns)
    total_samples = len(X_train)
    logger.info("Evaluating feature selection policies on %d training features (%d samples)...", total_features, total_samples)

    # Precompute variance for each column in X_train
    variances = X_train.var()
    constant_features = list(variances[variances <= 0.0].index)
    all_cols = list(X_train.columns)

    evaluations: Dict[str, Any] = {}

    # Policy 1: Retain all
    evaluations["all"] = asdict(
        SelectionThresholdEvaluation(
            strategy_name="all",
            retained_feature_count=total_features,
            dropped_feature_count=0,
            retained_percentage=100.0,
            description="Preserves all features without filtering (Baseline default).",
            retained_features=all_cols,
            dropped_features=[]
        )
    )

    # Policy 2: Remove constant features (variance <= 0)
    retained_const = [c for c in all_cols if c not in constant_features]
    evaluations["remove_constant"] = asdict(
        SelectionThresholdEvaluation(
            strategy_name="remove_constant",
            retained_feature_count=len(retained_const),
            dropped_feature_count=len(constant_features),
            retained_percentage=round((len(retained_const) / total_features) * 100.0, 2),
            description=f"Drops {len(constant_features)} zero-variance features where all training values are identical.",
            retained_features=retained_const,
            dropped_features=constant_features
        )
    )

    # Policy 3: Minimum detection counts
    for min_count in [5, 10, 20, 50]:
        eligible_aps = set(ap_stats_df[ap_stats_df["train_detection_count"] >= min_count]["ap_name"])
        # Columns to keep: eligible WAPs, their visibility masks, and any summary features
        retained = [
            c for c in X_train.columns
            if c.split("_")[0] in eligible_aps or (not c.startswith("WAP"))
        ]
        dropped = [c for c in X_train.columns if c not in retained]
        evaluations[f"min_detection_count_{min_count}"] = asdict(
            SelectionThresholdEvaluation(
                strategy_name=f"min_detection_count_{min_count}",
                retained_feature_count=len(retained),
                dropped_feature_count=len(dropped),
                retained_percentage=round((len(retained) / total_features) * 100.0, 2),
                description=f"Retains APs detected in at least {min_count} training samples (and all summary features).",
                retained_features=retained,
                dropped_features=dropped
            )
        )

    # Policy 4: Variance thresholds
    for var_th in [0.0001, 0.001, 0.01]:
        retained_var = list(variances[variances > var_th].index)
        dropped_var = [c for c in all_cols if c not in retained_var]
        evaluations[f"variance_threshold_{var_th}"] = asdict(
            SelectionThresholdEvaluation(
                strategy_name=f"variance_threshold_{var_th}",
                retained_feature_count=len(retained_var),
                dropped_feature_count=len(dropped_var),
                retained_percentage=round((len(retained_var) / total_features) * 100.0, 2),
                description=f"Retains features with empirical variance > {var_th} on training data.",
                retained_features=retained_var,
                dropped_features=dropped_var
            )
        )

    summary_result = {
        "total_input_features": total_features,
        "training_sample_count": total_samples,
        "constant_features_count": len(constant_features),
        "constant_feature_names": constant_features,
        "evaluations": evaluations,
        "strategies": evaluations,  # Alias for backward-compatibility
    }

    if save_json_path:
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_result, f, indent=2)
        logger.info("Saved feature selection summary to %s", save_json_path)

    if save_md_report_path:
        save_md_report_path.parent.mkdir(parents=True, exist_ok=True)
        generate_markdown_report(summary_result, save_md_report_path)

    return summary_result


def generate_markdown_report(summary: Dict[str, Any], path: Path) -> None:
    """Write publication-grade markdown summary for feature selection analysis."""
    lines = [
        "# Wi-Fi Feature Selection Analysis Report",
        "",
        "## 1. Executive Summary",
        "",
        f"This report evaluates training-only feature filtering policies for the **{summary['total_input_features']}** total features ",
        f"across **{summary['training_sample_count']:,}** training fingerprints. ",
        "All calculations were executed strictly on the training partition without utilizing validation labels or statistics.",
        "",
        "## 2. Policy Comparison Matrix",
        "",
        "| Strategy Identifier | Retained Features | Dropped Features | Retained % | Description |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for name, ev in summary["evaluations"].items():
        lines.append(
            f"| `{name}` | **{ev['retained_feature_count']:,}** | {ev['dropped_feature_count']:,} | {ev['retained_percentage']:.2f}% | {ev['description']} |"
        )

    lines.extend([
        "",
        "## 3. Zero-Variance Constant Features",
        "",
        f"Identified **{summary['constant_features_count']}** features with zero empirical variance on the training partition.",
        "Under the default configuration (`strategy: all`), these features are preserved to maintain cross-partition schema consistency.",
        "",
        "```text",
        ", ".join(summary["constant_feature_names"][:30]) + ("..." if len(summary["constant_feature_names"]) > 30 else ""),
        "```",
        "",
        "## 4. Recommendation for Downstream Modeling",
        "",
        "- **Tree-based Baselines (Random Forest, XGBoost):** Can naturally ignore zero-variance features through split-criterion optimization. Use `strategy: all`.",
        "- **Linear Models & Distance-based Methods (k-NN, SVM, MLP):** Benefit from dropping zero-variance features (`remove_constant` or `min_detection_count_10`) to reduce matrix dimension without losing informative variance.",
        ""
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote feature selection report to %s", path)


def apply_feature_selection(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    strategy: str = "all",
    min_detection_count: int = 10,
    variance_threshold: float = 0.001,
    ap_stats_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    """Filter features deterministically using rules fitted strictly on X_train.

    Validation features are subsetted using the exact same selected column list.

    Args:
        X_train: Training feature DataFrame.
        X_val: Optional validation feature DataFrame.
        strategy: 'all', 'remove_constant', 'min_detection_count', or 'variance_threshold'.
        min_detection_count: Threshold if strategy='min_detection_count'.
        variance_threshold: Threshold if strategy='variance_threshold'.
        ap_stats_df: Optional precomputed training AP statistics.

    Returns:
        Tuple of (selected_X_train, selected_X_val, selected_column_names).
    """
    resolved_strat = strategy.lower().strip()
    total_cols = list(X_train.columns)

    if resolved_strat == "all":
        logger.info("Feature Selection: Retaining all %d features (strategy='all').", len(total_cols))
        return X_train, X_val, total_cols

    elif resolved_strat in ("remove_constant", "drop_constant"):
        variances = X_train.var()
        selected = list(variances[variances > 0.0].index)
        logger.info("Feature Selection: Dropped %d constant features. Retained %d features.", len(total_cols) - len(selected), len(selected))

    elif resolved_strat == "min_detection_count":
        if ap_stats_df is not None:
            active_waps = set(ap_stats_df[ap_stats_df["train_detection_count"] >= min_detection_count]["ap_name"])
        else:
            active_waps = set()
        selected = [c for c in total_cols if c.split("_")[0] in active_waps or not c.startswith("WAP")]
        logger.info("Feature Selection: Retained %d features (min_count >= %d).", len(selected), min_detection_count)

    elif resolved_strat == "variance_threshold":
        variances = X_train.var()
        selected = list(variances[variances > variance_threshold].index)
        logger.info("Feature Selection: Retained %d features (variance > %.4f).", len(selected), variance_threshold)

    else:
        raise ValueError(f"Unknown feature selection strategy '{strategy}'. Valid options: ['all', 'remove_constant', 'min_detection_count', 'variance_threshold']")

    X_train_filtered = X_train[selected].copy()
    X_val_filtered = X_val[selected].copy() if X_val is not None else None

    return X_train_filtered, X_val_filtered, selected
