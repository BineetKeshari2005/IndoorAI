"""Comprehensive Inspection and Exploratory Data Analysis (EDA) module for UJIIndoorLoc.

Inspects raw dataset splits, calculates numerical and categorical statistics,
analyzes Wi-Fi visibility and signal strength characteristics, runs quality
and anomaly audits, and generates structured CSV/JSON artifacts and publication-grade plots.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is in sys.path
_current_dir = Path(__file__).resolve().parent
_repo_root = _current_dir.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Set writable MPLCONFIGDIR for sandboxed execution
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
Path("/tmp/matplotlib_cache").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")  # Headless backend for server execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.core.config import get_settings
from src.core.logging import setup_logger

logger = setup_logger("UJIIndoorLocInspector")

UNAVAILABLE_WAP_VALUE = 100
EXPECTED_NUM_WAPS = 520
CONTEXT_COLS = [
    "LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID",
    "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"
]


@dataclass
class QualityCheckResults:
    """Summary of data quality and consistency checks."""
    train_duplicates_count: int
    val_duplicates_count: int
    train_constant_waps_count: int
    val_constant_waps_count: int
    train_constant_waps: List[str]
    val_constant_waps: List[str]
    never_detected_waps_both_splits: List[str]
    train_unexpected_signals_count: int
    val_unexpected_signals_count: int
    train_nan_count: int
    val_nan_count: int
    schema_match: bool
    schema_diff_columns: List[str]
    unexpected_buildings_train: List[int]
    unexpected_floors_train: List[int]
    unexpected_buildings_val: List[int]
    unexpected_floors_val: List[int]
    notes: List[str]


def get_dataset_paths() -> Tuple[Path, Path, Path]:
    """Resolve data directory and raw train/validation file paths."""
    settings = get_settings()
    raw_dir = settings.paths.raw_data_dir / "ujiindoorloc"
    train_path = raw_dir / "trainingData.csv"
    val_path = raw_dir / "validationData.csv"
    output_dir = settings.paths.root_dir / "experiments" / "dataset_eda"
    return train_path, val_path, output_dir


def load_dataset(filepath: Path) -> pd.DataFrame:
    """Load UJIIndoorLoc CSV into a pandas DataFrame."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    logger.info("Loading dataset from %s ...", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), filepath.name)
    return df


def identify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Partition columns into WAP access point signal columns and context attributes."""
    wap_cols = [c for c in df.columns if c.startswith("WAP")]
    context_cols = [c for c in df.columns if not c.startswith("WAP")]
    return wap_cols, context_cols


def compute_column_information(df_train: pd.DataFrame, df_val: pd.DataFrame) -> pd.DataFrame:
    """Generate detailed column-by-column metadata table."""
    records = []
    wap_cols, context_cols = identify_columns(df_train)
    
    for col in df_train.columns:
        is_wap = col.startswith("WAP")
        t_dtype = str(df_train[col].dtype)
        v_dtype = str(df_val[col].dtype) if col in df_val.columns else "MISSING"
        t_nulls = int(df_train[col].isna().sum())
        v_nulls = int(df_val[col].isna().sum()) if col in df_val.columns else -1
        
        if is_wap:
            # For WAPs, calculate how many times the AP was observed (value != 100)
            t_visible = int((df_train[col] != UNAVAILABLE_WAP_VALUE).sum())
            v_visible = int((df_val[col] != UNAVAILABLE_WAP_VALUE).sum()) if col in df_val.columns else -1
            t_unavail = int((df_train[col] == UNAVAILABLE_WAP_VALUE).sum())
            category = "Wi-Fi Access Point (RSSI dBm)"
        else:
            t_visible = len(df_train) - t_nulls
            v_visible = len(df_val) - v_nulls if col in df_val.columns else -1
            t_unavail = 0
            category = "Spatial/Contextual Metadata"

        records.append({
            "column_name": col,
            "category": category,
            "train_dtype": t_dtype,
            "val_dtype": v_dtype,
            "train_null_count": t_nulls,
            "val_null_count": v_nulls,
            "train_observed_or_valid_count": t_visible,
            "val_observed_or_valid_count": v_visible,
            "train_unavailable_sentinel_100_count": t_unavail,
        })
    return pd.DataFrame(records)


def compute_numerical_statistics(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Calculate descriptive statistics for numerical columns in a split."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    stats_list = []
    
    for col in num_cols:
        series = df[col]
        is_wap = col.startswith("WAP")
        
        if is_wap:
            # Exclude sentinel 100 to show genuine RSSI characteristics
            valid_signals = series[series != UNAVAILABLE_WAP_VALUE]
            obs_count = len(valid_signals)
            obs_pct = (obs_count / len(series)) * 100.0
            
            if obs_count > 0:
                s_min = float(valid_signals.min())
                s_max = float(valid_signals.max())
                s_mean = float(valid_signals.mean())
                s_median = float(valid_signals.median())
                s_std = float(valid_signals.std()) if obs_count > 1 else 0.0
            else:
                s_min, s_max, s_mean, s_median, s_std = None, None, None, None, None
            
            stats_list.append({
                "split": split_name,
                "column": col,
                "type": "WAP_SIGNAL",
                "total_rows": len(series),
                "null_count": int(series.isna().sum()),
                "unavailable_100_count": int((series == UNAVAILABLE_WAP_VALUE).sum()),
                "valid_obs_count": obs_count,
                "valid_obs_pct": round(obs_pct, 4),
                "min": s_min,
                "max": s_max,
                "mean": round(s_mean, 4) if s_mean is not None else None,
                "median": round(s_median, 4) if s_median is not None else None,
                "std": round(s_std, 4) if s_std is not None else None,
            })
        else:
            stats_list.append({
                "split": split_name,
                "column": col,
                "type": "CONTEXT_FIELD",
                "total_rows": len(series),
                "null_count": int(series.isna().sum()),
                "unavailable_100_count": 0,
                "valid_obs_count": len(series) - int(series.isna().sum()),
                "valid_obs_pct": 100.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std()), 4),
            })

    return pd.DataFrame(stats_list)


def compute_categorical_statistics(df_train: pd.DataFrame, df_val: pd.DataFrame) -> Dict[str, Any]:
    """Compute frequency distributions and cardinality for discrete context columns."""
    cat_columns = ["BUILDINGID", "FLOOR", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID"]
    res: Dict[str, Any] = {}
    
    for col in cat_columns:
        train_counts = df_train[col].value_counts().to_dict() if col in df_train.columns else {}
        val_counts = df_val[col].value_counts().to_dict() if col in df_val.columns else {}
        
        # Sort keys as integers if possible
        def format_keys(d: Dict[Any, Any]) -> Dict[str, int]:
            return {str(k): int(v) for k, v in sorted(d.items(), key=lambda x: str(x[0]))}

        res[col] = {
            "train_unique_count": int(df_train[col].nunique()) if col in df_train.columns else 0,
            "val_unique_count": int(df_val[col].nunique()) if col in df_val.columns else 0,
            "train_frequencies": format_keys(train_counts),
            "val_frequencies": format_keys(val_counts),
        }
    return res


def compute_wap_visibility_statistics(df_train: pd.DataFrame, df_val: pd.DataFrame) -> pd.DataFrame:
    """Analyze visibility rates for all 520 Access Points across train and validation sets."""
    wap_cols, _ = identify_columns(df_train)
    records = []
    
    for wap in wap_cols:
        t_series = df_train[wap]
        t_valid = t_series[t_series != UNAVAILABLE_WAP_VALUE]
        t_count = len(t_valid)
        t_pct = (t_count / len(df_train)) * 100.0
        
        if wap in df_val.columns:
            v_series = df_val[wap]
            v_valid = v_series[v_series != UNAVAILABLE_WAP_VALUE]
            v_count = len(v_valid)
            v_pct = (v_count / len(df_val)) * 100.0
        else:
            v_count, v_pct = 0, 0.0

        records.append({
            "wap_id": wap,
            "train_detected_count": t_count,
            "train_detected_pct": round(t_pct, 4),
            "val_detected_count": v_count,
            "val_detected_pct": round(v_pct, 4),
            "total_detected_count": t_count + v_count,
            "train_rssi_min": float(t_valid.min()) if t_count > 0 else None,
            "train_rssi_max": float(t_valid.max()) if t_count > 0 else None,
            "train_rssi_mean": round(float(t_valid.mean()), 2) if t_count > 0 else None,
            "is_constant_100_in_train": t_count == 0,
            "is_constant_100_in_val": v_count == 0,
            "is_constant_in_both": (t_count == 0 and v_count == 0),
        })

    df_res = pd.DataFrame(records)
    return df_res.sort_values(by="total_detected_count", ascending=False).reset_index(drop=True)


def run_quality_checks(df_train: pd.DataFrame, df_val: pd.DataFrame) -> QualityCheckResults:
    """Perform structural and semantic quality checks on the dataset splits."""
    wap_cols_t, ctx_cols_t = identify_columns(df_train)
    wap_cols_v, ctx_cols_v = identify_columns(df_val)
    
    notes: List[str] = []

    # 1. Duplicate records check
    train_dups = int(df_train.duplicated().sum())
    val_dups = int(df_val.duplicated().sum())
    if train_dups > 0:
        notes.append(f"Training set contains {train_dups} exact duplicate rows.")
    if val_dups > 0:
        notes.append(f"Validation set contains {val_dups} exact duplicate rows.")

    # 2. Schema consistency
    schema_diff = list(set(df_train.columns) ^ set(df_val.columns))
    schema_match = len(schema_diff) == 0 and list(df_train.columns) == list(df_val.columns)
    if not schema_match:
        notes.append(f"Schema mismatch between train and val! Differing columns: {schema_diff}")

    # 3. Categorical bounds
    valid_buildings = {0, 1, 2}
    valid_floors = {0, 1, 2, 3, 4}
    unexp_bld_train = [int(b) for b in set(df_train["BUILDINGID"]) - valid_buildings]
    unexp_flr_train = [int(f) for f in set(df_train["FLOOR"]) - valid_floors]
    unexp_bld_val = [int(b) for b in set(df_val["BUILDINGID"]) - valid_buildings]
    unexp_flr_val = [int(f) for f in set(df_val["FLOOR"]) - valid_floors]

    # 4. Unexpected signal values (must be 100 or -104 <= RSSI <= 0)
    train_wap_mat = df_train[wap_cols_t].values
    val_wap_mat = df_val[wap_cols_v].values
    
    # Check invalid RSSI
    invalid_train_mask = (train_wap_mat != UNAVAILABLE_WAP_VALUE) & ((train_wap_mat > 0) | (train_wap_mat < -104))
    invalid_val_mask = (val_wap_mat != UNAVAILABLE_WAP_VALUE) & ((val_wap_mat > 0) | (val_wap_mat < -104))
    train_unexp_signals = int(invalid_train_mask.sum())
    val_unexp_signals = int(invalid_val_mask.sum())
    
    if train_unexp_signals > 0:
        notes.append(f"Training set has {train_unexp_signals} RSSI values outside [-104, 0] dBm (excluding 100).")
    if val_unexp_signals > 0:
        notes.append(f"Validation set has {val_unexp_signals} RSSI values outside [-104, 0] dBm (excluding 100).")

    # 5. Missing / NaN values
    train_nans = int(df_train.isna().sum().sum())
    val_nans = int(df_val.isna().sum().sum())

    # 6. Constant WAPs (never detected = always 100)
    t_constant = [c for c in wap_cols_t if (df_train[c] == UNAVAILABLE_WAP_VALUE).all()]
    v_constant = [c for c in wap_cols_v if (df_val[c] == UNAVAILABLE_WAP_VALUE).all()]
    both_constant = list(set(t_constant) & set(v_constant))

    notes.append(f"{len(t_constant)} WAPs in training set are never observed (constant value 100).")
    notes.append(f"{len(v_constant)} WAPs in validation set are never observed (constant value 100).")
    notes.append(f"{len(both_constant)} WAPs are completely unobserved across BOTH training and validation sets.")

    return QualityCheckResults(
        train_duplicates_count=train_dups,
        val_duplicates_count=val_dups,
        train_constant_waps_count=len(t_constant),
        val_constant_waps_count=len(v_constant),
        train_constant_waps=t_constant,
        val_constant_waps=v_constant,
        never_detected_waps_both_splits=sorted(both_constant),
        train_unexpected_signals_count=train_unexp_signals,
        val_unexpected_signals_count=val_unexp_signals,
        train_nan_count=train_nans,
        val_nan_count=val_nans,
        schema_match=schema_match,
        schema_diff_columns=schema_diff,
        unexpected_buildings_train=unexp_bld_train,
        unexpected_floors_train=unexp_flr_train,
        unexpected_buildings_val=unexp_bld_val,
        unexpected_floors_val=unexp_flr_val,
        notes=notes,
    )


def generate_eda_plots(df_train: pd.DataFrame, df_val: pd.DataFrame, plots_dir: Path) -> None:
    """Generate high-resolution visualization plots for dataset exploration."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    wap_cols, _ = identify_columns(df_train)

    # 1. Building Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    train_bld = df_train["BUILDINGID"].value_counts().sort_index().rename("Train")
    val_bld = df_val["BUILDINGID"].value_counts().sort_index().rename("Validation")
    bld_df = pd.concat([train_bld, val_bld], axis=1).fillna(0)
    bld_df.plot(kind="bar", ax=ax, width=0.7)
    ax.set_title("UJIIndoorLoc: Sample Distribution by Building ID", fontsize=14, fontweight="bold")
    ax.set_xlabel("Building ID", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_xticklabels([f"Building {int(x)}" for x in bld_df.index], rotation=0)
    for p in ax.patches:
        height = int(p.get_height())
        if height > 0:
            ax.annotate(f"{height:,}", (p.get_x() + p.get_width() / 2., height + 100),
                        ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    bld_plot_path = plots_dir / "building_distribution.png"
    plt.savefig(bld_plot_path, dpi=300)
    plt.close()
    logger.info("Saved plot: %s", bld_plot_path)

    # 2. Floor Distribution per Building
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    sns.countplot(data=df_train, x="FLOOR", hue="BUILDINGID", ax=axes[0], palette="Set2")
    axes[0].set_title("Training Set: Floor Distribution by Building", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Floor ID", fontsize=11)
    axes[0].set_ylabel("Sample Count", fontsize=11)
    axes[0].legend(title="Building ID")

    sns.countplot(data=df_val, x="FLOOR", hue="BUILDINGID", ax=axes[1], palette="Set2")
    axes[1].set_title("Validation Set: Floor Distribution by Building", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Floor ID", fontsize=11)
    axes[1].legend(title="Building ID")
    plt.tight_layout()
    floor_plot_path = plots_dir / "floor_distribution.png"
    plt.savefig(floor_plot_path, dpi=300)
    plt.close()
    logger.info("Saved plot: %s", floor_plot_path)

    # 3. Wi-Fi Visibility Distribution (Number of visible APs per record)
    train_visible_aps = (df_train[wap_cols] != UNAVAILABLE_WAP_VALUE).sum(axis=1)
    val_visible_aps = (df_val[wap_cols] != UNAVAILABLE_WAP_VALUE).sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(train_visible_aps, ax=ax, color="#1f77b4", label=f"Train (Mean: {train_visible_aps.mean():.1f})",
                 kde=True, binwidth=2, stat="density", alpha=0.5)
    sns.histplot(val_visible_aps, ax=ax, color="#ff7f0e", label=f"Validation (Mean: {val_visible_aps.mean():.1f})",
                 kde=True, binwidth=2, stat="density", alpha=0.5)
    ax.set_title("Distribution of Visible Access Points Detected per Record", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Detected WAPs (RSSI != +100)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    vis_plot_path = plots_dir / "wifi_visibility_distribution.png"
    plt.savefig(vis_plot_path, dpi=300)
    plt.close()
    logger.info("Saved plot: %s", vis_plot_path)

    # 4. Distribution of Valid RSSI Values (excluding sentinel +100)
    # Extract flattened array of genuine signals
    train_raw_signals = df_train[wap_cols].values
    train_valid_signals = train_raw_signals[train_raw_signals != UNAVAILABLE_WAP_VALUE]
    val_raw_signals = df_val[wap_cols].values
    val_valid_signals = val_raw_signals[val_raw_signals != UNAVAILABLE_WAP_VALUE]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(train_valid_signals, ax=ax, color="#2ca02c", label=f"Train Signals (N={len(train_valid_signals):,})",
                 kde=True, binwidth=2, stat="density", alpha=0.5)
    sns.histplot(val_valid_signals, ax=ax, color="#d62728", label=f"Val Signals (N={len(val_valid_signals):,})",
                 kde=True, binwidth=2, stat="density", alpha=0.5)
    ax.set_title("Distribution of Observed Wi-Fi Signal Strengths (Excluding Sentinel +100)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Signal Strength (RSSI dBm)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.axvline(-104, color="gray", linestyle="--", label="Receiver Noise Floor (-104 dBm)")
    ax.legend(fontsize=11)
    plt.tight_layout()
    rssi_plot_path = plots_dir / "rssi_distribution_valid_signals.png"
    plt.savefig(rssi_plot_path, dpi=300)
    plt.close()
    logger.info("Saved plot: %s", rssi_plot_path)

    # 5. Visible APs per Sample grouped by Building
    df_train_vis = df_train[["BUILDINGID", "FLOOR"]].copy()
    df_train_vis["visible_aps"] = train_visible_aps
    df_train_vis["Split"] = "Train"

    df_val_vis = df_val[["BUILDINGID", "FLOOR"]].copy()
    df_val_vis["visible_aps"] = val_visible_aps
    df_val_vis["Split"] = "Validation"

    combined_vis = pd.concat([df_train_vis, df_val_vis], axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=combined_vis, x="BUILDINGID", y="visible_aps", hue="Split", ax=ax, palette="Pastel1")
    ax.set_title("Detected AP Count per Sample Across Buildings", fontsize=14, fontweight="bold")
    ax.set_xlabel("Building ID", fontsize=12)
    ax.set_ylabel("Visible Access Points Count", fontsize=12)
    ax.set_xticklabels([f"Building {b}" for b in sorted(combined_vis["BUILDINGID"].unique())])
    ax.legend(title="Dataset Split")
    plt.tight_layout()
    aps_plot_path = plots_dir / "visible_aps_per_sample.png"
    plt.savefig(aps_plot_path, dpi=300)
    plt.close()
    logger.info("Saved plot: %s", aps_plot_path)


def run_inspection() -> Dict[str, Any]:
    """Execute complete dataset inspection, statistical profiling, and EDA artifact generation."""
    train_path, val_path, output_dir = get_dataset_paths()
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_train = load_dataset(train_path)
    df_val = load_dataset(val_path)

    wap_cols, context_cols = identify_columns(df_train)
    logger.info("Dataset feature breakdown: %d WAP columns, %d context attributes.", len(wap_cols), len(context_cols))

    # 1. Dataset Overview
    train_mem_mb = float(df_train.memory_usage(deep=True).sum() / (1024 * 1024))
    val_mem_mb = float(df_val.memory_usage(deep=True).sum() / (1024 * 1024))

    overview = {
        "dataset_name": "UJIIndoorLoc",
        "official_source": "https://archive.ics.uci.edu/dataset/310/ujiindoorloc",
        "splits": {
            "training": {
                "filename": train_path.name,
                "rows": int(len(df_train)),
                "columns": int(len(df_train.columns)),
                "memory_mb": round(train_mem_mb, 2),
            },
            "validation": {
                "filename": val_path.name,
                "rows": int(len(df_val)),
                "columns": int(len(df_val.columns)),
                "memory_mb": round(val_mem_mb, 2),
            },
            "combined_total_rows": int(len(df_train) + len(df_val)),
        },
        "feature_composition": {
            "wap_columns_count": len(wap_cols),
            "context_columns_count": len(context_cols),
            "context_column_names": context_cols,
            "wap_prefix": "WAP",
            "unavailable_wap_sentinel_value": UNAVAILABLE_WAP_VALUE,
        },
    }

    overview_path = output_dir / "dataset_overview.json"
    with open(overview_path, "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)
    logger.info("Wrote %s", overview_path)

    # 2. Column Information Table
    col_info_df = compute_column_information(df_train, df_val)
    col_info_path = output_dir / "column_information.csv"
    col_info_df.to_csv(col_info_path, index=False)
    logger.info("Wrote %s (%d columns profiled)", col_info_path, len(col_info_df))

    # 3. Numerical Statistics Table
    train_num_stats = compute_numerical_statistics(df_train, "training")
    val_num_stats = compute_numerical_statistics(df_val, "validation")
    num_stats_df = pd.concat([train_num_stats, val_num_stats], axis=0).reset_index(drop=True)
    num_stats_path = output_dir / "numerical_statistics.csv"
    num_stats_df.to_csv(num_stats_path, index=False)
    logger.info("Wrote %s", num_stats_path)

    # 4. Categorical Statistics JSON
    cat_stats = compute_categorical_statistics(df_train, df_val)
    cat_stats_path = output_dir / "categorical_statistics.json"
    with open(cat_stats_path, "w", encoding="utf-8") as f:
        json.dump(cat_stats, f, indent=2)
    logger.info("Wrote %s", cat_stats_path)

    # 5. WAP Visibility Statistics CSV
    wap_vis_df = compute_wap_visibility_statistics(df_train, df_val)
    wap_vis_path = output_dir / "wap_visibility_statistics.csv"
    wap_vis_df.to_csv(wap_vis_path, index=False)
    logger.info("Wrote %s (%d WAPs profiled)", wap_vis_path, len(wap_vis_df))

    # 6. Quality Checks
    quality_results = run_quality_checks(df_train, df_val)

    # 7. Dataset Split Summary JSON
    train_visible_aps = (df_train[wap_cols] != UNAVAILABLE_WAP_VALUE).sum(axis=1)
    val_visible_aps = (df_val[wap_cols] != UNAVAILABLE_WAP_VALUE).sum(axis=1)
    train_signals = df_train[wap_cols].values
    train_valid = train_signals[train_signals != UNAVAILABLE_WAP_VALUE]
    val_signals = df_val[wap_cols].values
    val_valid = val_signals[val_signals != UNAVAILABLE_WAP_VALUE]

    split_summary = {
        "training_set": {
            "samples": int(len(df_train)),
            "visible_aps_per_sample": {
                "min": int(train_visible_aps.min()),
                "max": int(train_visible_aps.max()),
                "mean": round(float(train_visible_aps.mean()), 2),
                "median": float(train_visible_aps.median()),
                "std": round(float(train_visible_aps.std()), 2),
            },
            "valid_rssi_stats_dbm": {
                "total_observations": int(len(train_valid)),
                "min": float(train_valid.min()),
                "max": float(train_valid.max()),
                "mean": round(float(train_valid.mean()), 2),
                "median": float(np.median(train_valid)),
                "std": round(float(train_valid.std()), 2),
            },
            "building_distribution": {str(k): int(v) for k, v in df_train["BUILDINGID"].value_counts().items()},
            "floor_distribution": {str(k): int(v) for k, v in df_train["FLOOR"].value_counts().items()},
            "unique_users_count": int(df_train["USERID"].nunique()),
            "unique_phones_count": int(df_train["PHONEID"].nunique()),
        },
        "validation_set": {
            "samples": int(len(df_val)),
            "visible_aps_per_sample": {
                "min": int(val_visible_aps.min()),
                "max": int(val_visible_aps.max()),
                "mean": round(float(val_visible_aps.mean()), 2),
                "median": float(val_visible_aps.median()),
                "std": round(float(val_visible_aps.std()), 2),
            },
            "valid_rssi_stats_dbm": {
                "total_observations": int(len(val_valid)),
                "min": float(val_valid.min()),
                "max": float(val_valid.max()),
                "mean": round(float(val_valid.mean()), 2),
                "median": float(np.median(val_valid)),
                "std": round(float(val_valid.std()), 2),
            },
            "building_distribution": {str(k): int(v) for k, v in df_val["BUILDINGID"].value_counts().items()},
            "floor_distribution": {str(k): int(v) for k, v in df_val["FLOOR"].value_counts().items()},
            "unique_users_count": int(df_val["USERID"].nunique()),
            "unique_phones_count": int(df_val["PHONEID"].nunique()),
        },
        "quality_audit": asdict(quality_results),
    }

    split_summary_path = output_dir / "dataset_split_summary.json"
    with open(split_summary_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    logger.info("Wrote %s", split_summary_path)

    # 8. Visualizations
    generate_eda_plots(df_train, df_val, plots_dir)

    return split_summary


if __name__ == "__main__":
    try:
        summary = run_inspection()
        print("\n=======================================================")
        print("UJIIndoorLoc Dataset Inspection Successfully Completed!")
        print("=======================================================")
        train_info = summary["training_set"]
        val_info = summary["validation_set"]
        print(f"Training Set:   {train_info['samples']:,} samples | Users: {train_info['unique_users_count']} | Phones: {train_info['unique_phones_count']}")
        print(f"Validation Set: {val_info['samples']:,} samples | Users: {val_info['unique_users_count']} | Phones: {val_info['unique_phones_count']}")
        print(f"Mean Visible APs/Sample: Train={train_info['visible_aps_per_sample']['mean']} | Val={val_info['visible_aps_per_sample']['mean']}")
        print(f"Quality Check: Duplicates (Train: {summary['quality_audit']['train_duplicates_count']}, Val: {summary['quality_audit']['val_duplicates_count']})")
        print(f"Constant 100 WAPs in Train: {summary['quality_audit']['train_constant_waps_count']} | Val: {summary['quality_audit']['val_constant_waps_count']}")
        print("All EDA artifacts and plots saved under experiments/dataset_eda/")
    except Exception as exc:
        logger.exception("Inspection pipeline execution failed: %s", exc)
        sys.exit(1)

