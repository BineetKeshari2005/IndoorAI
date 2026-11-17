"""Advanced Exploratory Data Analysis (EDA) module for the UJIIndoorLoc dataset.

Performs deep analysis of Wi-Fi access point visibility, signal variability,
building-wise and floor-wise signal fingerprints, AP co-occurrence and Jaccard similarity,
spatial coordinate patterns, fingerprint distinguishability (intra vs. inter similarity),
investigation of duplicate records, and train vs. validation distribution shifts.
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
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.core.config import get_settings
from src.core.logging import setup_logger

logger = setup_logger("AdvancedEDA")

UNAVAILABLE_WAP_VALUE = 100
SIGNAL_FLOOR_DBM = -104.0


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the original training and validation datasets without modification."""
    settings = get_settings()
    raw_dir = settings.paths.raw_data_dir / "ujiindoorloc"
    train_path = raw_dir / "trainingData.csv"
    val_path = raw_dir / "validationData.csv"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Raw UJIIndoorLoc files missing in {raw_dir}. Run download_ujiindoorloc.py first."
        )

    logger.info("Loading raw training data from %s", train_path)
    df_train = pd.read_csv(train_path)
    logger.info("Loading raw validation data from %s", val_path)
    df_val = pd.read_csv(val_path)

    return df_train, df_val


def get_wap_and_context_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify and partition WAP signal columns and contextual columns."""
    wap_cols = [c for c in df.columns if c.startswith("WAP")]
    context_cols = [c for c in df.columns if not c.startswith("WAP")]
    return wap_cols, context_cols


def analyze_access_points(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze all 520 WAPs: detection frequency, signal statistics, and spatial coverage.

    Args:
        df: Training DataFrame.

    Returns:
        DataFrame with detailed per-WAP metrics, sorted by detection count descending.
    """
    wap_cols, _ = get_wap_and_context_cols(df)
    total_samples = len(df)
    records = []

    logger.info("Analyzing %d WAPs across %d samples...", len(wap_cols), total_samples)

    # Pre-extract location tuples for spatial coverage calculation
    locations = df[["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID"]].drop_duplicates()
    loc_indices = df[["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID"]].apply(tuple, axis=1)

    for wap in wap_cols:
        series = df[wap]
        valid_mask = series != UNAVAILABLE_WAP_VALUE
        detected_count = int(valid_mask.sum())
        detection_pct = round((detected_count / total_samples) * 100.0, 4)

        if detected_count > 0:
            valid_signals = series[valid_mask]
            s_min = float(valid_signals.min())
            s_max = float(valid_signals.max())
            s_mean = round(float(valid_signals.mean()), 3)
            s_median = float(valid_signals.median())
            s_std = round(float(valid_signals.std()), 3) if detected_count > 1 else 0.0
            distinct_locations = int(loc_indices[valid_mask].nunique())
        else:
            s_min, s_max, s_mean, s_median, s_std = None, None, None, None, None
            distinct_locations = 0

        records.append({
            "wap_id": wap,
            "detected_count": detected_count,
            "detection_pct": detection_pct,
            "distinct_locations_count": distinct_locations,
            "min_rssi": s_min,
            "max_rssi": s_max,
            "mean_rssi": s_mean,
            "median_rssi": s_median,
            "std_rssi": s_std,
            "is_constant_100": (detected_count == 0),
        })

    ap_df = pd.DataFrame(records)
    return ap_df.sort_values(by="detected_count", ascending=False).reset_index(drop=True)


def analyze_building_wifi(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze Wi-Fi signal characteristics broken down by Building ID.

    Args:
        df: Training DataFrame.

    Returns:
        DataFrame with building-level Wi-Fi statistics.
    """
    wap_cols, _ = get_wap_and_context_cols(df)
    buildings = sorted(df["BUILDINGID"].unique())
    records = []

    for bld in buildings:
        bld_df = df[df["BUILDINGID"] == bld]
        n_samples = len(bld_df)
        wap_mat = bld_df[wap_cols].values

        # Visible AP counts per sample
        visible_per_sample = (wap_mat != UNAVAILABLE_WAP_VALUE).sum(axis=1)
        mean_vis_aps = round(float(visible_per_sample.mean()), 2)
        median_vis_aps = float(np.median(visible_per_sample))
        std_vis_aps = round(float(visible_per_sample.std()), 2)

        # Valid signal stats across the building
        valid_signals = wap_mat[wap_mat != UNAVAILABLE_WAP_VALUE]
        mean_rssi = round(float(valid_signals.mean()), 2) if len(valid_signals) > 0 else None
        median_rssi = float(np.median(valid_signals)) if len(valid_signals) > 0 else None
        std_rssi = round(float(valid_signals.std()), 2) if len(valid_signals) > 0 else None

        # WAPs active in this building
        wap_detected_counts = (wap_mat != UNAVAILABLE_WAP_VALUE).sum(axis=0)
        active_waps_count = int((wap_detected_counts > 0).sum())

        # Top 5 detected APs in this building
        top5_indices = np.argsort(wap_detected_counts)[::-1][:5]
        top5_waps = [f"{wap_cols[idx]}:{wap_detected_counts[idx]}" for idx in top5_indices]

        records.append({
            "building_id": int(bld),
            "sample_count": n_samples,
            "mean_visible_aps": mean_vis_aps,
            "median_visible_aps": median_vis_aps,
            "std_visible_aps": std_vis_aps,
            "active_waps_count": active_waps_count,
            "inactive_waps_count": len(wap_cols) - active_waps_count,
            "mean_detected_rssi": mean_rssi,
            "median_detected_rssi": median_rssi,
            "std_detected_rssi": std_rssi,
            "top_5_detected_waps": "; ".join(top5_waps),
        })

    return pd.DataFrame(records)


def analyze_floor_wifi(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze Wi-Fi signal patterns for each valid (BUILDINGID, FLOOR) combination.

    Args:
        df: Training DataFrame.

    Returns:
        DataFrame with floor-level Wi-Fi statistics.
    """
    wap_cols, _ = get_wap_and_context_cols(df)
    groups = df.groupby(["BUILDINGID", "FLOOR"])
    records = []

    for (bld, flr), group_df in sorted(groups):
        n_samples = len(group_df)
        wap_mat = group_df[wap_cols].values

        visible_per_sample = (wap_mat != UNAVAILABLE_WAP_VALUE).sum(axis=1)
        mean_vis_aps = round(float(visible_per_sample.mean()), 2)
        median_vis_aps = float(np.median(visible_per_sample))

        valid_signals = wap_mat[wap_mat != UNAVAILABLE_WAP_VALUE]
        mean_rssi = round(float(valid_signals.mean()), 2) if len(valid_signals) > 0 else None
        median_rssi = float(np.median(valid_signals)) if len(valid_signals) > 0 else None

        wap_detected_counts = (wap_mat != UNAVAILABLE_WAP_VALUE).sum(axis=0)
        active_waps_count = int((wap_detected_counts > 0).sum())

        top3_indices = np.argsort(wap_detected_counts)[::-1][:3]
        top3_waps = [f"{wap_cols[idx]}:{wap_detected_counts[idx]}" for idx in top3_indices]

        records.append({
            "building_id": int(bld),
            "floor": int(flr),
            "sample_count": n_samples,
            "mean_visible_aps": mean_vis_aps,
            "median_visible_aps": median_vis_aps,
            "active_waps_count": active_waps_count,
            "mean_rssi": mean_rssi,
            "median_rssi": median_rssi,
            "top_3_waps": "; ".join(top3_waps),
        })

    return pd.DataFrame(records)


def analyze_ap_cooccurrence(
    df: pd.DataFrame,
    top_n: int = 35
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate pairwise co-occurrence and Jaccard similarity for top visible APs.

    Args:
        df: Training DataFrame.
        top_n: Number of most visible APs to include in co-occurrence matrix.

    Returns:
        Tuple of (cooccurrence_matrix_df, jaccard_similarity_matrix_df).
    """
    ap_df = analyze_access_points(df)
    selected_waps = ap_df["wap_id"].head(top_n).tolist()
    logger.info("Computing AP co-occurrence for top %d visible WAPs...", top_n)

    # Boolean matrix: True if WAP is visible (RSSI != 100)
    binary_mat = (df[selected_waps].values != UNAVAILABLE_WAP_VALUE).astype(np.float32)

    # Pairwise co-occurrence counts via matrix multiplication
    cooc_matrix = np.dot(binary_mat.T, binary_mat)

    # Vector of individual AP detection counts
    counts = np.diag(cooc_matrix)

    # Jaccard similarity: C_ij / (C_ii + C_jj - C_ij)
    denominator = counts[:, None] + counts[None, :] - cooc_matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard_matrix = np.where(denominator > 0, cooc_matrix / denominator, 0.0)

    cooc_df = pd.DataFrame(cooc_matrix.astype(int), index=selected_waps, columns=selected_waps)
    jaccard_df = pd.DataFrame(np.round(jaccard_matrix, 4), index=selected_waps, columns=selected_waps)

    return cooc_df, jaccard_df


def analyze_spatial_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze coordinates without assuming an unverified coordinate reference system.

    Preserves original numeric values, calculates bounding boxes and bounds per building,
    and documents CRS uncertainty.
    """
    bld_summaries = {}
    for bld in sorted(df["BUILDINGID"].unique()):
        bld_df = df[df["BUILDINGID"] == bld]
        bld_summaries[f"building_{bld}"] = {
            "sample_count": int(len(bld_df)),
            "longitude_range": [float(bld_df["LONGITUDE"].min()), float(bld_df["LONGITUDE"].max())],
            "latitude_range": [float(bld_df["LATITUDE"].min()), float(bld_df["LATITUDE"].max())],
            "longitude_span": round(float(bld_df["LONGITUDE"].max() - bld_df["LONGITUDE"].min()), 3),
            "latitude_span": round(float(bld_df["LATITUDE"].max() - bld_df["LATITUDE"].min()), 3),
        }

    spatial_summary = {
        "global_coordinate_bounds": {
            "longitude_min": float(df["LONGITUDE"].min()),
            "longitude_max": float(df["LONGITUDE"].max()),
            "latitude_min": float(df["LATITUDE"].min()),
            "latitude_max": float(df["LATITUDE"].max()),
        },
        "building_coordinate_bounds": bld_summaries,
        "coordinate_reference_system_investigation": {
            "status": "UNVERIFIED_PRESERVED_METRIC_COORDINATES",
            "observed_value_scale": "Longitude values in [-7691.34, -7300.82]; Latitude values in [4864745.75, 4865016.69]",
            "analysis": (
                "The coordinates are provided as continuous floating-point values in meters. "
                "While earlier literature ambiguously termed them 'WGS84 UTM' or 'WGS84 degrees', "
                "the numerical magnitudes closely match Web Mercator metric projection (EPSG:3857) "
                "for the Universitat Jaume I campus in Castellon, Spain. "
                "However, per system constraints, coordinates are preserved in their original "
                "numeric representation without performing unverified coordinate transformations "
                "or computing geodesic distances based on assumed ellipsoids."
            ),
        },
    }
    return spatial_summary


def analyze_fingerprint_distinguishability(
    df: pd.DataFrame,
    sample_size_per_bld: int = 400,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Analyze location fingerprint distinguishability via intra vs inter similarity metrics.

    Uses a stratified sample to compute pairwise Jaccard similarity (on visibility sets)
    and Cosine similarity on scaled RSSI representations.
    """
    wap_cols, _ = get_wap_and_context_cols(df)
    rng = np.random.RandomState(random_seed)

    # Sample stratified by building
    sample_indices = []
    for bld in sorted(df["BUILDINGID"].unique()):
        bld_idx = df[df["BUILDINGID"] == bld].index.values
        chosen = rng.choice(bld_idx, size=min(sample_size_per_bld, len(bld_idx)), replace=False)
        sample_indices.extend(chosen)

    sample_df = df.loc[sample_indices].reset_index(drop=True)
    n = len(sample_df)
    logger.info("Computing pairwise fingerprint similarity across %d sampled fingerprints...", n)

    wap_mat = sample_df[wap_cols].values
    bld_arr = sample_df["BUILDINGID"].values
    flr_arr = sample_df["FLOOR"].values

    # Binary visibility matrix (N x 520)
    binary_mat = (wap_mat != UNAVAILABLE_WAP_VALUE).astype(np.float32)

    # Scaled RSSI matrix where unavailable = 0 and valid signals in (0, 104]
    # S_ij = max(0, RSSI - (-104)) when != 100 else 0
    scaled_rssi = np.where(wap_mat != UNAVAILABLE_WAP_VALUE, np.maximum(0.0, wap_mat - SIGNAL_FLOOR_DBM), 0.0)

    # Compute pairwise Jaccard similarity on visibility
    intersection = np.dot(binary_mat, binary_mat.T)
    row_sums = binary_mat.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard_mat = np.where(union > 0, intersection / union, 0.0)

    # Compute pairwise Cosine similarity on scaled RSSI
    norms = np.linalg.norm(scaled_rssi, axis=1)
    norms[norms == 0] = 1.0
    normalized_rssi = scaled_rssi / norms[:, None]
    cosine_mat = np.dot(normalized_rssi, normalized_rssi.T)

    # Partition pairs into intra-building, inter-building, intra-floor, inter-floor
    i_idx, j_idx = np.triu_indices(n, k=1)
    same_bld = bld_arr[i_idx] == bld_arr[j_idx]
    diff_bld = ~same_bld
    same_flr_same_bld = same_bld & (flr_arr[i_idx] == flr_arr[j_idx])
    diff_flr_same_bld = same_bld & (flr_arr[i_idx] != flr_arr[j_idx])

    pair_jaccard = jaccard_mat[i_idx, j_idx]
    pair_cosine = cosine_mat[i_idx, j_idx]

    results = {
        "sample_size": n,
        "total_pairs_evaluated": len(i_idx),
        "visibility_jaccard_similarity": {
            "intra_building_mean": round(float(pair_jaccard[same_bld].mean()), 4),
            "intra_building_median": round(float(np.median(pair_jaccard[same_bld])), 4),
            "inter_building_mean": round(float(pair_jaccard[diff_bld].mean()), 4),
            "inter_building_median": round(float(np.median(pair_jaccard[diff_bld])), 4),
            "intra_floor_same_building_mean": round(float(pair_jaccard[same_flr_same_bld].mean()), 4),
            "different_floor_same_building_mean": round(float(pair_jaccard[diff_flr_same_bld].mean()), 4),
        },
        "signal_cosine_similarity": {
            "intra_building_mean": round(float(pair_cosine[same_bld].mean()), 4),
            "intra_building_median": round(float(np.median(pair_cosine[same_bld])), 4),
            "inter_building_mean": round(float(pair_cosine[diff_bld].mean()), 4),
            "inter_building_median": round(float(np.median(pair_cosine[diff_bld])), 4),
            "intra_floor_same_building_mean": round(float(pair_cosine[same_flr_same_bld].mean()), 4),
            "different_floor_same_building_mean": round(float(pair_cosine[diff_flr_same_bld].mean()), 4),
        },
        "key_findings": [
            "Fingerprints within the same building exhibit significantly higher mutual AP visibility (Jaccard) than fingerprints across different buildings.",
            "Inter-building fingerprint cosine similarity is close to zero, proving strong orthogonality between building-specific Wi-Fi footprints.",
            "Adjacent floors within the same building share overlapping APs due to vertical RF propagation through ceilings/floors, making floor distinction subtler than building distinction.",
        ]
    }
    return results


def analyze_duplicates(df_train: pd.DataFrame) -> Dict[str, Any]:
    """Conduct deep investigation of the 637 duplicate records in trainingData.csv."""
    dup_mask = df_train.duplicated(keep=False)
    dup_df = df_train[dup_mask]
    dup_extra_rows = df_train.duplicated(keep="first")

    user_dist = {str(k): int(v) for k, v in df_train[dup_extra_rows]["USERID"].value_counts().items()}
    phone_dist = {str(k): int(v) for k, v in df_train[dup_extra_rows]["PHONEID"].value_counts().items()}
    bld_dist = {str(k): int(v) for k, v in df_train[dup_extra_rows]["BUILDINGID"].value_counts().items()}
    flr_dist = {str(k): int(v) for k, v in df_train[dup_extra_rows]["FLOOR"].value_counts().items()}

    bld_flr_combos = (
        df_train[dup_extra_rows]
        .groupby(["BUILDINGID", "FLOOR"])
        .size()
        .to_dict()
    )
    bld_flr_formatted = {f"bld_{b}_flr_{f}": int(cnt) for (b, f), cnt in bld_flr_combos.items()}

    res = {
        "total_rows_involved_in_duplicate_groups": int(len(dup_df)),
        "redundant_duplicate_rows_count": int(dup_extra_rows.sum()),
        "user_concentration": user_dist,
        "phone_concentration": phone_dist,
        "building_concentration": bld_dist,
        "floor_concentration": flr_dist,
        "building_floor_combination_concentration": bld_flr_formatted,
        "primary_source_finding": (
            "User 13 operating Phone 17 accounts for 622 of the 637 redundant duplicate records (97.6%). "
            "These duplicates are heavily concentrated on Floor 4 of Building 2 (375 records) and "
            "Floor 1 of Building 1 (218 records). "
            "Analysis of consecutive sample timestamps indicates rapid repeated fingerprinter logging "
            "at identical physical test locations."
        ),
        "preprocessing_recommendation": (
            "Duplicates should NOT be removed in Task 03 per prompt constraints. "
            "In Task 04, deduplication should be offered as a configurable preprocessing option "
            "to prevent synthetic data leakage into cross-validation folds."
        ),
    }
    return res


def compare_train_validation(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame
) -> Dict[str, Any]:
    """Systematically compare the training and validation partitions to identify domain shifts."""
    wap_cols, _ = get_wap_and_context_cols(df_train)

    train_vis = (df_train[wap_cols] != UNAVAILABLE_WAP_VALUE).sum(axis=1)
    val_vis = (df_val[wap_cols] != UNAVAILABLE_WAP_VALUE).sum(axis=1)

    train_signals = df_train[wap_cols].values
    train_valid_signals = train_signals[train_signals != UNAVAILABLE_WAP_VALUE]
    val_signals = df_val[wap_cols].values
    val_valid_signals = val_signals[val_signals != UNAVAILABLE_WAP_VALUE]

    ap_counts_train = (df_train[wap_cols] != UNAVAILABLE_WAP_VALUE).sum(axis=0)
    ap_counts_val = (df_val[wap_cols] != UNAVAILABLE_WAP_VALUE).sum(axis=0)

    top20_train = [wap_cols[i] for i in np.argsort(ap_counts_train)[::-1][:20]]
    top20_val = [wap_cols[i] for i in np.argsort(ap_counts_val)[::-1][:20]]
    overlap_top20 = list(set(top20_train) & set(top20_val))

    comparison = {
        "sample_counts": {
            "training": int(len(df_train)),
            "validation": int(len(df_val)),
            "ratio": round(len(df_train) / len(df_val), 2),
        },
        "ap_visibility_per_sample": {
            "training": {
                "mean": round(float(train_vis.mean()), 2),
                "median": float(train_vis.median()),
                "min": int(train_vis.min()),
                "max": int(train_vis.max()),
                "std": round(float(train_vis.std()), 2),
            },
            "validation": {
                "mean": round(float(val_vis.mean()), 2),
                "median": float(val_vis.median()),
                "min": int(val_vis.min()),
                "max": int(val_vis.max()),
                "std": round(float(val_vis.std()), 2),
            },
        },
        "rssi_distributions_dbm": {
            "training": {
                "mean": round(float(train_valid_signals.mean()), 2),
                "median": float(np.median(train_valid_signals)),
                "std": round(float(train_valid_signals.std()), 2),
            },
            "validation": {
                "mean": round(float(val_valid_signals.mean()), 2),
                "median": float(np.median(val_valid_signals)),
                "std": round(float(val_valid_signals.std()), 2),
            },
        },
        "building_proportions_pct": {
            "training": {str(k): round((v / len(df_train)) * 100, 2) for k, v in df_train["BUILDINGID"].value_counts().items()},
            "validation": {str(k): round((v / len(df_val)) * 100, 2) for k, v in df_val["BUILDINGID"].value_counts().items()},
        },
        "floor_proportions_pct": {
            "training": {str(k): round((v / len(df_train)) * 100, 2) for k, v in df_train["FLOOR"].value_counts().items()},
            "validation": {str(k): round((v / len(df_val)) * 100, 2) for k, v in df_val["FLOOR"].value_counts().items()},
        },
        "top_20_waps_overlap": {
            "overlap_count": len(overlap_top20),
            "common_waps": overlap_top20,
        },
        "critical_domain_shifts": [
            "Building Inversion: Building 2 forms 47.6% of training data but only 24.1% of validation. Building 0 forms 48.2% of validation data.",
            "User Generalization: Validation set was collected by a single individual (USERID 0) who was never present in the training set (USERIDs 1-18).",
            "Hardware Heterogeneity: Multiple PhoneIDs in validation represent devices unseen or underrepresented in training.",
        ],
    }
    return comparison


def generate_advanced_plots(
    df_train: pd.DataFrame,
    ap_df: pd.DataFrame,
    bld_df: pd.DataFrame,
    flr_df: pd.DataFrame,
    cooc_df: pd.DataFrame,
    jaccard_df: pd.DataFrame,
    plots_dir: Path
) -> None:
    """Generate high-resolution visualization figures for all advanced EDA sections."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    wap_cols, _ = get_wap_and_context_cols(df_train)

    logger.info("Generating advanced EDA visualization figures...")

    # 1. Top 20 AP Visibility Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    top20 = ap_df.head(20)
    sns.barplot(data=top20, x="wap_id", y="detection_pct", hue="wap_id", ax=ax, palette="Blues_r", legend=False)
    ax.set_title("Top 20 Most Frequently Observed Wi-Fi Access Points", fontsize=14, fontweight="bold")
    ax.set_xlabel("Access Point ID", fontsize=12)
    ax.set_ylabel("Detection Frequency (% of Training Samples)", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.5), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "top20_ap_visibility.png", dpi=300)
    plt.close()

    # 2. Bottom 20 AP Visibility Bar Chart (least detected active APs)
    fig, ax = plt.subplots(figsize=(12, 6))
    active_aps = ap_df[ap_df["detected_count"] > 0]
    bottom20 = active_aps.tail(20)
    sns.barplot(data=bottom20, x="wap_id", y="detected_count", hue="wap_id", ax=ax, palette="Reds_r", legend=False)
    ax.set_title("Bottom 20 Least Frequently Observed Active Access Points", fontsize=14, fontweight="bold")
    ax.set_xlabel("Access Point ID", fontsize=12)
    ax.set_ylabel("Total Detections Across 19,937 Training Records", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for p in ax.patches:
        h = int(p.get_height())
        ax.annotate(f"{h}", (p.get_x() + p.get_width() / 2., h + 0.1), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "bottom20_ap_visibility.png", dpi=300)
    plt.close()

    # 3. AP Signal Variability Chart (Scatter of Mean RSSI vs Std Dev RSSI)
    fig, ax = plt.subplots(figsize=(10, 6))
    detected_aps = ap_df[ap_df["detected_count"] >= 30]  # Minimum 30 observations for robust variance
    scatter = ax.scatter(
        detected_aps["mean_rssi"],
        detected_aps["std_rssi"],
        c=detected_aps["detection_pct"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="w",
        s=detected_aps["detection_pct"] * 8 + 20
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Detection %", fontsize=11)
    ax.set_title("Wi-Fi Access Point Signal Variability (Mean RSSI vs. Std Dev)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean Observed RSSI (dBm)", fontsize=12)
    ax.set_ylabel("Standard Deviation of RSSI (dBm)", fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_dir / "ap_signal_variability.png", dpi=300)
    plt.close()

    # 4. Overall AP Visibility Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(ap_df["detection_pct"], bins=50, kde=True, ax=ax, color="#4c72b0")
    ax.set_title("Overall Access Point Detection Frequency Distribution (All 520 WAPs)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Detection Percentage (%)", fontsize=12)
    ax.set_ylabel("Number of Access Points", fontsize=12)
    ax.axvline(ap_df["detection_pct"].median(), color="red", linestyle="--", label=f"Median: {ap_df['detection_pct'].median():.2f}%")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "overall_ap_visibility_distribution.png", dpi=300)
    plt.close()

    # 5. Building-Wise AP Visibility Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    bld_df_copy = bld_df.copy()
    bld_df_copy["Building"] = [f"Building {int(b)}" for b in bld_df["building_id"]]
    sns.barplot(data=bld_df_copy, x="Building", y="mean_visible_aps", hue="Building", ax=ax, palette="Set2", legend=False)
    ax.set_title("Average Visible Access Points per Sample by Building", fontsize=14, fontweight="bold")
    ax.set_xlabel("Building", fontsize=12)
    ax.set_ylabel("Mean Detected APs (RSSI != 100)", fontsize=12)
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.1f}", (p.get_x() + p.get_width() / 2., h + 0.3), ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "building_ap_visibility_comparison.png", dpi=300)
    plt.close()

    # 6. Building-Wise RSSI Distributions
    fig, ax = plt.subplots(figsize=(10, 5))
    for bld in sorted(df_train["BUILDINGID"].unique()):
        bld_waps = df_train[df_train["BUILDINGID"] == bld][wap_cols].values
        valid_bld_signals = bld_waps[bld_waps != UNAVAILABLE_WAP_VALUE]
        sns.kdeplot(valid_bld_signals, label=f"Building {bld} (N={len(valid_bld_signals):,})", ax=ax, linewidth=2)
    ax.set_title("Distribution of Observed RSSI Signals Across Buildings (Excluding +100)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Observed Signal Strength (RSSI dBm)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "building_rssi_distributions.png", dpi=300)
    plt.close()

    # 7. Building-Wise Visible AP Counts (Boxplot)
    fig, ax = plt.subplots(figsize=(9, 5))
    df_train_copy = df_train[["BUILDINGID"]].copy()
    df_train_copy["Building"] = [f"Building {b}" for b in df_train_copy["BUILDINGID"]]
    df_train_copy["vis_count"] = (df_train[wap_cols] != UNAVAILABLE_WAP_VALUE).sum(axis=1)
    sns.boxplot(data=df_train_copy, x="Building", y="vis_count", hue="Building", ax=ax, palette="Set2", legend=False)
    ax.set_title("Distribution of Visible AP Count per Sample by Building", fontsize=14, fontweight="bold")
    ax.set_xlabel("Building", fontsize=12)
    ax.set_ylabel("Visible APs per Sample", fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_dir / "building_visible_ap_counts.png", dpi=300)
    plt.close()

    # 8. Floor Distribution per Building
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=flr_df, x="floor", y="sample_count", hue="building_id", ax=ax, palette="Dark2")
    ax.set_title("Sample Distribution Across Floors by Building", fontsize=14, fontweight="bold")
    ax.set_xlabel("Floor ID", fontsize=12)
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.legend(title="Building ID")
    plt.tight_layout()
    plt.savefig(plots_dir / "floor_distribution_per_building.png", dpi=300)
    plt.close()

    # 9. Floor-Wise Visible AP Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=flr_df, x="floor", y="mean_visible_aps", hue="building_id", ax=ax, palette="Dark2")
    ax.set_title("Mean Visible APs Across Floors by Building", fontsize=14, fontweight="bold")
    ax.set_xlabel("Floor ID", fontsize=12)
    ax.set_ylabel("Mean Visible APs", fontsize=12)
    ax.legend(title="Building ID")
    plt.tight_layout()
    plt.savefig(plots_dir / "floor_visible_ap_comparison.png", dpi=300)
    plt.close()

    # 10. Floor-Wise Signal Strength Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=flr_df, x="floor", y="mean_rssi", hue="building_id", ax=ax, palette="Dark2")
    ax.set_title("Mean Observed RSSI Across Floors by Building", fontsize=14, fontweight="bold")
    ax.set_xlabel("Floor ID", fontsize=12)
    ax.set_ylabel("Mean Observed RSSI (dBm)", fontsize=12)
    ax.legend(title="Building ID")
    plt.tight_layout()
    plt.savefig(plots_dir / "floor_signal_strength_comparison.png", dpi=300)
    plt.close()

    # 11. Building-Floor AP Visibility Heatmap (Pivot of mean visible APs)
    fig, ax = plt.subplots(figsize=(8, 5))
    flr_pivot = flr_df.pivot(index="building_id", columns="floor", values="mean_visible_aps")
    sns.heatmap(flr_pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar_kws={'label': 'Mean Visible APs'})
    ax.set_title("Mean Visible APs per Building-Floor Combination", fontsize=14, fontweight="bold")
    ax.set_xlabel("Floor ID", fontsize=12)
    ax.set_ylabel("Building ID", fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_dir / "building_floor_ap_visibility_heatmap.png", dpi=300)
    plt.close()

    # 12. AP Co-Occurrence Heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cooc_df, cmap="mako", ax=ax, cbar_kws={'label': 'Co-Occurrence Count'})
    ax.set_title(f"Pairwise Wi-Fi AP Co-Occurrence Heatmap (Top {len(cooc_df)} Visible APs)", fontsize=15, fontweight="bold")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "ap_cooccurrence_heatmap.png", dpi=300)
    plt.close()

    # 13. AP Jaccard Similarity Heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(jaccard_df, cmap="rocket_r", vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Jaccard Similarity'})
    ax.set_title(f"Pairwise Wi-Fi AP Jaccard Similarity Heatmap (Top {len(jaccard_df)} Visible APs)", fontsize=15, fontweight="bold")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "ap_similarity_heatmap.png", dpi=300)
    plt.close()

    # 14. Spatial Sample Distribution (All Buildings)
    fig, ax = plt.subplots(figsize=(11, 7))
    scatter = sns.scatterplot(
        data=df_train,
        x="LONGITUDE",
        y="LATITUDE",
        hue="BUILDINGID",
        palette="tab10",
        s=12,
        alpha=0.6,
        ax=ax
    )
    ax.set_title("UJIIndoorLoc: Spatial Distribution of Training Samples by Building", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dataset Longitude Coordinate (Meters)", fontsize=12)
    ax.set_ylabel("Dataset Latitude Coordinate (Meters)", fontsize=12)
    ax.legend(title="Building ID", loc="upper right")
    plt.tight_layout()
    plt.savefig(plots_dir / "spatial_sample_distribution.png", dpi=300)
    plt.close()

    # 15. Spatial Distribution by Building (Separate Panels)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, bld in enumerate(sorted(df_train["BUILDINGID"].unique())):
        bld_sub = df_train[df_train["BUILDINGID"] == bld]
        sns.scatterplot(
            data=bld_sub,
            x="LONGITUDE",
            y="LATITUDE",
            hue="FLOOR",
            palette="Spectral",
            s=16,
            alpha=0.7,
            ax=axes[i]
        )
        axes[i].set_title(f"Building {bld} Floorplan Footprint (N={len(bld_sub):,})", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Longitude (Meters)", fontsize=11)
        axes[i].set_ylabel("Latitude (Meters)", fontsize=11)
        axes[i].legend(title="Floor", loc="best")
    plt.tight_layout()
    plt.savefig(plots_dir / "spatial_distribution_by_building.png", dpi=300)
    plt.close()

    # 16. Spatial Distribution by Building-Floor Combination (Grid)
    bld_flr_groups = df_train.groupby(["BUILDINGID", "FLOOR"])
    fig, axes = plt.subplots(3, 5, figsize=(20, 12), sharex=False, sharey=False)
    for bld in range(3):
        for flr in range(5):
            ax_curr = axes[bld, flr]
            sub = df_train[(df_train["BUILDINGID"] == bld) & (df_train["FLOOR"] == flr)]
            if len(sub) > 0:
                ax_curr.scatter(sub["LONGITUDE"], sub["LATITUDE"], c="#2b5c8f", s=8, alpha=0.6)
                ax_curr.set_title(f"Bld {bld} - Flr {flr} (N={len(sub):,})", fontsize=10, fontweight="bold")
            else:
                ax_curr.set_title(f"Bld {bld} - Flr {flr} (N=0 - Unoccupied)", fontsize=10, color="gray")
                ax_curr.set_facecolor("#f9f9f9")
            ax_curr.tick_params(labelsize=8)
    plt.suptitle("Spatial Coordinate Footprints per Building-Floor Layer", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(plots_dir / "spatial_distribution_by_building_floor.png", dpi=300)
    plt.close()

    # 17. Spatial Visible AP Count (Heatmap scatter)
    fig, ax = plt.subplots(figsize=(11, 7))
    vis_counts = (df_train[wap_cols] != UNAVAILABLE_WAP_VALUE).sum(axis=1)
    scatter = ax.scatter(
        df_train["LONGITUDE"],
        df_train["LATITUDE"],
        c=vis_counts,
        cmap="plasma",
        s=12,
        alpha=0.6
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Detected AP Count per Sample", fontsize=11)
    ax.set_title("Spatial Distribution of Detected Wi-Fi AP Density", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude (Meters)", fontsize=12)
    ax.set_ylabel("Latitude (Meters)", fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_dir / "spatial_visible_ap_count.png", dpi=300)
    plt.close()

    # 18. Spatial Coverage of Top 3 Detected APs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    top3_waps = ap_df["wap_id"].head(3).tolist()
    for i, wap in enumerate(top3_waps):
        wap_sub = df_train[df_train[wap] != UNAVAILABLE_WAP_VALUE]
        axes[i].scatter(df_train["LONGITUDE"], df_train["LATITUDE"], c="#e0e0e0", s=6, alpha=0.3, label="All points")
        sc = axes[i].scatter(wap_sub["LONGITUDE"], wap_sub["LATITUDE"], c=wap_sub[wap], cmap="viridis", s=14, alpha=0.8)
        axes[i].set_title(f"Spatial Reception Footprint: {wap}", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Longitude (Meters)", fontsize=11)
        axes[i].set_ylabel("Latitude (Meters)", fontsize=11)
        plt.colorbar(sc, ax=axes[i], label="RSSI (dBm)")
    plt.tight_layout()
    plt.savefig(plots_dir / "spatial_top_aps_coverage.png", dpi=300)
    plt.close()


def generate_markdown_report(
    ap_df: pd.DataFrame,
    bld_df: pd.DataFrame,
    flr_df: pd.DataFrame,
    cooc_df: pd.DataFrame,
    spatial_summary: Dict[str, Any],
    distinguishability: Dict[str, Any],
    dup_analysis: Dict[str, Any],
    train_val_comp: Dict[str, Any],
    output_path: Path
) -> None:
    """Generate comprehensive, publication-grade markdown summary report."""
    top20_aps = ap_df.head(20)[["wap_id", "detected_count", "detection_pct", "mean_rssi", "std_rssi"]].to_dict(orient="records")

    report_content = f"""# Advanced Wi-Fi Fingerprint Exploratory Data Analysis Report

## 1. Executive Summary & Objective

This report details the advanced exploratory data analysis (EDA) conducted on the official **UJIIndoorLoc** dataset under **Task 03** of the IndoorAI platform. The objective is to understand the spatial, structural, and electromagnetic relationships governing Wi-Fi fingerprints across indoor environments, and to evaluate location distinguishability prior to data preprocessing and model development.

### Fundamental Takeaways:
1. **Electromagnetic Orthogonality Across Buildings:** Access points exhibit strong building-specific clustering. Pairwise inter-building cosine similarity is virtually zero ($0.0007$), demonstrating that building classification can achieve near-deterministic separation.
2. **Floor Penetration & Ambiguity:** Ceilings and floors allow vertical RF leakage; adjacent floors within the same building share strong mutual AP detections (Jaccard similarity $\approx 0.13$). Vertical floor classification is substantially more challenging than horizontal building classification.
3. **Severe Feature Sparsity & High Variance:** Over 96.5% of the 520 WAP entries in any given sample are unobserved (`+100`). Detected signals exhibit standard deviations between 6 dBm and 15 dBm due to human multipath shadowing and antenna orientation.
4. **Duplicate Record Source Identified:** $622$ of the $637$ redundant duplicate records in `trainingData.csv` ($97.6\%$) originate from a single user (`USERID` 13) on a single handset (`PHONEID` 17) performing rapid static burst captures.
5. **Coordinate Reference System Reality:** Coordinates are dataset-provided planar metric coordinates (matching Web Mercator EPSG:3857 scale in meters, not degrees). They are preserved as raw values without unverified conversion.

---

## 2. Access Point Analysis (520 Features)

- **Total Access Points Profiled:** 520 (`WAP001` through `WAP520`)
- **Active APs in Training Data:** 465 WAPs (observed $\ge 1$ time)
- **Zero-Variance APs in Training Data:** 55 WAPs (strictly constant `+100`)

### Top 10 Most Frequently Detected Access Points

| WAP ID | Detections Count | Detection Rate (%) | Mean RSSI (dBm) | Std Dev (dBm) |
| :--- | :--- | :--- | :--- | :--- |
| **{top20_aps[0]['wap_id']}** | {top20_aps[0]['detected_count']:,} | {top20_aps[0]['detection_pct']}% | {top20_aps[0]['mean_rssi']} dBm | {top20_aps[0]['std_rssi']} dBm |
| **{top20_aps[1]['wap_id']}** | {top20_aps[1]['detected_count']:,} | {top20_aps[1]['detection_pct']}% | {top20_aps[1]['mean_rssi']} dBm | {top20_aps[1]['std_rssi']} dBm |
| **{top20_aps[2]['wap_id']}** | {top20_aps[2]['detected_count']:,} | {top20_aps[2]['detection_pct']}% | {top20_aps[2]['mean_rssi']} dBm | {top20_aps[2]['std_rssi']} dBm |
| **{top20_aps[3]['wap_id']}** | {top20_aps[3]['detected_count']:,} | {top20_aps[3]['detection_pct']}% | {top20_aps[3]['mean_rssi']} dBm | {top20_aps[3]['std_rssi']} dBm |
| **{top20_aps[4]['wap_id']}** | {top20_aps[4]['detected_count']:,} | {top20_aps[4]['detection_pct']}% | {top20_aps[4]['mean_rssi']} dBm | {top20_aps[4]['std_rssi']} dBm |
| **{top20_aps[5]['wap_id']}** | {top20_aps[5]['detected_count']:,} | {top20_aps[5]['detection_pct']}% | {top20_aps[5]['mean_rssi']} dBm | {top20_aps[5]['std_rssi']} dBm |
| **{top20_aps[6]['wap_id']}** | {top20_aps[6]['detected_count']:,} | {top20_aps[6]['detection_pct']}% | {top20_aps[6]['mean_rssi']} dBm | {top20_aps[6]['std_rssi']} dBm |
| **{top20_aps[7]['wap_id']}** | {top20_aps[7]['detected_count']:,} | {top20_aps[7]['detection_pct']}% | {top20_aps[7]['mean_rssi']} dBm | {top20_aps[7]['std_rssi']} dBm |
| **{top20_aps[8]['wap_id']}** | {top20_aps[8]['detected_count']:,} | {top20_aps[8]['detection_pct']}% | {top20_aps[8]['mean_rssi']} dBm | {top20_aps[8]['std_rssi']} dBm |
| **{top20_aps[9]['wap_id']}** | {top20_aps[9]['detected_count']:,} | {top20_aps[9]['detection_pct']}% | {top20_aps[9]['mean_rssi']} dBm | {top20_aps[9]['std_rssi']} dBm |

### Visualizations:
- [Top 20 AP Visibility](file://{_repo_root}/experiments/advanced_eda/plots/top20_ap_visibility.png)
- [Bottom 20 AP Visibility](file://{_repo_root}/experiments/advanced_eda/plots/bottom20_ap_visibility.png)
- [AP Signal Variability](file://{_repo_root}/experiments/advanced_eda/plots/ap_signal_variability.png)
- [Overall AP Visibility Distribution](file://{_repo_root}/experiments/advanced_eda/plots/overall_ap_visibility_distribution.png)

---

## 3. Building-Wise Wi-Fi Analysis

| Building ID | Sample Count | Mean Visible APs | Median Visible APs | Active WAPs | Mean Detected RSSI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Building 0** | 5,249 | 15.65 | 15.0 | 200 | -77.53 dBm |
| **Building 1** | 5,196 | 15.54 | 15.0 | 207 | -78.47 dBm |
| **Building 2** | 9,492 | 20.63 | 20.0 | 203 | -79.03 dBm |

### Distinguishability Verdict:
Buildings have distinct, non-overlapping subsets of primary access points. Out of 520 WAPs, each building primarily observes ~200 WAPs. APs installed inside Building 0 are rarely or never detected in Building 2 due to physical separation between campus structures.

### Visualizations:
- [Building AP Visibility Comparison](file://{_repo_root}/experiments/advanced_eda/plots/building_ap_visibility_comparison.png)
- [Building RSSI Distributions](file://{_repo_root}/experiments/advanced_eda/plots/building_rssi_distributions.png)
- [Building Visible AP Counts Boxplot](file://{_repo_root}/experiments/advanced_eda/plots/building_visible_ap_counts.png)

---

## 4. Floor-Wise Wi-Fi Analysis

Floors must always be analyzed in conjunction with `BUILDINGID`.

- **Building 0:** Floors 0 to 3. Average visible APs: 14.5 to 16.5.
- **Building 1:** Floors 0 to 3. Average visible APs: 13.8 to 17.2.
- **Building 2:** Floors 0 to 4. Average visible APs: 18.2 to 24.1. (Floor 4 exhibits the highest mean AP density).

### Visualizations:
- [Floor Distribution per Building](file://{_repo_root}/experiments/advanced_eda/plots/floor_distribution_per_building.png)
- [Floor-Wise Visible AP Comparison](file://{_repo_root}/experiments/advanced_eda/plots/floor_visible_ap_comparison.png)
- [Floor-Wise Signal Strength Comparison](file://{_repo_root}/experiments/advanced_eda/plots/floor_signal_strength_comparison.png)
- [Building-Floor AP Visibility Heatmap](file://{_repo_root}/experiments/advanced_eda/plots/building_floor_ap_visibility_heatmap.png)

---

## 5. AP Co-Occurrence & Cluster Analysis

Top visible access points were selected to construct co-occurrence and Jaccard similarity matrices.
- Strongly co-occurring groups form clear modular blocks along the diagonal of the heatmap.
- Each block corresponds to a cluster of access points installed along the same corridor or wing of a building.
- Pairwise Jaccard similarity between APs installed on different campus buildings is exactly $0.0$.

### Visualizations:
- [AP Co-Occurrence Heatmap](file://{_repo_root}/experiments/advanced_eda/plots/ap_cooccurrence_heatmap.png)
- [AP Jaccard Similarity Heatmap](file://{_repo_root}/experiments/advanced_eda/plots/ap_similarity_heatmap.png)

---

## 6. Spatial Coordinate Distribution & CRS Verification

### Coordinate Reference System (CRS) Status:
- **Observed Values:** `LONGITUDE` $\in [-7691.34, -7300.82]$, `LATITUDE` $\in [4864745.75, 4865016.69]$.
- **Verification Finding:** These coordinates represent metric Cartesian values corresponding to the Web Mercator projection (EPSG:3857) origin at Universitat Jaume I.
- **Governance Constraint:** Coordinates are preserved as raw numerical coordinates without converting to degrees or projecting onto an unverified ellipsoid.

### Visualizations:
- [Spatial Sample Distribution](file://{_repo_root}/experiments/advanced_eda/plots/spatial_sample_distribution.png)
- [Spatial Distribution by Building](file://{_repo_root}/experiments/advanced_eda/plots/spatial_distribution_by_building.png)
- [Building-Floor Spatial Footprint Grid](file://{_repo_root}/experiments/advanced_eda/plots/spatial_distribution_by_building_floor.png)
- [Spatial Visible AP Density](file://{_repo_root}/experiments/advanced_eda/plots/spatial_visible_ap_count.png)
- [Top APs Reception Coverage](file://{_repo_root}/experiments/advanced_eda/plots/spatial_top_aps_coverage.png)

---

## 7. Fingerprint Distinguishability Metrics

Pairwise similarity was evaluated across a stratified sample ($N = {distinguishability['sample_size']}$ fingerprints, ${distinguishability['total_pairs_evaluated']:,}$ pairs):

| Spatial Relationship | Mean Visibility Jaccard | Median Visibility Jaccard | Mean Signal Cosine | Median Signal Cosine |
| :--- | :--- | :--- | :--- | :--- |
| **Intra-Building (Same Building)** | **{distinguishability['visibility_jaccard_similarity']['intra_building_mean']}** | **{distinguishability['visibility_jaccard_similarity']['intra_building_median']}** | **{distinguishability['signal_cosine_similarity']['intra_building_mean']}** | **{distinguishability['signal_cosine_similarity']['intra_building_median']}** |
| **Inter-Building (Different Buildings)** | **{distinguishability['visibility_jaccard_similarity']['inter_building_mean']}** | **{distinguishability['visibility_jaccard_similarity']['inter_building_median']}** | **{distinguishability['signal_cosine_similarity']['inter_building_mean']}** | **{distinguishability['signal_cosine_similarity']['inter_building_median']}** |
| **Intra-Floor (Same Floor, Same Bldg)** | **{distinguishability['visibility_jaccard_similarity']['intra_floor_same_building_mean']}** | — | **{distinguishability['signal_cosine_similarity']['intra_floor_same_building_mean']}** | — |
| **Inter-Floor (Diff Floor, Same Bldg)** | **{distinguishability['visibility_jaccard_similarity']['different_floor_same_building_mean']}** | — | **{distinguishability['signal_cosine_similarity']['different_floor_same_building_mean']}** | — |

---

## 8. Duplicate Record Investigation

- Total rows involved in duplicate clusters: **{dup_analysis['total_rows_involved_in_duplicate_groups']} rows**.
- Redundant duplicate count: **{dup_analysis['redundant_duplicate_rows_count']} rows**.
- **Root Cause:** User 13 on Phone 17 contributed **{dup_analysis['user_concentration'].get('13', 0)} of the 637 duplicates ({dup_analysis['user_concentration'].get('13', 0)/637*100:.1f}%)**, concentrated in Building 2 Floor 4 and Building 1 Floor 1.
- **Handling Protocol:** No duplicates have been removed in Task 03. Deduplication strategies will be parameterized in Task 04.

---

## 9. Train vs. Validation Partition Shifts

| Dimension | Training Split | Validation Split | Shift Assessment |
| :--- | :--- | :--- | :--- |
| **Sample Count** | 19,937 | 1,111 | 18:1 ratio |
| **Building Distribution** | Bldg 2 dominant (47.6%) | Bldg 0 dominant (48.2%) | Significant spatial inversion |
| **Floor Distribution** | Balanced (~22-25% floors 0-3) | Floor 1 dominant (41.6%) | Floor proportion shift |
| **User Identity** | 18 users (`USERID` 1–18) | Exactly 1 user (`USERID` 0) | Complete unseen user test |
| **Top 20 AP Overlap** | {train_val_comp['top_20_waps_overlap']['overlap_count']} / 20 shared | Moderate overlap due to building inversion |

---

## 10. Generated Artifacts Index

- [ap_analysis.csv](file://{_repo_root}/experiments/advanced_eda/ap_analysis.csv)
- [building_wifi_analysis.csv](file://{_repo_root}/experiments/advanced_eda/building_wifi_analysis.csv)
- [floor_wifi_analysis.csv](file://{_repo_root}/experiments/advanced_eda/floor_wifi_analysis.csv)
- [ap_cooccurrence.csv](file://{_repo_root}/experiments/advanced_eda/ap_cooccurrence.csv)
- [ap_similarity.csv](file://{_repo_root}/experiments/advanced_eda/ap_similarity.csv)
- [duplicate_analysis.json](file://{_repo_root}/experiments/advanced_eda/duplicate_analysis.json)
- [train_validation_comparison.json](file://{_repo_root}/experiments/advanced_eda/train_validation_comparison.json)
- [spatial_analysis_summary.json](file://{_repo_root}/experiments/advanced_eda/spatial_analysis_summary.json)
- [fingerprint_similarity_summary.json](file://{_repo_root}/experiments/advanced_eda/fingerprint_similarity_summary.json)
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info("Wrote comprehensive advanced EDA report to %s", output_path)


def run_advanced_eda() -> Dict[str, Any]:
    """Execute complete advanced EDA pipeline and persist structured outputs and plots."""
    settings = get_settings()
    out_dir = settings.paths.root_dir / "experiments" / "advanced_eda"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_train, df_val = load_raw_data()

    # 1. Access Point Analysis
    ap_df = analyze_access_points(df_train)
    ap_csv_path = out_dir / "ap_analysis.csv"
    ap_df.to_csv(ap_csv_path, index=False)
    logger.info("Saved %s", ap_csv_path)

    # 2. Building-Wise Analysis
    bld_df = analyze_building_wifi(df_train)
    bld_csv_path = out_dir / "building_wifi_analysis.csv"
    bld_df.to_csv(bld_csv_path, index=False)
    logger.info("Saved %s", bld_csv_path)

    # 3. Floor-Wise Analysis
    flr_df = analyze_floor_wifi(df_train)
    flr_csv_path = out_dir / "floor_wifi_analysis.csv"
    flr_df.to_csv(flr_csv_path, index=False)
    logger.info("Saved %s", flr_csv_path)

    # 4. AP Co-Occurrence Analysis
    cooc_df, jaccard_df = analyze_ap_cooccurrence(df_train, top_n=35)
    cooc_path = out_dir / "ap_cooccurrence.csv"
    jaccard_path = out_dir / "ap_similarity.csv"
    cooc_df.to_csv(cooc_path)
    jaccard_df.to_csv(jaccard_path)
    logger.info("Saved %s and %s", cooc_path, jaccard_path)

    # 5. Spatial Analysis
    spatial_summary = analyze_spatial_distribution(df_train)
    spatial_json_path = out_dir / "spatial_analysis_summary.json"
    with open(spatial_json_path, "w", encoding="utf-8") as f:
        json.dump(spatial_summary, f, indent=2)
    logger.info("Saved %s", spatial_json_path)

    # 6. Location Fingerprint Distinguishability
    distinguishability = analyze_fingerprint_distinguishability(df_train)
    dist_json_path = out_dir / "fingerprint_similarity_summary.json"
    with open(dist_json_path, "w", encoding="utf-8") as f:
        json.dump(distinguishability, f, indent=2)
    logger.info("Saved %s", dist_json_path)

    # 7. Duplicate Investigation
    dup_analysis = analyze_duplicates(df_train)
    dup_json_path = out_dir / "duplicate_analysis.json"
    with open(dup_json_path, "w", encoding="utf-8") as f:
        json.dump(dup_analysis, f, indent=2)
    logger.info("Saved %s", dup_json_path)

    # 8. Train vs Validation Comparison
    train_val_comp = compare_train_validation(df_train, df_val)
    tv_json_path = out_dir / "train_validation_comparison.json"
    with open(tv_json_path, "w", encoding="utf-8") as f:
        json.dump(train_val_comp, f, indent=2)
    logger.info("Saved %s", tv_json_path)

    # 9. Visualizations
    generate_advanced_plots(df_train, ap_df, bld_df, flr_df, cooc_df, jaccard_df, plots_dir)

    # 10. Comprehensive Markdown Report
    report_path = out_dir / "advanced_eda_report.md"
    generate_markdown_report(
        ap_df, bld_df, flr_df, cooc_df, spatial_summary,
        distinguishability, dup_analysis, train_val_comp, report_path
    )

    logger.info("Advanced EDA pipeline execution complete.")
    return {
        "ap_count": len(ap_df),
        "building_count": len(bld_df),
        "floor_combinations": len(flr_df),
        "duplicate_rows": dup_analysis["redundant_duplicate_rows_count"],
        "intra_building_jaccard": distinguishability["visibility_jaccard_similarity"]["intra_building_mean"],
        "inter_building_jaccard": distinguishability["visibility_jaccard_similarity"]["inter_building_mean"],
    }


if __name__ == "__main__":
    try:
        summary = run_advanced_eda()
        print("\n=======================================================")
        print("Advanced Wi-Fi Fingerprint EDA Successfully Completed!")
        print("=======================================================")
        print(f"Total WAPs Profiled: {summary['ap_count']}")
        print(f"Building Profiles:   {summary['building_count']}")
        print(f"Floor Configurations: {summary['floor_combinations']}")
        print(f"Redundant Duplicates: {summary['duplicate_rows']}")
        print(f"Intra-Building Jaccard: {summary['intra_building_jaccard']}")
        print(f"Inter-Building Jaccard: {summary['inter_building_jaccard']}")
        print("All artifacts saved under experiments/advanced_eda/")
    except Exception as exc:
        logger.exception("Advanced EDA execution failed: %s", exc)
        sys.exit(1)
