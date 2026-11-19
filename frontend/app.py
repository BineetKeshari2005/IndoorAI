"""IndoorAI — Interactive Indoor Localization & Wi-Fi Fingerprinting Dashboard (Task 11).

Demonstrates pre-trained Random Forest, k-Nearest Neighbors, and Hybrid models
on the UJIIndoorLoc benchmark dataset. Provides fingerprint exploration,
signal strength analysis, real-time location inference, 2D coordinate mapping,
error evaluation, and verified model benchmarking.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on PYTHONPATH and configure Matplotlib cache
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import streamlit as st

# Global white theme for Plotly charts across both workspaces
pio.templates.default = "plotly_white"

# Global white theme for Matplotlib figures
plt.rcParams["figure.facecolor"] = "#FFFFFF"
plt.rcParams["axes.facecolor"] = "#FFFFFF"
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.labelcolor"] = "#0F172A"
plt.rcParams["text.color"] = "#0F172A"
plt.rcParams["xtick.color"] = "#475569"
plt.rcParams["ytick.color"] = "#475569"
plt.rcParams["grid.color"] = "#E2E8F0"
plt.rcParams["grid.alpha"] = 0.8
plt.rcParams["figure.autolayout"] = True

from backend.app.services.prediction_service import (
    PredictionService,
    get_prediction_service,
)
from frontend.api_client import DEFAULT_BACKEND_URL, DatasetApiClient
from frontend.components.comparison_view import render_dataset_comparison
from frontend.components.management_view import render_management
from frontend.components.overview import render_overview
from frontend.components.pdr_calibration_view import render_pdr_calibration_view
from frontend.components.pdr_view import render_pdr_view
from frontend.components.photo_view import render_photos_and_annotations
from frontend.components.preprocessing_view import render_preprocessing_view
from frontend.components.quality_view import render_quality_dashboard
from frontend.components.sensor_fusion_view import render_sensor_fusion_view
from frontend.components.computer_vision_view import render_computer_vision_view
from frontend.components.map_localization_view import render_map_localization_view
from frontend.components.mapping_view import render_mapping_view
from frontend.components.navigation_view import render_navigation_view
from frontend.components.realtime_view import render_realtime_view
from frontend.components.sensor_view import render_sensor_explorer
from frontend.components.timeline_view import render_timeline_explorer
from frontend.components.upload import render_upload
from frontend.components.wifi_localization_view import render_wifi_localization_view
from frontend.components.wifi_view import render_wifi_explorer

# Set page configuration
st.set_page_config(
    page_title="IndoorAI — Indoor Localization Dashboard",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern white theme styling across both workspaces
st.markdown(
    """
    <style>
    /* Global clean white theme */
    .stApp {
        background-color: #FFFFFF !important;
        color: #0F172A;
    }
    
    /* Global sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] * {
        color: #0F172A;
    }
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stCaption {
        color: #64748B;
    }

    /* Main container clean spacing */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Typography */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* Metric and KPI Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        box-shadow: 0 3px 6px rgba(0,0,0,0.06);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
        font-weight: 600;
    }

    /* Prediction Result Card */
    .prediction-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 22px;
        margin-top: 12px;
        margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    /* Badges */
    .badge-match {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #BBF7D0;
    }
    .badge-mismatch {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #FECACA;
    }
    .badge-experimental {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #FDE68A;
    }
    .badge-mlp {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #BFDBFE;
    }

    /* Restrained Disclaimer / Notice Box */
    .disclaimer-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
        color: #334155;
        font-size: 0.93rem;
        line-height: 1.5;
    }

    /* Dataframe tables styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading IndoorAI pre-trained models and datasets...")
def load_service() -> PredictionService:
    """Load and cache the centralized prediction service once."""
    import importlib
    import backend.app.services.prediction_service as ps_module
    importlib.reload(ps_module)
    service = ps_module.get_prediction_service(base_dir=REPO_ROOT)
    service.load_all()
    return service


def render_sidebar(service: PredictionService) -> str:
    """Render the sidebar with model selector and navigation for benchmark workspace."""
    with st.sidebar:
        st.markdown("### 🤖 Select Active Model")
        k_val = service.knn_model.params.n_neighbors if service.knn_model else 7
        model_options = [
            "Random Forest — Task 07 Baseline",
            f"k-Nearest Neighbors — Task 08 Baseline (k={k_val})",
            "Multitask MLP — Task 12 Deep Learning",
            "Hybrid RF + k-NN — Task 10 Experimental",
        ]
        selected_model = st.radio(
            "Prediction Model",
            model_options,
            index=0,
            help="Choose the model family to perform indoor localization.",
        )

        name_lower = selected_model.lower()
        if "mlp" in name_lower:
            st.caption("Active Config: Deep MLP (Backbone 512->256->128, 4 Task Heads, PyTorch CPU)")
        elif "neighbor" in name_lower or "knn" in name_lower:
            st.caption(f"Active Config: k={k_val}, metric=cosine, weights=distance")
        elif "hybrid" in name_lower:
            st.caption("Active Config: RF Stage 1 Macro + Local k-NN Stage 2 (tau=0.70)")
        else:
            st.caption("Active Config: 5-Head Ensemble (200 trees/head)")

        st.markdown("---")
        st.markdown("### 📑 Section Navigation")
        st.markdown(
            """
            - [A. System Overview](#a-system-dataset-overview)
            - [B. Fingerprint Explorer](#b-wi-fi-fingerprint-explorer)
            - [C. Location Prediction](#c-indoor-location-prediction)
            - [D. 2D Coordinate Map](#d-2d-indoor-coordinate-map)
            - [E. Prediction Evaluation](#e-prediction-accuracy-evaluation)
            - [F. Model Benchmarks](#f-model-benchmarking-selection)
            """
        )

        st.markdown("---")
        st.markdown("### 🛡️ System Status")
        st.success("● Pre-Trained Models: Ready (4/4)")
        st.info("● Dataset: UJIIndoorLoc Benchmark (19,937 Train / 1,111 Val)")
        st.info("● Feature Pipeline: 1,058 Features")
        st.caption("IndoorAI v1.2.0 — Task 12 Deep Learning Integrated")

    return selected_model


def render_section_a(service: PredictionService) -> None:
    """Section A: System & Dataset Overview."""
    st.markdown('<div class="main-title">IndoorAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Interactive Indoor Localization & Wi-Fi Fingerprinting Demonstration Engine</div>',
        unsafe_allow_html=True,
    )

    # Disclosure notice
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Benchmark Evaluation Notice:</strong> This application demonstrates indoor localization on pre-recorded Wi-Fi radio signal fingerprints from the <strong>UJIIndoorLoc</strong> academic benchmark dataset. It simulates positioning across verified validation fingerprints and custom CSV inputs. It <strong>does not</strong> perform live mobile tracking, GPS trilateration, or physical surveillance. All spatial coordinates are expressed strictly in <em>native dataset coordinate units</em>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### A. System & Dataset Overview")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

    with kpi_col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">19,937</div><div class="metric-label">Training Fingerprints</div></div>',
            unsafe_allow_html=True,
        )
    with kpi_col2:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">1,111</div><div class="metric-label">Validation Fingerprints</div></div>',
            unsafe_allow_html=True,
        )
    with kpi_col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">Campus Buildings (0–2)</div></div>',
            unsafe_allow_html=True,
        )
    with kpi_col4:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">5</div><div class="metric-label">Floor Levels (0–4)</div></div>',
            unsafe_allow_html=True,
        )
    with kpi_col5:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">520</div><div class="metric-label">Monitored WAPs</div></div>',
            unsafe_allow_html=True,
        )
    with kpi_col6:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">Trained ML Models</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)


def render_section_b(
    service: PredictionService,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]], List[Dict[str, Any]], int, str]:
    """Section B: Wi-Fi Fingerprint Explorer.

    Returns:
        Tuple of (selected_features_df, ground_truth_dict_or_None, detected_waps, visible_count, source_label)
    """
    st.markdown("### B. Wi-Fi Fingerprint Explorer")
    st.markdown(
        "Select a pre-recorded validation fingerprint from the UJIIndoorLoc dataset or upload a custom CSV fingerprint to inspect detected Wi-Fi signals."
    )

    source_tab = st.radio(
        "Select Input Source",
        ["Browse Validation Dataset (1,111 Samples)", "Upload Custom Wi-Fi CSV Fingerprint"],
        horizontal=True,
    )

    features_df: pd.DataFrame
    ground_truth: Optional[Dict[str, Any]] = None
    detected_waps: List[Dict[str, Any]] = []
    visible_count: int = 0
    sample_label: str = ""

    if source_tab == "Browse Validation Dataset (1,111 Samples)":
        b_col1, b_col2 = st.columns([1, 2])

        with b_col1:
            sample_idx = st.number_input(
                "Validation Record Index (0 – 1,110)",
                min_value=0,
                max_value=len(service.validation_features) - 1,
                value=0,
                step=1,
            )

            st.markdown("**Quick Preset Fingerprints:**")
            preset_cols = st.columns(3)
            with preset_cols[0]:
                if st.button("Sample #0 (B1 F1)"):
                    sample_idx = 0
            with preset_cols[1]:
                if st.button("Sample #150 (B0 F3)"):
                    sample_idx = 150
            with preset_cols[2]:
                if st.button("Sample #450 (B2 F3)"):
                    sample_idx = 450

        sample_data = service.get_validation_sample(sample_idx)
        features_df = sample_data["features"]
        ground_truth = sample_data["ground_truth"]
        detected_waps = sample_data["detected_waps"]
        visible_count = sample_data["visible_count"]
        sample_label = f"Validation Sample #{sample_idx}"

        with b_col2:
            st.markdown(f"#### Fingerprint Metadata: `{sample_label}`")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Ground Truth Building", f"Building {int(ground_truth['BUILDINGID'])}")
            with m_col2:
                st.metric("Ground Truth Floor", f"Floor {int(ground_truth['FLOOR'])}")
            with m_col3:
                st.metric("Detected WAPs", f"{visible_count} / 520")
            with m_col4:
                max_sig = detected_waps[0]["rssi_dbm"] if detected_waps else -105.0
                st.metric("Strongest Signal", f"{max_sig:.1f} dBm")

    else:
        st.info(
            "Upload a CSV containing Wi-Fi Access Point columns (`WAP001` through `WAP520`) with RSSI values in dBm "
            "(use `100` or `-105` for undetected APs). Optional columns: `BUILDINGID`, `FLOOR`, `LONGITUDE`, `LATITUDE`."
        )
        uploaded_file = st.file_uploader("Upload Wi-Fi Fingerprint CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.read()
                csv_buffer = io.BytesIO(raw_bytes)
                (
                    parsed_feats,
                    parsed_gt,
                    parsed_waps_list,
                ) = service.preprocess_raw_csv(csv_buffer)

                st.success(f"Successfully validated and preprocessed {len(parsed_feats)} fingerprint records from CSV!")

                row_idx = 0
                if len(parsed_feats) > 1:
                    row_idx = st.slider(
                        "Select Record from Uploaded CSV",
                        min_value=0,
                        max_value=len(parsed_feats) - 1,
                        value=0,
                    )

                features_df = parsed_feats.iloc[[row_idx]]
                detected_waps = parsed_waps_list[row_idx]
                visible_count = len(detected_waps)
                sample_label = f"Uploaded CSV Row #{row_idx}"

                if parsed_gt is not None:
                    ground_truth = parsed_gt.iloc[row_idx].to_dict()
                    st.caption(
                        f"Ground Truth extracted: Building {int(ground_truth['BUILDINGID'])}, Floor {int(ground_truth['FLOOR'])}"
                    )
                else:
                    ground_truth = None
                    st.caption("No ground truth target columns provided in uploaded CSV (evaluation disabled).")

            except Exception as e:
                st.error(f"CSV Validation Error: {str(e)}")
                # Fallback to sample 0
                fallback = service.get_validation_sample(0)
                return fallback["features"], fallback["ground_truth"], fallback["detected_waps"], fallback["visible_count"], "Validation Sample #0 (Fallback)"
        else:
            st.warning("No CSV uploaded yet. Displaying sample validation fingerprint #0 below.")
            sample_data = service.get_validation_sample(0)
            return sample_data["features"], sample_data["ground_truth"], sample_data["detected_waps"], sample_data["visible_count"], "Validation Sample #0"

    # WAPs Table and Signal Chart
    st.markdown("#### Detected Access Points & Signal Strengths")
    if visible_count == 0:
        st.warning("⚠️ No Wi-Fi Access Points detected in this fingerprint (all 520 WAPs are unobserved/100 dBm).")
    else:
        sig_col1, sig_col2 = st.columns([1, 1])

        with sig_col1:
            st.markdown(f"**Detected Access Points ({visible_count} observed):**")
            table_data = []
            for w in detected_waps:
                rssi = w["rssi_dbm"]
                quality = "🟢 Strong (≥ -60 dBm)" if rssi >= -60 else ("🟡 Medium (-80 to -60 dBm)" if rssi >= -80 else "🔴 Weak (< -80 dBm)")
                table_data.append({
                    "Access Point": w["wap_id"],
                    "Signal (dBm)": f"{rssi:.1f}",
                    "Quality": quality,
                })
            df_display = pd.DataFrame(table_data)
            st.dataframe(df_display, height=280, use_container_width=True)

        with sig_col2:
            st.markdown("**Top Observed Signal Strengths (dBm):**")
            top_waps = detected_waps[:15]
            if top_waps:
                names = [w["wap_id"] for w in top_waps][::-1]
                vals = [w["rssi_dbm"] for w in top_waps][::-1]
                colors = [
                    "#10B981" if v >= -60 else ("#F59E0B" if v >= -80 else "#EF4444")
                    for v in vals
                ]

                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.barh(names, vals, color=colors, height=0.6)
                ax.set_xlim(-105, -30)
                ax.set_xlabel("RSSI (dBm)", fontsize=9)
                ax.set_ylabel("Access Point", fontsize=9)
                ax.grid(axis="x", linestyle="--", alpha=0.5)
                ax.tick_params(labelsize=8)

                for bar in bars:
                    width = bar.get_width()
                    ax.text(
                        width + 1.5,
                        bar.get_y() + bar.get_height() / 2,
                        f"{width:.0f}",
                        va="center",
                        ha="left",
                        fontsize=7,
                        color="#374151",
                    )
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

    st.markdown("---")
    return features_df, ground_truth, detected_waps, visible_count, sample_label


def render_section_c(
    service: PredictionService,
    features_df: pd.DataFrame,
    active_model_name: str,
) -> Dict[str, Any]:
    """Section C: Indoor Location Prediction.

    Returns:
        Prediction result dictionary.
    """
    st.markdown("### C. Indoor Location Prediction")
    st.markdown(
        f"Generate continuous indoor coordinates, building identifier, and floor level for the selected fingerprint using **{active_model_name}**."
    )

    pred_res: Optional[Dict[str, Any]] = None

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_pred = st.button("📍 Run Localization Prediction", type="primary", use_container_width=True)

    name_lower = active_model_name.lower()
    with col_info:
        k_val = service.knn_model.params.n_neighbors if service.knn_model else 7
        if "mlp" in name_lower or "deep" in name_lower:
            st.markdown(
                '<span class="badge-mlp">DEEP LEARNING</span> <em>Multitask MLP with Shared Trunk (512 &rarr; 256 &rarr; 128) + 4 Task Heads (Building, Floor, Building-Floor, Longitude/Latitude). Checkpoint: 2.75 MB, 709,258 parameters.</em>',
                unsafe_allow_html=True,
            )
        elif "hybrid" in name_lower:
            st.markdown(
                '<span class="badge-experimental">EXPERIMENTAL PROTOTYPE</span> <em>Two-stage architecture: RF Macro Structural Routing + Localized k-NN Metric Interpolation (k=7, tau=0.70).</em>',
                unsafe_allow_html=True,
            )
        elif "forest" in name_lower or "rf" in name_lower:
            st.caption("Random Forest: 5-head ensemble predicting Building, Floor, Building-Floor, Longitude, and Latitude (200 trees/head). Size: ~561 MB.")
        else:
            st.caption(f"k-Nearest Neighbors: Cosine distance metric with distance-weighted continuous regression (k={k_val}, Variant B features). Size: ~17.7 MB.")

    # Run prediction automatically or on button click
    pred_res = service.predict(features_df, model_name=active_model_name)

    # Display prediction cards
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)

    with p_col1:
        st.metric("Predicted Building", f"Building {pred_res['pred_building']}")
    with p_col2:
        st.metric("Predicted Floor", f"Floor {pred_res['pred_floor']}")
    with p_col3:
        st.metric("Predicted Longitude", f"{pred_res['pred_longitude']:.2f}")
    with p_col4:
        st.metric("Predicted Latitude", f"{pred_res['pred_latitude']:.2f}")
    with p_col5:
        conf_label = "Confidence (Joint Softmax) *" if ("mlp" in name_lower or "deep" in name_lower) else "Confidence (Pmax) *"
        st.metric(conf_label, f"{pred_res['confidence_pct']:.1f}%")
    with p_col6:
        unc_val = pred_res.get("spatial_uncertainty")
        unc_text = f"± {unc_val:.1f} units" if unc_val is not None else "N/A (Point Regressor)"
        st.metric("Spatial Uncertainty", unc_text)

    # Sub-details
    if "mlp" in name_lower or "deep" in name_lower:
        b_conf = pred_res["details"].get("building_confidence", 0.0) * 100
        f_conf = pred_res["details"].get("floor_confidence", 0.0) * 100
        st.caption(
            f"**Deep Learning Output Breakdown:** Building Softmax: `{b_conf:.1f}%` | Floor Softmax: `{f_conf:.1f}%` | "
            f"Joint Uncalibrated Softmax Score: `{pred_res['confidence_pct']:.1f}%` | Model Size: `2.75 MB` (PyTorch CPU)"
        )
        st.caption(
            "*\\*Notice: Confidence reflects the uncalibrated joint softmax score ($P_{\\max,\\text{bldg}} \\times P_{\\max,\\text{floor}}$). "
            "Deep neural networks produce uncalibrated probabilities; this is not a calibrated Bayesian posterior. "
            "Continuous coordinates are point estimates without spatial dispersion.*"
        )
    elif "hybrid" in name_lower:
        used_local = pred_res["details"].get("used_local_knn", False)
        mode_text = (
            "✅ Routed to Localized Building-Floor k-NN Regressor"
            if used_local
            else "⚠️ High uncertainty fallback: routed to Global k-NN Regressor"
        )
        st.caption(f"**Hierarchical Routing Decision:** {mode_text} | *Note: Gating threshold tau=0.70 is experimental.*")
        st.caption("*\\*Confidence indicates uncalibrated maximum ensemble vote / neighbor consensus proportion ($P_{\\max}$). It does not represent a calibrated Bayesian probability of correctness.*")
    else:
        st.caption("*\\*Confidence indicates uncalibrated maximum ensemble vote / neighbor consensus proportion ($P_{\\max}$). It does not represent a calibrated Bayesian probability of correctness.*")

    st.markdown("</div>", unsafe_allow_html=True)

    # Uncertainty and outlier callout
    st.info(
        "ℹ️ **Notice on Spatial Uncertainty & Outliers:** A high confidence score ($P_{\\max} > 80\\%$) indicates local training consensus, "
        "but does **not** guarantee low Euclidean coordinate error. Physical indoor multipath, antenna variation, and vertical concrete slab attenuation "
        "can produce elevated spatial errors ($> 40\\text{ units}$) on challenging fingerprints (e.g. Sample #948 or multi-floor leakage)."
    )
    st.markdown("---")
    return pred_res


def render_section_d(
    service: PredictionService,
    prediction: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]],
) -> None:
    """Section D: 2D Indoor Coordinate Map."""
    st.markdown("### D. 2D Indoor Coordinate Map")
    st.markdown(
        "Interactive 2D spatial visualization comparing predicted coordinates against ambient validation fingerprints and ground truth."
    )

    map_filter = st.radio(
        "Map Spatial Filter",
        ["Show All Buildings (Campus Overview)", "Filter to Predicted Building"],
        horizontal=True,
    )

    # Prepare ambient data
    val_targets = service.validation_targets.copy()
    pred_lon = prediction["pred_longitude"]
    pred_lat = prediction["pred_latitude"]
    pred_b = prediction["pred_building"]

    if map_filter == "Filter to Predicted Building":
        plot_data = val_targets[val_targets["BUILDINGID"] == pred_b]
        title_suffix = f"Building {pred_b}"
    else:
        plot_data = val_targets
        title_suffix = "All Campus Buildings (0, 1, 2)"

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Clean color palette for buildings
    colors = {0: "#2563EB", 1: "#D97706", 2: "#7C3AED"}
    labels = {0: "Building 0 Ambient", 1: "Building 1 Ambient", 2: "Building 2 Ambient"}

    # Ambient points
    for b_id in sorted(plot_data["BUILDINGID"].unique()):
        subset = plot_data[plot_data["BUILDINGID"] == b_id]
        ax.scatter(
            subset["LONGITUDE"],
            subset["LATITUDE"],
            c=colors.get(b_id, "#94A3B8"),
            label=labels.get(b_id, f"Building {b_id}"),
            alpha=0.22,
            s=22,
            edgecolors="none",
        )

    # Plot Ground Truth if available
    if ground_truth is not None:
        gt_lon = float(ground_truth["LONGITUDE"])
        gt_lat = float(ground_truth["LATITUDE"])
        ax.scatter(
            gt_lon,
            gt_lat,
            c="#059669",
            s=160,
            marker="s",
            edgecolors="#064E3B",
            linewidth=2,
            label=f"Ground Truth (B{int(ground_truth['BUILDINGID'])} F{int(ground_truth['FLOOR'])})",
            zorder=6,
        )

        # Draw error vector
        ax.plot(
            [gt_lon, pred_lon],
            [gt_lat, pred_lat],
            color="#DC2626",
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            zorder=7,
            label="Localization Error Vector",
        )

        # Distance annotation
        mid_x = (gt_lon + pred_lon) / 2
        mid_y = (gt_lat + pred_lat) / 2
        err_dist = np.sqrt((pred_lon - gt_lon) ** 2 + (pred_lat - gt_lat) ** 2)
        ax.annotate(
            f"Error: {err_dist:.2f} units",
            xy=(mid_x, mid_y),
            xytext=(mid_x + 5, mid_y + 5),
            fontsize=8.5,
            fontweight="bold",
            color="#991B1B",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#FECACA", alpha=0.95),
            zorder=8,
        )

    # Plot Predicted Location
    ax.scatter(
        pred_lon,
        pred_lat,
        c="#DC2626",
        s=220,
        marker="*",
        edgecolors="#7F1D1D",
        linewidth=2,
        label=f"Predicted ({prediction['model_name']})",
        zorder=9,
    )

    ax.set_title(f"IndoorAI 2D Coordinate Map — {title_suffix}", fontsize=11, fontweight="bold", pad=10, color="#0F172A")
    ax.set_xlabel("Dataset Longitude (Coordinate X)", fontsize=9, color="#0F172A")
    ax.set_ylabel("Dataset Latitude (Coordinate Y)", fontsize=9, color="#0F172A")
    ax.grid(True, linestyle=":", color="#E2E8F0", alpha=0.9)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95, facecolor="#FFFFFF", edgecolor="#E2E8F0")
    plt.tight_layout()

    st.pyplot(fig, clear_figure=True)
    st.caption(
        "🗺️ **Coordinate Invariant Disclosure:** Coordinates are plotted directly in native UJIIndoorLoc coordinate space. "
        "Axes denote relative dataset coordinate units (not metric metres). No architectural walls, corridors, or room layouts are fabricated."
    )
    st.markdown("---")


def render_section_e(
    service: PredictionService,
    prediction: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]],
) -> None:
    """Section E: Prediction Accuracy Evaluation."""
    st.markdown("### E. Prediction Accuracy Evaluation")

    if ground_truth is None:
        st.info("ℹ️ Evaluation metrics require ground truth coordinates and labels. This uploaded record contains no target labels.")
        st.markdown("---")
        return

    metrics = service.calculate_evaluation_metrics(prediction, ground_truth)

    e_col1, e_col2, e_col3, e_col4 = st.columns(4)

    with e_col1:
        if metrics["building_correct"]:
            st.markdown(
                '<div class="metric-card"><div class="badge-match">MATCH</div><div class="metric-label">Building Classification</div><p style="margin-top:6px;font-size:0.9rem;">Building '
                f"{prediction['pred_building']} == {int(ground_truth['BUILDINGID'])}</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="metric-card"><div class="badge-mismatch">MISMATCH</div><div class="metric-label">Building Classification</div><p style="margin-top:6px;font-size:0.9rem;">Pred B'
                f"{prediction['pred_building']} != True B{int(ground_truth['BUILDINGID'])}</p></div>",
                unsafe_allow_html=True,
            )

    with e_col2:
        if metrics["floor_correct"]:
            st.markdown(
                '<div class="metric-card"><div class="badge-match">MATCH</div><div class="metric-label">Floor Classification</div><p style="margin-top:6px;font-size:0.9rem;">Floor '
                f"{prediction['pred_floor']} == {int(ground_truth['FLOOR'])}</p></div>",
                unsafe_allow_html=True,
            )
        else:
            diff = metrics["floor_difference"]
            st.markdown(
                '<div class="metric-card"><div class="badge-mismatch">MISMATCH (Δ '
                f'{diff} Flr)</div><div class="metric-label">Floor Classification</div><p style="margin-top:6px;font-size:0.9rem;">Pred F{prediction["pred_floor"]} != True F{int(ground_truth["FLOOR"])}</p></div>',
                unsafe_allow_html=True,
            )

    with e_col3:
        if metrics["building_floor_correct"]:
            st.markdown(
                '<div class="metric-card"><div class="badge-match">EXACT MATCH</div><div class="metric-label">Building-Floor Joint</div><p style="margin-top:6px;font-size:0.9rem;">Macro Location Correct</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="metric-card"><div class="badge-mismatch">ERROR</div><div class="metric-label">Building-Floor Joint</div><p style="margin-top:6px;font-size:0.9rem;">Macro Discrepancy</p></div>',
                unsafe_allow_html=True,
            )

    with e_col4:
        err_val = metrics["euclidean_error_2d"]
        err_status = "🟢 Excellent" if err_val < 5.0 else ("🟡 Good" if err_val < 10.0 else ("🟠 Moderate" if err_val < 20.0 else "🔴 Elevated"))
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{err_val:.2f}</div><div class="metric-label">2D Euclidean Error (Units)</div><p style="margin-top:4px;font-size:0.85rem;">{err_status}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"**Coordinate Decomposition:** Longitude Error: `{metrics['longitude_error']:.3f} units` | "
        f"Latitude Error: `{metrics['latitude_error']:.3f} units` | "
        f"Joint 2D Euclidean Distance: `{metrics['euclidean_error_2d']:.3f} units`"
    )
    st.markdown("---")


def render_section_f(service: PredictionService) -> None:
    """Section F: Model Benchmarking & Selection."""
    st.markdown("### F. Model Benchmarking & Selection")
    st.markdown(
        "Independently verified validation benchmarks on the 1,111 UJIIndoorLoc validation fingerprints across Tasks 07–12."
    )

    st.markdown("#### 1. Independently Audited Three-Way Baselines (Tasks 07, 08, & 12)")
    st.caption("Audited and verified across all 1,111 validation records without information leakage. All coordinate errors in native dataset coordinate units (not metric metres).")
    if hasattr(service, "get_benchmark_table"):
        bench_df = service.get_benchmark_table()
    else:
        import importlib
        import backend.app.services.prediction_service as ps_mod
        importlib.reload(ps_mod)
        bench_df = ps_mod.PredictionService.get_benchmark_table(base_dir=REPO_ROOT)
    st.dataframe(bench_df, use_container_width=True)

    st.markdown("#### 2. Comparative Strengths & Operational Trade-offs")
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    with t_col1:
        st.markdown("##### 🌲 Random Forest Baseline")
        st.markdown(
            """
            - **Task:** 07 Baseline
            - **Strengths:** Robust macro classification (99.82% building, 88.66% floor). Orthogonal decision boundaries capture vertical slab attenuation effectively.
            - **Limitations:** Massive serialized footprint (~561 MB); piecewise step functions in 2D coordinate regression (11.53 units mean error).
            - **Best Used For:** Campus macro structure identification and global fallback.
            """
        )
    with t_col2:
        st.markdown("##### 📍 k-Nearest Neighbors")
        st.markdown(
            """
            - **Task:** 08 Baseline (k=7)
            - **Strengths:** Lowest 2D coordinate error (10.05 units mean, 7.33 units median) due to continuous metric interpolation; compact size (~17.7 MB).
            - **Limitations:** Lower floor accuracy (84.43%) due to multi-floor RF bleed distorting cosine distance metrics across vertical slabs.
            - **Best Used For:** Continuous metric coordinate regression within confirmed floor regions.
            """
        )
    with t_col3:
        st.markdown("##### 🧠 Multitask MLP")
        st.markdown(
            """
            - **Task:** 12 Deep Learning
            - **Strengths:** Highest floor accuracy (90.37%) and joint building-floor accuracy (90.37%); ultra-compact footprint (~2.75 MB, 709K params); fast single-pass PyTorch inference.
            - **Limitations:** Point regression yields higher 2D coordinate error (12.18 units mean, 10.06 units median) than k-NN; no built-in spatial uncertainty bounds.
            - **Best Used For:** Resource-constrained mobile deployments prioritizing vertical floor discrimination.
            """
        )
    with t_col4:
        st.markdown("##### ⚡ Hybrid Localizer")
        st.markdown(
            """
            - **Task:** 10 Experimental
            - **Strengths:** Two-stage hierarchical routing (RF macro + local k-NN micro) achieves 9.98 units mean coordinate error on validation split.
            - **Limitations:** Prone to catastrophic error propagation if Stage 1 RF misclassifies floor and confidence gating fails. Gating threshold (tau=0.70) was evaluated post-hoc.
            - **Status:** Exploratory research prototype; unverified for production baselining.
            """
        )

    st.markdown("#### 3. Experimental Research Prototype (Task 10 Error Analysis)")
    if hasattr(service, "get_experimental_hybrid_table"):
        hybrid_df = service.get_experimental_hybrid_table()
    elif hasattr(PredictionService, "get_experimental_hybrid_table"):
        hybrid_df = PredictionService.get_experimental_hybrid_table(base_dir=REPO_ROOT)
    else:
        import importlib
        import backend.app.services.prediction_service as ps_mod
        importlib.reload(ps_mod)
        hybrid_df = ps_mod.PredictionService.get_experimental_hybrid_table(base_dir=REPO_ROOT)
    st.dataframe(hybrid_df, use_container_width=True)

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Experimental Integrity Notice:</strong> The Two-Stage Hybrid Localizer is an exploratory research prototype investigated in Task 10. Its confidence gating threshold (\\(\\tau = 0.70\\)) was evaluated post-hoc directly on the validation dataset rather than determined via training-only cross-validation. While it achieves a lower mean coordinate error (9.98 units) on the validation split, it is <strong>not</strong> an established baseline and is kept strictly separate from audited baselines.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Benchmark verified on official UJIIndoorLoc 1,111 validation records under zero data leakage conditions.")

    st.caption("Benchmark verified on official UJIIndoorLoc 1,111 validation records under zero data leakage conditions.")


def render_multimodal_workspace() -> None:
    """Renders the Multimodal Survey Dataset Explorer (Tasks 17-19)."""
    with st.sidebar:
        st.markdown("### 🔌 Backend Connection")
        backend_url = st.text_input(
            "FastAPI Base URL:",
            value=os.getenv("INDOORAI_BACKEND_URL", DEFAULT_BACKEND_URL),
            help="URL of the running IndoorAI FastAPI backend service."
        )
        client = DatasetApiClient(base_url=backend_url)
        is_online, health_msg = client.check_health()
        if is_online:
            st.success(f"● {health_msg}")
        else:
            st.warning(f"○ {health_msg}")
            st.caption("Start backend: `python -m uvicorn backend.app.main:app --port 8000`")

        st.markdown("---")
        st.markdown("### 📑 Explorer Sections")
        sections = [
            "1. Overview & Catalog",
            "2. Upload Dataset",
            "3. Quality Dashboard",
            "4. Wi-Fi Explorer",
            "5. Sensor Explorer",
            "6. Multimodal Timeline",
            "7. Photos & Annotations",
            "8. Dataset Comparison",
            "9. Dataset Management",
            "10. Preprocessing & Alignment",
            "11. PDR & Movement Analysis",
            "12. PDR Calibration & Validation",
            "13. Site-Specific Wi-Fi Localization",
            "14. Wi-Fi + PDR Sensor Fusion",
            "15. Computer Vision & Landmarks",
            "16. Indoor Mapping & Spatial Graphs",
            "17. Map-Constrained Localization",
            "18. Indoor Navigation & Route Planning",
            "19. Real-Time IndoorAI",
        ]
        selected_sec = st.radio("Navigation", sections, index=0)

        st.markdown("---")
        st.markdown("### 🛡️ System Status")
        st.info("● Ingestion Engine: Task 18 Compliant")
        st.info("● Preprocessing Engine: Task 20 Ready")
        st.info("● PDR Kinematics: Task 21 Ready")
        st.info("● PDR Calibration: Task 22 Ready")
        st.info("● Site Wi-Fi Radio Map: Task 23 Ready")
        st.info("● EKF Sensor Fusion: Task 24 Ready")
        st.info("● Computer Vision & Landmarks: Task 25 Ready")
        st.info("● Indoor Spatial Maps: Task 26 Ready")
        st.info("● Map-Constrained PF: Task 27 Ready")
        st.info("● Indoor Navigation: Task 28 Ready")
        st.info("● Real-Time Streaming: Task 29 Ready")
        st.caption("IndoorAI Multimodal Explorer v1.11.0")

    # Main Header
    st.markdown('<div class="main-title">IndoorAI — Multimodal Survey Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Interactive exploration and quality assurance for mobile Wi-Fi, IMU, and visual survey datasets (Tasks 17–29)</div>',
        unsafe_allow_html=True
    )

    selected_dataset_id = st.session_state.get("selected_dataset_id")

    if selected_sec == "1. Overview & Catalog":
        render_overview(client)
    elif selected_sec == "2. Upload Dataset":
        render_upload(client)
    elif selected_sec == "3. Quality Dashboard":
        render_quality_dashboard(client, selected_dataset_id)
    elif selected_sec == "4. Wi-Fi Explorer":
        render_wifi_explorer(client, selected_dataset_id)
    elif selected_sec == "5. Sensor Explorer":
        render_sensor_explorer(client, selected_dataset_id)
    elif selected_sec == "6. Multimodal Timeline":
        render_timeline_explorer(client, selected_dataset_id)
    elif selected_sec == "7. Photos & Annotations":
        render_photos_and_annotations(client, selected_dataset_id)
    elif selected_sec == "8. Dataset Comparison":
        render_dataset_comparison(client)
    elif selected_sec == "9. Dataset Management":
        render_management(client, selected_dataset_id)
    elif selected_sec == "10. Preprocessing & Alignment":
        render_preprocessing_view(client, selected_dataset_id)
    elif selected_sec == "11. PDR & Movement Analysis":
        render_pdr_view(client, selected_dataset_id)
    elif selected_sec == "12. PDR Calibration & Validation":
        render_pdr_calibration_view(client, selected_dataset_id)
    elif selected_sec == "13. Site-Specific Wi-Fi Localization":
        render_wifi_localization_view(client, selected_dataset_id)
    elif selected_sec == "14. Wi-Fi + PDR Sensor Fusion":
        render_sensor_fusion_view(client, selected_dataset_id)
    elif selected_sec == "15. Computer Vision & Landmarks":
        render_computer_vision_view(client)
    elif selected_sec == "16. Indoor Mapping & Spatial Graphs":
        render_mapping_view(client)
    elif selected_sec == "17. Map-Constrained Localization":
        render_map_localization_view(client, selected_dataset_id)
    elif selected_sec == "18. Indoor Navigation & Route Planning":
        render_navigation_view(client, selected_dataset_id)
    elif selected_sec == "19. Real-Time IndoorAI":
        render_realtime_view(client, selected_dataset_id)


def render_benchmark_workspace() -> None:
    """Renders the original UJIIndoorLoc ML Benchmark & Localization Workspace (Tasks 11-12)."""
    service = load_service()
    active_model = render_sidebar(service)

    render_section_a(service)
    features_df, ground_truth, detected_waps, visible_count, sample_label = render_section_b(service)
    prediction = render_section_c(service, features_df, active_model)
    render_section_d(service, prediction, ground_truth)
    render_section_e(service, prediction, ground_truth)
    render_section_f(service)


def main():
    with st.sidebar:
        st.markdown("## 📍 IndoorAI")
        st.caption("Multimodal Positioning & Survey Analytics")
        st.markdown("---")

        st.markdown("### 🗂️ Select Workspace")
        workspace_mode = st.radio(
            "Workspace View:",
            [
                "🌐 Multimodal Survey Explorer (Tasks 17–29)",
                "🔬 UJIIndoorLoc Benchmark & Inference (Tasks 07–12)",
            ],
            index=0,
            help="Switch between mobile multimodal survey data analytics and the academic ML benchmark dashboard."
        )
        st.markdown("---")

    if workspace_mode == "🌐 Multimodal Survey Explorer (Tasks 17–29)":
        render_multimodal_workspace()
    else:
        render_benchmark_workspace()


if __name__ == "__main__":
    main()

