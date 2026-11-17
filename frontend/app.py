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
from typing import Any, Dict, List, Optional

# Ensure repository root is on PYTHONPATH and configure Matplotlib cache
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from backend.app.services.prediction_service import (
    PredictionService,
    get_prediction_service,
)
from frontend.api_client import DEFAULT_BACKEND_URL, DatasetApiClient
from frontend.components.comparison_view import render_dataset_comparison
from frontend.components.management_view import render_management
from frontend.components.overview import render_overview
from frontend.components.pdr_view import render_pdr_view
from frontend.components.photo_view import render_photos_and_annotations
from frontend.components.preprocessing_view import render_preprocessing_view
from frontend.components.quality_view import render_quality_dashboard
from frontend.components.sensor_view import render_sensor_explorer
from frontend.components.timeline_view import render_timeline_explorer
from frontend.components.upload import render_upload
from frontend.components.wifi_view import render_wifi_explorer

# Set page configuration
st.set_page_config(
    page_title="IndoorAI — Indoor Localization Dashboard",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #F0FDF4 100%);
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 20px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .badge-match {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-mismatch {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-experimental {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 4px;
        margin-bottom: 20px;
        color: #78350F;
        font-size: 0.95rem;
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
            "Random Forest (Task 07 Baseline)",
            f"k-Nearest Neighbors (Task 08 Baseline, k={k_val})",
            "Hybrid RF + k-NN (Task 10 Experimental)",
        ]
        selected_model = st.radio(
            "Prediction Model",
            model_options,
            index=0,
            help="Choose the model family to perform indoor localization.",
        )
        st.caption(f"Active k-NN Config: k={k_val}, metric=cosine, weights=distance")

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
        st.success("● Pre-Trained Models: Ready (3/3)")
        st.info("● Dataset: UJIIndoorLoc Benchmark")
        st.info("● Feature Pipeline: 1,058 Features")
        st.caption("IndoorAI v1.0.1 — Task 11.1 Audited")

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
            '<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">Trained ML Models</div></div>',
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

    with col_info:
        k_val = service.knn_model.params.n_neighbors if service.knn_model else 7
        if "Hybrid" in active_model_name:
            st.markdown(
                '<span class="badge-experimental">EXPERIMENTAL PROTOTYPE</span> <em>Two-stage architecture: RF Macro Structural Routing + Localized k-NN Metric Interpolation (k=7, tau=0.70).</em>',
                unsafe_allow_html=True,
            )
        elif "Forest" in active_model_name:
            st.caption("Random Forest: 5-head ensemble predicting Building, Floor, Building-Floor, Longitude, and Latitude (200 trees/head).")
        else:
            st.caption(f"k-Nearest Neighbors: Cosine distance metric with distance-weighted continuous regression (k={k_val}, Variant B features).")

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
        st.metric("Confidence (Pmax) *", f"{pred_res['confidence_pct']:.1f}%")
    with p_col6:
        st.metric("Spatial Uncertainty", f"± {pred_res['spatial_uncertainty']:.1f} units")

    # Sub-details if Hybrid
    if "Hybrid" in active_model_name:
        used_local = pred_res["details"].get("used_local_knn", False)
        mode_text = (
            "✅ Routed to Localized Building-Floor k-NN Regressor"
            if used_local
            else "⚠️ High uncertainty fallback: routed to Global k-NN Regressor"
        )
        st.caption(f"**Hierarchical Routing Decision:** {mode_text} | *Note: Gating threshold tau=0.70 is experimental.*")

    st.caption("*\t*Confidence indicates uncalibrated maximum ensemble vote / neighbor consensus proportion ($P_{\\max}$). It does not represent a calibrated Bayesian probability of correctness.*")
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

    # Color palette for buildings
    colors = {0: "#3B82F6", 1: "#F59E0B", 2: "#8B5CF6"}
    labels = {0: "Building 0 Ambient", 1: "Building 1 Ambient", 2: "Building 2 Ambient"}

    # Ambient points
    for b_id in sorted(plot_data["BUILDINGID"].unique()):
        subset = plot_data[plot_data["BUILDINGID"] == b_id]
        ax.scatter(
            subset["LONGITUDE"],
            subset["LATITUDE"],
            c=colors.get(b_id, "#9CA3AF"),
            label=labels.get(b_id, f"Building {b_id}"),
            alpha=0.25,
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
            c="#10B981",
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
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEE2E2", edgecolor="#F87171", alpha=0.9),
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

    ax.set_title(f"IndoorAI 2D Coordinate Map — {title_suffix}", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Dataset Longitude (Coordinate X)", fontsize=9)
    ax.set_ylabel("Dataset Latitude (Coordinate Y)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
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
        "Independently verified validation benchmarks on the 1,111 UJIIndoorLoc validation fingerprints across Tasks 07–10."
    )

    st.markdown("#### 1. Independently Audited Baselines (Task 09 Source of Truth)")
    st.caption("Audited and verified across all 1,111 validation records without information leakage.")
    if hasattr(service, "get_benchmark_table"):
        bench_df = service.get_benchmark_table()
    else:
        import importlib
        import backend.app.services.prediction_service as ps_mod
        importlib.reload(ps_mod)
        bench_df = ps_mod.PredictionService.get_benchmark_table(base_dir=REPO_ROOT)
    st.dataframe(bench_df, use_container_width=True)

    st.markdown("#### 2. Experimental Research Prototype (Task 10 Error Analysis)")
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

    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.markdown("#### 🌲 Random Forest Baseline")
        st.markdown(
            """
            - **Strengths:** Outstanding macro classification (99.82% building, 88.66% floor accuracy). Decision trees capture vertical slab attenuation effectively.
            - **Limitations:** Large memory footprint (561 MB serialization); piecewise orthogonal step functions in coordinate regression (11.53 units mean error).
            - **Best Used For:** Macro campus structure identification and global fallback.
            """
        )
    with t_col2:
        st.markdown("#### 📍 k-Nearest Neighbors Baseline")
        st.markdown(
            """
            - **Strengths:** Continuous metric coordinate regression (10.05 units mean error, 7.33 units median); ultra-compact storage (17.7 MB).
            - **Limitations:** Lower floor accuracy (84.43%) due to multi-floor RF bleed and vertical distance metric distortion.
            - **Best Used For:** Continuous metric coordinate regression within confirmed floor regions.
            """
        )
    with t_col3:
        st.markdown("#### ⚡ Hybrid Localizer (Experimental)")
        st.markdown(
            """
            - **Strengths:** Demonstrates potential of two-stage hierarchical localization: combines RF macro classification with localized micro k-NN regression (9.98 units mean error).
            - **Limitations:** Suffers from catastrophic error propagation if Stage 1 RF misclassifies floor and confidence gating fails. Requires rigorous training-split threshold calibration.
            - **Status:** Experimental research prototype; unverified for production baselining.
            """
        )

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
        ]
        selected_sec = st.radio("Navigation", sections, index=0)

        st.markdown("---")
        st.markdown("### 🛡️ System Status")
        st.info("● Ingestion Engine: Task 18 Compliant")
        st.info("● Preprocessing Engine: Task 20 Ready")
        st.info("● PDR Kinematics: Task 21 Ready")
        st.caption("IndoorAI Multimodal Explorer v1.4.0")

    # Main Header
    st.markdown('<div class="main-title">IndoorAI — Multimodal Survey Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Interactive exploration and quality assurance for mobile Wi-Fi, IMU, and visual survey datasets (Tasks 17–21)</div>',
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
                "🌐 Multimodal Survey Explorer (Tasks 17–19)",
                "🔬 UJIIndoorLoc Benchmark & Inference (Tasks 07–12)",
            ],
            index=0,
            help="Switch between mobile multimodal survey data analytics and the academic ML benchmark dashboard."
        )
        st.markdown("---")

    if workspace_mode == "🌐 Multimodal Survey Explorer (Tasks 17–19)":
        render_multimodal_workspace()
    else:
        render_benchmark_workspace()


if __name__ == "__main__":
    main()

