"""Streamlit component for Wi-Fi + PDR Extended Kalman Filter Sensor Fusion (Task 24)."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import DatasetApiClient, DatasetApiClientException


def render_sensor_fusion_view(client: DatasetApiClient, dataset_id: Optional[str]) -> None:
    """Renders the comprehensive Wi-Fi + PDR Sensor Fusion dashboard."""
    st.markdown("## 🧬 Wi-Fi + PDR Sensor Fusion (Extended Kalman Filter)")
    st.caption(
        "Combines continuous relative movement from Pedestrian Dead Reckoning (PDR) with absolute site-specific "
        "Wi-Fi fingerprint corrections using an Extended Kalman Filter, Joseph form covariance updates, and "
        "statistical Chi-Square outlier gating (Task 24)."
    )

    if not dataset_id:
        st.info("💡 Please select an active survey dataset in **1. Overview & Catalog** first.")
        return

    # Check fusion status and existing artifacts
    try:
        fusion_status = client.get_fusion_status(dataset_id)
        is_available = fusion_status.get("is_available", False)
        summary = fusion_status.get("summary") or {}
        fused_trajectory = client.get_fused_trajectory(dataset_id) if is_available else []
    except DatasetApiClientException as e:
        st.error(f"Error communicating with sensor fusion backend: {e}")
        return

    # Top KPI Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Fused States", len(fused_trajectory) if is_available else 0)
    with m_col2:
        pdr_steps = summary.get("pdr_steps_count", 0)
        st.metric("PDR Steps Integrated", pdr_steps)
    with m_col3:
        w_acc = summary.get("wifi_scans_accepted", 0)
        w_tot = summary.get("wifi_scans_total", 0)
        acc_pct = (w_acc / w_tot * 100.0) if w_tot > 0 else 0.0
        st.metric("Wi-Fi Corrections", f"{w_acc} / {w_tot} ({acc_pct:.0f}%)")
    with m_col4:
        mean_rad = summary.get("mean_uncertainty_radius_m", 0.0)
        st.metric("Mean 95% Uncertainty", f"±{mean_rad:.2f} m" if is_available else "N/A")

    # Main Tabs
    tab_map, tab_run, tab_diag, tab_bench = st.tabs([
        "🗺️ 2D Fused Trajectory & Uncertainty Map",
        "⚡ Fusion Controls & Execution",
        "📈 Innovation & Outlier Gating Analysis",
        "📊 PDR vs Wi-Fi vs Fused Benchmark",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: 2D Fused Trajectory & Uncertainty Map
    # -------------------------------------------------------------------------
    with tab_map:
        st.markdown("### 🗺️ 2D Trajectory Reconstruction & Spatial Uncertainty")
        st.caption(
            "Comparing relative PDR drift, raw Wi-Fi measurements (accepted vs rejected), "
            "and the fused EKF trajectory with 95% spatial confidence ellipses."
        )

        if not is_available or not fused_trajectory:
            st.info("No fused trajectory has been computed yet. Use the **Fusion Controls & Execution** tab to run fusion.")
        else:
            df_traj = pd.DataFrame(fused_trajectory)

            # Separate into PDR-only points, Wi-Fi accepted, Wi-Fi rejected
            fig = go.Figure()

            # 1. Continuous Fused Path
            fig.add_trace(
                go.Scatter(
                    x=df_traj["x_m"],
                    y=df_traj["y_m"],
                    mode="lines+markers",
                    line=dict(color="#1E3A8A", width=3),
                    marker=dict(size=4, color="#1E3A8A"),
                    name="Fused EKF Trajectory",
                    hoverinfo="text",
                    hovertext=[
                        f"Point #{i+1}<br>Status: {r['status']}<br>Pos: ({r['x_m']:.2f}, {r['y_m']:.2f})m<br>"
                        f"Heading: {r['theta_deg']:.1f}°<br>95% Radius: ±{r['conservative_95_radius_m']:.2f}m"
                        for i, r in df_traj.iterrows()
                    ],
                )
            )

            # 2. Accepted Wi-Fi Corrections
            df_accepted = df_traj[df_traj["status"] == "WIFI_CORRECTED"]
            if not df_accepted.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_accepted["x_m"],
                        y=df_accepted["y_m"],
                        mode="markers",
                        marker=dict(size=11, color="#10B981", symbol="diamond", line=dict(color="black", width=1)),
                        name="Accepted Wi-Fi Correction",
                        hoverinfo="text",
                        hovertext=[
                            f"Wi-Fi Accepted<br>Pos: ({r['x_m']:.2f}, {r['y_m']:.2f})m<br>"
                            f"Mahalanobis d²: {r.get('mahalanobis_dist_sq', 0):.2f}"
                            for _, r in df_accepted.iterrows()
                        ],
                    )
                )

            # 3. Rejected Wi-Fi Measurements (Outliers)
            df_rejected = df_traj[df_traj["status"] == "WIFI_REJECTED"]
            if not df_rejected.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_rejected["x_m"],
                        y=df_rejected["y_m"],
                        mode="markers",
                        marker=dict(size=12, color="#EF4444", symbol="x", line=dict(color="darkred", width=2)),
                        name="Rejected Wi-Fi Outlier",
                        hoverinfo="text",
                        hovertext=[
                            f"Wi-Fi REJECTED<br>Reason: {r.get('rejection_reason', 'Gated Out')}<br>"
                            f"Pos: ({r['x_m']:.2f}, {r['y_m']:.2f})m<br>"
                            f"Mahalanobis d²: {r.get('mahalanobis_dist_sq', 0):.2f}"
                            for _, r in df_rejected.iterrows()
                        ],
                    )
                )

            # 4. Periodic 95% Confidence Ellipses
            step_interval = max(1, len(df_traj) // 15)
            for idx in range(0, len(df_traj), step_interval):
                pt = df_traj.iloc[idx]
                cx, cy = pt["x_m"], pt["y_m"]
                a = pt["semi_major_axis_m"]
                b = pt["semi_minor_axis_m"]
                ang_rad = math.radians(pt["ellipse_angle_deg"])

                # Parametric ellipse points
                t_vals = np.linspace(0, 2 * math.pi, 30)
                ex = a * np.cos(t_vals)
                ey = b * np.sin(t_vals)
                # Rotate
                rx = cx + ex * math.cos(ang_rad) - ey * math.sin(ang_rad)
                ry = cy + ex * math.sin(ang_rad) + ey * math.cos(ang_rad)

                fig.add_trace(
                    go.Scatter(
                        x=rx,
                        y=ry,
                        mode="lines",
                        line=dict(color="rgba(30, 58, 138, 0.25)", width=1, dash="dot"),
                        showlegend=(idx == 0),
                        name="95% Confidence Ellipse" if idx == 0 else None,
                        hoverinfo="skip",
                    )
                )

            # 5. Start and End Points
            fig.add_trace(
                go.Scatter(
                    x=[df_traj["x_m"].iloc[0]],
                    y=[df_traj["y_m"].iloc[0]],
                    mode="markers+text",
                    marker=dict(size=14, color="green", symbol="circle"),
                    text=["START"],
                    textposition="bottom center",
                    name="Start Origin",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[df_traj["x_m"].iloc[-1]],
                    y=[df_traj["y_m"].iloc[-1]],
                    mode="markers+text",
                    marker=dict(size=16, color="red", symbol="star"),
                    text=["FINISH"],
                    textposition="top center",
                    name="Finish Endpoint",
                )
            )

            fig.update_layout(
                title=f"2D Fused Indoor Trajectory ({summary.get('total_distance_m', 0):.1f}m, {summary.get('duration_s', 0):.1f}s)",
                xaxis_title="Building Local X (meters)",
                yaxis_title="Building Local Y (meters)",
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(scaleanchor="x", scaleratio=1),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Fused Trajectory States Table"):
                disp_cols = ["timestamp_epoch_ms", "status", "x_m", "y_m", "theta_deg", "sigma_x_m", "sigma_y_m", "conservative_95_radius_m", "rejection_reason"]
                avail_cols = [c for c in disp_cols if c in df_traj.columns]
                st.dataframe(df_traj[avail_cols], use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 2: Fusion Controls & Execution
    # -------------------------------------------------------------------------
    with tab_run:
        st.markdown("### ⚡ Execute Sensor Fusion")
        st.caption("Configure building coordinates, initial pose, and EKF noise parameters for this dataset.")

        default_site = f"SITE-{dataset_id[:8].upper()}"

        with st.form("execute_fusion_form"):
            c_site, c_bldg, c_floor = st.columns(3)
            with c_site:
                run_site = st.text_input("Target Site ID", value=default_site)
            with c_bldg:
                run_bldg = st.text_input("Building ID", value="BLDG-1")
            with c_floor:
                run_floor = st.number_input("Floor Number", value=1, step=1)

            st.markdown("#### Initial Pose & Coordinate Alignment")
            c_x, c_y, c_head = st.columns(3)
            with c_x:
                run_init_x = st.number_input("Initial X (m, leave 0 to auto-init from Wi-Fi)", value=0.0, step=1.0)
            with c_y:
                run_init_y = st.number_input("Initial Y (m, leave 0 to auto-init from Wi-Fi)", value=0.0, step=1.0)
            with c_head:
                run_init_head = st.number_input("Initial Heading (degrees)", value=0.0, step=5.0)

            st.markdown("#### Extended Kalman Filter Parameters")
            c_gate, c_sig, c_ver = st.columns(3)
            with c_gate:
                run_chi2 = st.number_input("Chi-Square Outlier Gate (95% = 5.991)", value=5.991, min_value=1.0, max_value=25.0, step=0.5)
            with c_sig:
                run_sigma = st.number_input("Default Wi-Fi Sigma (m)", value=4.0, min_value=0.5, max_value=15.0, step=0.5)
            with c_ver:
                run_ver = st.text_input("Site Radio Map Version", value="v1")

            run_gating = st.checkbox("Enable Statistical Outlier Gating", value=True)

            submitted = st.form_submit_button("🚀 Run Sensor Fusion")
            if submitted:
                payload = {
                    "site_id": run_site,
                    "building_id": run_bldg,
                    "floor_id": int(run_floor),
                    "initial_x_m": float(run_init_x) if run_init_x != 0.0 else None,
                    "initial_y_m": float(run_init_y) if run_init_y != 0.0 else None,
                    "initial_heading_deg": float(run_init_head),
                    "chi2_threshold": float(run_chi2),
                    "enable_outlier_gating": run_gating,
                    "default_wifi_sigma_m": float(run_sigma),
                    "radio_map_version": run_ver,
                }
                with st.spinner("Executing Wi-Fi + PDR Extended Kalman Filter fusion..."):
                    try:
                        res = client.run_sensor_fusion(dataset_id, payload)
                        st.success(
                            f"Fusion completed successfully! Generated {res.get('total_points', 0)} fused points "
                            f"({res.get('summary', {}).get('wifi_scans_accepted', 0)} Wi-Fi corrections accepted)."
                        )
                        st.rerun()
                    except DatasetApiClientException as e:
                        st.error(f"Sensor fusion execution failed: {e}")

    # -------------------------------------------------------------------------
    # TAB 3: Innovation & Outlier Gating Analysis
    # -------------------------------------------------------------------------
    with tab_diag:
        st.markdown("### 📈 Innovation & Mahalanobis Distance Analysis")
        st.caption(
            "Inspect innovation vectors r_k = z_k - H x_{k|k-1} and Mahalanobis distances d_k^2 to diagnose "
            "measurement consistency and verify statistical outlier rejection."
        )

        if not is_available or not fused_trajectory:
            st.info("No fusion execution data available. Run fusion in the **Fusion Controls** tab first.")
        else:
            df_traj = pd.DataFrame(fused_trajectory)
            df_wifi = df_traj[df_traj["status"].isin(["WIFI_CORRECTED", "WIFI_REJECTED"])].copy()

            if df_wifi.empty:
                st.info("No Wi-Fi observations were encountered during this fusion session.")
            else:
                diag_c1, diag_c2 = st.columns([1, 1])

                with diag_c1:
                    # Mahalanobis Distance Gating Chart
                    chi2_thresh = summary.get("chi2_threshold", 5.991)
                    fig_maha = go.Figure()
                    fig_maha.add_trace(
                        go.Scatter(
                            x=list(range(1, len(df_wifi) + 1)),
                            y=df_wifi["mahalanobis_dist_sq"],
                            mode="markers",
                            marker=dict(
                                size=10,
                                color=df_wifi["status"].map({"WIFI_CORRECTED": "#10B981", "WIFI_REJECTED": "#EF4444"}),
                                symbol=df_wifi["status"].map({"WIFI_CORRECTED": "circle", "WIFI_REJECTED": "x"}),
                            ),
                            text=df_wifi["rejection_reason"].fillna("Accepted"),
                            name="Mahalanobis d²",
                        )
                    )
                    # Gating threshold line
                    fig_maha.add_hline(
                        y=5.991,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Chi2 Gate (95% = 5.991)",
                        annotation_position="top right",
                    )
                    fig_maha.update_layout(
                        title="Mahalanobis Distance per Wi-Fi Scan",
                        xaxis_title="Wi-Fi Observation Index",
                        yaxis_title="Mahalanobis Distance Squared (d²)",
                        height=400,
                    )
                    st.plotly_chart(fig_maha, use_container_width=True)

                with diag_c2:
                    # Rejection Breakdown Pie / Bar Chart
                    rej_breakdown = summary.get("rejection_breakdown", {})
                    if rej_breakdown:
                        df_rej = pd.DataFrame(list(rej_breakdown.items()), columns=["Reason", "Count"])
                        fig_rej = px.pie(
                            df_rej,
                            names="Reason",
                            values="Count",
                            title="Wi-Fi Rejection Root Cause Breakdown",
                            color_discrete_sequence=px.colors.sequential.RdBu,
                        )
                        st.plotly_chart(fig_rej, use_container_width=True)
                    else:
                        st.success("🎉 All Wi-Fi observations were statistically accepted (zero outliers detected)!")

    # -------------------------------------------------------------------------
    # TAB 4: PDR vs Wi-Fi vs Fused Benchmark
    # -------------------------------------------------------------------------
    with tab_bench:
        st.markdown("### 📊 PDR-only vs Wi-Fi-only vs Fused Benchmark")
        st.caption(
            "Empirical verification quantifying positioning accuracy, trajectory drift, and uncertainty coverage "
            "across independent test trajectories."
        )

        with st.form("eval_benchmark_form"):
            b_c1, b_c2 = st.columns(2)
            with b_c1:
                b_site = st.text_input("Evaluation Site ID", value=default_site)
            with b_c2:
                b_synth = st.checkbox(
                    "Use Mathematically Controlled Synthetic Benchmark Fixture",
                    value=True,
                    help="Generates an L-shaped corridor trajectory with simulated drift and injected Wi-Fi outlier.",
                )
            eval_btn = st.form_submit_button("⚡ Run Comparative Trajectory Evaluation")

        if eval_btn:
            eval_payload = {
                "site_id": b_site,
                "chi2_threshold": 5.991,
                "use_synthetic_fixture": b_synth,
            }
            with st.spinner("Computing comparative benchmark across PDR-only, Wi-Fi-only, and Fused..."):
                try:
                    res_eval = client.evaluate_sensor_fusion(dataset_id, eval_payload)
                    rep = res_eval.get("report", {})

                    st.success("Evaluation completed successfully!")

                    # Benchmark Table
                    pdr_m = rep.get("pdr_only", {})
                    wifi_m = rep.get("wifi_only", {})
                    fused_m = rep.get("fused", {})

                    comparison_data = [
                        {"Metric": "Mean Error (m)", "PDR-Only": f"{pdr_m.get('mean_error_m', 0):.2f}", "Wi-Fi Only": f"{wifi_m.get('mean_error_m', 0):.2f}", "Fused EKF": f"{fused_m.get('mean_error_m', 0):.2f}"},
                        {"Metric": "Median Error (m)", "PDR-Only": f"{pdr_m.get('median_error_m', 0):.2f}", "Wi-Fi Only": f"{wifi_m.get('median_error_m', 0):.2f}", "Fused EKF": f"{fused_m.get('median_error_m', 0):.2f}"},
                        {"Metric": "95th Percentile (m)", "PDR-Only": f"{pdr_m.get('p95_error_m', 0):.2f}", "Wi-Fi Only": f"{wifi_m.get('p95_error_m', 0):.2f}", "Fused EKF": f"{fused_m.get('p95_error_m', 0):.2f}"},
                        {"Metric": "Max Error (m)", "PDR-Only": f"{pdr_m.get('max_error_m', 0):.2f}", "Wi-Fi Only": f"{wifi_m.get('max_error_m', 0):.2f}", "Fused EKF": f"{fused_m.get('max_error_m', 0):.2f}"},
                        {"Metric": "Endpoint Error (m)", "PDR-Only": f"{pdr_m.get('endpoint_error_m', 0):.2f}", "Wi-Fi Only": f"{wifi_m.get('endpoint_error_m', 0):.2f}", "Fused EKF": f"{fused_m.get('endpoint_error_m', 0):.2f}"},
                        {"Metric": "Drift per Meter", "PDR-Only": f"{pdr_m.get('drift_per_meter', 0):.4f}", "Wi-Fi Only": "N/A", "Fused EKF": f"{fused_m.get('drift_per_meter', 0):.4f}"},
                        {"Metric": "95% Ellipse Coverage", "PDR-Only": "N/A", "Wi-Fi Only": "N/A", "Fused EKF": f"{fused_m.get('uncertainty_95_coverage', 0)*100:.1f}%" if fused_m.get('uncertainty_95_coverage') is not None else "N/A"},
                    ]
                    df_comp = pd.DataFrame(comparison_data)
                    st.dataframe(df_comp, use_container_width=True, hide_index=True)

                    # Bar chart comparison
                    metrics_chart = pd.DataFrame([
                        {"Approach": "PDR-Only", "Mean Error (m)": pdr_m.get("mean_error_m", 0), "Endpoint Error (m)": pdr_m.get("endpoint_error_m", 0)},
                        {"Approach": "Wi-Fi Only", "Mean Error (m)": wifi_m.get("mean_error_m", 0), "Endpoint Error (m)": wifi_m.get("endpoint_error_m", 0)},
                        {"Approach": "Fused EKF", "Mean Error (m)": fused_m.get("mean_error_m", 0), "Endpoint Error (m)": fused_m.get("endpoint_error_m", 0)},
                    ])
                    fig_bar = px.bar(
                        metrics_chart,
                        x="Approach",
                        y=["Mean Error (m)", "Endpoint Error (m)"],
                        barmode="group",
                        title="Positioning Error Comparison Across Fusion Approaches",
                        color_discrete_sequence=["#3B82F6", "#10B981"],
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    st.caption(
                        f"Evaluation fixture: {'Controlled Synthetic Test Fixture' if rep.get('is_synthetic_fixture') else 'Real Ground Truth Annotations'} | "
                        f"Duration: {rep.get('duration_s')}s | Wi-Fi Acceptance: {rep.get('wifi_scans_accepted')}/{rep.get('wifi_scans_total')} "
                        f"({rep.get('acceptance_rate', 0)*100:.0f}%) | Outliers Rejected: {rep.get('wifi_scans_rejected')}"
                    )

                except DatasetApiClientException as e:
                    st.error(f"Evaluation failed: {e}")
