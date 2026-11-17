"""Streamlit component for Pedestrian Dead Reckoning (PDR) & Movement Analysis (Task 21)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import DatasetApiClient, DatasetApiClientException


def render_pdr_view(client: DatasetApiClient, dataset_id: Optional[str]) -> None:
    """Renders the Pedestrian Dead Reckoning (PDR) dashboard screen."""
    st.markdown("## 👣 Pedestrian Dead Reckoning (PDR) & Movement Analysis")
    st.caption("Relative motion tracking, step kinematics, heading estimation, and trajectory reconstruction (Task 21).")

    st.warning("⚠️ **Notice**: Relative PDR trajectory — not absolute indoor coordinates.")

    if not dataset_id:
        st.info("💡 Please select an active survey dataset in **1. Overview & Catalog** first.")
        return

    # Check PDR status
    with st.spinner("Checking PDR status..."):
        try:
            status_data = client.get_pdr_status(dataset_id)
        except DatasetApiClientException as e:
            st.error(f"Failed to communicate with PDR backend: {e}")
            return

    is_computed = status_data.get("is_computed", False)
    report = status_data.get("report")
    manifest = status_data.get("manifest")

    # 1. PDR Configuration Expander
    with st.expander("🛠️ PDR Engine Configuration", expanded=not is_computed):
        st.markdown("Adjust step detection sensitivity, Weinberg stride calibration, and coordinate parameters.")
        c1, c2 = st.columns(2)
        with c1:
            step_signal = st.selectbox(
                "Step Detection Signal:",
                ["vertical", "magnitude"],
                index=0,
                help="Vertical acceleration (from gravity compensation) or magnitude."
            )
            weinberg_k = st.slider(
                "Weinberg Stride Factor (K):",
                min_value=0.30,
                max_value=0.60,
                value=0.42,
                step=0.01,
                help="Calibration constant relating vertical bounce to step length: L = K * (a_max - a_min)^0.25."
            )
            min_interval_s = st.slider(
                "Min Step Interval / Refractory Period (s):",
                min_value=0.25,
                max_value=0.60,
                value=0.33,
                step=0.01,
                help="Minimum time between adjacent steps (rejects bounce noise)."
            )

        with c2:
            min_prominence = st.slider(
                "Min Peak Prominence (m/s²):",
                min_value=0.3,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="Minimum vertical acceleration peak prominence required to validate a step."
            )
            init_heading_deg = st.number_input(
                "Initial Heading (degrees):",
                min_value=-180.0,
                max_value=180.0,
                value=0.0,
                step=5.0,
                help="Starting heading angle in local navigation frame."
            )
            st.caption("Coordinate convention: Local Cartesian (X=East, Y=North, counterclockwise angle).")

        config_payload = {
            "step_detection_signal": step_signal,
            "weinberg_k": float(weinberg_k),
            "min_step_interval_s": float(min_interval_s),
            "min_peak_prominence_mps2": float(min_prominence),
            "initial_heading_rad": float(np.radians(init_heading_deg)),
            "origin_x_m": 0.0,
            "origin_y_m": 0.0,
        }

        st.markdown("---")
        btn_col, note_col = st.columns([1, 2])
        with btn_col:
            run_btn = st.button("▶ Execute PDR Engine", type="primary", use_container_width=True)
        with note_col:
            st.caption("Requires preprocessed sensor data (Task 20). Persists artifacts in `data/processed/multimodal/<dataset_id>/pdr/`.")

        if run_btn:
            with st.spinner("Executing Pedestrian Dead Reckoning pipeline..."):
                try:
                    res = client.run_pdr(dataset_id, config_payload)
                    st.success(f"✅ {res.get('message', 'PDR finished.')}")
                    # Refresh status
                    status_data = client.get_pdr_status(dataset_id)
                    is_computed = status_data.get("is_computed", True)
                    report = status_data.get("report")
                    manifest = status_data.get("manifest")
                except DatasetApiClientException as e:
                    st.error(f"❌ PDR execution failed: {e}")

    # 2. Movement Diagnostics & KPI Cards
    if not is_computed or not report:
        st.info("ℹ️ PDR has not been computed for this survey yet. Click **'Execute PDR Engine'** above to run it.")
        return

    st.markdown("---")
    st.markdown("### 📊 Movement Summary")

    mov_sum = report.get("movement_summary", {})
    grav_sum = report.get("gravity_summary", {})
    orient_sum = report.get("orientation_summary", {})
    step_sum = report.get("step_summary", {})

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("Total Steps", f"{mov_sum.get('total_steps', 0):,}")
    with k2:
        st.metric("Mean Cadence", f"{step_sum.get('mean_cadence_spm', 0.0)} SPM")
    with k3:
        st.metric("Mean Step Length", f"{mov_sum.get('mean_step_length_m', 0.0)} m")
    with k4:
        st.metric("Total Distance", f"{mov_sum.get('total_distance_m', 0.0)} m")
    with k5:
        st.metric("Net Displacement", f"{mov_sum.get('net_displacement_m', 0.0)} m")
    with k6:
        st.metric("Final Uncertainty", f"±{mov_sum.get('final_uncertainty_radius_m', 0.0)} m")

    # 3. Interactive Visualization Tabs
    tab_traj, tab_steps, tab_orient, tab_diag, tab_json = st.tabs([
        "🗺️ 2D Relative Trajectory",
        "👣 Step Kinematics",
        "🧭 Orientation & Heading",
        "⚠️ Quality & Uncertainty",
        "📜 Report & Manifest JSON"
    ])

    with tab_traj:
        st.markdown("#### Reconstructed 2D Relative Movement Path")
        st.caption("Local Cartesian coordinate plane starting at (0, 0). Does not represent absolute building coordinates.")

        with st.spinner("Loading trajectory points..."):
            points = client.get_pdr_trajectory(dataset_id)

        if points:
            df_pts = pd.DataFrame(points)
            x_vals = df_pts["rel_x_m"].values
            y_vals = df_pts["rel_y_m"].values
            conf_vals = df_pts["confidence"].values
            unc_vals = df_pts["pos_uncertainty_radius_m"].values

            fig = go.Figure()

            # Path line
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name="PDR Path",
                line=dict(color="#1f77b4", width=3)
            ))

            # Step points
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                name="Step Events",
                marker=dict(
                    size=8,
                    color=conf_vals,
                    colorscale="Viridis",
                    colorbar=dict(title="Confidence"),
                    showscale=True
                ),
                text=[f"Step: {p.get('step_id')}<br>Dist: {p.get('cum_distance_m')} m<br>Uncert: ±{p.get('pos_uncertainty_radius_m')} m" for p in points],
                hoverinfo="text"
            ))

            # Origin marker
            fig.add_trace(go.Scatter(
                x=[x_vals[0]],
                y=[y_vals[0]],
                mode="markers+text",
                name="Start Origin (0,0)",
                text=["START"],
                textposition="top center",
                marker=dict(color="green", size=14, symbol="star")
            ))

            # End marker
            fig.add_trace(go.Scatter(
                x=[x_vals[-1]],
                y=[y_vals[-1]],
                mode="markers+text",
                name="Current End",
                text=["END"],
                textposition="bottom center",
                marker=dict(color="red", size=14, symbol="diamond")
            ))

            fig.update_layout(
                title=f"Relative PDR Path ({len(points)} points, {mov_sum.get('total_distance_m', 0.0)} m)",
                xaxis_title="Relative X (meters)",
                yaxis_title="Relative Y (meters)",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                template="plotly_white",
                height=550,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No trajectory points found.")

    with tab_steps:
        st.markdown("#### Detected Step Events")
        with st.spinner("Loading step records..."):
            steps = client.get_pdr_steps(dataset_id)

        if steps:
            df_steps = pd.DataFrame(steps)
            st.dataframe(df_steps, use_container_width=True)

            # Step interval histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_steps["step_interval_ms"],
                nbinsx=25,
                marker_color="#ff7f0e",
                name="Step Interval (ms)"
            ))
            fig_hist.update_layout(
                title="Step Interval Distribution (ms)",
                xaxis_title="Step Interval (ms)",
                yaxis_title="Count",
                template="plotly_white",
                height=350
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No step events recorded.")

    with tab_orient:
        st.markdown("#### Heading & Attitude Estimation")
        st.write(f"- **Primary Orientation Source**: `{orient_sum.get('primary_source', 'UNKNOWN')}`")
        st.write(f"- **Mean Orientation Confidence**: {orient_sum.get('mean_confidence', 0.0):.2f}")
        st.write(f"- **Magnetic Disturbances Detected**: {orient_sum.get('magnetic_disturbance_count', 0)}")

        with st.spinner("Loading trajectory heading values..."):
            points = client.get_pdr_trajectory(dataset_id)

        if points:
            df_pts = pd.DataFrame(points)
            t_s = (df_pts["timestamp_ms"] - df_pts["timestamp_ms"].iloc[0]) / 1000.0

            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(
                x=t_s,
                y=df_pts["heading_deg"],
                mode="lines+markers",
                name="Heading (degrees)",
                line=dict(color="#2ca02c", width=2)
            ))
            fig_h.update_layout(
                title="Estimated Heading over Time",
                xaxis_title="Elapsed Time (seconds)",
                yaxis_title="Heading (degrees)",
                yaxis=dict(range=[-185, 185]),
                template="plotly_white",
                height=350
            )
            st.plotly_chart(fig_h, use_container_width=True)

    with tab_diag:
        st.markdown("#### Quality & Uncertainty Diagnostics")
        col_g, col_o = st.columns(2)
        with col_g:
            st.markdown("##### Gravity Separation Quality")
            st.write(f"- **Method**: `{grav_sum.get('method_used', 'N/A')}`")
            st.write(f"- **Quality Score**: {grav_sum.get('mean_compensation_quality', 0.0):.2f}")
            st.write(f"- **Rotation Vector Coverage**: {grav_sum.get('rotation_vector_coverage_ratio', 0.0)*100:.1f}%")

        with col_o:
            st.markdown("##### Heading & Drift Metrics")
            st.write(f"- **Accumulated Spatial Uncertainty Radius**: ±{mov_sum.get('final_uncertainty_radius_m', 0.0)} m")
            st.write(f"- **Candidate Steps Detected**: {step_sum.get('total_candidates_detected', 0)}")
            st.write(f"- **Accepted Steps**: {step_sum.get('total_steps_accepted', 0)}")
            st.write(f"- **Refractory Rejections**: {step_sum.get('total_rejected_refractory', 0)}")

        if report.get("warnings"):
            st.markdown("##### Diagnostic Warnings")
            for w in report.get("warnings", []):
                st.warning(f"⚠️ {w}")

    with tab_json:
        st.markdown("#### Raw Report and Audit Manifest")
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            st.markdown("##### PDR Report JSON")
            st.json(report)
        with c_r2:
            st.markdown("##### PDR Manifest JSON")
            st.json(manifest or {})
