"""Streamlit component for Map-Constrained Multimodal Particle Filter Localization (Task 27)."""

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


def render_map_localization_view(client: DatasetApiClient, dataset_id: Optional[str]) -> None:
    """Renders the comprehensive Map-Constrained Multimodal Localization dashboard."""
    st.markdown("## 🧭 Map-Constrained Multimodal Localization (Particle Filter)")
    st.caption(
        "Sequential Monte Carlo estimation enforcing verified building boundaries, solid wall collision "
        "rejection, doorway passage tolerance, Wi-Fi observation likelihoods, and visual landmarks to eliminate "
        "physically implausible indoor trajectories (Task 27)."
    )

    if not dataset_id:
        st.info("💡 Please select an active survey dataset in **1. Overview & Catalog** first.")
        return

    # 1. Fetch current status and trajectory
    is_available = False
    summary: Dict[str, Any] = {}
    trajectory_data: List[Dict[str, Any]] = []

    try:
        status_resp = client.get_map_localization_status(dataset_id)
        is_available = status_resp.get("is_available", False)
        summary = status_resp.get("summary") or {}
        if is_available:
            traj_resp = client.get_map_localization_trajectory(dataset_id)
            trajectory_data = traj_resp.get("trajectory", [])
    except Exception as e:
        st.error(f"Error communicating with map localization backend: {e}")

    # Top KPI Metrics Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Filtered States", len(trajectory_data) if is_available else 0)
    with kpi2:
        wall_rej = summary.get("total_wall_crossings_rejected", 0)
        st.metric("Wall Collisions Rejected", wall_rej)
    with kpi3:
        resamp_cnt = summary.get("resampling_events_count", 0)
        st.metric("Resampling Events", resamp_cnt)
    with kpi4:
        unc_m = summary.get("mean_position_uncertainty_m", 0.0)
        st.metric("Mean 95% Uncertainty", f"±{unc_m:.2f} m" if is_available else "N/A")

    # Tabs
    tab_map, tab_run, tab_diag, tab_bench = st.tabs([
        "🗺️ 2D Trajectory & Floorplan Canvas",
        "⚡ Filter Controls & Execution",
        "🔬 Particle Spread & Diagnostics",
        "📊 Comparative Benchmark Evaluation",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: 2D Trajectory & Floorplan Canvas
    # -------------------------------------------------------------------------
    with tab_map:
        st.markdown("### 🗺️ Map-Constrained Trajectory Canvas")
        st.caption(
            "Visualizing the particle filter state estimates overlaid against verified building walls, "
            "doorway openings, and comparing with the unconstrained EKF baseline."
        )

        if not is_available or not trajectory_data:
            st.info("No map-constrained trajectory computed yet. Configure parameters and run in **Filter Controls & Execution**.")
        else:
            df_traj = pd.DataFrame(trajectory_data)

            # Interactive Layer Controls
            ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
            with ctrl1:
                show_ekf = st.checkbox("Show EKF Baseline", value=True)
            with ctrl2:
                show_uncertainty = st.checkbox("Show 95% Uncertainty", value=False)
            with ctrl3:
                color_by = st.selectbox("Color Trace By", ["Observation Type", "Floor Index", "Particle Spread"], index=0)

            # Build Plotly Figure
            fig = go.Figure()

            # Retrieve building map for background geometry
            building_id = df_traj["building_id"].iloc[0] if "building_id" in df_traj.columns else "BLDG-1"
            try:
                bmap = client.get_indoor_map(dataset_id=dataset_id, building_id=building_id)
                if bmap and "floors" in bmap:
                    for f in bmap["floors"]:
                        # Outer boundary
                        boundary_pts = f.get("boundary", [])
                        if boundary_pts and len(boundary_pts) > 2:
                            bx = [p["x_m"] for p in boundary_pts] + [boundary_pts[0]["x_m"]]
                            by = [p["y_m"] for p in boundary_pts] + [boundary_pts[0]["y_m"]]
                            fig.add_trace(go.Scatter(
                                x=bx, y=by, mode="lines",
                                line=dict(color="#64748b", width=2, dash="solid"),
                                name=f"Floor {f.get('floor_id', 0)} Perimeter",
                                hoverinfo="text",
                                text=f"Floor {f.get('floor_id', 0)} Boundary",
                            ))
                        # Rooms
                        for rm in f.get("rooms", []):
                            poly = rm.get("polygon", [])
                            if poly and len(poly) > 2:
                                rx = [p["x_m"] for p in poly] + [poly[0]["x_m"]]
                                ry = [p["y_m"] for p in poly] + [poly[0]["y_m"]]
                                fig.add_trace(go.Scatter(
                                    x=rx, y=ry, mode="lines",
                                    fill="toself",
                                    fillcolor="rgba(226, 232, 240, 0.25)",
                                    line=dict(color="#334155", width=2),
                                    name=f"Room: {rm.get('name', 'Room')}",
                                    text=f"{rm.get('name', 'Room')} ({rm.get('room_type', '')})",
                                    hoverinfo="text",
                                ))
                        # Doors
                        for dr in f.get("doors", []):
                            dr_poly = dr.get("polygon", [])
                            if dr_poly and len(dr_poly) >= 2:
                                dx = [p["x_m"] for p in dr_poly]
                                dy = [p["y_m"] for p in dr_poly]
                                fig.add_trace(go.Scatter(
                                    x=dx, y=dy, mode="lines",
                                    line=dict(color="#10b981", width=5),
                                    name="Door Opening",
                                    text=f"Door: {dr.get('door_id', '')}",
                                    hoverinfo="text",
                                ))
            except Exception as e:
                st.caption(f"(Building map overlay notice: {e})")

            # Add EKF Baseline trajectory if toggled
            if show_ekf and "ekf_reference_x_m" in df_traj.columns:
                ekf_pts = df_traj.dropna(subset=["ekf_reference_x_m", "ekf_reference_y_m"])
                if not ekf_pts.empty:
                    fig.add_trace(go.Scatter(
                        x=ekf_pts["ekf_reference_x_m"],
                        y=ekf_pts["ekf_reference_y_m"],
                        mode="lines+markers",
                        line=dict(color="#f97316", width=2, dash="dot"),
                        marker=dict(size=4, color="#ea580c"),
                        name="EKF Baseline (Unconstrained)",
                        hoverinfo="text",
                        hovertext=[f"EKF Ref: ({x:.2f}, {y:.2f})" for x, y in zip(ekf_pts["ekf_reference_x_m"], ekf_pts["ekf_reference_y_m"])],
                    ))

            # Add Map-Constrained PF Trajectory
            marker_color: Any = "#2563eb"
            if color_by == "Floor Index" and "floor_id" in df_traj.columns:
                marker_color = df_traj["floor_id"]
            elif color_by == "Particle Spread" and "particle_spread_m" in df_traj.columns:
                marker_color = df_traj["particle_spread_m"]

            fig.add_trace(go.Scatter(
                x=df_traj["x_m"],
                y=df_traj["y_m"],
                mode="lines+markers",
                line=dict(color="#2563eb", width=3),
                marker=dict(
                    size=6,
                    color=marker_color,
                    colorscale="Viridis" if color_by != "Observation Type" else None,
                    showscale=color_by != "Observation Type",
                ),
                name="Map-Constrained PF Trajectory",
                hoverinfo="text",
                hovertext=[
                    f"Step {idx}<br>Type: {r.get('observation_type', 'PDR')}<br>"
                    f"Pos: ({r.get('x_m', 0.0):.2f}, {r.get('y_m', 0.0):.2f}) m<br>"
                    f"Heading: {r.get('theta_deg', 0.0):.1f}°<br>"
                    f"Neff: {r.get('effective_sample_size', 0.0):.1f}<br>"
                    f"95% Unc: ±{r.get('position_uncertainty_m', 0.0):.2f} m"
                    for idx, r in df_traj.iterrows()
                ],
            ))

            # Add Start and End markers
            fig.add_trace(go.Scatter(
                x=[df_traj["x_m"].iloc[0]],
                y=[df_traj["y_m"].iloc[0]],
                mode="markers+text",
                marker=dict(size=14, color="#16a34a", symbol="star"),
                text=["START"],
                textposition="top center",
                name="Start Location",
            ))

            fig.add_trace(go.Scatter(
                x=[df_traj["x_m"].iloc[-1]],
                y=[df_traj["y_m"].iloc[-1]],
                mode="markers+text",
                marker=dict(size=14, color="#dc2626", symbol="square"),
                text=["END"],
                textposition="top center",
                name="End Location",
            ))

            # Add 95% uncertainty circles if requested
            if show_uncertainty:
                step_sample = df_traj.iloc[::max(1, len(df_traj) // 20)]
                for _, r in step_sample.iterrows():
                    radius = r.get("position_uncertainty_m", 1.0)
                    theta_circ = np.linspace(0, 2 * np.pi, 25)
                    cx = r["x_m"] + radius * np.cos(theta_circ)
                    cy = r["y_m"] + radius * np.sin(theta_circ)
                    fig.add_trace(go.Scatter(
                        x=cx, y=cy, mode="lines",
                        line=dict(color="rgba(37, 99, 235, 0.3)", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

            fig.update_layout(
                xaxis_title="East / X (meters)",
                yaxis_title="North / Y (meters)",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                height=650,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: Filter Controls & Execution
    # -------------------------------------------------------------------------
    with tab_run:
        st.markdown("### ⚡ Particle Filter Execution Parameters")
        st.caption("Adjust Monte Carlo parameters and trigger localization run.")

        # Map and building options
        building_options = ["BLDG-1", "SYNTH_BENCHMARK", "BUILDING_A"]
        try:
            stored_maps = client.list_indoor_maps(dataset_id=dataset_id)
            if stored_maps:
                building_options = [m.get("building_id", "BLDG-1") for m in stored_maps]
        except Exception:
            pass

        with st.form("pf_run_form"):
            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1:
                sel_bldg = st.selectbox("Building Map", building_options, index=0)
                sel_floor = st.number_input("Floor Index", min_value=-5, max_value=20, value=1)
                p_count = st.slider("Particle Count (N)", min_value=100, max_value=2000, value=500, step=50)
                resample_ratio = st.slider("Resampling Ratio (Neff / N)", min_value=0.20, max_value=0.80, value=0.50, step=0.05)

            with col_cfg2:
                sigma_l = st.slider("Step Length Noise Std Dev (m)", min_value=0.02, max_value=0.30, value=0.10, step=0.01)
                sigma_h = st.slider("Heading Noise Std Dev (deg)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
                wall_penalty = st.slider("Wall Crossing Weight (0.0 = Hard Reject)", min_value=0.0, max_value=0.5, value=0.0, step=0.05)
                allow_unknown = st.checkbox("Allow Unmapped / Unknown Areas", value=True, help="Prevents eliminating particles if room geometry is partially surveyed.")

            with st.expander("Initial Pose Override (Optional)"):
                init_c1, init_c2, init_c3 = st.columns(3)
                with init_c1:
                    override_x = st.text_input("Initial X (m)", value="")
                with init_c2:
                    override_y = st.text_input("Initial Y (m)", value="")
                with init_c3:
                    override_head = st.number_input("Initial Heading (°)", value=0.0, step=5.0)

            run_submitted = st.form_submit_button("🚀 Run Map-Constrained Particle Filter", type="primary", use_container_width=True)

        if run_submitted:
            with st.spinner("Executing Map-Constrained Particle Filter..."):
                try:
                    req_params: Dict[str, Any] = {
                        "building_id": sel_bldg,
                        "building_map_id": sel_bldg,
                        "floor_id": int(sel_floor),
                        "particle_count": int(p_count),
                        "resample_threshold_ratio": float(resample_ratio),
                        "sigma_step_length_m": float(sigma_l),
                        "sigma_heading_rad": math.radians(float(sigma_h)),
                        "wall_crossing_penalty_factor": float(wall_penalty),
                        "allow_unknown_areas": bool(allow_unknown),
                        "random_seed": 42,
                    }
                    if override_x.strip() and override_y.strip():
                        req_params["initial_x_m"] = float(override_x)
                        req_params["initial_y_m"] = float(override_y)
                        req_params["initial_heading_deg"] = float(override_head)

                    res = client.run_map_localization(dataset_id=dataset_id, run_params=req_params)
                    st.success(
                        f"✅ Map-constrained particle filter completed successfully! "
                        f"Computed {res.get('total_points', 0)} states."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error running particle filter: {e}")

    # -------------------------------------------------------------------------
    # TAB 3: Particle Spread & Diagnostics
    # -------------------------------------------------------------------------
    with tab_diag:
        st.markdown("### 🔬 Particle Dynamics & Degeneracy Diagnostics")
        st.caption("Monitoring Effective Sample Size (Neff), spatial cloud spread, and wall collision occurrences.")

        if not is_available or not trajectory_data:
            st.info("No localization run to diagnose.")
        else:
            df_diag = pd.DataFrame(trajectory_data)
            df_diag["step_num"] = range(len(df_diag))

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                # Effective sample size plot
                fig_neff = go.Figure()
                fig_neff.add_trace(go.Scatter(
                    x=df_diag["step_num"],
                    y=df_diag["effective_sample_size"],
                    mode="lines",
                    name="Neff (Effective Samples)",
                    line=dict(color="#2563eb", width=2),
                ))
                threshold_val = float(df_diag["particle_count"].iloc[0]) * 0.5
                fig_neff.add_hline(
                    y=threshold_val,
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text="Resampling Threshold (50%)",
                )
                fig_neff.update_layout(
                    title="Effective Sample Size (Neff) vs Step",
                    xaxis_title="Trajectory Step",
                    yaxis_title="Effective Particle Count",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_neff, use_container_width=True)

            with col_d2:
                # Particle Spread and 95% uncertainty
                fig_spread = go.Figure()
                fig_spread.add_trace(go.Scatter(
                    x=df_diag["step_num"],
                    y=df_diag["particle_spread_m"],
                    mode="lines",
                    name="1-Sigma Spread (m)",
                    line=dict(color="#10b981", width=2),
                ))
                fig_spread.add_trace(go.Scatter(
                    x=df_diag["step_num"],
                    y=df_diag["position_uncertainty_m"],
                    mode="lines",
                    name="95% Confidence Radius (m)",
                    line=dict(color="#f59e0b", width=2, dash="dot"),
                ))
                fig_spread.update_layout(
                    title="Particle Cloud Spread & 95% Uncertainty",
                    xaxis_title="Trajectory Step",
                    yaxis_title="Uncertainty Radius (m)",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_spread, use_container_width=True)

            # EKF Innovation Tracking
            if "innovation_x_m" in df_diag.columns and df_diag["innovation_x_m"].notnull().any():
                df_diag["ekf_deviation_m"] = np.hypot(
                    df_diag["innovation_x_m"].fillna(0.0),
                    df_diag["innovation_y_m"].fillna(0.0),
                )
                fig_dev = go.Figure()
                fig_dev.add_trace(go.Scatter(
                    x=df_diag["step_num"],
                    y=df_diag["ekf_deviation_m"],
                    mode="lines+markers",
                    name="Distance to EKF State (m)",
                    line=dict(color="#8b5cf6", width=2),
                    marker=dict(size=4),
                ))
                fig_dev.update_layout(
                    title="Map Constraint Offset relative to Unconstrained EKF Baseline (m)",
                    xaxis_title="Trajectory Step",
                    yaxis_title="Deviation Magnitude (m)",
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_dev, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: Comparative Benchmark Evaluation
    # -------------------------------------------------------------------------
    with tab_bench:
        st.markdown("### 📊 Comparative Benchmark Evaluation")
        st.caption(
            "Quantitative evaluation of Absolute Trajectory Error (ATE), RMSE, and physical wall crossing "
            "violations between the Unconstrained Baseline and Map-Constrained Particle Filter."
        )

        b_c1, b_c2 = st.columns([2, 1])
        with b_c1:
            use_synthetic = st.checkbox("Evaluate on Multi-Floor Synthetic Ground-Truth Benchmark", value=True)
        with b_c2:
            run_eval_btn = st.button("🧪 Run Comparative Evaluation", type="primary", use_container_width=True)

        if run_eval_btn:
            with st.spinner("Computing comparative trajectory metrics..."):
                try:
                    eval_res = client.evaluate_map_localization(
                        dataset_id=dataset_id,
                        eval_params={
                            "use_synthetic_fixture": use_synthetic,
                            "particle_count": 500,
                            "allow_unknown_areas": True,
                        },
                    )
                    report = eval_res.get("report", {})
                    st.session_state["map_pf_eval_report"] = report
                    st.success("Evaluation completed successfully!")
                except Exception as e:
                    st.error(f"Failed to evaluate: {e}")

        # Display cached or freshly generated evaluation report
        report = st.session_state.get("map_pf_eval_report")
        if not report:
            # Try fetching default
            try:
                def_res = client.get_map_localization_evaluation(dataset_id=dataset_id, use_synthetic_fixture=True)
                report = def_res.get("report", {})
            except Exception:
                pass

        if report:
            st.markdown("#### 🎯 Quantitative Error Summary")

            metrics = report.get("metrics", {})
            pf_m = metrics.get("particle_filter", {})
            unc_m = metrics.get("unconstrained_baseline", {})

            # Metric Comparison Table
            summary_table = [
                {
                    "Metric": "Mean Absolute Trajectory Error (ATE)",
                    "Map-Constrained PF": f"{pf_m.get('mean_error_m', 0.0):.2f} m",
                    "Unconstrained Baseline": f"{unc_m.get('mean_error_m', 0.0):.2f} m" if unc_m else "N/A",
                },
                {
                    "Metric": "Root Mean Squared Error (RMSE)",
                    "Map-Constrained PF": f"{pf_m.get('rmse_m', 0.0):.2f} m",
                    "Unconstrained Baseline": f"{unc_m.get('rmse_m', 0.0):.2f} m" if unc_m else "N/A",
                },
                {
                    "Metric": "Median Error",
                    "Map-Constrained PF": f"{pf_m.get('median_error_m', 0.0):.2f} m",
                    "Unconstrained Baseline": f"{unc_m.get('median_error_m', 0.0):.2f} m" if unc_m else "N/A",
                },
                {
                    "Metric": "95th Percentile Error",
                    "Map-Constrained PF": f"{pf_m.get('p95_error_m', 0.0):.2f} m",
                    "Unconstrained Baseline": f"{unc_m.get('p95_error_m', 0.0):.2f} m" if unc_m else "N/A",
                },
                {
                    "Metric": "Maximum Error",
                    "Map-Constrained PF": f"{pf_m.get('max_error_m', 0.0):.2f} m",
                    "Unconstrained Baseline": f"{unc_m.get('max_error_m', 0.0):.2f} m" if unc_m else "N/A",
                },
                {
                    "Metric": "Wall Crossing Violations",
                    "Map-Constrained PF": f"{pf_m.get('wall_crossing_violations', 0)}",
                    "Unconstrained Baseline": f"{unc_m.get('wall_crossing_violations', 0)}" if unc_m else "N/A",
                },
                {
                    "Metric": "Wall Violation Reduction",
                    "Map-Constrained PF": f"{report.get('wall_violation_reduction_pct', 100.0):.1f}%",
                    "Unconstrained Baseline": "0.0%",
                },
            ]
            st.dataframe(pd.DataFrame(summary_table), use_container_width=True)

            # Bar chart comparison
            fig_bar = go.Figure(data=[
                go.Bar(
                    name="Map-Constrained PF",
                    x=["Mean Error (m)", "RMSE (m)", "P95 Error (m)", "Max Error (m)"],
                    y=[
                        pf_m.get("mean_error_m", 0.0),
                        pf_m.get("rmse_m", 0.0),
                        pf_m.get("p95_error_m", 0.0),
                        pf_m.get("max_error_m", 0.0),
                    ],
                    marker_color="#2563eb",
                ),
                go.Bar(
                    name="Unconstrained Baseline",
                    x=["Mean Error (m)", "RMSE (m)", "P95 Error (m)", "Max Error (m)"],
                    y=[
                        unc_m.get("mean_error_m", 0.0) if unc_m else 0.0,
                        unc_m.get("rmse_m", 0.0) if unc_m else 0.0,
                        unc_m.get("p95_error_m", 0.0) if unc_m else 0.0,
                        unc_m.get("max_error_m", 0.0) if unc_m else 0.0,
                    ],
                    marker_color="#f97316",
                ),
            ])
            fig_bar.update_layout(
                barmode="group",
                title="Error Metric Comparison (Meters)",
                yaxis_title="Error (m)",
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.download_button(
                label="📥 Download Benchmark Evaluation Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name=f"map_localization_eval_{dataset_id}.json",
                mime="application/json",
            )
