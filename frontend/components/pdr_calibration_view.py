"""Streamlit component for PDR Calibration, Ground-Truth Validation & Drift Analysis (Task 22)."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import DatasetApiClient, DatasetApiClientException


def render_pdr_calibration_view(client: DatasetApiClient, dataset_id: Optional[str]) -> None:
    """Renders the comprehensive PDR Calibration & Validation view."""
    st.markdown("## 📐 PDR Calibration, Ground-Truth Validation & Drift Analysis")
    st.caption("Empirical Weinberg stride calibration, step/heading validation, trajectory drift, and 95% spatial confidence ellipses (Task 22).")

    if not dataset_id:
        st.info("💡 Please select an active survey dataset in **1. Overview & Catalog** first.")
        return

    # Fetch calibration profiles and registered trials
    try:
        profiles = client.get_pdr_calibration_profiles(dataset_id)
        trials = client.get_pdr_trials(dataset_id)
        evaluation_reports = client.get_pdr_evaluation_reports(dataset_id)
    except DatasetApiClientException as e:
        st.error(f"Failed to communicate with calibration service: {e}")
        return

    # Top Status Row
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    with status_col1:
        st.metric("Registered Trials", len(trials))
    with status_col2:
        st.metric("Calibration Profiles", len(profiles))
    with status_col3:
        active_prof = profiles[0] if profiles else None
        active_status = active_prof.get("calibration_status", "UNCALIBRATED") if active_prof else "UNCALIBRATED"
        badge_color = {
            "CALIBRATED": "🟢",
            "VALIDATED": "🌟",
            "UNCALIBRATED": "⚪",
            "INSUFFICIENT_DATA": "⚠️"
        }.get(active_status, "⚪")
        st.metric("Active Status", f"{badge_color} {active_status}")
    with status_col4:
        active_k = active_prof.get("weinberg_k", 0.42) if active_prof else 0.42
        st.metric("Active Weinberg K", f"{active_k:.4f}")

    tab_calib, tab_trials, tab_drift, tab_step_head = st.tabs([
        "🔬 Weinberg Stride Calibration",
        "📋 Walking Trials Registry",
        "🗺️ Trajectory Drift & 95% Ellipses",
        "🎯 Step & Heading Validation"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Weinberg Stride Calibration
    # -------------------------------------------------------------------------
    with tab_calib:
        st.markdown("### 🔬 Empirical Weinberg Step-Length Optimization")
        st.markdown(
            r"Estimates the optimal stride coefficient $K$ in $L_k = K \cdot (a_{\max} - a_{\min})^{0.25}$ "
            r"by minimizing weighted relative distance error against manually measured ground-truth walking trials."
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### 1. Select Calibration Trials")
            if not trials:
                st.warning("⚠️ No ground-truth walking trials registered yet. Please register trials in the **Walking Trials Registry** tab first.")
            else:
                trial_options = {f"{t['trial_id']} ({t['trial_type']}, {t['phone_placement']}, {t['measured_distance_m']}m)": t["trial_id"] for t in trials}
                selected_labels = st.multiselect(
                    "Training Trials (used to fit K):",
                    options=list(trial_options.keys()),
                    default=list(trial_options.keys())[:min(2, len(trial_options))],
                    help="Select at least 1-2 walking trials with known ground-truth distances."
                )
                train_ids = [trial_options[lbl] for lbl in selected_labels]

                held_out_labels = st.multiselect(
                    "Held-out Validation Trials (strictly separate):",
                    options=[lbl for lbl in trial_options.keys() if lbl not in selected_labels],
                    help="Trials held out from optimization to verify generalization and prevent overfitting."
                )
                val_ids = [trial_options[lbl] for lbl in held_out_labels]

                prof_name = st.text_input("New Profile Name:", value="Handheld Walking Profile")
                placement_opt = st.selectbox("Target Phone Placement:", ["HAND_HELD", "POCKET", "BAG", "TEXTING", "EAR", "UNKNOWN"], index=0)

                if st.button("🚀 Run Weinberg Calibration", type="primary"):
                    if not train_ids:
                        st.error("Please select at least 1 training trial.")
                    else:
                        with st.spinner("Optimizing stride parameter K via bounded scalar minimization..."):
                            try:
                                res_prof = client.calibrate_pdr(dataset_id, {
                                    "training_trial_ids": train_ids,
                                    "held_out_trial_ids": val_ids if val_ids else None,
                                    "profile_name": prof_name,
                                    "profile_type": "PHONE_PLACEMENT_SPECIFIC",
                                    "phone_placement": placement_opt
                                })
                                st.success(f"Calibration finished! Fitted K = {res_prof.get('weinberg_k'):.4f} (Status: {res_prof.get('calibration_status')})")
                                st.rerun()
                            except DatasetApiClientException as err:
                                st.error(f"Calibration failed: {err}")

        with c2:
            st.markdown("#### 2. Calibration Profiles Comparison")
            if profiles:
                prof_data = []
                for p in profiles:
                    train_m = p.get("training_metrics", {})
                    prof_data.append({
                        "Profile ID": p.get("profile_id"),
                        "Name": p.get("profile_name"),
                        "Weinberg K": f"{p.get('weinberg_k'):.4f}",
                        "Status": p.get("calibration_status"),
                        "Placement": p.get("phone_placement") or "ALL",
                        "Training Trials": len(p.get("source_trial_ids", [])),
                        "Train MAE (m)": f"{train_m.get('mae_m', 0.0):.2f}" if "mae_m" in train_m else "N/A",
                        "Train Rel Err (%)": f"{train_m.get('mean_relative_error_pct', 0.0):.1f}%" if "mean_relative_error_pct" in train_m else "N/A"
                    })
                st.dataframe(pd.DataFrame(prof_data), use_container_width=True)

                st.markdown("##### 💡 Impact of Stride Calibration")
                st.info(
                    "Calibrating $K$ per participant or phone carriage mode significantly reduces cumulative distance drift. "
                    "Uncalibrated default ($K=0.42$) typically has $\\pm 8\\text{--}15\\%$ distance error, while calibrated profiles achieve $\\le 3\\text{--}5\\%$ error on held-out trials."
                )

    # -------------------------------------------------------------------------
    # TAB 2: Walking Trials Registry
    # -------------------------------------------------------------------------
    with tab_trials:
        st.markdown("### 📋 Measured Walking Trials Registry")
        st.markdown("Register and inspect manually measured walking trials with documented ground-truth geometry.")

        with st.expander("➕ Register New Walking Trial", expanded=not bool(trials)):
            with st.form("trial_registration_form"):
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    t_session = st.text_input("Session ID:", value="SES-CALIB-01")
                    t_part = st.text_input("Pseudonymous Participant ID:", value="PARTICIPANT-01")
                    t_dev = st.text_input("Device Model:", value="Pixel 7 Pro")
                with col_r2:
                    t_place = st.selectbox("Phone Placement:", ["HAND_HELD", "POCKET", "BAG", "TEXTING", "EAR"], index=0)
                    t_cond = st.selectbox("Walking Condition:", ["NORMAL_PACE", "SLOW_WALK", "BRISK_WALK", "STAIRS_EXCLUDED", "SHUFFLE"], index=0)
                    t_type = st.selectbox("Trial Geometry:", ["STRAIGHT_LINE", "FORWARD_RETURN", "L_SHAPED", "MULTIPLE_TURNS", "SHORT_DISTANCE", "LONG_DISTANCE"], index=0)
                with col_r3:
                    t_dist = st.number_input("Measured Distance (m):", min_value=1.0, max_value=500.0, value=20.0, step=0.5)
                    t_method = st.selectbox("Measurement Method:", ["LASER_RANGEFINDER", "SURVEY_TAPE", "ARCHITECTURAL_BLUEPRINT", "TILED_FLOOR_COUNT"], index=0)
                    t_unc = st.number_input("Measurement Uncertainty (m):", min_value=0.01, max_value=2.0, value=0.05, step=0.01)

                col_r4, col_r5 = st.columns(2)
                with col_r4:
                    t_start_ms = st.number_input("Start Timestamp (epoch ms):", min_value=0, value=1731855600000, step=1000)
                    t_end_ms = st.number_input("End Timestamp (epoch ms):", min_value=0, value=1731855620000, step=1000)
                    t_steps = st.number_input("Manual Step Count (optional):", min_value=0, value=28, step=1)
                with col_r5:
                    t_start_h = st.number_input("Start Reference Heading (deg, optional):", min_value=0.0, max_value=360.0, value=0.0, step=1.0)
                    t_end_h = st.number_input("End Reference Heading (deg, optional):", min_value=0.0, max_value=360.0, value=0.0, step=1.0)
                    t_turns = st.number_input("Number of Turns:", min_value=0, max_value=20, value=0, step=1)

                t_notes = st.text_area("Trial Notes:", "Standard straight-line corridor walk with laser measurement.")
                is_synth = st.checkbox("Mark as Synthetic Test Fixture (strictly distinguished from real trials)", value=False)

                submit_btn = st.form_submit_button("💾 Save Walking Trial", type="primary")
                if submit_btn:
                    req_payload = {
                        "session_id": t_session,
                        "participant_id": t_part,
                        "device_model": t_dev,
                        "phone_placement": t_place,
                        "walking_condition": t_cond,
                        "trial_type": t_type,
                        "start_timestamp_ms": int(t_start_ms),
                        "end_timestamp_ms": int(t_end_ms),
                        "ground_truth": {
                            "measured_distance_m": float(t_dist),
                            "measurement_method": t_method,
                            "measurement_uncertainty_m": float(t_unc),
                            "measured_start_heading_deg": float(t_start_h) if t_start_h > 0 else None,
                            "measured_end_heading_deg": float(t_end_h) if t_end_h > 0 else None,
                            "ground_truth_step_count": int(t_steps) if t_steps > 0 else None,
                            "num_turns": int(t_turns)
                        },
                        "trial_notes": t_notes,
                        "is_synthetic": is_synth
                    }
                    try:
                        saved_trial = client.register_pdr_trial(dataset_id, req_payload)
                        st.success(f"Registered trial '{saved_trial.get('trial_id')}' successfully!")
                        st.rerun()
                    except DatasetApiClientException as err:
                        st.error(f"Failed to register trial: {err}")

        if trials:
            st.markdown("#### Registered Trials Catalog")
            trial_rows = []
            for t in trials:
                trial_rows.append({
                    "Trial ID": t.get("trial_id"),
                    "Type": t.get("trial_type"),
                    "Placement": t.get("phone_placement"),
                    "Pace": t.get("walking_condition"),
                    "Distance (m)": f"{t.get('measured_distance_m', 0.0):.2f}",
                    "Method": t.get("measurement_method"),
                    "Steps": t.get("ground_truth_step_count") or "N/A",
                    "Synthetic?": "Yes" if t.get("is_synthetic") else "No"
                })
            st.dataframe(pd.DataFrame(trial_rows), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Trajectory Drift & 95% Confidence Ellipses
    # -------------------------------------------------------------------------
    with tab_drift:
        st.markdown("### 🗺️ Trajectory Drift & 95% Confidence Ellipses")
        st.markdown(
            r"Visualizes the 2D estimated PDR trajectory alongside ground-truth path reference. "
            r"Error ellipses represent the **rigorous 95% confidence region** derived from spatial covariance "
            r"using the 2-DOF Chi-square quantile $\chi^2_{2, 0.95} = 5.991$ ($a = 2.448\sqrt{\lambda_1}, b = 2.448\sqrt{\lambda_2}$)."
        )

        with st.spinner("Loading PDR trajectory points..."):
            traj_points = client.get_pdr_trajectory(dataset_id)

        if not traj_points:
            st.warning("⚠️ No PDR trajectory points computed for this dataset. Please run PDR in Section 11 first.")
        else:
            df_traj = pd.DataFrame(traj_points)
            xs = df_traj["rel_x_m"].values
            ys = df_traj["rel_y_m"].values

            # Plotly Visualization
            fig = go.Figure()

            # 1. PDR Estimated Path
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines+markers",
                name="Estimated PDR Path",
                line=dict(color="#2563EB", width=3),
                marker=dict(size=6, color="#1D4ED8")
            ))

            # 2. 95% Chi-Square Confidence Ellipses at every 3rd step
            chi2_scale = 5.99146
            for idx in range(0, len(df_traj), max(1, len(df_traj) // 8)):
                row = df_traj.iloc[idx]
                px = row["rel_x_m"]
                py = row["rel_y_m"]
                major = row.get("semi_major_axis_m")
                minor = row.get("semi_minor_axis_m")
                angle_deg = row.get("ellipse_angle_deg", 0.0)

                if major and minor and major > 0 and minor > 0:
                    t_vals = np.linspace(0, 2 * np.pi, 36)
                    rot_rad = math.radians(angle_deg)
                    # Parametric ellipse
                    el_x = px + (major * np.cos(t_vals) * np.cos(rot_rad) - minor * np.sin(t_vals) * np.sin(rot_rad))
                    el_y = py + (major * np.cos(t_vals) * np.sin(rot_rad) + minor * np.sin(t_vals) * np.cos(rot_rad))

                    fig.add_trace(go.Scatter(
                        x=el_x, y=el_y,
                        mode="lines",
                        name="95% Confidence Ellipse" if idx == 0 else None,
                        showlegend=(idx == 0),
                        line=dict(color="rgba(239, 68, 68, 0.4)", width=1, dash="dot"),
                        fill="toself",
                        fillcolor="rgba(239, 68, 68, 0.05)"
                    ))

            # 3. Reference Path (if trials available)
            if trials:
                sel_trial = trials[0]
                t_gt_dist = sel_trial.get("measured_distance_m", 0.0)
                # Nominal reference along Y axis
                fig.add_trace(go.Scatter(
                    x=[0.0, 0.0], y=[0.0, t_gt_dist],
                    mode="lines+markers",
                    name=f"Ground-Truth Reference ({sel_trial.get('trial_id')})",
                    line=dict(color="#10B981", width=3, dash="dash"),
                    marker=dict(size=8, color="#059669")
                ))

            fig.update_layout(
                title="PDR Trajectory Drift vs 95% Spatial Confidence Ellipses",
                xaxis_title="Relative East (m)",
                yaxis_title="Relative North (m)",
                template="plotly_white",
                height=520,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Drift KPI Cards
            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            tot_dist = df_traj["cum_distance_m"].iloc[-1] if "cum_distance_m" in df_traj.columns else 0.0
            end_x = df_traj["rel_x_m"].iloc[-1]
            end_y = df_traj["rel_y_m"].iloc[-1]
            net_disp = math.sqrt(end_x**2 + end_y**2)

            with d_col1:
                st.metric("Total Traversed Distance", f"{tot_dist:.2f} m")
            with d_col2:
                st.metric("Net Displacement", f"{net_disp:.2f} m")
            with d_col3:
                final_major = df_traj["semi_major_axis_m"].iloc[-1] if "semi_major_axis_m" in df_traj.columns else 0.0
                st.metric("Final 95% Semi-Major Axis", f"±{final_major:.2f} m")
            with d_col4:
                final_minor = df_traj["semi_minor_axis_m"].iloc[-1] if "semi_minor_axis_m" in df_traj.columns else 0.0
                st.metric("Final 95% Semi-Minor Axis", f"±{final_minor:.2f} m")

    # -------------------------------------------------------------------------
    # TAB 4: Step & Heading Validation
    # -------------------------------------------------------------------------
    with tab_step_head:
        st.markdown("### 🎯 Step Detection & Heading Evaluation")
        st.markdown("Run validation suite against selected ground-truth trials to generate statistical precision and drift reports.")

        if not trials:
            st.warning("Please register at least one walking trial to execute statistical validation.")
        else:
            sel_trial_id = st.selectbox(
                "Select Trial to Validate:",
                [t["trial_id"] for t in trials],
                format_func=lambda x: f"{x} - {next((t['trial_type'] + ' (' + str(t['measured_distance_m']) + 'm)' for t in trials if t['trial_id'] == x), '')}"
            )
            sel_prof_id = st.selectbox(
                "Select Calibration Profile for Evaluation:",
                [p["profile_id"] for p in profiles],
                format_func=lambda x: f"{x} - {next((p['profile_name'] for p in profiles if p['profile_id'] == x), '')}"
            )

            if st.button("📊 Evaluate PDR Performance", type="primary"):
                with st.spinner("Evaluating step classification, heading errors, and spatial drift..."):
                    try:
                        rep_list = client.evaluate_pdr(dataset_id, {
                            "trial_ids": [sel_trial_id],
                            "calibration_profile_id": sel_prof_id
                        })
                        if rep_list:
                            st.session_state["active_eval_report"] = rep_list[0]
                            st.success("Evaluation completed successfully!")
                    except DatasetApiClientException as err:
                        st.error(f"Evaluation failed: {err}")

        # Display active report if available
        active_rep = st.session_state.get("active_eval_report")
        if active_rep:
            st.markdown("#### Evaluation Results")
            sm = active_rep.get("step_metrics", {})
            tm = active_rep.get("trajectory_metrics", {})
            hm = active_rep.get("heading_metrics", {})
            um = active_rep.get("uncertainty_metrics", {})

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Step F1-Score", f"{sm.get('f1_score', 0.0):.3f}")
                st.caption(f"Precision: {sm.get('precision', 0.0):.2f} | Recall: {sm.get('recall', 0.0):.2f}")
            with col_m2:
                st.metric("Distance Error", f"{tm.get('absolute_distance_error_m', 0.0):.2f} m")
                st.caption(f"Rel Error: {tm.get('relative_distance_error', 0.0)*100.0:.1f}%")
            with col_m3:
                st.metric("Endpoint Drift", f"{tm.get('endpoint_error_m', 0.0):.2f} m")
                st.caption(f"Drift Rate: {tm.get('drift_per_meter', 0.0):.3f} m/m")
            with col_m4:
                mean_h_err = hm.get("mean_absolute_error_deg") if hm else None
                st.metric("Mean Heading Error", f"{mean_h_err:.1f}°" if mean_h_err is not None else "N/A")
                h_drift = hm.get("drift_rate_deg_per_min", 0.0) if hm else 0.0
                st.caption(f"Heading Drift: {h_drift:.1f}°/min")

            if um:
                st.markdown("##### 📐 Spatial Uncertainty & 95% Ellipse Verification")
                u_in = um.get("ground_truth_within_95_ellipse", False)
                u_badge = "✅ Ground-Truth Endpoint inside 95% Ellipse" if u_in else "❌ Endpoint outside 95% Ellipse (Underestimated Covariance)"
                st.info(f"**Verification**: {u_badge}")

            st.download_button(
                "📥 Download Evaluation Report (JSON)",
                data=json.dumps(active_rep, indent=2),
                file_name=f"{active_rep.get('report_id')}.json",
                mime="application/json"
            )
