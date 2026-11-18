"""Streamlit component for Site-Specific Wi-Fi Radio Map & Fingerprint Localization (Task 23)."""

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


def render_wifi_localization_view(client: DatasetApiClient, dataset_id: Optional[str]) -> None:
    """Renders the comprehensive Site-Specific Wi-Fi Localization & Radio Map view."""
    st.markdown("## 📶 Site-Specific Wi-Fi Radio Map & Fingerprint Localization")
    st.caption(
        "Build reference radio maps from real Android survey sweeps, inspect AP signal statistics, "
        "localize fingerprints with Weighted k-NN or Probabilistic Gaussian matching, and run "
        "leakage-resistant session-partitioned evaluations (Task 23)."
    )

    if not dataset_id:
        st.info("💡 Please select an active survey dataset in **1. Overview & Catalog** first.")
        return

    # Derive default site_id from active dataset or session
    default_site_id = f"SITE-{dataset_id[:8].upper()}"

    # Top-level controls
    col_site, col_ver, col_refresh = st.columns([2, 1, 1])
    with col_site:
        site_id = st.text_input("Active Physical Site ID", value=default_site_id, key="wifi_survey_site_id")
    with col_ver:
        map_version = st.text_input("Radio Map Version", value="v1", key="wifi_survey_map_version")
    with col_refresh:
        st.write("")
        st.write("")
        st.button("🔄 Refresh Data", key="wifi_refresh_btn")

    # Fetch existing locations and radio maps
    try:
        registered_locations = client.list_wifi_survey_locations(dataset_id, site_id)
        available_maps = client.list_wifi_radio_maps(dataset_id, site_id=site_id)
        current_radio_map = client.get_wifi_radio_map(dataset_id, site_id, version=map_version)
    except DatasetApiClientException as e:
        st.error(f"Error communicating with Wi-Fi Survey backend: {e}")
        return

    # Top Status Bar
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        st.metric("Reference Locations", len(registered_locations))
    with s_col2:
        st.metric("Compiled Radio Maps", len(available_maps))
    with s_col3:
        n_aps = len(current_radio_map.get("all_bssids", [])) if current_radio_map else 0
        st.metric("Unique APs in Active Map", n_aps)
    with s_col4:
        total_sw = current_radio_map.get("metadata", {}).get("total_sweeps", 0) if current_radio_map else 0
        st.metric("Survey Sweeps in Map", total_sw)

    # Main Tabs
    tab_reg, tab_map, tab_loc, tab_eval = st.tabs([
        "🏢 Reference Locations & Survey Registry",
        "📡 Radio Map Compiler & Signal Profiles",
        "🎯 Live Fingerprint Localizer Testbed",
        "📊 Leakage-Resistant Evaluation & Confusion Matrix"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Reference Locations Registry
    # -------------------------------------------------------------------------
    with tab_reg:
        st.markdown("### 🏢 Site Reference Locations")
        st.caption(
            "Designated survey landmarks with verified physical coordinates (x, y), building, and floor labels."
        )

        col_table, col_form = st.columns([3, 2])

        with col_table:
            if registered_locations:
                df_locs = pd.DataFrame(registered_locations)
                display_cols = [c for c in ["location_id", "building_id", "floor_id", "room_id", "x_m", "y_m", "is_verified", "description"] if c in df_locs.columns]
                st.dataframe(df_locs[display_cols], use_container_width=True, hide_index=True)

                # 2D Map of reference locations
                fig_locs = px.scatter(
                    df_locs,
                    x="x_m",
                    y="y_m",
                    color=df_locs["floor_id"].astype(str),
                    text="location_id",
                    title=f"Reference Locations Spatial Layout ({site_id})",
                    labels={"color": "Floor", "x_m": "Local X (m)", "y_m": "Local Y (m)"},
                    hover_data=["building_id", "room_id", "description"],
                )
                fig_locs.update_traces(marker=dict(size=14, symbol="diamond"), textposition="top center")
                fig_locs.update_layout(xaxis_title="Local X Coordinate (meters)", yaxis_title="Local Y Coordinate (meters)")
                st.plotly_chart(fig_locs, use_container_width=True)
            else:
                st.info(
                    f"No reference locations registered for site '{site_id}' yet. "
                    "Register reference points below or auto-discover them when building a radio map from dataset."
                )

        with col_form:
            st.markdown("#### ➕ Register New Reference Location")
            with st.form("register_location_form"):
                new_loc_id = st.text_input("Location ID", value=f"LOC-{len(registered_locations)+1:02d}")
                new_bldg = st.text_input("Building ID", value="BLDG-1")
                new_floor = st.number_input("Floor Number", value=0, step=1)
                new_room = st.text_input("Room / Zone ID (optional)", value="")
                c_x, c_y = st.columns(2)
                with c_x:
                    new_x = st.number_input("X (meters)", value=0.0, step=1.0)
                with c_y:
                    new_y = st.number_input("Y (meters)", value=0.0, step=1.0)
                new_desc = st.text_area("Description / Landmark Notes", value="Survey ground landmark")
                new_verified = st.checkbox("Physically Verified Coordinates", value=True)

                submitted = st.form_submit_button("Save Reference Location")
                if submitted:
                    loc_payload = {
                        "location_id": new_loc_id.strip(),
                        "site_id": site_id,
                        "building_id": new_bldg.strip(),
                        "floor_id": int(new_floor),
                        "room_id": new_room.strip() if new_room.strip() else None,
                        "x_m": float(new_x),
                        "y_m": float(new_y),
                        "description": new_desc.strip(),
                        "is_verified": new_verified,
                        "metadata": {},
                    }
                    try:
                        res = client.register_wifi_survey_location(dataset_id, loc_payload)
                        st.success(f"Registered reference location '{res.get('location_id')}' successfully!")
                        st.rerun()
                    except DatasetApiClientException as e:
                        st.error(f"Failed to register location: {e}")

    # -------------------------------------------------------------------------
    # TAB 2: Radio Map Compiler & Signal Profiles
    # -------------------------------------------------------------------------
    with tab_map:
        st.markdown("### 📡 Radio Map Compiler & Access Point Statistics")
        st.caption(
            "Compile versioned radio maps by aggregating multi-sweep Wi-Fi scans into mean/median/std signal distributions and visibility rates."
        )

        with st.expander("⚙️ Radio Map Compilation Settings", expanded=not bool(current_radio_map)):
            with st.form("compile_radio_map_form"):
                b_c1, b_c2, b_c3 = st.columns(3)
                with b_c1:
                    b_site = st.text_input("Site ID", value=site_id)
                    b_ver = st.text_input("Version Tag", value=map_version)
                with b_c2:
                    b_penalty = st.number_input("Missing AP Floor RSSI (dBm)", value=-105.0, step=1.0, min_value=-120.0, max_value=-80.0)
                    b_min_sweeps = st.number_input("Min Sweeps per Location", value=1, min_value=1, step=1)
                with b_c3:
                    b_min_vis = st.slider("Min AP Visibility Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
                    b_std_floor = st.number_input("Std RSSI Floor (dBm)", value=2.0, min_value=0.5, step=0.5)

                b_auto_reg = st.checkbox(
                    "Auto-register reference locations from dataset annotations / metadata if missing",
                    value=True,
                )
                b_desc = st.text_input("Description", value=f"Radio map for {site_id} from dataset {dataset_id}")

                build_submitted = st.form_submit_button("🔨 Compile Radio Map")
                if build_submitted:
                    payload = {
                        "site_id": b_site,
                        "version": b_ver,
                        "missing_penalty_rssi": float(b_penalty),
                        "min_sweeps_per_location": int(b_min_sweeps),
                        "min_visibility_rate": float(b_min_vis),
                        "std_rssi_floor": float(b_std_floor),
                        "description": b_desc,
                        "auto_register_locations": b_auto_reg,
                    }
                    with st.spinner("Compiling site radio map from survey observations..."):
                        try:
                            compiled = client.build_wifi_radio_map(dataset_id, payload)
                            st.success(
                                f"Radio map '{b_ver}' for site '{b_site}' compiled successfully! "
                                f"({compiled['metadata']['total_locations']} locations, "
                                f"{compiled['metadata']['unique_bssids']} unique BSSIDs)"
                            )
                            st.rerun()
                        except DatasetApiClientException as e:
                            st.error(f"Failed to compile radio map: {e}")

        # Radio Map Profile Inspector
        if current_radio_map:
            st.markdown(f"#### 📻 Radio Map Summary: `{site_id}` ({map_version})")
            meta = current_radio_map.get("metadata", {})
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Total Locations", meta.get("total_locations", 0))
            m_c2.metric("Total Sweeps", meta.get("total_sweeps", 0))
            m_c3.metric("Unique BSSIDs", meta.get("unique_bssids", 0))
            m_c4.metric("Missing AP Penalty", f"{meta.get('missing_penalty_rssi', -105.0)} dBm")

            profiles = current_radio_map.get("profiles", {})
            if profiles:
                loc_options = list(profiles.keys())
                selected_loc_id = st.selectbox("Select Reference Location to Inspect", loc_options)
                selected_prof = profiles[selected_loc_id]

                ap_stats = selected_prof.get("ap_stats", {})
                if ap_stats:
                    stats_records = []
                    for bssid, s in ap_stats.items():
                        stats_records.append({
                            "BSSID": bssid,
                            "SSID": s.get("ssid") or "Hidden",
                            "Visibility Rate": s.get("visibility_rate", 0.0),
                            "Mean RSSI (dBm)": s.get("mean_rssi"),
                            "Median RSSI (dBm)": s.get("median_rssi"),
                            "Std RSSI (dBm)": s.get("std_rssi"),
                            "Min RSSI (dBm)": s.get("min_rssi"),
                            "Max RSSI (dBm)": s.get("max_rssi"),
                            "Band": s.get("channel_band") or "unknown",
                        })

                    df_stats = pd.DataFrame(stats_records).sort_values(by="Mean RSSI (dBm)", ascending=False)

                    p_c1, p_c2 = st.columns([1, 1])
                    with p_c1:
                        # Top APs Mean RSSI
                        top_aps = df_stats.head(15)
                        fig_rssi = px.bar(
                            top_aps,
                            x="Mean RSSI (dBm)",
                            y="BSSID",
                            color="Visibility Rate",
                            orientation="h",
                            title=f"Top 15 APs by Mean RSSI at {selected_loc_id}",
                            color_continuous_scale="Viridis",
                            hover_data=["SSID", "Std RSSI (dBm)", "Band"],
                        )
                        fig_rssi.update_layout(yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_rssi, use_container_width=True)

                    with p_c2:
                        # AP Visibility Rate Histogram
                        fig_vis = px.histogram(
                            df_stats,
                            x="Visibility Rate",
                            nbins=10,
                            title=f"AP Visibility Distribution at {selected_loc_id}",
                            labels={"Visibility Rate": "Observation Visibility Rate (0.0 - 1.0)"},
                        )
                        st.plotly_chart(fig_vis, use_container_width=True)

                    with st.expander(f"📋 Full AP Statistics Table ({len(df_stats)} APs)"):
                        st.dataframe(df_stats, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No AP statistics recorded at {selected_loc_id}.")
        else:
            st.info(f"No compiled radio map found for {site_id} ({map_version}). Use the compiler above.")

    # -------------------------------------------------------------------------
    # TAB 3: Live Fingerprint Localizer Testbed
    # -------------------------------------------------------------------------
    with tab_loc:
        st.markdown("### 🎯 Live Fingerprint Localizer Testbed")
        st.caption(
            "Test indoor positioning from input Wi-Fi query scans. Compare Weighted k-NN and Probabilistic Gaussian matching."
        )

        if not current_radio_map:
            st.warning("Please compile a radio map in the compiler tab first before running live localization.")
        else:
            # Algorithm & Parameters Controls
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                algo_choice = st.selectbox(
                    "Matching Algorithm",
                    ["weighted_knn", "probabilistic"],
                    format_func=lambda x: "Weighted k-NN" if x == "weighted_knn" else "Probabilistic Gaussian",
                )
            with p_col2:
                k_val = st.slider("Nearest Neighbors (k)", min_value=1, max_value=10, value=3)
            with p_col3:
                metric_choice = st.selectbox(
                    "Distance Metric",
                    ["euclidean", "manhattan"],
                    disabled=(algo_choice == "probabilistic"),
                )
            with p_col4:
                st.write("")
                st.write("")
                query_source = st.radio(
                    "Query Fingerprint Source",
                    ["Simulate from Radio Map Location", "Custom Manual AP Inputs"],
                    horizontal=True,
                )

            query_observations: Dict[str, float] = {}

            if query_source == "Simulate from Radio Map Location":
                profiles = current_radio_map.get("profiles", {})
                sim_loc = st.selectbox("Select Source Reference Location to Simulate Query From", list(profiles.keys()))
                noise_dbm = st.slider("Simulated Measurement Gaussian Noise (dBm)", min_value=0.0, max_value=8.0, value=1.5, step=0.5)

                if sim_loc and sim_loc in profiles:
                    base_vec = profiles[sim_loc].get("mean_rssi_vector", {})
                    # Add simulated noise
                    for b, mean_r in base_vec.items():
                        noisy_r = float(mean_r + np.random.normal(0, noise_dbm))
                        query_observations[b] = round(noisy_r, 2)
                    st.caption(f"Generated query fingerprint with {len(query_observations)} APs from location '{sim_loc}'.")
            else:
                st.markdown("#### Enter Manual AP Observations")
                custom_text = st.text_area(
                    "Enter one AP per line (Format: `BSSID, RSSI_DBM` or JSON `{\"bssid\": rssi}`)",
                    value="""00:11:22:33:44:55, -65.0
00:11:22:33:44:56, -72.0
00:11:22:33:44:57, -81.0""",
                    height=100,
                )
                if custom_text.strip():
                    try:
                        if custom_text.strip().startswith("{"):
                            query_observations = json.loads(custom_text.strip())
                        else:
                            for line in custom_text.strip().split("\n"):
                                parts = line.split(",")
                                if len(parts) >= 2:
                                    query_observations[parts[0].strip()] = float(parts[1].strip())
                    except Exception as e:
                        st.error(f"Error parsing manual inputs: {e}")

            if st.button("🚀 Localize Fingerprint", type="primary"):
                if not query_observations:
                    st.error("No valid AP observations provided in query.")
                else:
                    payload = {
                        "site_id": site_id,
                        "version": map_version,
                        "observations": query_observations,
                        "algorithm": algo_choice,
                        "k": k_val,
                        "distance_metric": metric_choice,
                    }
                    try:
                        pred = client.localize_wifi_fingerprint(dataset_id, payload)
                        st.markdown("---")
                        st.markdown("#### 📍 Localization Results")

                        # Prediction KPI row
                        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
                        with res_c1:
                            st.metric("Predicted Building", pred.get("predicted_building_id"))
                        with res_c2:
                            st.metric("Predicted Floor", f"Floor {pred.get('predicted_floor_id')}")
                        with res_c3:
                            st.metric(
                                "Predicted (X, Y)",
                                f"({pred.get('predicted_x_m'):.1f}m, {pred.get('predicted_y_m'):.1f}m)",
                            )
                        with res_c4:
                            conf = pred.get("confidence", 0.0)
                            conf_badge = "🟢" if conf >= 0.6 else "🟡" if conf >= 0.35 else "🔴"
                            st.metric("Confidence Score", f"{conf_badge} {conf:.2f}")

                        # Low confidence or missing AP warnings
                        warnings = pred.get("warnings", [])
                        for w in warnings:
                            st.warning(f"⚠️ {w}")

                        st.caption(
                            f"Matched {pred.get('matched_ap_count')} of {pred.get('total_query_aps')} "
                            f"query APs against site radio map."
                        )

                        # Candidate Matches Table
                        cand_list = pred.get("candidate_matches", [])
                        if cand_list:
                            st.markdown("##### 🏆 Top Candidate Reference Locations")
                            df_cand = pd.DataFrame(cand_list)
                            st.dataframe(df_cand, use_container_width=True, hide_index=True)

                            # 2D Spatial Visualization of Prediction & Candidates
                            loc_dict = current_radio_map.get("locations", {})
                            all_loc_records = [
                                {
                                    "location_id": l["location_id"],
                                    "x_m": l["x_m"],
                                    "y_m": l["y_m"],
                                    "floor_id": l["floor_id"],
                                    "type": "Reference Landmark",
                                }
                                for l in loc_dict.values()
                            ]
                            df_map_all = pd.DataFrame(all_loc_records)

                            fig_2d = go.Figure()
                            # Reference locations
                            fig_2d.add_trace(
                                go.Scatter(
                                    x=df_map_all["x_m"],
                                    y=df_map_all["y_m"],
                                    mode="markers+text",
                                    marker=dict(size=10, color="lightblue", symbol="circle"),
                                    text=df_map_all["location_id"],
                                    textposition="bottom center",
                                    name="Reference Landmarks",
                                )
                            )

                            # Candidate matches
                            fig_2d.add_trace(
                                go.Scatter(
                                    x=df_cand["x_m"],
                                    y=df_cand["y_m"],
                                    mode="markers+text",
                                    marker=dict(size=14, color="orange", symbol="diamond"),
                                    text=df_cand["location_id"],
                                    textposition="top center",
                                    name="Top Candidates",
                                )
                            )

                            # Estimated Location
                            est_x = pred.get("predicted_x_m", 0.0)
                            est_y = pred.get("predicted_y_m", 0.0)
                            fig_2d.add_trace(
                                go.Scatter(
                                    x=[est_x],
                                    y=[est_y],
                                    mode="markers+text",
                                    marker=dict(size=18, color="red", symbol="star"),
                                    text=["★ ESTIMATED POSITION"],
                                    textposition="top right",
                                    name="Estimated Position",
                                )
                            )

                            # Draw connection lines from estimate to top candidates
                            for _, c in df_cand.iterrows():
                                fig_2d.add_trace(
                                    go.Scatter(
                                        x=[est_x, c["x_m"]],
                                        y=[est_y, c["y_m"]],
                                        mode="lines",
                                        line=dict(color="rgba(255, 100, 0, 0.4)", dash="dash"),
                                        showlegend=False,
                                    )
                                )

                            fig_2d.update_layout(
                                title="2D Indoor Positioning Estimate vs Candidates",
                                xaxis_title="Local X Coordinate (meters)",
                                yaxis_title="Local Y Coordinate (meters)",
                                height=500,
                            )
                            st.plotly_chart(fig_2d, use_container_width=True)

                    except DatasetApiClientException as e:
                        st.error(f"Localization failed: {e}")

    # -------------------------------------------------------------------------
    # TAB 4: Leakage-Resistant Evaluation & Confusion Matrix
    # -------------------------------------------------------------------------
    with tab_eval:
        st.markdown("### 📊 Leakage-Resistant Evaluation & Confusion Matrix")
        st.caption(
            "Rigorous offline validation enforcing session-level train/test separation so repeated stationary scans "
            "never leak between training radio maps and test evaluation sets."
        )

        with st.form("offline_evaluation_form"):
            e_c1, e_c2, e_c3 = st.columns(3)
            with e_c1:
                eval_algo = st.selectbox(
                    "Evaluation Algorithm",
                    ["weighted_knn", "probabilistic"],
                    format_func=lambda x: "Weighted k-NN" if x == "weighted_knn" else "Probabilistic Gaussian",
                )
            with e_c2:
                eval_k = st.slider("Top-k Nearest Neighbors", min_value=1, max_value=10, value=3)
            with e_c3:
                eval_metric = st.selectbox("Distance Metric", ["euclidean", "manhattan"])

            e_ratio = st.slider("Holdout Test Session Fraction", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
            eval_submitted = st.form_submit_button("⚡ Run Leakage-Resistant Evaluation")

        if eval_submitted:
            eval_payload = {
                "site_id": site_id,
                "algorithm": eval_algo,
                "k": eval_k,
                "distance_metric": eval_metric,
                "test_ratio": float(e_ratio),
            }
            with st.spinner("Executing session-partitioned offline evaluation..."):
                try:
                    eval_res = client.evaluate_wifi_localization(dataset_id, eval_payload)
                    metrics = eval_res.get("metrics", {})

                    st.success("Evaluation completed successfully!")

                    # Metric Row 1: Classification Accuracies
                    st.markdown("#### 🎯 Classification Accuracies")
                    acc_c1, acc_c2, acc_c3, acc_c4 = st.columns(4)
                    acc_c1.metric("Building Accuracy", f"{metrics.get('building_accuracy', 0.0)*100:.1f}%")
                    acc_c2.metric("Floor Accuracy", f"{metrics.get('floor_accuracy', 0.0)*100:.1f}%")
                    acc_c3.metric("Building-Floor Exact", f"{metrics.get('building_floor_accuracy', 0.0)*100:.1f}%")
                    acc_c4.metric("Location ID Accuracy", f"{metrics.get('location_id_accuracy', 0.0)*100:.1f}%")

                    # Metric Row 2: Coordinate Errors
                    st.markdown("#### 📐 Coordinate Estimation Errors (Meters)")
                    err_c1, err_c2, err_c3, err_c4 = st.columns(4)
                    err_c1.metric("Mean Error", f"{metrics.get('mean_coordinate_error_m', 0.0):.2f} m")
                    err_c2.metric("Median Error", f"{metrics.get('median_coordinate_error_m', 0.0):.2f} m")
                    err_c3.metric("95th Percentile", f"{metrics.get('p95_coordinate_error_m', 0.0):.2f} m")
                    err_c4.metric("Low Confidence Rate", f"{metrics.get('low_confidence_rate', 0.0)*100:.1f}%")

                    # Sessions Partition Details
                    st.caption(
                        f"Evaluated {metrics.get('total_test_samples')} test sweeps. "
                        f"Train Sessions: {metrics.get('train_session_ids')} | "
                        f"Test Sessions: {metrics.get('test_session_ids')}"
                    )

                    # Floor Confusion Matrix
                    conf_matrix = metrics.get("confusion_matrix_floors", {})
                    if conf_matrix:
                        st.markdown("#### 🔄 Floor Classification Confusion Matrix")
                        all_floors = sorted(list(conf_matrix.keys()))
                        matrix_data = []
                        for tf in all_floors:
                            row = []
                            for pf in all_floors:
                                row.append(conf_matrix[tf].get(pf, 0))
                            matrix_data.append(row)

                        fig_conf = px.imshow(
                            matrix_data,
                            x=all_floors,
                            y=all_floors,
                            labels=dict(x="Predicted Floor", y="True Floor", color="Count"),
                            text_auto=True,
                            color_continuous_scale="Blues",
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)

                    # Per-Location Error Breakdown
                    per_loc = metrics.get("per_location_errors", [])
                    if per_loc:
                        st.markdown("#### 📍 Per-Reference-Location Error Breakdown")
                        df_per_loc = pd.DataFrame(per_loc)
                        st.dataframe(df_per_loc, use_container_width=True, hide_index=True)

                        fig_loc_err = px.bar(
                            df_per_loc,
                            x="location_id",
                            y="mean_error_m",
                            color="sample_count",
                            title="Mean Euclidean Error by Reference Location",
                            labels={"location_id": "Reference Location", "mean_error_m": "Mean Error (meters)"},
                        )
                        st.plotly_chart(fig_loc_err, use_container_width=True)

                except DatasetApiClientException as e:
                    st.error(f"Evaluation failed: {e}")
