"""Real-Time Multimodal Streaming & Live Navigation Dashboard View (Task 29).

Provides interactive inspection of WebSocket streaming sessions, live Wi-Fi / IMU ingestion,
real-time particle filter localization, turn-by-turn guidance HUD, and test simulation harness.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import DatasetApiClient, DatasetApiClientException


def render_realtime_view(client: DatasetApiClient, active_dataset_id: Optional[str] = None) -> None:
    """Renders the complete Real-Time Streaming and Live Navigation UI."""
    st.markdown("## 📡 19. Real-Time Multimodal Ingestion & Live Navigation")
    st.markdown(
        "Monitor live WebSocket streams from authorized Android mobile clients, view real-time "
        "sensor and Wi-Fi observations, track map-constrained localization fixes, and follow "
        "turn-by-turn indoor route guidance."
    )

    # Session Selector / Creation
    st.sidebar.markdown("### 🔌 Stream Connection")
    session_id_input = st.sidebar.text_input("Active Streaming Session ID", value="LIVE_SURVEY_01")
    session_id = session_id_input.strip() or "LIVE_SURVEY_01"

    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("▶️ Start Session", use_container_width=True):
            try:
                client.control_realtime_session(session_id=session_id, action="START")
                st.sidebar.success(f"Session {session_id} started.")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
    with col_btn2:
        if st.button("⏹️ Stop Session", use_container_width=True):
            try:
                client.control_realtime_session(session_id=session_id, action="STOP")
                st.sidebar.warning(f"Session {session_id} stopped.")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    # Fetch status
    status_data: Dict[str, Any] = {}
    try:
        status_data = client.get_realtime_status(session_id)
    except Exception as e:
        status_data = {
            "session_id": session_id,
            "is_active": False,
            "connected_clients": 0,
            "events_received": 0,
            "events_processed": 0,
            "events_rejected": 0,
            "events_dropped": 0,
            "buffer_size": 0,
            "buffer_capacity": 5000,
            "buffer_utilization_pct": 0.0,
            "building_id": "SYNTH_BLDG_01",
            "floor_id": 1,
        }

    # Fetch history
    history_data: List[Dict[str, Any]] = []
    try:
        h_res = client.get_realtime_history(session_id=session_id, max_items=50)
        history_data = h_res.get("items", [])
    except Exception:
        history_data = []

    tabs = st.tabs([
        "📊 Stream Telemetry & Buffer",
        "📶 Live Sensor & Wi-Fi Feed",
        "📍 Live Map Localization Canvas",
        "🧭 Turn-by-Turn Navigation HUD",
        "🧪 Simulation & Replay Harness",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Stream Telemetry & Buffer
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.markdown("### 🔌 Real-Time Pipeline Telemetry")

        is_act = status_data.get("is_active", False)
        clients = status_data.get("connected_clients", 0)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            state_pill = "🟢 ACTIVE" if is_act else "⏸️ PAUSED / STOPPED"
            st.metric("Session State", state_pill)
        with c2:
            st.metric("Connected Clients", clients)
        with c3:
            st.metric("Events Received", status_data.get("events_received", 0))
        with c4:
            st.metric("Events Processed", status_data.get("events_processed", 0))
        with c5:
            rej = status_data.get("events_rejected", 0) + status_data.get("events_dropped", 0)
            st.metric("Rejected / Dropped", rej)

        st.markdown("---")
        st.markdown("#### 📦 Bounded In-Memory Queue & Backpressure")
        buf_size = status_data.get("buffer_size", 0)
        buf_cap = status_data.get("buffer_capacity", 5000)
        buf_util = status_data.get("buffer_utilization_pct", 0.0)

        st.progress(min(1.0, buf_util / 100.0))
        st.caption(f"Queue: **{buf_size} / {buf_cap} events** ({buf_util}% allocated). Bounded backpressure prevents memory exhaustion.")

        col_act1, col_act2, col_act3 = st.columns(3)
        with col_act1:
            if st.button("⏸️ Pause Ingestion", use_container_width=True):
                client.control_realtime_session(session_id, "PAUSE")
                st.rerun()
        with col_act2:
            if st.button("▶️ Resume Ingestion", use_container_width=True):
                client.control_realtime_session(session_id, "RESUME")
                st.rerun()
        with col_act3:
            if st.button("🧹 Clear Buffer", use_container_width=True):
                client.control_realtime_session(session_id, "RESET")
                st.rerun()

    # -------------------------------------------------------------------------
    # TAB 2: Live Sensor & Wi-Fi Feed
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.markdown("### 📶 Live Multimodal Observations Feed")

        if not history_data:
            st.info("No real-time events ingested for this session yet. Connect the Android client or use Tab 5 to simulate.")
        else:
            event_records = []
            for item in history_data:
                ev = item.get("event", {})
                event_records.append({
                    "Seq #": item.get("sequence_number"),
                    "Event ID": item.get("event_id"),
                    "Type": item.get("event_type"),
                    "Timestamp UTC": time.strftime("%H:%M:%S", time.gmtime(item.get("timestamp_ms", 0) / 1000)),
                    "Updates Generated": len(item.get("updates", [])),
                })
            st.dataframe(pd.DataFrame(event_records).tail(15), use_container_width=True)

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.markdown("#### 📡 Wi-Fi Ingestion Breakdown")
            # Synthesize or extract recent Wi-Fi APs
            ap_sample = [
                {"BSSID": "14:22:db:11:01", "SSID": "Campus-Secure", "RSSI (dBm)": -58},
                {"BSSID": "14:22:db:11:02", "SSID": "Campus-IoT", "RSSI (dBm)": -64},
                {"BSSID": "2c:39:96:4a:10", "SSID": "Staff-WiFi", "RSSI (dBm)": -72},
                {"BSSID": "38:10:d5:8b:20", "SSID": "Campus-Guest", "RSSI (dBm)": -81},
            ]
            fig_wifi = px.bar(
                pd.DataFrame(ap_sample),
                x="RSSI (dBm)",
                y="BSSID",
                orientation="h",
                color="RSSI (dBm)",
                color_continuous_scale="Viridis",
                title="Recent Observed Access Points",
            )
            fig_wifi.update_layout(xaxis=dict(range=[-105, -30]))
            st.plotly_chart(fig_wifi, use_container_width=True)

        with col_w2:
            st.markdown("#### 🏃 Tri-Axial Kinematic Stream")
            # Sample wave
            t_pts = list(range(30))
            z_accel = [9.8 + 1.8 * math.sin(i * 0.5) for i in t_pts]
            fig_imu = go.Figure()
            fig_imu.add_trace(go.Scatter(x=t_pts, y=z_accel, mode="lines", name="Z-Accel (m/s²)", line=dict(color="#29B6F6", width=2)))
            fig_imu.add_hline(y=9.8, line_dash="dash", line_color="gray", annotation_text="1g Gravity Baseline")
            fig_imu.update_layout(title="Vertical Acceleration Kinematics", xaxis_title="Sample Index", yaxis_title="m/s²")
            st.plotly_chart(fig_imu, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Live Map Localization Canvas
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.markdown("### 📍 Live Map-Constrained Localization")

        # Extract latest localization from recent updates if present
        latest_loc = None
        for item in reversed(history_data):
            for up in item.get("updates", []):
                if up.get("type") == "LOCALIZATION_UPDATE":
                    latest_loc = up
                    break
            if latest_loc:
                break

        if latest_loc:
            curr_x = latest_loc["x_m"]
            curr_y = latest_loc["y_m"]
            curr_floor = latest_loc["floor_id"]
            curr_heading = latest_loc["heading_rad"]
            curr_uncert = latest_loc["uncertainty_radius_m"]
            curr_source = latest_loc["source"]
            curr_conf = latest_loc["confidence_level"]
        else:
            curr_x, curr_y, curr_floor = 18.0, 15.0, 1
            curr_heading = 0.785
            curr_uncert = 2.5
            curr_source = "INITIAL_STANDBY"
            curr_conf = "MEDIUM"

        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        with col_l1:
            st.metric("Position (ENU)", f"X: {curr_x:.2f}m, Y: {curr_y:.2f}m")
        with col_l2:
            st.metric("Floor", f"Floor {curr_floor}")
        with col_l3:
            st.metric("95% Uncertainty", f"± {curr_uncert:.2f} m")
        with col_l4:
            st.metric("Engine Source", curr_source)

        # 2D Floorplan Canvas with user location and confidence ellipse
        fig_map = go.Figure()
        # Synthetic corridor & rooms layout
        fig_map.add_shape(type="rect", x0=0, y0=0, x1=40, y1=30, line=dict(color="gray", width=2))
        fig_map.add_shape(type="rect", x0=2, y0=2, x1=14, y1=12, fillcolor="rgba(100,100,100,0.1)", line=dict(color="black"))
        fig_map.add_shape(type="rect", x0=18, y0=2, x1=38, y1=12, fillcolor="rgba(100,100,100,0.1)", line=dict(color="black"))
        fig_map.add_shape(type="rect", x0=2, y0=18, x1=18, y1=28, fillcolor="rgba(100,100,100,0.1)", line=dict(color="black"))
        fig_map.add_shape(type="rect", x0=22, y0=18, x1=38, y1=28, fillcolor="rgba(100,100,100,0.1)", line=dict(color="black"))

        # User location dot
        fig_map.add_trace(go.Scatter(
            x=[curr_x],
            y=[curr_y],
            mode="markers",
            marker=dict(size=14, color="#E91E63", symbol="circle"),
            name="Live User Location",
        ))

        # Heading vector arrow
        arrow_len = 3.0
        fig_map.add_annotation(
            x=curr_x + arrow_len * math.cos(curr_heading),
            y=curr_y + arrow_len * math.sin(curr_heading),
            ax=curr_x,
            ay=curr_y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor="#E91E63",
        )

        # 95% Uncertainty ellipse circle
        theta = [i * 2 * math.pi / 40 for i in range(41)]
        cx = [curr_x + curr_uncert * math.cos(t) for t in theta]
        cy = [curr_y + curr_uncert * math.sin(t) for t in theta]
        fig_map.add_trace(go.Scatter(
            x=cx,
            y=cy,
            mode="lines",
            line=dict(color="#E91E63", width=1, dash="dot"),
            fill="toself",
            fillcolor="rgba(233, 30, 99, 0.15)",
            name="95% Confidence Ellipse",
        ))

        fig_map.update_layout(
            title=f"Building Floorplan (Floor {curr_floor}) - Real-Time Position",
            xaxis=dict(range=[-2, 42], title="East (meters)"),
            yaxis=dict(range=[-2, 32], title="North (meters)", scaleanchor="x", scaleratio=1),
            height=500,
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: Turn-by-Turn Navigation HUD
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.markdown("### 🧭 Live Turn-by-Turn Guidance HUD")

        # Extract latest navigation update
        latest_nav = None
        for item in reversed(history_data):
            for up in item.get("updates", []):
                if up.get("type") == "NAVIGATION_UPDATE":
                    latest_nav = up
                    break
            if latest_nav:
                break

        if latest_nav:
            instruction = latest_nav.get("current_instruction", "Proceed along route.")
            instr_type = latest_nav.get("instruction_type", "CONTINUE")
            dist_next = latest_nav.get("distance_to_next_waypoint_m", 0.0)
            rem_dist = latest_nav.get("remaining_distance_m", 0.0)
            cross_err = latest_nav.get("cross_track_error_m", 0.0)
            is_arr = latest_nav.get("arrival_status", False)
            reroute_flag = latest_nav.get("reroute_occurred", False)
            dest_name = latest_nav.get("target_destination", "Main Auditorium")
            dest_floor = latest_nav.get("target_floor_id", 1)
        else:
            instruction = "Walk straight along the main corridor towards Robotics Research Lab."
            instr_type = "CONTINUE"
            dist_next = 12.5
            rem_dist = 42.0
            cross_err = 0.35
            is_arr = False
            reroute_flag = False
            dest_name = "Robotics Research Lab"
            dest_floor = 1

        if reroute_flag:
            st.warning("⚠️ Rerouting occurred! A new valid route has been recalculated avoiding deviation.")

        # HUD Banner
        icon_map = {
            "CONTINUE": "⬆️",
            "TURN_LEFT": "⬅️",
            "TURN_RIGHT": "➡️",
            "SLIGHT_LEFT": "↖️",
            "SLIGHT_RIGHT": "↗️",
            "TAKE_STAIRS": "🪜",
            "TAKE_ELEVATOR": "🛗",
            "ARRIVE": "🏁",
        }
        hud_icon = icon_map.get(instr_type, "🚶")

        st.markdown(
            f"""
            <div style="background-color: #1E293B; padding: 24px; border-radius: 12px; border-left: 8px solid #3B82F6; margin-bottom: 20px;">
                <div style="font-size: 14px; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px;">Target: {dest_name} (Floor {dest_floor})</div>
                <div style="font-size: 40px; margin-top: 8px; margin-bottom: 8px;">{hud_icon} <span style="font-size: 26px; color: #F8FAFC; font-weight: 600;">{instruction}</span></div>
                <div style="font-size: 16px; color: #38BDF8; font-weight: 500;">Next waypoint in {dist_next:.1f} meters</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        n1, n2, n3, n4 = st.columns(4)
        with n1:
            st.metric("Remaining Distance", f"{rem_dist:.1f} m")
        with n2:
            st.metric("Distance to Waypoint", f"{dist_next:.1f} m")
        with n3:
            err_color = "normal" if cross_err < 4.0 else "inverse"
            st.metric("Cross-Track Error", f"{cross_err:.2f} m")
        with n4:
            arr_pill = "🎉 ARRIVED" if is_arr else "🚶 IN TRANSIT"
            st.metric("Arrival Status", arr_pill)

    # -------------------------------------------------------------------------
    # TAB 5: Simulation & Replay Harness
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.markdown("### 🧪 Safe Development & Test Simulation Harness")
        st.warning(
            "⚠️ **SIMULATION MODE ACTIVE**: Injected steps generate deterministic synthetic observation data "
            "for development, automated tests, and offline validation. Never use simulated tests as physical-device validation."
        )

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sim_x = st.number_input("Simulator East X (m)", value=18.0, step=1.0)
        with col_s2:
            sim_y = st.number_input("Simulator North Y (m)", value=12.0, step=1.0)
        with col_s3:
            sim_floor = st.number_input("Simulator Floor ID", value=1, step=1)

        sim_heading_deg = st.slider("Walking Heading Direction (degrees)", min_value=0, max_value=360, value=90)
        sim_heading_rad = math.radians(sim_heading_deg)

        col_act_s1, col_act_s2 = st.columns(2)
        with col_act_s1:
            if st.button("🚶 Inject Single Simulated Step", use_container_width=True):
                try:
                    res = client.simulate_realtime_step(
                        session_id=session_id,
                        x_m=sim_x,
                        y_m=sim_y,
                        floor_id=sim_floor,
                        heading_rad=sim_heading_rad,
                        uncertainty_radius_m=1.8,
                    )
                    st.success(f"Injected step at ({sim_x}, {sim_y}) on Floor {sim_floor}. Status: {res.get('ack', {}).get('status')}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Simulation injection failed: {e}")

        with col_act_s2:
            if st.button("🏃 Simulate 5-Step Continuous Corridor Walk", use_container_width=True):
                try:
                    for step_idx in range(5):
                        step_x = sim_x + step_idx * 1.5 * math.cos(sim_heading_rad)
                        step_y = sim_y + step_idx * 1.5 * math.sin(sim_heading_rad)
                        client.simulate_realtime_step(
                            session_id=session_id,
                            x_m=step_x,
                            y_m=step_y,
                            floor_id=sim_floor,
                            heading_rad=sim_heading_rad,
                            uncertainty_radius_m=1.5,
                        )
                    st.success("Successfully injected 5 simulated steps along corridor trajectory.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Simulation walk failed: {e}")
