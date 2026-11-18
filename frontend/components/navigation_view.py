"""Streamlit component for Indoor Navigation & Intelligent Route Planning (Task 28)."""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import DatasetApiClient, DatasetApiClientException


def render_navigation_view(client: DatasetApiClient, dataset_id: Optional[str]) -> None:
    """Renders the comprehensive Indoor Navigation & Intelligent Route Planning dashboard."""
    st.markdown("## 🧭 Indoor Navigation & Intelligent Route Planning")
    st.caption(
        "Topologically constrained indoor route planning, Dijkstra & A* multi-floor pathfinding, "
        "turn-by-turn pedestrian guidance, real-time progress tracking, and debounced dynamic rerouting (Task 28)."
    )

    if not dataset_id:
        st.info("💡 Please select an active survey dataset in **1. Overview & Catalog** first.")
        return

    # Initialize session state for navigation
    if "nav_active_route" not in st.session_state:
        st.session_state.nav_active_route = None
    if "nav_session_id" not in st.session_state:
        st.session_state.nav_session_id = None
    if "nav_sim_step" not in st.session_state:
        st.session_state.nav_sim_step = 0
    if "nav_sim_progress" not in st.session_state:
        st.session_state.nav_sim_progress = None

    # Top KPI Metrics Row
    active_route = st.session_state.nav_active_route
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Active Route ID", active_route.get("route_id")[:12] if active_route else "None")
    with kpi2:
        st.metric("Total Distance", f"{active_route.get('total_distance_m', 0.0):.1f} m" if active_route else "—")
    with kpi3:
        st.metric("Estimated Time", f"{active_route.get('estimated_time_s', 0.0):.0f} s" if active_route else "—")
    with kpi4:
        st.metric("Floors Traversed", len(active_route.get("floors_traversed", [])) if active_route else "—")

    st.markdown("---")

    tabs = st.tabs([
        "📍 1. Destination Directory",
        "🗺️ 2. Route Planning & Algorithms",
        "📐 3. Interactive Route Canvas",
        "🚶 4. Turn-by-Turn Guidance & Sim",
        "📊 5. Rerouting & Benchmark",
    ])

    # =========================================================================
    # TAB 1: Destination Directory & Search
    # =========================================================================
    with tabs[0]:
        st.subheader("Verified Indoor Destinations")
        st.markdown(
            "Destinations extracted strictly from verified building elements (rooms, doors/exits, vertical transitions, "
            "and visual landmarks) without spatial fabrication."
        )

        col_b, col_f, col_cat, col_q = st.columns([1.5, 1.0, 1.5, 2.0])
        with col_b:
            bldg_id = st.text_input("Building ID", value="SYNTH_BLDG_01", key="nav_dir_bldg")
        with col_f:
            floor_filter = st.selectbox("Floor", options=["All", "0", "1"], index=0, key="nav_dir_floor")
        with col_cat:
            cat_filter = st.selectbox(
                "Category",
                options=["All", "ROOM", "VERTICAL_TRANSITION", "DOORWAY_EXIT", "LANDMARK"],
                index=0,
                key="nav_dir_cat",
            )
        with col_q:
            search_query = st.text_input("Search Keyword / Room #", value="", key="nav_dir_q")

        floor_arg = int(floor_filter) if floor_filter != "All" else None
        cat_arg = cat_filter if cat_filter != "All" else None

        try:
            dest_res = client.list_destinations(
                dataset_id=dataset_id,
                building_id=bldg_id,
                floor_id=floor_arg,
                category=cat_arg,
                query=search_query,
                limit=50,
            )
            dests = dest_res.get("destinations", [])
            st.success(f"Found **{len(dests)}** verified destinations in `{bldg_id}`.")

            if dests:
                rows = []
                for d in dests:
                    coords = d.get("verified_coordinates_m", [0, 0])
                    rows.append({
                        "ID": d.get("destination_id"),
                        "Name": d.get("display_name"),
                        "Category": d.get("category"),
                        "Floor": d.get("floor_id"),
                        "Coordinates (m)": f"({coords[0]:.1f}, {coords[1]:.1f})",
                        "Room #": d.get("room_number") or "—",
                        "Accessible": "✅ Yes" if d.get("is_accessible") else "⚠️ No",
                        "Verification": d.get("verification_status"),
                    })
                df_dest = pd.DataFrame(rows)
                st.dataframe(df_dest, use_container_width=True)
            else:
                st.warning("No verified destinations match the selected criteria.")
        except Exception as e:
            st.error(f"Error fetching destinations: {e}")

    # =========================================================================
    # TAB 2: Route Planning & Algorithm Comparison
    # =========================================================================
    with tabs[1]:
        st.subheader("Topological Route Planning")
        st.markdown(
            "Plan optimal indoor paths using **Dijkstra's Algorithm** (guaranteed shortest) or **A\* Search** "
            "(admissible Euclidean + floor-penalty heuristic) with strict wall non-penetration enforcement."
        )

        plan_col1, plan_col2 = st.columns(2)
        with plan_col1:
            st.markdown("##### 🏁 Origin & Destination")
            plan_bldg = st.text_input("Building ID", value="SYNTH_BLDG_01", key="nav_plan_bldg")
            
            # Origin choices
            origin_type = st.radio("Origin Mode", ["Topology Node", "Coordinates (x, y, floor)"], horizontal=True)
            orig_node = "NODE-RM-room_101"
            orig_x, orig_y, orig_f = 2.0, 5.0, 0
            if origin_type == "Topology Node":
                orig_node = st.text_input("Origin Node ID", value="NODE-RM-room_101", key="nav_orig_node")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    orig_x = st.number_input("X (m)", value=2.0, step=0.5, key="nav_orig_x")
                with c2:
                    orig_y = st.number_input("Y (m)", value=5.0, step=0.5, key="nav_orig_y")
                with c3:
                    orig_f = st.number_input("Floor", value=0, step=1, key="nav_orig_f")

            # Destination choices
            dest_type = st.radio("Destination Mode", ["Destination ID", "Coordinates (x, y, floor)"], horizontal=True)
            dest_id = "DEST-RM-room_102"
            dest_x, dest_y, dest_f = 22.0, 5.0, 0
            if dest_type == "Destination ID":
                # Quick pick destinations
                quick_dest_opts = [
                    "DEST-RM-room_102 (Room 102 - Floor 0)",
                    "DEST-VT-elevator_101 (Elevator 101 - Floor 0)",
                    "DEST-RM-room_201 (Room 201 - Floor 1)",
                    "DEST-RM-room_202 (Room 202 - Floor 1)",
                    "DEST-EXIT-exit_101 (Main Exit - Floor 0)",
                ]
                sel_opt = st.selectbox("Select Target Destination", options=quick_dest_opts, index=0)
                dest_id = sel_opt.split(" ")[0]
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    dest_x = st.number_input("Dest X (m)", value=22.0, step=0.5, key="nav_dest_x")
                with c2:
                    dest_y = st.number_input("Dest Y (m)", value=5.0, step=0.5, key="nav_dest_y")
                with c3:
                    dest_f = st.number_input("Dest Floor", value=0, step=1, key="nav_dest_f")

        with plan_col2:
            st.markdown("##### ⚙️ Routing Configuration")
            routing_alg = st.selectbox("Search Algorithm", ["ASTAR", "DIJKSTRA"], index=0)
            routing_pref = st.selectbox(
                "Routing Preference",
                ["SHORTEST", "ACCESSIBLE", "AVOID_STAIRS", "AVOID_ELEVATORS", "FEWEST_TURNS"],
                index=0,
            )

            st.markdown("###### Heuristic & Cost Weights")
            with st.expander("Adjust Cost Penalties", expanded=False):
                w_dist = st.slider("Distance Weight (wd)", 0.5, 2.0, 1.0, 0.1)
                w_turn = st.slider("Turn Penalty Weight (wt)", 0.0, 5.0, 1.0, 0.5)
                w_stair = st.slider("Stair Penalty Weight (ws)", 0.0, 20.0, 2.0, 1.0)
                w_acc = st.slider("Accessibility Barrier Penalty (wa)", 0.0, 50.0, 10.0, 5.0)

            btn_plan = st.button("🚀 Plan Indoor Route", type="primary", use_container_width=True)

        if btn_plan:
            with st.spinner("Computing topological graph search and wall collision audit..."):
                t_start = time.perf_counter()
                payload = {
                    "building_id": plan_bldg,
                    "algorithm": routing_alg,
                    "preference": routing_pref,
                }
                if origin_type == "Topology Node":
                    payload["origin_node_id"] = orig_node
                else:
                    payload["origin_x_m"] = orig_x
                    payload["origin_y_m"] = orig_y
                    payload["origin_floor_id"] = int(orig_f)

                if dest_type == "Destination ID":
                    payload["destination_id"] = dest_id
                else:
                    payload["destination_x_m"] = dest_x
                    payload["destination_y_m"] = dest_y
                    payload["destination_floor_id"] = int(dest_f)

                try:
                    route_res = client.plan_indoor_route(dataset_id=dataset_id, plan_params=payload)
                    plan_time_ms = (time.perf_counter() - t_start) * 1000.0
                    st.session_state.nav_active_route = route_res
                    st.session_state.nav_session_id = None
                    st.session_state.nav_sim_step = 0
                    st.session_state.nav_sim_progress = None

                    st.success(
                        f"Route generated successfully in **{plan_time_ms:.1f} ms**! "
                        f"Total distance: **{route_res['total_distance_m']:.1f} m**, "
                        f"Waypoints: **{len(route_res['waypoints'])}**, "
                        f"Instructions: **{len(route_res['instructions'])}**"
                    )
                except DatasetApiClientException as dace:
                    st.error(f"Routing Error: {dace}")
                except Exception as e:
                    st.error(f"Failed to plan route: {e}")

        # Display route summary if route is present
        if st.session_state.nav_active_route:
            rt = st.session_state.nav_active_route
            st.markdown("---")
            st.markdown("##### 📋 Route Overview")
            rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns(5)
            rcol1.metric("Route ID", rt["route_id"][:12])
            rcol2.metric("Distance", f"{rt['total_distance_m']:.1f} m")
            rcol3.metric("Est. Walk Time", f"{rt['estimated_time_s']:.0f} s")
            rcol4.metric("Algorithm", rt["algorithm_used"])
            rcol5.metric("Accessible", "✅ Yes" if rt["is_accessible"] else "⚠️ No")

    # =========================================================================
    # TAB 3: 2D Interactive Route & Floor Map Canvas
    # =========================================================================
    with tabs[2]:
        st.subheader("2D Interactive Floorplan & Route Canvas")
        st.markdown("Visual overlay of verified rooms, doorways, vertical transitions, and the planned route polyline.")

        if not st.session_state.nav_active_route:
            st.info("ℹ️ No active route planned yet. Use **Tab 2 (Route Planning)** to compute a route first.")
        else:
            rt = st.session_state.nav_active_route
            floors_in_route = rt.get("floors_traversed", [0])
            active_canvas_floor = st.selectbox("Select Floor to Inspect", options=floors_in_route, index=0)

            # Filter waypoints for this floor
            wp_list = [w for w in rt.get("waypoints", []) if w.get("floor_id") == active_canvas_floor]

            fig = go.Figure()

            # Render synthetic rooms for reference
            # Synthetic Floor 0: Room 101 (0,0 to 10,10), Room 102 (15,0 to 25,10), Corridor (0,4 to 25,6)
            if active_canvas_floor == 0:
                # Room 101
                fig.add_trace(go.Scatter(
                    x=[0, 10, 10, 0, 0], y=[0, 0, 10, 10, 0],
                    mode="lines", fill="toself", fillcolor="rgba(100, 149, 237, 0.15)",
                    line=dict(color="#4169E1", width=2),
                    name="Room 101 (Lab)", hoverinfo="name"
                ))
                # Room 102
                fig.add_trace(go.Scatter(
                    x=[15, 25, 25, 15, 15], y=[0, 0, 10, 10, 0],
                    mode="lines", fill="toself", fillcolor="rgba(100, 149, 237, 0.15)",
                    line=dict(color="#4169E1", width=2),
                    name="Room 102 (Office)", hoverinfo="name"
                ))
                # Corridor
                fig.add_trace(go.Scatter(
                    x=[0, 25, 25, 0, 0], y=[4, 4, 6, 6, 4],
                    mode="lines", fill="toself", fillcolor="rgba(240, 230, 140, 0.2)",
                    line=dict(color="#DAA520", width=1, dash="dot"),
                    name="Corridor 0", hoverinfo="name"
                ))
                # Doors & Elevator
                fig.add_trace(go.Scatter(
                    x=[10.0, 15.0, 12.0], y=[5.0, 5.0, 8.0],
                    mode="markers+text", text=["Door 101", "Door 102", "Elevator 101"],
                    textposition="top center",
                    marker=dict(symbol=["square", "square", "star"], size=[10, 10, 14], color=["#2E8B57", "#2E8B57", "#9932CC"]),
                    name="Portals / Transitions"
                ))
            elif active_canvas_floor == 1:
                # Room 201
                fig.add_trace(go.Scatter(
                    x=[0, 10, 10, 0, 0], y=[0, 0, 10, 10, 0],
                    mode="lines", fill="toself", fillcolor="rgba(100, 149, 237, 0.15)",
                    line=dict(color="#4169E1", width=2),
                    name="Room 201", hoverinfo="name"
                ))
                # Room 202
                fig.add_trace(go.Scatter(
                    x=[15, 25, 25, 15, 15], y=[0, 0, 10, 10, 0],
                    mode="lines", fill="toself", fillcolor="rgba(100, 149, 237, 0.15)",
                    line=dict(color="#4169E1", width=2),
                    name="Room 202", hoverinfo="name"
                ))
                # Corridor
                fig.add_trace(go.Scatter(
                    x=[0, 25, 25, 0, 0], y=[4, 4, 6, 6, 4],
                    mode="lines", fill="toself", fillcolor="rgba(240, 230, 140, 0.2)",
                    line=dict(color="#DAA520", width=1, dash="dot"),
                    name="Corridor 1", hoverinfo="name"
                ))
                # Doors & Elevator
                fig.add_trace(go.Scatter(
                    x=[10.0, 15.0, 12.0], y=[5.0, 5.0, 8.0],
                    mode="markers+text", text=["Door 201", "Door 202", "Elevator 201"],
                    textposition="top center",
                    marker=dict(symbol=["square", "square", "star"], size=[10, 10, 14], color=["#2E8B57", "#2E8B57", "#9932CC"]),
                    name="Portals / Transitions"
                ))

            # Route polyline for this floor
            if wp_list:
                wx = [w["x_m"] for w in wp_list]
                wy = [w["y_m"] for w in wp_list]
                wlabels = [f"#{i}: {w['label']} ({w['x_m']:.1f}, {w['y_m']:.1f})" for i, w in enumerate(wp_list)]

                fig.add_trace(go.Scatter(
                    x=wx, y=wy,
                    mode="lines+markers",
                    line=dict(color="#FF4500", width=4),
                    marker=dict(size=8, color="#FF4500"),
                    text=wlabels,
                    hoverinfo="text",
                    name="Planned Route"
                ))

                # Highlight origin and destination on this floor if present
                if wp_list[0] == rt["waypoints"][0]:
                    fig.add_trace(go.Scatter(
                        x=[wx[0]], y=[wy[0]],
                        mode="markers+text",
                        marker=dict(size=14, color="#32CD32", symbol="circle"),
                        text=["🚩 Start"],
                        textposition="bottom center",
                        name="Start Point"
                    ))
                if wp_list[-1] == rt["waypoints"][-1]:
                    fig.add_trace(go.Scatter(
                        x=[wx[-1]], y=[wy[-1]],
                        mode="markers+text",
                        marker=dict(size=14, color="#DC143C", symbol="flag"),
                        text=["🏁 Goal"],
                        textposition="top center",
                        name="Destination"
                    ))

            # If user simulated position is active on this floor, display it
            sim_prog = st.session_state.nav_sim_progress
            if sim_prog and sim_prog.get("current_floor_id") == active_canvas_floor:
                fig.add_trace(go.Scatter(
                    x=[sim_prog["current_x_m"]],
                    y=[sim_prog["current_y_m"]],
                    mode="markers+text",
                    marker=dict(size=16, color="#00FFFF", symbol="cross", line=dict(width=2, color="#00008B")),
                    text=["📍 User Pose"],
                    textposition="top center",
                    name="Current User Fix"
                ))

            fig.update_layout(
                title=f"Floor {active_canvas_floor} — Topological Geometry & Navigation Path",
                xaxis_title="East (m)",
                yaxis_title="North (m)",
                xaxis=dict(scaleanchor="y", scaleratio=1),
                height=550,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # TAB 4: Turn-by-Turn Guidance & Progress Simulation
    # =========================================================================
    with tabs[3]:
        st.subheader("Pedestrian Turn-by-Turn Guidance & Live Trip Simulation")
        st.markdown(
            "Natural human-readable instructions compiled along topological geometry with arrival detection (1.5 m), "
            "cross-track tracking, and interactive simulation."
        )

        if not st.session_state.nav_active_route:
            st.info("ℹ️ Please plan a route in **Tab 2** to inspect instructions and test simulation.")
        else:
            rt = st.session_state.nav_active_route
            instructions = rt.get("instructions", [])

            sim_col1, sim_col2 = st.columns([1.2, 1.8])

            with sim_col1:
                st.markdown("##### 🚶 Trip Session Stepper")
                if not st.session_state.nav_session_id:
                    if st.button("🏁 Start Live Navigation Session", type="primary", use_container_width=True):
                        try:
                            sess_res = client.create_navigation_session(
                                dataset_id=dataset_id,
                                session_params={"route_id": rt["route_id"]},
                            )
                            st.session_state.nav_session_id = sess_res["session_id"]
                            st.session_state.nav_sim_step = 0
                            st.session_state.nav_sim_progress = None
                            st.success(f"Session `{sess_res['session_id'][:12]}` active!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to create session: {e}")
                else:
                    st.info(f"Active Session: `{st.session_state.nav_session_id[:16]}...`")

                    total_wp = len(rt["waypoints"])
                    step_slider = st.slider(
                        "Advance Waypoint Step",
                        0, total_wp - 1,
                        value=st.session_state.nav_sim_step,
                        key="nav_step_slider"
                    )

                    btn_c1, btn_c2 = st.columns(2)
                    with btn_c1:
                        if st.button("➡️ Step to Waypoint", use_container_width=True):
                            st.session_state.nav_sim_step = step_slider
                            wp = rt["waypoints"][step_slider]
                            try:
                                prog_res = client.update_navigation_progress(
                                    dataset_id=dataset_id,
                                    session_id=st.session_state.nav_session_id,
                                    progress_params={
                                        "x_m": wp["x_m"],
                                        "y_m": wp["y_m"],
                                        "floor_id": wp["floor_id"],
                                        "localization_status": "TRACKING",
                                    }
                                )
                                st.session_state.nav_sim_progress = prog_res
                                if prog_res.get("updated_route"):
                                    st.session_state.nav_active_route = prog_res["updated_route"]
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to step: {e}")

                    with btn_c2:
                        if st.button("💥 Simulate Deviation (+6m)", type="secondary", use_container_width=True):
                            # Perturb location
                            wp = rt["waypoints"][st.session_state.nav_sim_step]
                            try:
                                prog_res = client.update_navigation_progress(
                                    dataset_id=dataset_id,
                                    session_id=st.session_state.nav_session_id,
                                    progress_params={
                                        "x_m": wp["x_m"] + 6.0,
                                        "y_m": wp["y_m"] + 6.0,
                                        "floor_id": wp["floor_id"],
                                        "localization_status": "TRACKING",
                                    }
                                )
                                st.session_state.nav_sim_progress = prog_res
                                if prog_res.get("updated_route"):
                                    st.session_state.nav_active_route = prog_res["updated_route"]
                                    st.warning(f"⚠️ Deviation detected! Automatic rerouting triggered ({prog_res['status_message']}).")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error during deviation simulation: {e}")

                    if st.button("🛑 Complete / Reset Session", use_container_width=True):
                        try:
                            client.complete_navigation_session(
                                dataset_id=dataset_id,
                                session_id=st.session_state.nav_session_id
                            )
                        except Exception:
                            pass
                        st.session_state.nav_session_id = None
                        st.session_state.nav_sim_step = 0
                        st.session_state.nav_sim_progress = None
                        st.rerun()

                # Progress Indicators
                if st.session_state.nav_sim_progress:
                    prog = st.session_state.nav_sim_progress
                    st.markdown("---")
                    st.markdown("##### 📊 Live Trip Feedback")
                    st.metric("Completion", f"{prog['completion_percentage']:.1f} %")
                    st.metric("Remaining Distance", f"{prog['remaining_distance_m']:.1f} m")
                    st.metric("Cross-Track Deviation", f"{prog['cross_track_deviation_m']:.2f} m")
                    if prog["is_off_route"]:
                        st.error(f"Off Route: {prog['status_message']}")
                    else:
                        st.success("On Route: Tracking normally")

            with sim_col2:
                st.markdown("##### 📜 Turn-by-Turn Instruction Sequence")
                active_idx = 0
                if st.session_state.nav_sim_progress:
                    active_idx = st.session_state.nav_sim_progress.get("next_instruction_index", 0)

                for idx, ins in enumerate(instructions):
                    itype = ins.get("instruction_type", "CONTINUE")
                    icon = "🚶"
                    if "START" in itype:
                        icon = "🚩"
                    elif "LEFT" in itype:
                        icon = "⬅️"
                    elif "RIGHT" in itype:
                        icon = "➡️"
                    elif "STAIRS" in itype:
                        icon = "🪜"
                    elif "ELEVATOR" in itype:
                        icon = "🛗"
                    elif "DOOR" in itype:
                        icon = "🚪"
                    elif "DESTINATION" in itype:
                        icon = "🏁"

                    is_current = (idx == active_idx)
                    card_border = "border: 2px solid #FF4500; background-color: rgba(255,69,0,0.08);" if is_current else "border: 1px solid #444;"

                    st.markdown(
                        f"""
                        <div style="padding: 10px; margin-bottom: 8px; border-radius: 6px; {card_border}">
                            <div style="font-weight: bold; font-size: 1.05rem;">
                                {icon} Step {ins['sequence_number']}: {ins['text']}
                            </div>
                            <div style="font-size: 0.85rem; color: #aaa; margin-top: 4px;">
                                Distance: <b>{ins['distance_to_next_m']:.1f} m</b> &nbsp;|&nbsp; 
                                Target Heading: <b>{ins['target_heading_deg'] if ins['target_heading_deg'] is not None else '—'}°</b> &nbsp;|&nbsp; 
                                Cumulative: <b>{ins['cumulative_distance_m']:.1f} m</b> &nbsp;|&nbsp;
                                Floor: <b>{ins['floor_id']}</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # =========================================================================
    # TAB 5: Rerouting Diagnostics & Evaluation Benchmark
    # =========================================================================
    with tabs[4]:
        st.subheader("Rerouting Diagnostics & Comparative Evaluation Benchmark")
        st.markdown(
            "Empirical verification of Dijkstra vs A* optimality, admissibility ($h(u, v) \\le h^*(u, v)$), "
            "node expansion efficiency, and simulated trip rerouting resilience."
        )

        bldg_eval_id = st.text_input("Building ID to Benchmark", value="SYNTH_BLDG_01", key="nav_eval_bldg")
        if st.button("📊 Run Comparative Navigation Benchmark", type="primary"):
            with st.spinner("Executing multi-pair Dijkstra vs A* benchmarks and simulated trips..."):
                try:
                    eval_resp = client.evaluate_navigation(
                        dataset_id=dataset_id,
                        eval_params={"building_id": bldg_eval_id, "use_synthetic_benchmark": True}
                    )
                    rep = eval_resp.get("report", {})
                    st.success(f"Benchmark `{eval_resp.get('evaluation_id')}` completed!")

                    # Summary cards
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("Evaluated Pairs", rep.get("evaluated_route_pairs", 0))
                    sc2.metric("A* Optimality", f"{rep.get('mean_astar_optimality_ratio', 1.0):.4f}")
                    sc3.metric("A* Speedup", f"{rep.get('astar_mean_speedup', 1.0):.2f}x")
                    sc4.metric("Admissibility Check", "✅ 100% Passed" if rep.get("heuristic_admissible") else "❌ Violated")

                    # Comparative table of routes
                    st.markdown("---")
                    st.markdown("##### 🔬 Algorithm Performance per Pair")
                    rt_metrics = rep.get("route_metrics", [])
                    if rt_metrics:
                        r_rows = []
                        for m in rt_metrics:
                            r_rows.append({
                                "Route ID": m["route_id"],
                                "Origin -> Dest": f"{m['origin_name']} ➔ {m['destination_name']}",
                                "Dijkstra Time (ms)": f"{m['dijkstra_execution_time_ms']:.2f}",
                                "A* Time (ms)": f"{m['astar_execution_time_ms']:.2f}",
                                "Dijkstra Nodes": m["dijkstra_nodes_expanded"],
                                "A* Nodes": m["astar_nodes_expanded"],
                                "Path Length (m)": f"{m['total_distance_m']:.2f}",
                                "Optimality Ratio": f"{m['optimality_ratio']:.4f}",
                                "Heuristic Admissible": "✅ Yes" if m["heuristic_admissible"] else "❌ No",
                            })
                        df_perf = pd.DataFrame(r_rows)
                        st.dataframe(df_perf, use_container_width=True)

                    # Simulated trips metrics
                    st.markdown("---")
                    st.markdown("##### 🚶 Simulated Trip Navigation Metrics")
                    sim_metrics = rep.get("simulation_metrics", [])
                    if sim_metrics:
                        s_rows = []
                        for sm in sim_metrics:
                            s_rows.append({
                                "Trip ID": sm["trip_id"],
                                "Total Steps": sm["total_steps"],
                                "Trip Length (m)": f"{sm['total_distance_m']:.1f}",
                                "Avg Cross-Track (m)": f"{sm['mean_cross_track_deviation_m']:.2f}",
                                "Max Cross-Track (m)": f"{sm['max_cross_track_deviation_m']:.2f}",
                                "Reroutes Triggered": sm["reroutes_triggered"],
                                "Completed": "✅ Yes" if sm["trip_completed"] else "❌ No",
                                "Avg Step Latency (ms)": f"{sm['mean_step_processing_time_ms']:.2f}",
                            })
                        df_sim = pd.DataFrame(s_rows)
                        st.dataframe(df_sim, use_container_width=True)

                except Exception as e:
                    st.error(f"Benchmark error: {e}")
