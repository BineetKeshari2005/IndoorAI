"""Streamlit Dashboard component for Indoor Map Generation & Spatial Representation (Task 26)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import DatasetApiClient, DatasetApiClientException


def render_mapping_view(client: DatasetApiClient):
    """Renders Section 16: Indoor Map Generation & Spatial Representation."""
    st.header("16. Indoor Map Generation & Spatial Representation")
    st.markdown(
        """
        Constructs and manages structured multi-floor indoor spatial representations, metric 2D floorplans,
        topological navigation graphs, vertical transitions (stairs and elevators), and multimodal survey attachments
        (Wi-Fi reference locations and computer vision landmarks). Supports geometric validation, version snapshot publishing,
        and standard RFC 7946 GeoJSON export.
        """
    )

    # 1. Dataset / Building Map Selection
    try:
        buildings = client.list_indoor_maps()
    except Exception as e:
        st.error(f"Failed to connect to backend mapping service: {e}")
        return

    if not buildings:
        st.info("No indoor maps found. Initializing synthetic benchmark...")
        return

    building_options = {b["building_id"]: f"{b['name']} ({b['building_id']})" for b in buildings}
    col_bldg, col_refresh = st.columns([4, 1])
    with col_bldg:
        selected_building_id = st.selectbox(
            "Select Building Map",
            options=list(building_options.keys()),
            format_func=lambda x: building_options[x],
            key="mapping_building_select",
        )
    with col_refresh:
        if st.button("🔄 Refresh Maps", use_container_width=True):
            st.rerun()

    if not selected_building_id:
        return

    # Load complete building map
    try:
        building_map = client.get_indoor_map("default", selected_building_id)
    except Exception as e:
        st.error(f"Error loading building map {selected_building_id}: {e}")
        return

    if not building_map:
        st.warning("Selected building map data could not be retrieved.")
        return

    # Display High-Level Building Summary
    floors = building_map.get("floors", {})
    vts = building_map.get("vertical_transitions", {})
    conns = building_map.get("connections", {})

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("Floors", len(floors))
    with col_m2:
        st.metric("Total Rooms", building_map.get("total_rooms_count", 0))
    with col_m3:
        st.metric("Total Doors", building_map.get("total_doors_count", 0))
    with col_m4:
        st.metric("Vertical Links", len(vts))
    with col_m5:
        st.metric("Map Version", building_map.get("map_version", "1.0.0"))

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ 2D Floorplan & Spatial Geometry",
        "🕸️ Topological Navigation Graph",
        "✏️ Map Editor & Multimodal Ingestion",
        "✅ Validation, Evaluation & Export",
    ])

    # =========================================================================
    # Tab 1: 2D Floorplan & Spatial Geometry
    # =========================================================================
    with tab1:
        st.subheader("2D Metric Floorplan Viewer")
        st.markdown(
            "Floor coordinates are specified in `BUILDING_LOCAL_CARTESIAN_ENU` (+X East, +Y North, meters)."
        )

        if not floors:
            st.info("No floors defined for this building yet.")
        else:
            floor_keys = sorted(floors.keys(), key=lambda k: int(k))
            floor_labels = {k: f"{floors[k].get('label', f'Floor {k}')} (ID: {k}, Elev: {floors[k].get('elevation_m', 0.0)}m)" for k in floor_keys}

            col_fl, col_lyr = st.columns([2, 3])
            with col_fl:
                selected_floor_key = st.selectbox(
                    "Select Floor Level",
                    options=floor_keys,
                    format_func=lambda k: floor_labels[k],
                    key="map_floor_select",
                )
            with col_lyr:
                layers = st.multiselect(
                    "Active Map Layers",
                    options=["Boundary", "Rooms", "Corridors", "Doors", "Vertical Landings", "Wi-Fi Refs", "Landmarks"],
                    default=["Boundary", "Rooms", "Corridors", "Doors", "Vertical Landings", "Wi-Fi Refs", "Landmarks"],
                    key="map_layer_toggles",
                )

            selected_floor = floors[selected_floor_key]
            fig_fp = go.Figure()

            # 1. Floor Boundary
            if "Boundary" in layers and selected_floor.get("boundary"):
                bound = selected_floor["boundary"]
                verts = bound.get("vertices", [])
                if verts:
                    bx = [p["x"] for p in verts] + [verts[0]["x"]]
                    by = [p["y"] for p in verts] + [verts[0]["y"]]
                    fig_fp.add_trace(go.Scatter(
                        x=bx, y=by,
                        mode="lines",
                        line=dict(color="#1f77b4", width=3, dash="dash"),
                        name="Floor Boundary",
                        hoverinfo="text",
                        hovertext=f"Boundary: {selected_floor.get('label')} (Area: {selected_floor.get('total_floor_area_sq_m', 0)} m²)",
                    ))

            # 2. Rooms
            category_colors = {
                "OFFICE": "#2ca02c",
                "LAB": "#17becf",
                "CLASSROOM": "#9467bd",
                "RESTROOM": "#d62728",
                "CONFERENCE": "#ff7f0e",
                "LOBBY": "#bcbd22",
                "OTHER": "#7f7f7f",
            }
            if "Rooms" in layers:
                for r_id, room in selected_floor.get("rooms", {}).items():
                    poly = room.get("polygon", {})
                    verts = poly.get("vertices", [])
                    if verts:
                        rx = [p["x"] for p in verts] + [verts[0]["x"]]
                        ry = [p["y"] for p in verts] + [verts[0]["y"]]
                        cat = room.get("category", "OTHER")
                        color = category_colors.get(cat, "#7f7f7f")
                        v_status = room.get("verification_status", "UNVERIFIED")

                        # Polygon boundary
                        fig_fp.add_trace(go.Scatter(
                            x=rx, y=ry,
                            mode="lines",
                            fill="toself",
                            fillcolor=color,
                            opacity=0.35,
                            line=dict(color=color, width=2),
                            name=f"Room: {room.get('name')}",
                            hoverinfo="text",
                            hovertext=(
                                f"<b>{room.get('name')}</b><br>"
                                f"ID: {r_id}<br>"
                                f"Number: {room.get('room_number', 'N/A')}<br>"
                                f"Category: {cat}<br>"
                                f"Area: {room.get('area_sq_m')} m²<br>"
                                f"Status: {v_status}"
                            ),
                        ))

                        # Room Centroid label
                        c = room.get("centroid", {})
                        if c:
                            fig_fp.add_trace(go.Scatter(
                                x=[c.get("x", 0)],
                                y=[c.get("y", 0)],
                                mode="text",
                                text=[room.get("name")],
                                textfont=dict(size=10, color="#ffffff"),
                                showlegend=False,
                                hoverinfo="skip",
                            ))

            # 3. Corridors
            if "Corridors" in layers:
                for c_id, corr in selected_floor.get("corridors", {}).items():
                    cline = corr.get("centerline", [])
                    if len(cline) >= 2:
                        cx = [p["x"] for p in cline]
                        cy = [p["y"] for p in cline]
                        fig_fp.add_trace(go.Scatter(
                            x=cx, y=cy,
                            mode="lines+markers",
                            line=dict(color="#e377c2", width=4),
                            marker=dict(size=6, color="#e377c2"),
                            name=f"Corridor: {corr.get('name')}",
                            hoverinfo="text",
                            hovertext=f"Corridor: {corr.get('name')}<br>Length: {corr.get('length_m')}m<br>Width: {corr.get('width_m')}m",
                        ))

            # 4. Doors
            if "Doors" in layers:
                dx, dy, dtext = [], [], []
                for d_id, door in selected_floor.get("doors", {}).items():
                    pos = door.get("position", {})
                    dx.append(pos.get("x", 0))
                    dy.append(pos.get("y", 0))
                    dtext.append(
                        f"<b>Door: {door.get('name')}</b><br>"
                        f"ID: {d_id}<br>"
                        f"From: {door.get('connects_from_id')}<br>"
                        f"To: {door.get('connects_to_id')}<br>"
                        f"Type: {door.get('door_type')}<br>"
                        f"Status: {door.get('verification_status')}"
                    )
                if dx:
                    fig_fp.add_trace(go.Scatter(
                        x=dx, y=dy,
                        mode="markers",
                        marker=dict(size=11, color="#ffff00", symbol="square", line=dict(color="#000000", width=1.5)),
                        name="Doors",
                        hoverinfo="text",
                        hovertext=dtext,
                    ))

            # 5. Vertical Landings (Stairs / Elevators)
            if "Vertical Landings" in layers:
                cur_fid = int(selected_floor_key)
                for vt_id, vt in vts.items():
                    fpos = vt.get("floor_positions", {})
                    if str(cur_fid) in fpos or cur_fid in fpos:
                        pos = fpos.get(str(cur_fid)) or fpos.get(cur_fid)
                        ttype = vt.get("transition_type", "STAIRCASE")
                        symbol = "triangle-up" if ttype == "STAIRCASE" else "diamond"
                        color = "#ff7f0e" if ttype == "STAIRCASE" else "#8c564b"
                        fig_fp.add_trace(go.Scatter(
                            x=[pos.get("x", 0)],
                            y=[pos.get("y", 0)],
                            mode="markers+text",
                            marker=dict(size=13, color=color, symbol=symbol, line=dict(color="#ffffff", width=1.5)),
                            text=[vt.get("name")],
                            textposition="top center",
                            name=f"{ttype.capitalize()}: {vt.get('name')}",
                            hoverinfo="text",
                            hovertext=(
                                f"<b>{vt.get('name')}</b><br>"
                                f"Type: {ttype}<br>"
                                f"Floors: {vt.get('connected_floors')}<br>"
                                f"Accessible: {vt.get('is_accessible')}"
                            ),
                        ))

            # 6. Wi-Fi Reference Locations
            if "Wi-Fi Refs" in layers:
                wx, wy, wtext = [], [], []
                for w_id, w_data in selected_floor.get("wifi_reference_locations", {}).items():
                    pos = w_data.get("position", {})
                    wx.append(pos.get("x", 0))
                    wy.append(pos.get("y", 0))
                    wtext.append(f"Wi-Fi Ref: {w_data.get('label')}<br>ID: {w_id}<br>Fingerprints: {w_data.get('fingerprint_count', 0)}")
                if wx:
                    fig_fp.add_trace(go.Scatter(
                        x=wx, y=wy,
                        mode="markers",
                        marker=dict(size=8, color="#00ffcc", symbol="circle-open", line=dict(width=2)),
                        name="Wi-Fi Survey Refs",
                        hoverinfo="text",
                        hovertext=wtext,
                    ))

            # 7. Landmarks
            if "Landmarks" in layers:
                lx, ly, ltext = [], [], []
                for lm_id, lm_data in selected_floor.get("landmarks", {}).items():
                    pos = lm_data.get("position", {})
                    lx.append(pos.get("x", 0))
                    ly.append(pos.get("y", 0))
                    ltext.append(f"Landmark: {lm_data.get('name')}<br>Category: {lm_data.get('category')}<br>Status: {lm_data.get('verification_status')}")
                if lx:
                    fig_fp.add_trace(go.Scatter(
                        x=lx, y=ly,
                        mode="markers",
                        marker=dict(size=10, color="#ff00ff", symbol="star", line=dict(color="#ffffff", width=1)),
                        name="Visual Landmarks",
                        hoverinfo="text",
                        hovertext=ltext,
                    ))

            fig_fp.update_layout(
                title=f"2D Floorplan: {selected_floor.get('label')} (Building: {building_map.get('name')})",
                xaxis=dict(title="Local East (+X) [meters]", scaleanchor="y", scaleratio=1, gridcolor="#333333"),
                yaxis=dict(title="Local North (+Y) [meters]", gridcolor="#333333"),
                height=650,
                showlegend=True,
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig_fp, use_container_width=True)

            # Floor statistics table
            rooms_df = pd.DataFrame([
                {
                    "Room ID": r_id,
                    "Name": r.get("name"),
                    "Number": r.get("room_number", "-"),
                    "Category": r.get("category"),
                    "Area (m²)": r.get("area_sq_m"),
                    "Status": r.get("verification_status"),
                    "Doors": len(r.get("entrance_door_ids", [])),
                    "Wi-Fi Refs": len(r.get("wifi_reference_ids", [])),
                }
                for r_id, r in selected_floor.get("rooms", {}).items()
            ])
            if not rooms_df.empty:
                st.write("**Rooms Breakdown for Selected Floor:**")
                st.dataframe(rooms_df, use_container_width=True)

    # =========================================================================
    # Tab 2: Topological Navigation Graph
    # =========================================================================
    with tab2:
        st.subheader("Indoor Topological Navigation Graph")
        st.markdown(
            "Graph topology representation where rooms (centroids), corridor intersections, doors, and vertical transitions "
            "form nodes connected by bidirectional traversal edges."
        )

        try:
            metrics_resp = client.evaluate_indoor_map("default", selected_building_id)
        except Exception as e:
            metrics_resp = {}

        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        with col_g1:
            st.metric("Graph Nodes", metrics_resp.get("graph_node_count", 0))
        with col_g2:
            st.metric("Graph Edges", metrics_resp.get("graph_edge_count", 0))
        with col_g3:
            st.metric("Connected Components", metrics_resp.get("graph_connected_components", 1))
        with col_g4:
            st.metric("Accessibility Ratio", f"{metrics_resp.get('accessibility_ratio', 1.0)*100:.1f}%")

        # Floor filter for graph view
        all_floor_ids = sorted([int(k) for k in floors.keys()]) if floors else [0]
        selected_graph_floor = st.selectbox(
            "Graph Floor Filter (or All Floors)",
            options=["ALL"] + all_floor_ids,
            key="map_graph_floor_select",
        )

        # Plot topological graph in 2D
        fig_graph = go.Figure()

        # Nodes extraction
        node_types_map = {
            "ROOM": "#17becf",
            "CORRIDOR": "#e377c2",
            "DOOR": "#ffff00",
            "VERTICAL_TRANSITION": "#ff7f0e",
        }

        # Gather node positions across floors
        nodes_dict = {}
        for f_id_str, fl in floors.items():
            fid = int(f_id_str)
            if selected_graph_floor != "ALL" and fid != selected_graph_floor:
                continue

            # Rooms
            for r_id, rm in fl.get("rooms", {}).items():
                c = rm.get("centroid", {})
                if c:
                    nodes_dict[r_id] = (c.get("x", 0), c.get("y", 0), "ROOM", rm.get("name"), fid)

            # Corridors
            for c_id, cr in fl.get("corridors", {}).items():
                cline = cr.get("centerline", [])
                if cline:
                    mid = cline[len(cline) // 2]
                    nodes_dict[c_id] = (mid.get("x", 0), mid.get("y", 0), "CORRIDOR", cr.get("name"), fid)

            # Doors
            for d_id, dr in fl.get("doors", {}).items():
                pos = dr.get("position", {})
                door_nid = f"door_{d_id}"
                nodes_dict[door_nid] = (pos.get("x", 0), pos.get("y", 0), "DOOR", dr.get("name"), fid)

        # Vertical transition landings
        for vt_id, vt in vts.items():
            fpos = vt.get("floor_positions", {})
            for fid_k, p in fpos.items():
                fid_int = int(fid_k)
                if selected_graph_floor != "ALL" and fid_int != selected_graph_floor:
                    continue
                v_nid = f"vt_{vt_id}_f{fid_int}"
                nodes_dict[v_nid] = (p.get("x", 0), p.get("y", 0), "VERTICAL_TRANSITION", vt.get("name"), fid_int)

        # Edges
        edge_x, edge_y = [], []
        edge_conns = building_map.get("connections", {})
        for cid, conn in edge_conns.items():
            s_id = conn.get("source_id")
            d_id = conn.get("destination_id")
            if s_id in nodes_dict and d_id in nodes_dict:
                p1 = nodes_dict[s_id]
                p2 = nodes_dict[d_id]
                edge_x.extend([p1[0], p2[0], None])
                edge_y.extend([p1[1], p2[1], None])

        if edge_x:
            fig_graph.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(color="#555555", width=2),
                hoverinfo="none",
                name="Topology Edges",
            ))

        # Add node markers by type
        for n_type, n_color in node_types_map.items():
            nx = [p[0] for nid, p in nodes_dict.items() if p[2] == n_type]
            ny = [p[1] for nid, p in nodes_dict.items() if p[2] == n_type]
            ntext = [f"<b>{p[3]}</b><br>ID: {nid}<br>Type: {n_type}<br>Floor: {p[4]}" for nid, p in nodes_dict.items() if p[2] == n_type]
            if nx:
                fig_graph.add_trace(go.Scatter(
                    x=nx, y=ny,
                    mode="markers+text",
                    marker=dict(size=10, color=n_color, line=dict(color="#ffffff", width=1)),
                    text=[p[3] for nid, p in nodes_dict.items() if p[2] == n_type],
                    textposition="top right",
                    name=f"Nodes: {n_type}",
                    hoverinfo="text",
                    hovertext=ntext,
                ))

        fig_graph.update_layout(
            title="Indoor Navigation Graph Topology",
            xaxis=dict(title="Local East (+X) [meters]", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Local North (+Y) [meters]"),
            height=600,
            showlegend=True,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig_graph, use_container_width=True)

    # =========================================================================
    # Tab 3: Map Editor & Multimodal Ingestion
    # =========================================================================
    with tab3:
        st.subheader("Incremental Map Editing & Multimodal Integration")
        st.markdown(
            "Associate surveyed observations, Task 23 Wi-Fi fingerprint reference points, and Task 25 computer vision "
            "landmarks into the spatial model. Automatically link doors and transitions into navigation graph edges."
        )

        col_ed1, col_ed2 = st.columns(2)

        # Left Column: Add Floor or Room
        with col_ed1:
            st.write("##### ➕ Add Floor Level")
            with st.form("form_add_floor"):
                f_id = st.number_input("Floor Index", min_value=-5, max_value=50, value=len(floors), step=1)
                f_label = st.text_input("Floor Label", value=f"Floor {f_id}")
                f_elev = st.number_input("Elevation (meters)", value=float(f_id * 3.5), step=0.5)
                f_status = st.selectbox("Verification Status", ["VERIFIED", "PARTIALLY_SURVEYED", "UNVERIFIED"])
                btn_add_fl = st.form_submit_button("Add Floor")
                if btn_add_fl:
                    try:
                        client.add_map_floor("default", selected_building_id, {
                            "floor_id": int(f_id),
                            "label": f_label,
                            "elevation_m": f_elev,
                            "verification_status": f_status,
                        })
                        st.success(f"Floor {f_id} added successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding floor: {e}")

            st.write("##### 🚪 Add Door Portal")
            with st.form("form_add_door"):
                d_id = st.text_input("Door ID", value="door_new_01")
                d_floor = st.selectbox("Floor ID", options=sorted([int(k) for k in floors.keys()]))
                d_name = st.text_input("Door Name", value="New Office Door")
                d_x = st.number_input("X Coordinate (m)", value=10.0, step=0.5)
                d_y = st.number_input("Y Coordinate (m)", value=10.0, step=0.5)
                d_from = st.text_input("Connects From ID", value="room_101")
                d_to = st.text_input("Connects To ID", value="corr_f0_main")
                d_type = st.selectbox("Door Type", ["SINGLE_SWING", "DOUBLE_SWING", "SLIDING", "REVOLVING", "AUTOMATIC"])
                btn_add_dr = st.form_submit_button("Add Door")
                if btn_add_dr:
                    try:
                        client.add_map_door("default", selected_building_id, {
                            "door_id": d_id,
                            "floor_id": d_floor,
                            "name": d_name,
                            "position": {"x": d_x, "y": d_y},
                            "connects_from_id": d_from,
                            "connects_to_id": d_to,
                            "door_type": d_type,
                            "verification_status": "VERIFIED",
                        })
                        st.success(f"Door {d_id} added successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding door: {e}")

        # Right Column: Multimodal Attachments & Auto-link
        with col_ed2:
            st.write("##### 📶 Import Task 23 Wi-Fi Reference Locations")
            st.markdown("Load surveyed Wi-Fi reference points from radio maps into the floorplan.")
            with st.form("form_import_wifi"):
                imp_fid = st.selectbox("Target Floor", options=sorted([int(k) for k in floors.keys()]), key="wifi_imp_fid")
                sample_wifi = st.text_area(
                    "Wi-Fi Reference Points JSON",
                    value=json.dumps([
                        {"reference_id": "wifi_pt_01", "label": "AP North Zone", "x": 12.0, "y": 8.0, "fingerprint_count": 8},
                        {"reference_id": "wifi_pt_02", "label": "AP South Zone", "x": 22.0, "y": 8.0, "fingerprint_count": 12},
                    ], indent=2),
                    height=130,
                )
                auto_assoc = st.checkbox("Auto-associate enclosing room polygons", value=True)
                btn_imp_wifi = st.form_submit_button("Import Wi-Fi References")
                if btn_imp_wifi:
                    try:
                        data = json.loads(sample_wifi)
                        client.import_wifi_references_to_map("default", selected_building_id, {
                            "floor_id": imp_fid,
                            "wifi_references": data,
                            "auto_associate_rooms": auto_assoc,
                        })
                        st.success("Wi-Fi references imported successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to import Wi-Fi references: {e}")

            st.write("##### 👁️ Import Task 25 Computer Vision Landmarks")
            st.markdown("Attach verified landmark detections (door signs, room numbers, exit signs).")
            with st.form("form_import_lms"):
                imp_lm_fid = st.selectbox("Target Floor", options=sorted([int(k) for k in floors.keys()]), key="lm_imp_fid")
                sample_lms = st.text_area(
                    "Landmark Detections JSON",
                    value=json.dumps([
                        {"landmark_id": "lm_sign_101", "name": "Room 101 Plaque", "category": "ROOM_SIGN", "x": 7.0, "y": 9.0},
                        {"landmark_id": "lm_exit_east", "name": "Exit Directional Sign", "category": "SIGNBOARD", "x": 27.0, "y": 10.0},
                    ], indent=2),
                    height=130,
                )
                btn_imp_lms = st.form_submit_button("Import Vision Landmarks")
                if btn_imp_lms:
                    try:
                        data = json.loads(sample_lms)
                        client.import_landmarks_to_map("default", selected_building_id, {
                            "floor_id": imp_lm_fid,
                            "landmarks": data,
                            "auto_associate_doors": True,
                        })
                        st.success("Vision landmarks imported successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to import landmarks: {e}")

    # =========================================================================
    # Tab 4: Validation, Evaluation & Export
    # =========================================================================
    with tab4:
        st.subheader("Map Validation, Quality Metrics & Standards-Compliant Export")
        st.markdown(
            "Execute rigorous structural validation checks, monitor map completeness metrics, publish immutable numbered releases, "
            "and export spatial features to standard RFC 7946 GeoJSON or JSON formats."
        )

        col_val, col_pub = st.columns(2)

        # Validation Section
        with col_val:
            st.write("##### 🧪 Structural Map Validation")
            if st.button("Run Full Map Validation", type="primary", use_container_width=True):
                try:
                    val_rep = client.validate_indoor_map("default", selected_building_id)
                    if val_rep.get("is_valid", False):
                        st.success(f"✅ Map Passed Validation! ({val_rep.get('warning_count', 0)} warnings)")
                    else:
                        st.error(f"❌ Map Validation Failed: {val_rep.get('error_count', 0)} errors detected.")

                    if val_rep.get("errors"):
                        st.write("**Validation Errors:**")
                        for err in val_rep["errors"]:
                            st.error(f"• [{err.get('error_type')}] {err.get('message')}")

                    if val_rep.get("warnings"):
                        st.write("**Validation Warnings:**")
                        for warn in val_rep["warnings"]:
                            st.warning(f"• [{warn.get('error_type')}] {warn.get('message')}")
                except Exception as e:
                    st.error(f"Validation failed: {e}")

        # Version Publishing Section
        with col_pub:
            st.write("##### 🏷️ Publish Immutable Map Release")
            with st.form("form_publish_ver"):
                v_tag = st.text_input("Version Tag (optional)", placeholder="e.g. v1.1.0")
                btn_pub = st.form_submit_button("Publish Version Release")
                if btn_pub:
                    try:
                        res = client.publish_indoor_map_version(
                            "default", selected_building_id, {"version_tag": v_tag if v_tag else None}
                        )
                        st.success(f"Version snapshot {res.get('map_version')} published successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Publish failed: {e}")

            # List Version Snapshots
            versions = client.list_indoor_map_versions("default", selected_building_id)
            if versions:
                st.write("**Published Version History:**")
                st.dataframe(pd.DataFrame(versions)[["version", "published_at", "building_id"]], use_container_width=True)

        st.divider()

        # Export Section
        st.write("##### 💾 Map Serialization & GeoJSON Export")
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            st.write("**Export GeoJSON FeatureCollection (RFC 7946):**")
            st.caption("Exports room boundaries, corridors, doors, vertical transitions, and topology edges in standard GeoJSON.")
            if st.button("Generate GeoJSON Export", use_container_width=True):
                try:
                    geojson_data = client.export_indoor_map("default", selected_building_id, export_format="geojson")
                    geojson_str = json.dumps(geojson_data, indent=2)
                    st.download_button(
                        label="⬇️ Download GeoJSON File",
                        data=geojson_str,
                        file_name=f"{selected_building_id}_map.geojson",
                        mime="application/geo+json",
                        use_container_width=True,
                    )
                    st.json(geojson_data.get("metadata", {}))
                except Exception as e:
                    st.error(f"Export failed: {e}")

        with col_exp2:
            st.write("**Export Full Hierarchical JSON:**")
            st.caption("Exports complete BuildingMap Python domain schema including all floor definitions and metadata.")
            if st.button("Generate JSON Export", use_container_width=True):
                try:
                    json_data = client.export_indoor_map("default", selected_building_id, export_format="json")
                    json_str = json.dumps(json_data, indent=2) if isinstance(json_data, dict) else str(json_data)
                    st.download_button(
                        label="⬇️ Download JSON File",
                        data=json_str,
                        file_name=f"{selected_building_id}_map.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                    st.success("JSON export generated successfully!")
                except Exception as e:
                    st.error(f"Export failed: {e}")
