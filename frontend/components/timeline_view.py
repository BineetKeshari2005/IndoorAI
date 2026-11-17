"""Unified multimodal chronological timeline and cross-stream temporal overlap explorer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import DatasetApiClient


def render_timeline_explorer(client: DatasetApiClient, dataset_id: Optional[str]) -> None:
    """Renders cross-modal temporal overlap and unified chronological observation stream."""
    st.markdown("### ⏱️ Multimodal Timeline Explorer")

    if not dataset_id:
        st.info("Please select or upload a dataset in the Overview tab to view timeline alignment.")
        return

    # Fetch quality report for overlap data
    try:
        report = client.get_quality_report(dataset_id)
    except Exception as e:
        st.error(f"Failed to fetch quality report: {e}")
        return

    if not report:
        st.warning("No quality report available for dataset.")
        return

    # 1. Cross-Modal Temporal Overlap Visualization
    st.markdown("#### 🔄 Cross-Modal Concurrent Streaming Overlap")
    overlap_info = report.get("cross_modal_overlap", {})
    time_ranges = overlap_info.get("modality_time_ranges", {})
    timeline_summary = overlap_info.get("timeline_summary", "No timeline summary available.")
    all_overlap_s = overlap_info.get("all_modalities_overlap_seconds", 0.0)

    st.info(f"**Timeline Summary:** {timeline_summary}")

    # Build Gantt / horizontal interval chart of streams
    gantt_data = []
    base_epoch = None

    for mod, rng in time_ranges.items():
        start = rng.get("start_ms")
        end = rng.get("end_ms")
        if start is not None and end is not None:
            if base_epoch is None or start < base_epoch:
                base_epoch = start

    if base_epoch is not None:
        for mod, rng in time_ranges.items():
            start = rng.get("start_ms")
            end = rng.get("end_ms")
            if start is not None and end is not None:
                gantt_data.append({
                    "Modality": mod.capitalize(),
                    "Start (s)": round((start - base_epoch) / 1000.0, 2),
                    "End (s)": round((end - base_epoch) / 1000.0, 2),
                    "Duration (s)": round((end - start) / 1000.0, 2)
                })

    if gantt_data:
        gantt_df = pd.DataFrame(gantt_data)
        fig_gantt = px.bar(
            gantt_df,
            base="Start (s)",
            x="Duration (s)",
            y="Modality",
            orientation="h",
            color="Modality",
            title="Active Streaming Span by Modality (seconds from session start)",
            text="Duration (s)"
        )
        fig_gantt.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_gantt, use_container_width=True)

        ov_c1, ov_c2, ov_c3 = st.columns(3)
        with ov_c1:
            st.metric("Concurrent Multimodal Overlap", f"{all_overlap_s:.1f} s")
        with ov_c2:
            pairwise = overlap_info.get("pairwise_overlap_seconds", {})
            st.metric("Pairwise Stream Intersections", len(pairwise))
        with ov_c3:
            unc = report.get("timestamp_quality", {}).get("avg_reported_uncertainty_ms")
            st.metric("Avg Clock Uncertainty", f"±{unc or 3.5:.1f} ms", delta="Producer-Reported", delta_color="off")

    st.markdown("---")

    # 2. Unified Chronological Observation Feed
    st.markdown("#### 📜 Chronological Observation Stream")

    mod_filters = st.multiselect(
        "Filter Stream by Modality:",
        options=["Wi-Fi", "Sensors", "Photos", "Annotations"],
        default=["Wi-Fi", "Photos", "Annotations"],
        help="Select which sensor streams to include in the unified timeline feed."
    )

    feed_items = []

    if "Wi-Fi" in mod_filters:
        try:
            wifi_rows = client.get_observations(dataset_id, "wifi", limit=100)
            seen_sweeps = set()
            for r in wifi_rows:
                oid = r.get("observation_id")
                if oid and oid not in seen_sweeps:
                    seen_sweeps.add(oid)
                    feed_items.append({
                        "Time (epoch ms)": float(r.get("normalized_epoch_ms") or 0),
                        "Formatted Time": r.get("normalized_iso") or str(r.get("original_timestamp")),
                        "Modality": "📶 Wi-Fi",
                        "Event": f"Wi-Fi Sweep `{oid}`",
                        "Details": f"Visible APs in sweep. Uncertainty: ±{r.get('uncertainty_ms') or '30.0'} ms.",
                        "Raw Dict": r
                    })
        except Exception:
            pass

    if "Photos" in mod_filters:
        try:
            photo_rows = client.get_observations(dataset_id, "photos", limit=50)
            for r in photo_rows:
                feed_items.append({
                    "Time (epoch ms)": float(r.get("normalized_epoch_ms") or 0),
                    "Formatted Time": r.get("normalized_iso") or str(r.get("original_timestamp")),
                    "Modality": "📷 Photo",
                    "Event": f"Landmark Photo: {r.get('landmark_type_name', 'Photo')}",
                    "Details": f"Filename: {r.get('filename')}, Building: {r.get('building') or '-'}, Floor: {r.get('floor') or '-'}",
                    "Raw Dict": r
                })
        except Exception:
            pass

    if "Annotations" in mod_filters:
        try:
            ann_rows = client.get_observations(dataset_id, "annotations", limit=50)
            for r in ann_rows:
                feed_items.append({
                    "Time (epoch ms)": float(r.get("normalized_epoch_ms") or 0),
                    "Formatted Time": r.get("normalized_iso") or str(r.get("original_timestamp")),
                    "Modality": "🏷️ Annotation",
                    "Event": f"Note: {r.get('type_name', 'Annotation')}",
                    "Details": r.get("description", "Surveyor note"),
                    "Raw Dict": r
                })
        except Exception:
            pass

    if "Sensors" in mod_filters:
        try:
            # Sample sensor readings to avoid overwhelming table
            imu_rows = client.get_observations(dataset_id, "sensors", limit=50)
            for r in imu_rows:
                feed_items.append({
                    "Time (epoch ms)": float(r.get("normalized_epoch_ms") or 0),
                    "Formatted Time": r.get("normalized_iso") or str(r.get("original_timestamp_ns")),
                    "Modality": "🧭 Sensor",
                    "Event": f"{r.get('sensor_type')}",
                    "Details": f"x={r.get('x')}, y={r.get('y')}, z={r.get('z')}, mag={r.get('magnitude')}",
                    "Raw Dict": r
                })
        except Exception:
            pass

    if feed_items:
        feed_items.sort(key=lambda item: item["Time (epoch ms)"])
        feed_df = pd.DataFrame([
            {
                "Time": item["Formatted Time"],
                "Modality": item["Modality"],
                "Event": item["Event"],
                "Details": item["Details"]
            }
            for item in feed_items
        ])
        st.dataframe(feed_df, use_container_width=True)
    else:
        st.info("No timeline items available for selected filters.")

    st.caption(
        "⚠️ Timestamp Notice: Synchronization is mathematically reconciled via the mobile device ClockAnchor. "
        "Reported uncertainty bounds are producer-reported and do not imply hardware phase locking across radio and IMU buses."
    )
