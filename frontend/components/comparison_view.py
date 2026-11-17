"""Multi-dataset comparison component for survey datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.api_client import DatasetApiClient


def render_dataset_comparison(client: DatasetApiClient) -> None:
    """Renders side-by-side comparison across two or more ingested survey datasets."""
    st.markdown("### ⚖️ Multimodal Dataset Comparison")
    st.markdown(
        "Compare sensing density, modality availability, telemetry sources, and data quality across multiple survey sessions."
    )

    try:
        datasets = client.list_datasets()
    except Exception as e:
        st.error(f"Failed to fetch datasets: {e}")
        return

    if len(datasets) < 2:
        st.info("At least two registered datasets are required to perform comparative analysis.")
        return

    dataset_options = {
        f"{d.get('session_name')} ({d.get('session_id')}) — {d.get('dataset_id')[:8]}...": d.get("dataset_id")
        for d in datasets
    }

    selected_labels = st.multiselect(
        "Select Datasets to Compare (Select 2 or more):",
        options=list(dataset_options.keys()),
        default=list(dataset_options.keys())[:min(3, len(dataset_options))],
        help="Choose multiple survey datasets to inspect side-by-side metrics and sensor coverage."
    )

    if len(selected_labels) < 2:
        st.warning("Please select at least two datasets above.")
        return

    selected_ids = [dataset_options[lbl] for lbl in selected_labels]
    selected_datasets = [d for d in datasets if d.get("dataset_id") in selected_ids]

    # Build comparison summary table
    comparison_rows = []
    for d in selected_datasets:
        did = d.get("dataset_id", "")
        counts = d.get("observation_counts", {})
        mods = d.get("modality_availability", {})

        # Fetch quality report if available
        q_rep = None
        try:
            q_rep = client.get_quality_report(did)
        except Exception:
            pass

        wifi_q = q_rep.get("modalities", {}).get("wifi") if q_rep else {}
        sensor_q = q_rep.get("modalities", {}).get("sensors") if q_rep else {}

        avg_aps = wifi_q.get("avg_aps_per_scan", 0.0) if wifi_q else 0.0
        accel_hz = sensor_q.get("avg_sampling_freq_hz", {}).get("ACCELEROMETER", 0.0) if sensor_q else 0.0
        overlap_s = q_rep.get("cross_modal_overlap", {}).get("all_modalities_overlap_seconds", 0.0) if q_rep else 0.0

        comparison_rows.append({
            "Session Name": d.get("session_name", "Survey"),
            "Session ID": d.get("session_id", ""),
            "Validation Status": d.get("validation_status", "UNKNOWN"),
            "Source Type": d.get("data_source_type", "REAL"),
            "Wi-Fi Sweeps": counts.get("wifi", 0),
            "IMU Samples": counts.get("imu", 0),
            "Photos": counts.get("photos", 0),
            "Annotations": counts.get("annotations", 0),
            "Total Observations": counts.get("total", 0),
            "Avg APs/Sweep": f"{avg_aps:.1f}",
            "IMU Accel (Hz)": f"{accel_hz:.1f} Hz" if accel_hz > 0 else "N/A",
            "Concurrent Overlap": f"{overlap_s:.1f} s",
            "Warnings": d.get("warnings_count", 0),
            "Errors": d.get("errors_count", 0)
        })

    comp_df = pd.DataFrame(comparison_rows)
    st.dataframe(comp_df, use_container_width=True)

    st.markdown("---")

    # Visual Comparison Charts
    st.markdown("#### 📊 Comparative Telemetry Breakdown")
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("##### Total Observations by Dataset")
        fig_total = px.bar(
            comp_df,
            x="Session Name",
            y="Total Observations",
            color="Source Type",
            title="Total Observation Count Comparison",
            text="Total Observations"
        )
        fig_total.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_total, use_container_width=True)

    with c_col2:
        st.markdown("##### Wi-Fi vs IMU Coverage Breakdown")
        melted_df = pd.melt(
            comp_df,
            id_vars=["Session Name"],
            value_vars=["Wi-Fi Sweeps", "Photos", "Annotations"],
            var_name="Modality",
            value_name="Count"
        )
        fig_mod = px.bar(
            melted_df,
            x="Session Name",
            y="Count",
            color="Modality",
            barmode="group",
            title="Modality Sweep & Photo Comparison"
        )
        fig_mod.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_mod, use_container_width=True)

    st.caption(
        "⚠️ Benchmark Comparison Note: This table compares sensor observation quantities, density, and sampling rates. "
        "It does not evaluate indoor positioning accuracy or model performance."
    )
