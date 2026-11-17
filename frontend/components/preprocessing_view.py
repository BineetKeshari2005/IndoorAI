"""Streamlit component for multimodal data preprocessing and temporal alignment (Task 20)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from frontend.api_client import DatasetApiClient, DatasetApiClientException


def render_preprocessing_view(client: DatasetApiClient, dataset_id: Optional[str]) -> None:
    """Renders the Preprocessing & Alignment dashboard screen."""
    st.markdown("## ⚙️ Multimodal Data Preprocessing & Alignment")
    st.markdown(
        "Transform raw field observations into clean, structured, and synchronized analytical datasets (Task 20). "
        "Preserves original provenance without synthetic interpolation."
    )

    if not dataset_id:
        st.info("💡 Please select an active survey dataset in **1. Overview & Catalog** first.")
        return

    # Check current preprocessing status
    with st.spinner("Checking preprocessing status..."):
        try:
            status_data = client.get_preprocessing_status(dataset_id)
        except DatasetApiClientException as e:
            st.error(f"Failed to communicate with preprocessing service: {e}")
            return

    is_preprocessed = status_data.get("is_preprocessed", False)
    report = status_data.get("report")
    manifest = status_data.get("manifest")

    # 1. Configuration Panel
    with st.expander("🛠️ Preprocessing & Alignment Configuration", expanded=not is_preprocessed):
        st.markdown("Configure signal thresholds, temporal windowing, and session segmentation parameters.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Wi-Fi & Sensor Parameters")
            min_rssi = st.slider("Min RSSI Threshold (dBm):", -120, -50, -105, 5)
            max_rssi = st.slider("Max RSSI Threshold (dBm):", -50, -10, -20, 5)
            dedup_ap = st.selectbox(
                "AP Deduplication Policy (same sweep):",
                ["keep_strongest", "keep_first", "reject"],
                index=0,
                help="How to handle multiple detections of the same BSSID in a single scan sweep."
            )
            sensor_gap_ms = st.number_input(
                "Sensor Gap Threshold (ms):",
                min_value=100.0,
                max_value=5000.0,
                value=500.0,
                step=50.0,
                help="Sampling intervals exceeding this threshold are flagged as discontinuities without interpolation."
            )

        with c2:
            st.markdown("##### Temporal Alignment & Segmentation")
            window_size_ms = st.selectbox(
                "Temporal Window Size:",
                [100.0, 250.0, 500.0, 1000.0],
                index=2,
                format_func=lambda x: f"{int(x)} ms window",
                help="Duration of discrete non-interpolating cross-modal synchronization windows."
            )
            seg_strategy = st.selectbox(
                "Session Segmentation Strategy:",
                ["fixed_duration", "observation_count", "annotation_boundaries", "temporal_gaps"],
                index=0,
                help="Strategy used to divide continuous surveys into analytical segments."
            )
            seg_duration_s = st.number_input(
                "Segment Duration (seconds):",
                min_value=10.0,
                max_value=600.0,
                value=60.0,
                step=10.0,
                help="Duration of each segment when using fixed_duration strategy."
            )
            filter_photos = st.checkbox(
                "Enforce Privacy Sanitization Gate on Photos",
                value=True,
                help="Exclude photos whose EXIF sanitization is false or unverified."
            )

        config_payload = {
            "min_rssi_dbm": float(min_rssi),
            "max_rssi_dbm": float(max_rssi),
            "deduplicate_wifi_ap": dedup_ap,
            "sensor_gap_threshold_ms": float(sensor_gap_ms),
            "temporal_window_size_ms": float(window_size_ms),
            "segmentation_strategy": seg_strategy,
            "segmentation_duration_seconds": float(seg_duration_s),
            "filter_unsanitized_photos": filter_photos,
        }

        st.markdown("---")
        col_btn, col_note = st.columns([1, 2])
        with col_btn:
            run_btn = st.button("▶ Run Preprocessing Pipeline", type="primary", use_container_width=True)
        with col_note:
            st.caption("Writes structured Parquet feature tables and JSON manifests to `data/processed/multimodal/<dataset_id>/`.")

        if run_btn:
            with st.spinner("Executing multimodal preprocessing and alignment pipeline..."):
                try:
                    res = client.preprocess_dataset(dataset_id, config_payload)
                    st.success(f"✅ {res.get('message', 'Preprocessing finished.')}")
                    # Re-fetch updated status
                    status_data = client.get_preprocessing_status(dataset_id)
                    is_preprocessed = status_data.get("is_preprocessed", True)
                    report = status_data.get("report")
                    manifest = status_data.get("manifest")
                except DatasetApiClientException as e:
                    st.error(f"❌ Preprocessing failed: {e}")

    # 2. Results & Diagnostics Section
    if not is_preprocessed or not report:
        st.info("ℹ️ This dataset has not been preprocessed yet. Click **'Run Preprocessing Pipeline'** above to process it.")
        return

    st.markdown("---")
    st.markdown("### 📊 Preprocessing Diagnostic Summary")

    status_str = report.get("status", "UNKNOWN")
    if status_str == "COMPLETED":
        st.success(f"● Preprocessing Status: **{status_str}** — Cleaned without errors.")
    elif status_str == "WARNINGS":
        st.warning(f"● Preprocessing Status: **{status_str}** — Completed with non-fatal warnings.")
    else:
        st.error(f"● Preprocessing Status: **{status_str}** — Execution errors reported.")

    feat_sum = report.get("feature_summary", {})
    align_sum = report.get("alignment_summary", {})

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Wi-Fi Sweeps", f"{feat_sum.get('wifi_sweeps', 0):,}")
    with m2:
        st.metric("Unique BSSIDs", f"{feat_sum.get('unique_bssids', 0):,}")
    with m3:
        st.metric("Sensor Streams", f"{feat_sum.get('sensor_streams', 0):,}")
    with m4:
        st.metric("Temporal Windows", f"{feat_sum.get('temporal_windows', 0):,}")
    with m5:
        st.metric("Session Segments", f"{feat_sum.get('session_segments', 0):,}")

    # 3. Tabbed Detailed Analysis
    tab_clean, tab_align, tab_seg, tab_files, tab_manifest = st.tabs([
        "🧹 Cleaning & Quality",
        "⏱️ Temporal Alignment",
        "📍 Session Segments",
        "📦 Generated Artifacts",
        "📜 Report & Manifest"
    ])

    with tab_clean:
        st.markdown("#### Modality Cleaning Breakdown")
        clean_sum = report.get("cleaning_summary", {})

        col_w, col_s = st.columns(2)
        with col_w:
            st.markdown("##### 📶 Wi-Fi Observations")
            w_info = clean_sum.get("wifi", {})
            if w_info:
                st.write(f"- **Input Records**: {w_info.get('records_input', 0)}")
                st.write(f"- **Retained Records**: {w_info.get('records_retained', 0)}")
                st.write(f"- **Removed Missing BSSID**: {w_info.get('records_removed_missing_bssid', 0)}")
                st.write(f"- **Removed Out-of-Range RSSI**: {w_info.get('records_removed_invalid_rssi', 0)}")
                st.write(f"- **Removed Duplicates**: {w_info.get('records_removed_duplicates', 0)}")
            else:
                st.caption("No Wi-Fi data in dataset.")

        with col_s:
            st.markdown("##### 🧭 IMU & Magnetometer")
            s_info = clean_sum.get("sensors", {})
            if s_info:
                st.write(f"- **Input Records**: {s_info.get('records_input', 0)}")
                st.write(f"- **Retained Records**: {s_info.get('records_retained', 0)}")
                st.write(f"- **Removed NaN / Inf**: {s_info.get('records_removed_nan_inf', 0)}")
                st.write(f"- **Total Sampling Gaps (> {sensor_gap_ms} ms)**: {s_info.get('total_gaps_detected', 0)}")
                st.write(f"- **Sampling Frequencies**: {s_info.get('sampling_freq_by_sensor', {})}")
            else:
                st.caption("No Sensor data in dataset.")

        col_p, col_a = st.columns(2)
        with col_p:
            st.markdown("##### 📷 Photos")
            p_info = clean_sum.get("photos", {})
            if p_info:
                st.write(f"- **Input Records**: {p_info.get('records_input', 0)}")
                st.write(f"- **Approved Sanitized**: {p_info.get('records_approved', 0)}")
                st.write(f"- **Excluded Unsanitized**: {p_info.get('records_excluded_unsanitized', 0)}")
                st.write(f"- **Missing Disk Files**: {p_info.get('records_excluded_missing_file', 0)}")
            else:
                st.caption("No Photo data in dataset.")

        with col_a:
            st.markdown("##### 🏷️ Annotations")
            a_info = clean_sum.get("annotations", {})
            if a_info:
                st.write(f"- **Input Records**: {a_info.get('records_input', 0)}")
                st.write(f"- **Retained Records**: {a_info.get('records_retained', 0)}")
                st.write(f"- **Type Breakdown**: {a_info.get('type_counts', {})}")
            else:
                st.caption("No Annotation data in dataset.")

    with tab_align:
        st.markdown("#### Cross-Modal Synchronization Windows")
        st.write(f"- **Window Size**: {align_sum.get('window_size_ms', 500)} ms")
        st.write(f"- **Session Span**: {align_sum.get('total_session_duration_ms', 0):,} ms")
        st.write(f"- **Concurrent Windows (>= 2 modalities)**: {align_sum.get('windows_multimodal_concurrent', 0)}")
        st.write(f"- **All-Modality Windows**: {align_sum.get('windows_with_all_modalities', 0)}")
        st.write(f"- **Mean Clock Uncertainty**: ±{align_sum.get('mean_clock_uncertainty_ms', 0.0):.2f} ms")
        st.write(f"- **Ambiguous Boundary Observations**: {align_sum.get('total_ambiguous_timing_observations', 0)}")
        st.info("ℹ️ Clock uncertainty is producer-reported and monitored near window boundaries to prevent false alignment.")

    with tab_seg:
        st.markdown("#### Analytical Session Segments")
        st.caption("Survey segmented based on selected strategy. Observations strictly reference source IDs.")
        segments_url = f"{client.base_url}/api/datasets/{dataset_id}/observations/sensors" # or fetch segments
        try:
            files = client.get_processed_files(dataset_id)
            seg_file = next((f for f in files if "segments.json" in f.get("filename", "")), None)
            if seg_file:
                st.success(f"Segments artifact generated ({seg_file.get('record_count', 0)} segments).")
            else:
                st.info("No segments file generated.")
        except Exception as e:
            st.warning(f"Could not load segment details: {e}")

    with tab_files:
        st.markdown("#### Generated Parquet & JSON Artifacts")
        try:
            files = client.get_processed_files(dataset_id)
            if files:
                table_rows = []
                for f in files:
                    table_rows.append({
                        "Artifact File": f.get("relative_path", ""),
                        "Records": f.get("record_count", 0),
                        "Size (KB)": round(f.get("file_size_bytes", 0) / 1024.0, 2),
                        "SHA-256 Digest": f.get("sha256", "")[:16] + "..."
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
            else:
                st.info("No artifact files found.")
        except Exception as e:
            st.error(f"Failed to list artifact files: {e}")

    with tab_manifest:
        st.markdown("#### Audit Manifest & Diagnostics")
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown("##### Preprocessing Report JSON")
            st.json(report)
        with c_m2:
            st.markdown("##### Preprocessing Manifest JSON")
            st.json(manifest or {})
