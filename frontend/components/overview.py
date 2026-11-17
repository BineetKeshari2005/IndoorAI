"""Overview and catalog summary component for multimodal survey datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from frontend.api_client import DatasetApiClient


def render_overview(client: DatasetApiClient) -> Optional[str]:
    """Renders the top-level summary metrics, dataset table, and active selection.

    Returns:
        Selected dataset_id or None.
    """
    st.markdown("### 📊 Multimodal Dataset Catalog Overview")
    st.markdown(
        "Inspect and explore survey datasets collected by the IndoorAI mobile application "
        "and validated through the Task 18 ingestion pipeline."
    )

    # 1. Fetch datasets from API
    try:
        datasets = client.list_datasets()
    except Exception as e:
        st.error(f"Failed to communicate with backend API: {e}")
        st.info("Ensure the FastAPI backend is running: `python -m uvicorn backend.app.main:app --port 8000`")
        return None

    if not datasets:
        st.info("No survey datasets are currently registered in the catalog.")
        st.markdown(
            "Upload an `indoorai_session_<id>.zip` archive in the **Upload Dataset** tab to begin exploring."
        )
        return None

    # 2. Compute aggregate KPI metrics
    total_datasets = len(datasets)
    passed_cnt = sum(1 for d in datasets if d.get("validation_status") == "PASSED")
    warnings_cnt = sum(1 for d in datasets if d.get("validation_status") == "WARNINGS")
    failed_cnt = sum(1 for d in datasets if d.get("validation_status") in ("FAILED", "CORRUPTED"))

    total_wifi = sum(d.get("observation_counts", {}).get("wifi", 0) for d in datasets)
    total_imu = sum(d.get("observation_counts", {}).get("imu", 0) for d in datasets)
    total_photos = sum(d.get("observation_counts", {}).get("photos", 0) for d in datasets)
    total_annotations = sum(d.get("observation_counts", {}).get("annotations", 0) for d in datasets)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Datasets", total_datasets)
    with col2:
        st.metric("Valid Datasets (Passed)", passed_cnt, delta=f"{passed_cnt}/{total_datasets}")
    with col3:
        st.metric("Datasets with Warnings", warnings_cnt)
    with col4:
        st.metric("Failed / Corrupted", failed_cnt)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Wi-Fi Sweeps", f"{total_wifi:,}")
    with c2:
        st.metric("Total IMU Observations", f"{total_imu:,}")
    with c3:
        st.metric("Total Landmark Photos", f"{total_photos:,}")
    with c4:
        st.metric("Total Annotations", f"{total_annotations:,}")

    st.markdown("---")

    # 3. Interactive Dataset Table
    st.markdown("#### 📁 Registered Datasets")

    table_rows = []
    for d in datasets:
        mods = d.get("modality_availability", {})
        active_mods = []
        if mods.get("wifi"): active_mods.append("Wi-Fi")
        if mods.get("sensors"): active_mods.append("IMU")
        if mods.get("photos"): active_mods.append("Photos")
        if mods.get("annotations"): active_mods.append("Notes")

        counts = d.get("observation_counts", {})
        table_rows.append({
            "Dataset ID": d.get("dataset_id", "")[:8] + "...",
            "Full Dataset ID": d.get("dataset_id", ""),
            "Session Name": d.get("session_name", "Survey"),
            "Session ID": d.get("session_id", ""),
            "Ingestion Date": (d.get("ingested_at_iso") or "")[:19].replace("T", " "),
            "Status": d.get("validation_status", "UNKNOWN"),
            "Source Type": d.get("data_source_type", "REAL"),
            "Modalities": ", ".join(active_mods) if active_mods else "None",
            "Total Obs": counts.get("total", 0),
            "Location": f"{d.get('building_id') or '-'} / {d.get('floor_id') or '-'}"
        })

    df = pd.DataFrame(table_rows)
    display_df = df.drop(columns=["Full Dataset ID"])
    st.dataframe(display_df, use_container_width=True)

    # 4. Dataset Selector
    dataset_options = {
        f"{row['Session Name']} ({row['Session ID']}, Status: {row['Status']}) — {row['Full Dataset ID'][:8]}...": row["Full Dataset ID"]
        for _, row in df.iterrows()
    }

    current_selection = st.session_state.get("selected_dataset_id")
    current_index = 0
    if current_selection and current_selection in dataset_options.values():
        current_index = list(dataset_options.values()).index(current_selection)

    selected_label = st.selectbox(
        "Select Active Dataset for Deep Inspection:",
        options=list(dataset_options.keys()),
        index=current_index,
        help="Selects the survey dataset used across the Wi-Fi, Sensor, Timeline, and Photo explorers."
    )

    selected_id = dataset_options[selected_label]
    st.session_state["selected_dataset_id"] = selected_id

    # 5. Selected Dataset Metadata Strip
    selected_meta = next((d for d in datasets if d.get("dataset_id") == selected_id), None)
    if selected_meta:
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.caption(f"**Session:** {selected_meta.get('session_name')}")
        with m2:
            st.caption(f"**Session ID:** `{selected_meta.get('session_id')}`")
        with m3:
            st.caption(f"**Source:** `{selected_meta.get('data_source_type')}`")
        with m4:
            status = selected_meta.get('validation_status')
            color = "green" if status == "PASSED" else ("orange" if status == "WARNINGS" else "red")
            st.caption(f"**Status:** :{color}[{status}]")
        with m5:
            st.caption(f"**Size:** {selected_meta.get('file_size_bytes', 0) // 1024} KB")

    st.markdown(
        """
        <div class="disclaimer-box" style="margin-top: 15px;">
            <strong>⚠️ Quality Distinction Notice:</strong> A <code>PASSED</code> status confirms ZIP archive integrity, manifest SHA-256 verification, CSV schema adherence, and client EXIF sanitization tags. It indicates <em>structural and cryptographic validity</em>. It does <strong>not</strong> signify that an independent server-side visual privacy audit has been conducted on photographs.
        </div>
        """,
        unsafe_allow_html=True
    )

    return selected_id
