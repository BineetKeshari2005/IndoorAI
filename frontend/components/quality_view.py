"""Quality assurance and integrity diagnostic dashboard component for survey datasets."""

from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

from frontend.api_client import DatasetApiClient


def render_quality_dashboard(client: DatasetApiClient, dataset_id: Optional[str]) -> None:
    """Renders comprehensive quality, integrity, and privacy diagnostic views."""
    st.markdown("### 🔍 Dataset Quality & Integrity Dashboard")

    if not dataset_id:
        st.info("Please select or upload a dataset in the Overview or Upload tabs to view quality diagnostics.")
        return

    try:
        report = client.get_quality_report(dataset_id)
    except Exception as e:
        st.error(f"Failed to fetch quality report for dataset `{dataset_id}`: {e}")
        return

    if not report:
        st.warning(f"No quality report found for dataset `{dataset_id}`.")
        return

    status = report.get("validation_status", "UNKNOWN")
    color = "green" if status == "PASSED" else ("orange" if status == "WARNINGS" else "red")

    top_c1, top_c2, top_c3 = st.columns([2, 1, 1])
    with top_c1:
        st.markdown(f"#### Target Session: **{report.get('session_name')}** (`{report.get('session_id')}`)")
        st.caption(f"Dataset ID: `{report.get('dataset_id')}`")
    with top_c2:
        st.markdown(f"**Validation Status:** :{color}[{status}]")
    with top_c3:
        warns = len(report.get("warnings", []))
        errs = len(report.get("errors", []))
        st.markdown(f"**Diagnostics:** {errs} Errors | {warns} Warnings")

    st.markdown("---")

    # Tabs for domain quality areas
    tab_int, tab_wifi, tab_sensor, tab_photo, tab_ann, tab_diag = st.tabs([
        "🛡️ File Integrity",
        "📶 Wi-Fi Quality",
        "🧭 Sensor Quality",
        "📷 Photo Quality",
        "🏷️ Annotations",
        "⚠️ Diagnostic Log"
    ])

    # 1. Integrity Tab
    with tab_int:
        st.markdown("#### Cryptographic Manifest & File Verification")
        integrity = report.get("integrity", {})
        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            st.metric("Manifest Schema", "✅ Valid" if integrity.get("manifest_valid") else "❌ Invalid")
        with ic2:
            st.metric("SHA-256 Hashes", "✅ Verified" if integrity.get("checksums_valid") else "❌ Failed")
        with ic3:
            st.metric("Record Counts", "✅ Aligned" if integrity.get("record_counts_valid") else "❌ Discrepancy")
        with ic4:
            st.metric("Files Verified", integrity.get("files_verified_count", 0))

        if integrity.get("mismatched_files"):
            st.error("Failed Integrity Checks:")
            for m in integrity["mismatched_files"]:
                st.write(f"- {m}")
        else:
            st.success("All manifest entries matched bitwise SHA-256 hashes and file byte sizes.")

    # 2. Wi-Fi Quality Tab
    with tab_wifi:
        st.markdown("#### Wi-Fi Signal Coverage & Density Quality")
        wifi_q = report.get("modalities", {}).get("wifi") or {}
        if not wifi_q or wifi_q.get("total_scans", 0) == 0:
            st.info("No Wi-Fi observations recorded in this survey session.")
        else:
            w1, w2, w3, w4 = st.columns(4)
            with w1:
                st.metric("Total Scan Sweeps", wifi_q.get("total_scans", 0))
            with w2:
                st.metric("Unique BSSIDs Visible", wifi_q.get("unique_bssids_count", 0))
            with w3:
                st.metric("Avg APs per Sweep", wifi_q.get("avg_aps_per_scan", 0.0))
            with w4:
                min_r = wifi_q.get("min_rssi_dbm")
                max_r = wifi_q.get("max_rssi_dbm")
                st.metric("RSSI Dynamic Range", f"{min_r or '-'} to {max_r or '-'} dBm")

            w5, w6, w7, w8 = st.columns(4)
            with w5:
                st.metric("Fresh Sweeps", wifi_q.get("fresh_scan_count", 0))
            with w6:
                st.metric("Cached / Throttled Sweeps", wifi_q.get("cached_scan_count", 0))
            with w7:
                st.metric("Invalid RSSI Records", wifi_q.get("invalid_rssi_count", 0))
            with w8:
                st.metric("Intra-Sweep Duplicates", wifi_q.get("duplicate_ap_count", 0))

    # 3. Sensor Quality Tab
    with tab_sensor:
        st.markdown("#### Motion & Magnetic Sensor Sampling Quality")
        sensor_q = report.get("modalities", {}).get("sensors") or {}
        if not sensor_q or sensor_q.get("total_records", 0) == 0:
            st.info("No IMU or magnetometer records recorded in this survey session.")
        else:
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Total Sensor Records", sensor_q.get("total_records", 0))
            with s2:
                st.metric("Accelerometer Events", sensor_q.get("accelerometer_count", 0))
            with s3:
                st.metric("Gyroscope Events", sensor_q.get("gyroscope_count", 0))
            with s4:
                st.metric("Magnetometer Events", sensor_q.get("magnetometer_count", 0))

            st.markdown("##### Sampling Rates by Sensor Type")
            freqs = sensor_q.get("avg_sampling_freq_hz", {})
            min_intervals = sensor_q.get("min_interval_ms", {})
            max_intervals = sensor_q.get("max_interval_ms", {})

            freq_rows = []
            for stype, hz in freqs.items():
                freq_rows.append({
                    "Sensor": stype,
                    "Mean Frequency (Hz)": f"{hz:.1f} Hz",
                    "Min Interval (ms)": f"{min_intervals.get(stype, 0.0):.1f} ms",
                    "Max Interval (ms)": f"{max_intervals.get(stype, 0.0):.1f} ms",
                })
            if freq_rows:
                st.dataframe(pd.DataFrame(freq_rows), use_container_width=True)

            sq1, sq2, sq3 = st.columns(3)
            with sq1:
                st.metric("Sampling Dropouts (> 5x interval)", sensor_q.get("timestamp_gap_count", 0))
            with sq2:
                st.metric("Missing Axes (x, y, z)", sensor_q.get("missing_values_count", 0))
            with sq3:
                st.metric("Invalid Numerics (NaN/Inf)", sensor_q.get("invalid_numeric_count", 0))

            st.caption("Hardware acceleration and gyroscope readings do not alone provide absolute position.")

    # 4. Photo Quality Tab
    with tab_photo:
        st.markdown("#### Landmark Photo Metadata & Privacy Quality")
        photo_q = report.get("modalities", {}).get("photos") or {}
        if not photo_q or photo_q.get("total_photo_records", 0) == 0:
            st.info("No landmark photos recorded in this survey session.")
        else:
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("Photo Metadata Rows", photo_q.get("total_photo_records", 0))
            with p2:
                st.metric("Images Present on Disk", photo_q.get("image_files_present", 0))
            with p3:
                st.metric("EXIF Sanitized Count", photo_q.get("exif_sanitized_count", 0))
            with p4:
                st.metric("Faces Detected & Blurred", f"{photo_q.get('faces_detected_count', 0)} / {photo_q.get('face_blur_applied_count', 0)}")

            st.markdown("##### Privacy & Sanitization Auditing")
            pp1, pp2 = st.columns(2)
            with pp1:
                meta_san = photo_q.get("metadata_sanitization_verified")
                meta_label = "✅ PASSED (Client Tagged)" if meta_san else "❌ FAILED"
                st.info(f"**Client EXIF Sanitization Status:** {meta_label}")
            with pp2:
                audit_status = photo_q.get("visual_privacy_status", "PENDING_AUDIT")
                st.warning(f"**Independent Visual Privacy Audit:** ⚠️ {audit_status}")

            st.caption(
                "Notice: Client-side EXIF stripping does not prove automated redaction of faces or sensitive document text. "
                "An independent server-side deep learning audit is pending."
            )

    # 5. Annotations Tab
    with tab_ann:
        st.markdown("#### Manual Ground-Truth Annotations Quality")
        ann_q = report.get("modalities", {}).get("annotations") or {}
        if not ann_q or ann_q.get("total_annotations", 0) == 0:
            st.info("No manual ground-truth annotations recorded in this survey session.")
        else:
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                st.metric("Total Annotations", ann_q.get("total_annotations", 0))
            with a2:
                st.metric("Building Labels Present", ann_q.get("building_labels_present", 0))
            with a3:
                st.metric("Floor Labels Present", ann_q.get("floor_labels_present", 0))
            with a4:
                st.metric("Room Labels Present", ann_q.get("room_labels_present", 0))

            types = ann_q.get("type_breakdown", {})
            if types:
                st.markdown("##### Breakdown by Annotation Type")
                st.bar_chart(pd.Series(types))

    # 6. Diagnostic Log Tab
    with tab_diag:
        st.markdown("#### Diagnostic Warnings & Errors")
        errs = report.get("errors", [])
        warns = report.get("warnings", [])

        if not errs and not warns:
            st.success("No validation errors or quality warnings detected in this dataset.")
        else:
            if errs:
                st.error("Validation Errors:")
                for e in errs:
                    st.write(f"- {e}")
            if warns:
                st.warning("Quality Warnings:")
                for w in warns:
                    st.write(f"- {w}")
