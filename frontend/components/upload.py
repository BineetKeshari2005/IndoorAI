"""Dataset upload and ingestion interface for survey session ZIP archives."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import DatasetApiClient


def render_upload(client: DatasetApiClient) -> None:
    """Renders the file uploader and ingestion diagnostics interface."""
    st.markdown("### 📤 Upload Multimodal Survey Dataset")
    st.markdown(
        "Upload a `.zip` archive exported by the IndoorAI mobile survey client (Task 17). "
        "The archive will undergo rigorous security inspection, cryptographic manifest verification, "
        "and multimodal quality validation."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Select IndoorAI Survey ZIP Archive",
            type=["zip"],
            help="Select an indoorai_session_<id>.zip file containing session.json, observations, and manifest."
        )

        validate_only = st.checkbox(
            "Dry-Run Validation Only (validate without persisting to catalog)",
            value=False,
            help="Extracts and evaluates schema, checksums, and quality diagnostics without adding the dataset to the permanent registry."
        )

    with col2:
        st.markdown("#### 🔒 Ingestion Security Gate")
        st.caption("All uploaded archives are strictly validated against:")
        st.caption("• ZIP magic byte verification (`PK\\x03\\x04`)")
        st.caption("• Path traversal defense (no `..` or root paths)")
        st.caption("• ZIP bomb protection ($< 100:1$ ratio, max 2 GB)")
        st.caption("• File extension whitelist (`.json`, `.csv`, `.jpg`)")
        st.caption("• Cryptographic SHA-256 manifest integrity")

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.caption(f"Selected file: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")

        if file_size_mb > 500:
            st.error("File size exceeds maximum upload limit of 500 MB.")
            return

        if st.button("🚀 Upload and Ingest Dataset", type="primary"):
            with st.spinner("Uploading archive and executing quality assurance pipeline..."):
                file_bytes = uploaded_file.read()
                success, data, msg = client.upload_dataset(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    validate_only=validate_only
                )

            if success:
                st.success(f"✅ {msg}")
                report = data.get("quality_report", {})
                status_val = report.get("validation_status", "UNKNOWN")

                status_color = "green" if status_val == "PASSED" else ("orange" if status_val == "WARNINGS" else "red")
                st.markdown(f"### Validation Outcome: :{status_color}[{status_val}]")

                dataset_id = data.get("dataset_id")
                if not validate_only and dataset_id:
                    st.session_state["selected_dataset_id"] = dataset_id
                    st.info(f"Dataset registered with ID: `{dataset_id}`. Set as active dataset in explorer!")

                # Summary metrics from report
                integrity = report.get("integrity", {})
                i1, i2, i3, i4 = st.columns(4)
                with i1:
                    m_valid = "✅ Valid" if integrity.get("manifest_valid") else "❌ Invalid"
                    st.metric("Manifest Schema", m_valid)
                with i2:
                    c_valid = "✅ Verified" if integrity.get("checksums_valid") else "❌ Mismatched"
                    st.metric("SHA-256 Checksums", c_valid)
                with i3:
                    r_valid = "✅ Aligned" if integrity.get("record_counts_valid") else "❌ Discrepancy"
                    st.metric("Record Counts", r_valid)
                with i4:
                    st.metric("Files Verified", integrity.get("files_verified_count", 0))

                # Display warnings and errors if any
                warnings = report.get("warnings", [])
                errors = report.get("errors", [])

                if errors:
                    st.error("#### ❌ Validation Errors Detected:")
                    for err in errors:
                        st.write(f"- {err}")

                if warnings:
                    st.warning("#### ⚠️ Quality Warnings Detected:")
                    for warn in warnings:
                        st.write(f"- {warn}")

                if not errors and not warnings:
                    st.balloons()

            else:
                st.error(f"❌ Ingestion Failed: {msg}")
