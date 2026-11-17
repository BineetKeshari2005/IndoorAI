"""Dataset lifecycle management, revalidation, and safe deletion component."""

from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st

from frontend.api_client import DatasetApiClient


def render_management(client: DatasetApiClient, dataset_id: Optional[str]) -> None:
    """Renders dataset lifecycle management controls: revalidation, diagnostic audit, and safe deletion."""
    st.markdown("### ⚙️ Dataset Lifecycle Management")

    if not dataset_id:
        st.info("Please select or upload a dataset in the Overview tab to manage its lifecycle.")
        return

    try:
        dataset = client.get_dataset(dataset_id)
    except Exception as e:
        st.error(f"Failed to fetch dataset details: {e}")
        return

    if not dataset:
        st.warning("Selected dataset could not be found.")
        return

    st.markdown(f"#### Managing: **{dataset.get('session_name')}** (`{dataset.get('session_id')}`)")
    st.caption(f"Dataset UUID: `{dataset_id}` | Status: `{dataset.get('validation_status')}`")

    st.markdown("---")

    # 1. On-Demand Revalidation
    st.markdown("#### 🔄 Re-run Dataset Validation")
    st.write(
        "Re-runs the complete verification pipeline against disk records: cryptographic SHA-256 integrity, "
        "record count consistency, schema adherence, sampling rate diagnostics, and privacy sanitization tags."
    )

    if st.button("Re-run Quality & Integrity Validation", type="secondary"):
        with st.spinner("Re-evaluating cryptographic checksums and quality metrics..."):
            success, res_data, msg = client.revalidate_dataset(dataset_id)

        if success:
            st.success(f"✅ {msg}")
            new_status = res_data.get("validation_status", "UNKNOWN")
            st.info(f"Updated Validation Status: **{new_status}** ({res_data.get('warnings_count')} warnings, {res_data.get('errors_count')} errors)")
            st.rerun()
        else:
            st.error(f"❌ {msg}")

    st.markdown("---")

    # 2. Ingestion Diagnostics & Error Inspector
    st.markdown("#### 📋 Diagnostic Errors & Warnings Log")
    q_rep = None
    try:
        q_rep = client.get_quality_report(dataset_id)
    except Exception:
        pass

    if q_rep:
        errs = q_rep.get("errors", [])
        warns = q_rep.get("warnings", [])
        if not errs and not warns:
            st.success("No validation errors or quality warnings recorded for this dataset.")
        else:
            if errs:
                st.error("Validation Errors:")
                for e in errs:
                    st.write(f"- {e}")
            if warns:
                st.warning("Quality Warnings:")
                for w in warns:
                    st.write(f"- {w}")
    else:
        st.caption("No quality report available to display.")

    st.markdown("---")

    # 3. Permanent Deletion with Confirmation Gate
    st.markdown("#### 🗑️ Danger Zone: Delete Dataset")
    st.write(
        "Permanently removes this survey dataset from the catalog, deletes extracted observations, "
        "and wipes the preserved raw ZIP archive from disk."
    )

    confirm_check = st.checkbox(
        f"I confirm that I want to permanently delete dataset '{dataset.get('session_name')}' (`{dataset_id[:8]}...`)",
        value=False,
        help="Check this confirmation box to enable the permanent deletion button."
    )

    if confirm_check:
        if st.button("Permanently Delete Dataset", type="primary"):
            with st.spinner("Purging dataset workspace and updating catalog..."):
                success, msg = client.delete_dataset(dataset_id)

            if success:
                st.success(msg)
                # Clear session state selection
                st.session_state["selected_dataset_id"] = None
                st.rerun()
            else:
                st.error(msg)
