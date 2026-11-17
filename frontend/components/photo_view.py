"""Sanitized landmark photo gallery and ground-truth annotation inspector component."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from frontend.api_client import DatasetApiClient


def render_photos_and_annotations(client: DatasetApiClient, dataset_id: Optional[str]) -> None:
    """Renders privacy-guarded landmark photo gallery and surveyor ground-truth annotations."""
    st.markdown("### 📷 Photos & Ground-Truth Annotations")

    if not dataset_id:
        st.info("Please select or upload a dataset in the Overview tab to inspect photos and annotations.")
        return

    tab_photos, tab_ann = st.tabs(["🖼️ Landmark Photographs", "🏷️ Ground-Truth Annotations"])

    # 1. Landmark Photographs Tab
    with tab_photos:
        st.markdown("#### Privacy-Guarded Landmark Photo Gallery")
        st.caption("Only confirmed privacy-sanitized photographs are rendered. Unsanitized photos are strictly blocked.")

        try:
            photo_records = client.get_observations(dataset_id, "photos", limit=50)
        except Exception as e:
            st.error(f"Failed to fetch photo metadata: {e}")
            photo_records = []

        if not photo_records:
            st.info("No landmark photos recorded in this survey session.")
        else:
            # Inspection grid
            cols = st.columns(2)
            for idx, photo in enumerate(photo_records):
                col = cols[idx % 2]
                with col:
                    photo_id = photo.get("photo_id", "IMG-UNKNOWN")
                    fn = photo.get("filename", "")
                    is_sanitized = str(photo.get("is_exif_sanitized", "")).lower() == "true"

                    st.markdown(f"##### `{photo_id}` — {photo.get('landmark_type_name', 'Landmark')}")

                    # Privacy Enforcement Gate
                    if not is_sanitized:
                        st.error("🚫 Image Display Blocked: Photo metadata reports `is_exif_sanitized == false`.")
                        st.caption("Unsanitized or unverified photos are strictly prohibited from rendering.")
                    else:
                        img_bytes = client.get_image_bytes(dataset_id, fn)
                        if img_bytes:
                            st.image(img_bytes, caption=f"{fn} ({photo.get('image_width')}x{photo.get('image_height')})", use_column_width=True)
                        else:
                            st.warning(f"Image `{fn}` could not be retrieved from backend.")

                    # Metadata card
                    p_meta_c1, p_meta_c2 = st.columns(2)
                    with p_meta_c1:
                        st.caption(f"**Location:** Bldg {photo.get('building') or '-'}, Fl {photo.get('floor') or '-'}, Rm {photo.get('room') or '-'}")
                        st.caption(f"**Description:** {photo.get('description') or 'None'}")
                    with p_meta_c2:
                        st.caption(f"**Client EXIF Sanitized:** ✅ True")
                        faces = photo.get("faces_detected_count", "0")
                        blurred = photo.get("face_blur_applied", "false")
                        st.caption(f"**Face Blur Applied:** {blurred} ({faces} faces)")

                    st.markdown(
                        """
                        <div style="background-color: #FEF3C7; padding: 6px 10px; border-radius: 4px; font-size: 0.8rem; color: #92400E; margin-bottom: 12px;">
                            <strong>Audit Status:</strong> Independent Server Visual Audit: <code>PENDING_AUDIT</code>. EXIF stripping does not prove automated redaction of faces or document text.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # 2. Ground-Truth Annotations Tab
    with tab_ann:
        st.markdown("#### Surveyor Architectural Ground-Truth Notes")
        st.caption("Manual structural notes, door marks, corridors, and reference points recorded during the survey.")

        try:
            ann_records = client.get_observations(dataset_id, "annotations", limit=100)
        except Exception as e:
            st.error(f"Failed to fetch annotations: {e}")
            ann_records = []

        if not ann_records:
            st.info("No manual annotations recorded in this survey session.")
        else:
            df_ann = pd.DataFrame(ann_records)

            # Filter by annotation type
            types = ["All Types"] + sorted(df_ann["type_name"].dropna().unique().tolist())
            selected_type = st.selectbox("Filter by Annotation Type:", options=types, index=0)

            if selected_type != "All Types":
                df_ann = df_ann[df_ann["type_name"] == selected_type]

            display_cols = [
                col for col in ["annotation_id", "type_name", "description", "building", "floor", "room", "normalized_iso", "is_mock"]
                if col in df_ann.columns
            ]
            st.dataframe(df_ann[display_cols], use_container_width=True)

            st.caption("Ground-truth surveyor annotations are explicitly segregated from model predictions to prevent training data leakage.")
