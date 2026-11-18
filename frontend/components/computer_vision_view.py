"""Streamlit Dashboard component for Indoor Computer Vision & Landmark Detection (Task 25)."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from frontend.api_client import DatasetApiClient, DatasetApiClientException


def render_computer_vision_view(client: DatasetApiClient):
    """Renders Section 15: Indoor Computer Vision & Landmark Detection."""
    st.header("15. Indoor Computer Vision & Landmark Detection")
    st.markdown(
        """
        Detects indoor architectural landmarks (doors, room entrances, staircases, elevators, corridors)
        and signboards (room numbers, direction signs, exit signs) from privacy-sanitized survey photographs.
        Supports OCR text recognition, persistent landmark registration with manual verification, and visual feature matching.
        """
    )

    # 1. Dataset Selection
    try:
        datasets_list = client.list_datasets()
    except Exception as e:
        st.error(f"Failed to connect to backend API: {e}")
        return

    if not datasets_list:
        st.info("No ingested survey datasets found. Please upload a dataset in Section 7.")
        return

    dataset_options = {d["dataset_id"]: f"{d.get('session_name', d['dataset_id'])} ({d['dataset_id'][:8]}...)" for d in datasets_list}
    selected_dataset_id = st.selectbox(
        "Select Multimodal Survey Dataset",
        options=list(dataset_options.keys()),
        format_func=lambda x: dataset_options[x],
        key="vision_dataset_select",
    )

    if not selected_dataset_id:
        return

    # Fetch photo observations
    photo_records = []
    try:
        photo_records = client.get_dataset_observations(selected_dataset_id, "photos", limit=500)
    except Exception:
        pass

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📸 Image Gallery & Detections",
        "⚙️ Pipeline Execution",
        "🏛️ Landmark Registry",
        "🔍 Visual Matcher",
        "📊 Evaluation & Confusion Matrix",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Image Gallery & Detection Viewer
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Survey Photographs & Detected Landmark Overlay")

        if not photo_records:
            st.warning("No photograph records found in this dataset (photo_metadata.csv is empty or missing).")
        else:
            col_sel, col_filt = st.columns([2, 1])
            with col_sel:
                photo_choices = {r["photo_id"]: f"{r.get('filename', r['photo_id'])} (t={r.get('normalized_epoch_ms', 'N/A')})" for r in photo_records}
                selected_photo_id = st.selectbox("Select Survey Photo", options=list(photo_choices.keys()), format_func=lambda x: photo_choices[x])
            with col_filt:
                cat_filter = st.selectbox("Category Filter", ["ALL", "DOOR", "ROOM_ENTRANCE", "STAIRCASE", "ELEVATOR", "CORRIDOR", "ROOM_NUMBER_SIGN", "DIRECTION_SIGN", "EXIT_SIGN"])

            # Retrieve photo record details
            chosen_rec = next((r for r in photo_records if r["photo_id"] == selected_photo_id), None)
            if chosen_rec:
                col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
                with col_meta1:
                    is_san = str(chosen_rec.get("is_exif_sanitized", "")).lower() in ("true", "1")
                    if is_san:
                        st.success("🛡️ EXIF Sanitized: YES")
                    else:
                        st.error("⚠️ Sanitization: UNVERIFIED")
                with col_meta2:
                    st.metric("Building / Floor", f"{chosen_rec.get('building', 'N/A')} / F{chosen_rec.get('floor', 'N/A')}")
                with col_meta3:
                    st.metric("Room / Reference", chosen_rec.get("room") or "N/A")
                with col_meta4:
                    st.metric("Resolution", f"{chosen_rec.get('image_width', 'N/A')}x{chosen_rec.get('image_height', 'N/A')}")

                # Fetch detections for this dataset
                detections = client.get_vision_detections(selected_dataset_id)
                photo_dets = [d for d in detections if d.get("photo_id") in (selected_photo_id, chosen_rec.get("filename"))]

                if cat_filter != "ALL":
                    photo_dets = [d for d in photo_dets if d.get("category") == cat_filter]

                col_img, col_det_list = st.columns([3, 2])
                with col_img:
                    # Attempt loading photo from backend
                    fn = chosen_rec.get("filename", f"{selected_photo_id}.jpg")
                    try:
                        img_bytes = client.get_photo_bytes(selected_dataset_id, fn)
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        # Draw bounding boxes
                        annotated = _overlay_boxes(pil_img, photo_dets)
                        st.image(annotated, use_container_width=True, caption=f"Annotated: {fn} ({len(photo_dets)} landmarks)")
                    except Exception as e:
                        st.warning(f"Could not load image bytes: {e}. (Ensure image is EXIF-sanitized).")

                with col_det_list:
                    st.markdown("#### Detected Landmark Candidates")
                    if not photo_dets:
                        st.info("No landmark detections found for this photograph. Run Pipeline Execution in Tab 2.")
                    else:
                        for d in photo_dets:
                            cat = d.get("category", "UNKNOWN")
                            conf = d.get("confidence", 0.0)
                            ocr = d.get("ocr_record")
                            with st.container(border=True):
                                st.markdown(f"**{cat}** — Confidence: `{int(conf*100)}%`")
                                bb = d.get("bounding_box", {})
                                st.caption(f"Box: [{bb.get('x_min'):.2f}, {bb.get('y_min'):.2f}] to [{bb.get('x_max'):.2f}, {bb.get('y_max'):.2f}]")
                                if ocr and ocr.get("text"):
                                    st.markdown(f"🔤 **OCR Text:** `{ocr.get('text')}` (conf: {int(ocr.get('confidence',0)*100)}%)")
                                    st.caption("*(Unverified text candidate)*")

    # -------------------------------------------------------------------------
    # TAB 2: Pipeline Execution & Batch Analysis
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Run Computer Vision Landmark Detection Pipeline")
        st.markdown(
            """
            Analyzes photographs associated with the survey session. Automatically enforces EXIF sanitization checks,
            applies contrast normalization, detects architectural elements, and performs OCR on signboards.
            """
        )

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            conf_thresh = st.slider("Confidence Threshold", 0.20, 0.90, 0.40, 0.05)
        with col_c2:
            enable_ocr = st.checkbox("Enable Signboard OCR", value=True)
        with col_c3:
            auto_reg = st.checkbox("Auto-Register Detections", value=False, help="Adds detections to Landmark Registry as UNVERIFIED")

        if st.button("🚀 Run Landmark Detection Pipeline", type="primary"):
            with st.spinner("Analyzing survey photographs with computer vision & OCR..."):
                try:
                    res = client.analyze_vision(
                        selected_dataset_id,
                        {
                            "confidence_threshold": conf_thresh,
                            "enable_ocr": enable_ocr,
                            "auto_register": auto_reg,
                        }
                    )
                    st.success("Vision analysis completed successfully!")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Photos Scanned", res.get("total_photos_scanned", 0))
                    c2.metric("Valid Processed", res.get("photos_processed_valid", 0))
                    c3.metric("Blocked (Privacy)", res.get("photos_blocked_privacy", 0))
                    c4.metric("Landmarks Detected", res.get("total_landmarks_detected", 0))
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    # -------------------------------------------------------------------------
    # TAB 3: Landmark Registry & Manual Verification
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Physical Landmark Registry")
        st.markdown(
            """
            Persistent catalog of confirmed and candidate indoor landmarks.
            **Strict Rule:** Detections remain `UNVERIFIED` until explicitly confirmed with physical coordinates.
            """
        )

        col_b, col_f, col_s = st.columns(3)
        with col_b:
            b_filter = st.text_input("Building ID Filter", value="")
        with col_f:
            f_filter = st.text_input("Floor ID Filter", value="")
        with col_s:
            s_filter = st.selectbox("Verification Status", ["ALL", "UNVERIFIED", "MANUALLY_VERIFIED", "REJECTED"])

        bldg_val = b_filter.strip() if b_filter.strip() else None
        floor_val = int(f_filter.strip()) if f_filter.strip().isdigit() else None
        stat_val = s_filter if s_filter != "ALL" else None

        landmarks = client.list_landmarks(
            selected_dataset_id,
            building_id=bldg_val,
            floor_id=floor_val,
            status=stat_val,
        )

        st.caption(f"Showing {len(landmarks)} landmark records.")

        if landmarks:
            # Display summary table
            table_rows = []
            for lm in landmarks:
                table_rows.append({
                    "Landmark ID": lm.get("landmark_id"),
                    "Status": "✅ VERIFIED" if lm.get("verification_status") == "MANUALLY_VERIFIED" else ("❌ REJECTED" if lm.get("verification_status") == "REJECTED" else "⏳ UNVERIFIED"),
                    "Category": lm.get("category"),
                    "Name": lm.get("name"),
                    "Building": lm.get("building_id"),
                    "Floor": lm.get("floor_id"),
                    "Coords (X, Y)": f"({lm['verified_coordinates_m'][0]:.1f}, {lm['verified_coordinates_m'][1]:.1f})" if lm.get("verified_coordinates_m") else "None",
                    "OCR Text": lm.get("ocr_text") or "None",
                    "Templates": lm.get("template_count", 0),
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            # Verification edit form
            st.markdown("#### Manually Verify / Update Landmark")
            with st.form("verify_landmark_form"):
                lm_ids = [l["landmark_id"] for l in landmarks]
                target_lm_id = st.selectbox("Select Landmark to Verify", options=lm_ids)
                target_lm = next((l for l in landmarks if l["landmark_id"] == target_lm_id), None)

                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    new_name = st.text_input("Verified Name", value=target_lm.get("name", "") if target_lm else "")
                    new_status = st.selectbox("Verification Status", ["MANUALLY_VERIFIED", "UNVERIFIED", "REJECTED"], index=0)
                with col_v2:
                    c_x = st.number_input("Verified East Coordinate (X meters)", value=float(target_lm["verified_coordinates_m"][0]) if (target_lm and target_lm.get("verified_coordinates_m")) else 0.0)
                    c_y = st.number_input("Verified North Coordinate (Y meters)", value=float(target_lm["verified_coordinates_m"][1]) if (target_lm and target_lm.get("verified_coordinates_m")) else 0.0)

                new_desc = st.text_area("Description / Landmark Notes", value=target_lm.get("description", "") if target_lm else "")
                submit_verify = st.form_submit_button("Confirm Landmark Verification")

                if submit_verify and target_lm:
                    try:
                        client.verify_landmark(
                            selected_dataset_id,
                            {
                                "landmark_id": target_lm_id,
                                "building_id": target_lm.get("building_id", "BLDG-1"),
                                "status": new_status,
                                "verified_name": new_name,
                                "coordinates_m": [c_x, c_y],
                                "description": new_desc,
                            }
                        )
                        st.success(f"Successfully updated landmark {target_lm_id}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update landmark: {e}")
        else:
            st.info("No landmarks in registry matching filters. Process survey photos in Tab 2 with 'Auto-Register Detections' checked.")

    # -------------------------------------------------------------------------
    # TAB 4: Visual Landmark Matching
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("Visual Descriptor Landmark Matching")
        st.markdown(
            """
            Matches an indoor query photograph against registered physical landmarks using ORB local descriptors
            and appearance color embeddings.
            
            > **Note:** Similarity scores represent descriptor correspondence and are **not** probabilities.
            > Insufficient feature overlap returns `NO_MATCH`.
            """
        )

        if not photo_records:
            st.warning("No survey photographs available to match.")
        else:
            col_m_sel, col_m_btn = st.columns([3, 1])
            with col_m_sel:
                q_photo_choices = {r["photo_id"]: r.get("filename", r["photo_id"]) for r in photo_records}
                query_photo_id = st.selectbox("Select Query Photo to Match", options=list(q_photo_choices.keys()), format_func=lambda x: q_photo_choices[x])
            with col_m_btn:
                st.write("")
                st.write("")
                run_match = st.button("🔍 Match against Registry")

            if run_match:
                q_rec = next((r for r in photo_records if r["photo_id"] == query_photo_id), None)
                fn = q_rec.get("filename", f"{query_photo_id}.jpg") if q_rec else query_photo_id
                with st.spinner("Matching visual descriptors against landmark registry..."):
                    try:
                        match_res = client.match_survey_photo(
                            selected_dataset_id,
                            {
                                "photo_filename": fn,
                                "max_candidates": 5,
                            }
                        )
                        st.markdown("### Match Outcome")
                        st_status = match_res.get("match_status", "NO_MATCH")
                        if st_status == "STRONG_MATCH":
                            st.success(f"🎯 **Status:** `{st_status}` (Best Candidate: `{match_res.get('best_candidate_id')}`, Score: `{match_res.get('similarity_score'):.3f}`)")
                        elif st_status == "WEAK_MATCH":
                            st.warning(f"⚠️ **Status:** `{st_status}` (Candidate: `{match_res.get('best_candidate_id')}`, Score: `{match_res.get('similarity_score'):.3f}`)")
                        else:
                            st.error(f"❌ **Status:** `{st_status}` — Insufficient visual feature correspondence.")

                        st.caption(match_res.get("notes", ""))

                        cands = match_res.get("candidates", [])
                        if cands:
                            cand_df = pd.DataFrame([
                                {
                                    "Landmark ID": c.get("landmark_id"),
                                    "Name": c.get("landmark_name"),
                                    "Category": c.get("category"),
                                    "Similarity Score": f"{c.get('similarity_score', 0):.3f}",
                                    "ORB Inliers": c.get("inlier_count"),
                                    "Status": c.get("match_status"),
                                    "Building / Floor": f"{c.get('building_id')} / F{c.get('floor_id')}",
                                }
                                for c in cands
                            ])
                            st.dataframe(cand_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Visual matching error: {e}")

    # -------------------------------------------------------------------------
    # TAB 5: Performance Evaluation & Confusion Matrix
    # -------------------------------------------------------------------------
    with tab5:
        st.subheader("Landmark Detection & OCR Performance Evaluation")
        st.markdown(
            """
            Evaluates multi-class detection precision, recall, and F1 at IoU $\ge 0.50$,
            OCR Character Error Rate (CER), and visual matching accuracy.
            """
        )

        col_ev1, col_ev2 = st.columns([2, 1])
        with col_ev1:
            eval_mode = st.radio("Evaluation Benchmark Target", ["Controlled Synthetic Ground-Truth Fixture", "Real Survey Labeled Callset"], horizontal=True)
        with col_ev2:
            iou_cut = st.slider("IoU Evaluation Threshold", 0.30, 0.80, 0.50, 0.05)

        is_synth = (eval_mode == "Controlled Synthetic Ground-Truth Fixture")

        if st.button("📊 Run Evaluation Benchmark"):
            with st.spinner("Computing precision, recall, confusion matrix, and OCR error..."):
                try:
                    eval_rep = client.evaluate_vision(
                        selected_dataset_id,
                        {
                            "is_synthetic": is_synth,
                            "iou_threshold": iou_cut,
                        }
                    )

                    if eval_rep.get("is_synthetic_fixture"):
                        st.info("🧪 **Benchmark Mode:** Controlled synthetic ground-truth fixture with strictly partitioned image sessions.")
                    else:
                        st.caption("Benchmark Mode: Real survey manual labels.")

                    # Metrics cards
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Overall Precision", f"{eval_rep.get('overall_precision', 0)*100:.1f}%")
                    c2.metric("Overall Recall", f"{eval_rep.get('overall_recall', 0)*100:.1f}%")
                    c3.metric("Overall F1-Score", f"{eval_rep.get('overall_f1', 0)*100:.1f}%")
                    c4.metric("Mean Detection IoU", f"{eval_rep.get('mean_iou', 0):.2f}")

                    if eval_rep.get("ocr_exact_match_rate") is not None:
                        co1, co2 = st.columns(2)
                        co1.metric("OCR Exact Match Rate", f"{eval_rep.get('ocr_exact_match_rate', 0)*100:.1f}%")
                        co2.metric("OCR Character Error Rate (CER)", f"{eval_rep.get('ocr_character_error_rate', 0)*100:.1f}%")

                    # Per-category chart
                    per_cat = eval_rep.get("per_category", {})
                    if per_cat:
                        st.markdown("#### Per-Category Detection Performance")
                        cat_rows = []
                        for cat, m in per_cat.items():
                            if m.get("true_positives", 0) > 0 or m.get("false_positives", 0) > 0 or m.get("false_negatives", 0) > 0:
                                cat_rows.append({
                                    "Category": cat,
                                    "Precision": m.get("precision", 0),
                                    "Recall": m.get("recall", 0),
                                    "F1-Score": m.get("f1_score", 0),
                                    "TP": m.get("true_positives", 0),
                                    "FP": m.get("false_positives", 0),
                                    "FN": m.get("false_negatives", 0),
                                })

                        if cat_rows:
                            cat_df = pd.DataFrame(cat_rows)
                            fig = px.bar(
                                cat_df,
                                x="Category",
                                y=["Precision", "Recall", "F1-Score"],
                                barmode="group",
                                title="Precision, Recall, and F1 by Landmark Category",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(cat_df, use_container_width=True)

                    # Confusion Matrix Heatmap
                    conf_mat = eval_rep.get("confusion_matrix")
                    if conf_mat:
                        st.markdown("#### Category Confusion Matrix")
                        actual_labels = [k for k in conf_mat.keys() if any(conf_mat[k].values())]
                        pred_labels = [k for k in next(iter(conf_mat.values())).keys() if any(conf_mat[a].get(k, 0) for a in actual_labels)]

                        if actual_labels and pred_labels:
                            matrix_vals = [[conf_mat[a].get(p, 0) for p in pred_labels] for a in actual_labels]
                            fig_cm = px.imshow(
                                matrix_vals,
                                x=pred_labels,
                                y=actual_labels,
                                labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Count"),
                                text_auto=True,
                                color_continuous_scale="Blues",
                                title="Detection Classification Confusion Matrix",
                            )
                            st.plotly_chart(fig_cm, use_container_width=True)

                except Exception as e:
                    st.error(f"Evaluation failed: {e}")


def _overlay_boxes(pil_img: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    """Draws bounding boxes and labels onto a PIL Image."""
    import cv2
    np_img = np.array(pil_img)
    if len(np_img.shape) == 2:
        bgr = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
    elif np_img.shape[2] == 4:
        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    for d in detections:
        bb = d.get("bounding_box", {})
        px_min = int(round(bb.get("x_min", 0) * w))
        py_min = int(round(bb.get("y_min", 0) * h))
        px_max = int(round(bb.get("x_max", 1) * w))
        py_max = int(round(bb.get("y_max", 1) * h))

        cat = d.get("category", "UNKNOWN")
        conf = int(d.get("confidence", 0) * 100)
        label = f"{cat} {conf}%"
        if d.get("ocr_record") and d["ocr_record"].get("text"):
            label += f" [{d['ocr_record']['text']}]"

        color = (0, 255, 0)
        if "SIGN" in cat:
            color = (0, 200, 255)
        elif "DOOR" in cat:
            color = (0, 140, 255)
        elif "STAIR" in cat:
            color = (255, 100, 0)

        cv2.rectangle(bgr, (px_min, py_min), (px_max, py_max), color, 2)
        cv2.putText(bgr, label, (px_min + 3, max(15, py_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    rgb_res = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_res)
