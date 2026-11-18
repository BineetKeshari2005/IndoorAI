"""Evaluation framework for indoor landmark detection, OCR recognition, and visual matching."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.computer_vision.models import (
    BoundingBox,
    CategoryEvaluationMetrics,
    LandmarkCategory,
    LandmarkDetection,
    VisionEvaluationReport,
    VisualMatchReport,
    VisualMatchStatus,
)
from src.core.logging import setup_logger

logger = setup_logger("VisionEvaluation")


@dataclass
class GroundTruthLandmark:
    """Ground-truth bounding box and category for an annotated survey image."""
    photo_id: str
    category: LandmarkCategory
    bounding_box: BoundingBox
    text: Optional[str] = None
    session_id: str = ""
    building_id: Optional[str] = None
    floor_id: Optional[int] = None


class VisionEvaluator:
    """Evaluates landmark detector precision/recall, OCR error rates, and visual matching accuracy."""

    def __init__(self, iou_threshold: float = 0.50):
        self.iou_threshold = iou_threshold

    def evaluate_detections(
        self,
        predictions_by_photo: Dict[str, List[LandmarkDetection]],
        ground_truth_by_photo: Dict[str, List[GroundTruthLandmark]],
        is_synthetic_fixture: bool = False,
        matching_reports: Optional[List[VisualMatchReport]] = None,
        ground_truth_matches: Optional[Dict[str, str]] = None,
        notes: str = "",
    ) -> VisionEvaluationReport:
        """Computes multi-class precision, recall, F1, confusion matrix, and OCR error metrics."""
        all_categories = [c.value for c in LandmarkCategory if c != LandmarkCategory.UNKNOWN]
        cat_metrics: Dict[str, CategoryEvaluationMetrics] = {
            cat: CategoryEvaluationMetrics(category=cat) for cat in all_categories
        }

        # Confusion matrix: [actual][predicted]
        conf_matrix: Dict[str, Dict[str, int]] = {
            act: {pred: 0 for pred in all_categories + ["BACKGROUND"]}
            for act in all_categories + ["BACKGROUND"]
        }

        total_gt = 0
        total_preds = 0
        iou_sum = 0.0
        iou_count = 0

        # OCR metrics accumulators
        ocr_total = 0
        ocr_exact_matches = 0
        ocr_cer_sum = 0.0

        all_photo_ids = set(ground_truth_by_photo.keys()).union(predictions_by_photo.keys())

        for pid in all_photo_ids:
            gts = ground_truth_by_photo.get(pid, [])
            preds = predictions_by_photo.get(pid, [])

            total_gt += len(gts)
            total_preds += len(preds)

            matched_gt_indices: Set[int] = set()

            # Match predictions to ground truth by greedy highest IoU
            for pred in preds:
                pred_cat = pred.category.value
                best_iou = 0.0
                best_gt_idx = -1

                for g_idx, gt in enumerate(gts):
                    if g_idx in matched_gt_indices:
                        continue
                    iou = pred.bounding_box.iou(gt.bounding_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_gt_idx >= 0 and best_iou >= self.iou_threshold:
                    matched_gt = gts[best_gt_idx]
                    matched_gt_indices.add(best_gt_idx)
                    gt_cat = matched_gt.category.value

                    iou_sum += best_iou
                    iou_count += 1

                    if pred_cat == gt_cat:
                        if pred_cat in cat_metrics:
                            cat_metrics[pred_cat].true_positives += 1
                        conf_matrix[gt_cat][pred_cat] += 1

                        # Evaluate OCR if ground truth text is provided
                        if matched_gt.text and pred.ocr_record and pred.ocr_record.text:
                            ocr_total += 1
                            pred_txt = pred.ocr_record.text.strip().upper()
                            gt_txt = matched_gt.text.strip().upper()
                            if pred_txt == gt_txt:
                                ocr_exact_matches += 1
                            # Character Error Rate (CER) via Levenshtein edit distance
                            cer = self._calculate_cer(pred_txt, gt_txt)
                            ocr_cer_sum += cer
                    else:
                        # Classification mismatch
                        if pred_cat in cat_metrics:
                            cat_metrics[pred_cat].false_positives += 1
                        conf_matrix[gt_cat][pred_cat] += 1
                else:
                    # False positive (hallucinated detection or low IoU)
                    if pred_cat in cat_metrics:
                        cat_metrics[pred_cat].false_positives += 1
                    conf_matrix["BACKGROUND"][pred_cat] += 1

            # Any unmatched ground truth is a false negative
            for g_idx, gt in enumerate(gts):
                if g_idx not in matched_gt_indices:
                    gt_cat = gt.category.value
                    if gt_cat in cat_metrics:
                        cat_metrics[gt_cat].false_negatives += 1
                    conf_matrix[gt_cat]["BACKGROUND"] += 1

        # Finalize per-category metrics
        total_tp = sum(m.true_positives for m in cat_metrics.values())
        total_fp = sum(m.false_positives for m in cat_metrics.values())
        total_fn = sum(m.false_negatives for m in cat_metrics.values())

        for m in cat_metrics.values():
            m.compute()

        # Overall Micro-Averaged Precision, Recall, F1
        overall_prec = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        overall_rec = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2.0 * overall_prec * overall_rec / (overall_prec + overall_rec)) if (overall_prec + overall_rec) > 0 else 0.0
        mean_iou = (iou_sum / iou_count) if iou_count > 0 else 0.0

        # OCR summary
        ocr_exact_rate = (ocr_exact_matches / ocr_total) if ocr_total > 0 else None
        ocr_mean_cer = (ocr_cer_sum / ocr_total) if ocr_total > 0 else None

        # Visual matching evaluation
        match_prec, match_rec = None, None
        if matching_reports and ground_truth_matches:
            match_prec, match_rec = self._evaluate_visual_matching(matching_reports, ground_truth_matches)

        report = VisionEvaluationReport(
            is_synthetic_fixture=is_synthetic_fixture,
            total_images_evaluated=len(all_photo_ids),
            total_ground_truth_boxes=total_gt,
            total_predictions=total_preds,
            overall_precision=overall_prec,
            overall_recall=overall_rec,
            overall_f1=overall_f1,
            mean_iou=mean_iou,
            per_category=cat_metrics,
            confusion_matrix=conf_matrix,
            ocr_exact_match_rate=ocr_exact_rate,
            ocr_character_error_rate=ocr_mean_cer,
            matching_precision=match_prec,
            matching_recall=match_rec,
            notes=notes or ("Controlled synthetic fixture" if is_synthetic_fixture else "Survey evaluation"),
        )
        return report

    @staticmethod
    def _calculate_cer(reference: str, hypothesis: str) -> float:
        """Calculates normalized Character Error Rate (CER) using Levenshtein distance."""
        matcher = difflib.SequenceMatcher(None, reference, hypothesis)
        # Edit distance = insertions + deletions + substitutions
        dist = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                dist += max(i2 - i1, j2 - j1)
            elif tag == "delete":
                dist += (i2 - i1)
            elif tag == "insert":
                dist += (j2 - j1)
        return dist / float(max(1, len(reference)))

    @staticmethod
    def _evaluate_visual_matching(
        reports: List[VisualMatchReport],
        ground_truth: Dict[str, str],  # query_id -> expected_landmark_id
    ) -> Tuple[float, float]:
        """Calculates Precision and Recall for visual landmark matching."""
        tp, fp, fn = 0, 0, 0
        for r in reports:
            expected = ground_truth.get(r.query_image_id)
            if not expected:
                continue

            if r.match_status != VisualMatchStatus.NO_MATCH and r.best_candidate_id:
                if r.best_candidate_id == expected:
                    tp += 1
                else:
                    fp += 1
            else:
                fn += 1

        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        return round(prec, 4), round(rec, 4)

    @classmethod
    def generate_controlled_synthetic_vision_fixture(
        cls,
    ) -> Tuple[Dict[str, List[LandmarkDetection]], Dict[str, List[GroundTruthLandmark]], List[VisualMatchReport], Dict[str, str]]:
        """Generates a strictly labeled synthetic evaluation fixture for unit testing and CI verification.

        Guarantees session-level partition and deterministic ground-truth without fabricating real results.
        """
        # Session A: Training/Gallery photos
        # Session B: Independent Evaluation photos (no leakage)
        ground_truth: Dict[str, List[GroundTruthLandmark]] = {
            "IMG-SES-B-001": [
                GroundTruthLandmark(
                    photo_id="IMG-SES-B-001",
                    category=LandmarkCategory.DOOR,
                    bounding_box=BoundingBox(0.20, 0.15, 0.50, 0.85),
                    session_id="SES-B",
                    building_id="BLDG-1",
                    floor_id=1,
                ),
                GroundTruthLandmark(
                    photo_id="IMG-SES-B-001",
                    category=LandmarkCategory.ROOM_NUMBER_SIGN,
                    bounding_box=BoundingBox(0.55, 0.30, 0.70, 0.45),
                    text="ROOM 101",
                    session_id="SES-B",
                    building_id="BLDG-1",
                    floor_id=1,
                ),
            ],
            "IMG-SES-B-002": [
                GroundTruthLandmark(
                    photo_id="IMG-SES-B-002",
                    category=LandmarkCategory.STAIRCASE,
                    bounding_box=BoundingBox(0.10, 0.20, 0.90, 0.90),
                    session_id="SES-B",
                    building_id="BLDG-1",
                    floor_id=1,
                )
            ],
            "IMG-SES-B-003": [
                GroundTruthLandmark(
                    photo_id="IMG-SES-B-003",
                    category=LandmarkCategory.EXIT_SIGN,
                    bounding_box=BoundingBox(0.40, 0.05, 0.60, 0.20),
                    text="EXIT",
                    session_id="SES-B",
                    building_id="BLDG-1",
                    floor_id=1,
                )
            ],
        }

        # Simulated predictions (with 1 true positive door, 1 true positive sign, 1 true positive stair, 1 false positive)
        from src.computer_vision.models import OcrTextRecord
        predictions: Dict[str, List[LandmarkDetection]] = {
            "IMG-SES-B-001": [
                LandmarkDetection(
                    category=LandmarkCategory.DOOR,
                    bounding_box=BoundingBox(0.21, 0.14, 0.49, 0.86),  # IoU > 0.85
                    confidence=0.88,
                    photo_id="IMG-SES-B-001",
                    building_id="BLDG-1",
                    floor_id=1,
                ),
                LandmarkDetection(
                    category=LandmarkCategory.ROOM_NUMBER_SIGN,
                    bounding_box=BoundingBox(0.54, 0.31, 0.71, 0.44),  # IoU > 0.80
                    confidence=0.92,
                    photo_id="IMG-SES-B-001",
                    ocr_record=OcrTextRecord(text="ROOM 101", confidence=0.95),
                    building_id="BLDG-1",
                    floor_id=1,
                ),
            ],
            "IMG-SES-B-002": [
                LandmarkDetection(
                    category=LandmarkCategory.STAIRCASE,
                    bounding_box=BoundingBox(0.12, 0.22, 0.88, 0.88),  # IoU > 0.85
                    confidence=0.79,
                    photo_id="IMG-SES-B-002",
                    building_id="BLDG-1",
                    floor_id=1,
                ),
                LandmarkDetection(
                    category=LandmarkCategory.CORRIDOR,  # Spurious false positive
                    bounding_box=BoundingBox(0.05, 0.05, 0.95, 0.95),
                    confidence=0.42,
                    photo_id="IMG-SES-B-002",
                    building_id="BLDG-1",
                    floor_id=1,
                ),
            ],
            "IMG-SES-B-003": [
                LandmarkDetection(
                    category=LandmarkCategory.EXIT_SIGN,
                    bounding_box=BoundingBox(0.41, 0.06, 0.59, 0.19),  # IoU > 0.80
                    confidence=0.94,
                    photo_id="IMG-SES-B-003",
                    ocr_record=OcrTextRecord(text="EXIT", confidence=0.98),
                    building_id="BLDG-1",
                    floor_id=1,
                )
            ],
        }

        # Simulated visual matching reports
        match_reports = [
            VisualMatchReport(
                query_image_id="IMG-SES-B-001",
                match_status=VisualMatchStatus.STRONG_MATCH,
                best_candidate_id="LM-BLDG-1-F01-DOO-0001",
                similarity_score=0.84,
            ),
            VisualMatchReport(
                query_image_id="IMG-SES-B-002",
                match_status=VisualMatchStatus.STRONG_MATCH,
                best_candidate_id="LM-BLDG-1-F01-STA-0002",
                similarity_score=0.78,
            ),
        ]
        match_gt = {
            "IMG-SES-B-001": "LM-BLDG-1-F01-DOO-0001",
            "IMG-SES-B-002": "LM-BLDG-1-F01-STA-0002",
        }

        return predictions, ground_truth, match_reports, match_gt
