"""Visual feature extraction using ORB local descriptors and HSV color histograms."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.core.logging import setup_logger

logger = setup_logger("FeatureExtraction")


class VisualFeatureExtractor:
    """Extracts local ORB keypoints/descriptors and global appearance embeddings for landmark matching."""

    def __init__(self, max_orb_features: int = 500):
        self.max_orb_features = max_orb_features
        # Initialize ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=max_orb_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,
        )

    def extract_orb_descriptors(self, bgr_img: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Extracts ORB keypoints and (N, 32) uint8 descriptors from an image or crop."""
        if bgr_img is None or bgr_img.size == 0:
            return [], None

        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY) if len(bgr_img.shape) == 3 else bgr_img
        # Enhance local contrast for keypoint detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        keypoints, descriptors = self.orb.detectAndCompute(enhanced, None)
        return keypoints, descriptors

    def extract_global_embedding(self, bgr_img: np.ndarray, bins: Tuple[int, int, int] = (8, 8, 8)) -> List[float]:
        """Extracts a normalized 3D HSV color histogram representation as a compact 512-dim embedding."""
        if bgr_img is None or bgr_img.size == 0:
            return [0.0] * (bins[0] * bins[1] * bins[2])

        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            list(bins),
            [0, 180, 0, 256, 0, 256],
        )
        hist_flat = hist.flatten().astype(np.float64)
        norm = np.linalg.norm(hist_flat)
        if norm > 1e-7:
            hist_flat /= norm
        return [round(float(v), 5) for v in hist_flat]

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Computes cosine similarity between two 1D feature vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        a = np.array(vec1, dtype=np.float64)
        b = np.array(vec2, dtype=np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a <= 1e-9 or norm_b <= 1e-9:
            return 0.0
        sim = float(np.dot(a, b) / (norm_a * norm_b))
        return max(0.0, min(1.0, sim))
