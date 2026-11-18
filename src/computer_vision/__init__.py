"""IndoorAI Computer Vision Module: Indoor landmark detection, OCR recognition, and visual matching."""

from src.computer_vision.evaluation import (
    GroundTruthLandmark,
    VisionEvaluator,
)
from src.computer_vision.feature_extraction import VisualFeatureExtractor
from src.computer_vision.image_loader import ImageLoadingError, SafeImageLoader
from src.computer_vision.image_validation import (
    ImagePrivacyValidator,
    ImageValidationResult,
)
from src.computer_vision.landmark_classifier import LandmarkClassifier
from src.computer_vision.landmark_detector import LandmarkDetector
from src.computer_vision.landmark_matching import VisualLandmarkMatcher
from src.computer_vision.landmark_registry import LandmarkRegistry
from src.computer_vision.models import (
    BoundingBox,
    CategoryEvaluationMetrics,
    LandmarkCategory,
    LandmarkDetection,
    LandmarkRecord,
    OcrTextRecord,
    VerificationStatus,
    VisionEvaluationReport,
    VisualMatchCandidate,
    VisualMatchReport,
    VisualMatchStatus,
)
from src.computer_vision.ocr_engine import SignboardOcrEngine
from src.computer_vision.pipeline import VisionPipeline
from src.computer_vision.preprocessing import ImagePreprocessor

__all__ = [
    "BoundingBox",
    "CategoryEvaluationMetrics",
    "GroundTruthLandmark",
    "ImageLoadingError",
    "ImagePreprocessor",
    "ImagePrivacyValidator",
    "ImageValidationResult",
    "LandmarkCategory",
    "LandmarkClassifier",
    "LandmarkDetection",
    "LandmarkDetector",
    "LandmarkRecord",
    "LandmarkRegistry",
    "OcrTextRecord",
    "SafeImageLoader",
    "SignboardOcrEngine",
    "VerificationStatus",
    "VisionEvaluationReport",
    "VisionEvaluator",
    "VisionPipeline",
    "VisualFeatureExtractor",
    "VisualLandmarkMatcher",
    "VisualMatchCandidate",
    "VisualMatchReport",
    "VisualMatchStatus",
]
