"""Multimodal data preprocessing and temporal alignment package (Task 20)."""

from src.multimodal_preprocessing.annotations import AnnotationProcessor, AnnotationProcessingStats
from src.multimodal_preprocessing.config import MultimodalPreprocessingConfig
from src.multimodal_preprocessing.photos import PhotoMetadataProcessor, PhotoProcessingStats
from src.multimodal_preprocessing.pipeline import MultimodalPreprocessingPipeline
from src.multimodal_preprocessing.reporting import (
    PreprocessingManifest,
    PreprocessingReport,
    calculate_sha256,
)
from src.multimodal_preprocessing.segmentation import SegmentRecord, SessionSegmenter
from src.multimodal_preprocessing.sensors import SensorCleaningStats, SensorPreprocessor
from src.multimodal_preprocessing.timestamps import AlignmentStats, TimestampAligner
from src.multimodal_preprocessing.validation import MultimodalPreValidator, PreValidationResult
from src.multimodal_preprocessing.wifi import WifiCleaningStats, WifiPreprocessor

__all__ = [
    "MultimodalPreprocessingConfig",
    "MultimodalPreprocessingPipeline",
    "WifiPreprocessor",
    "WifiCleaningStats",
    "SensorPreprocessor",
    "SensorCleaningStats",
    "TimestampAligner",
    "AlignmentStats",
    "PhotoMetadataProcessor",
    "PhotoProcessingStats",
    "AnnotationProcessor",
    "AnnotationProcessingStats",
    "SessionSegmenter",
    "SegmentRecord",
    "MultimodalPreValidator",
    "PreValidationResult",
    "PreprocessingReport",
    "PreprocessingManifest",
    "calculate_sha256",
]
