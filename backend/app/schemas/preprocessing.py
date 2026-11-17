"""Pydantic schemas for multimodal preprocessing endpoints (Task 20)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PreprocessRequest(BaseModel):
    """Optional configuration parameters passed when requesting preprocessing."""

    min_rssi_dbm: float = Field(-105.0, description="Minimum acceptable RSSI in dBm")
    max_rssi_dbm: float = Field(-20.0, description="Maximum acceptable RSSI in dBm")
    deduplicate_wifi_ap: str = Field("keep_strongest", description="AP deduplication: 'keep_strongest', 'keep_first', 'reject'")
    sensor_gap_threshold_ms: float = Field(500.0, description="Sensor sampling gap threshold in ms")
    temporal_window_size_ms: float = Field(500.0, description="Temporal alignment window size in ms")
    segmentation_strategy: str = Field("fixed_duration", description="Segmentation strategy: 'fixed_duration', 'observation_count', 'annotation_boundaries', 'temporal_gaps'")
    segmentation_duration_seconds: float = Field(60.0, description="Fixed segment duration in seconds")
    filter_unsanitized_photos: bool = Field(True, description="Whether to exclude unsanitized photos")


class PreprocessResponse(BaseModel):
    """Result of dataset preprocessing execution."""

    dataset_id: str
    status: str
    message: str
    report: Dict[str, Any]


class PreprocessingStatusResponse(BaseModel):
    """Status and report of previously executed preprocessing."""

    dataset_id: str
    is_preprocessed: bool
    report: Optional[Dict[str, Any]] = None
    manifest: Optional[Dict[str, Any]] = None


class ProcessedFileInfo(BaseModel):
    """Metadata about a generated processed file."""

    filename: str
    relative_path: str
    record_count: int
    file_size_bytes: int
    sha256: str


class ProcessedFilesResponse(BaseModel):
    """Catalog of generated Parquet and JSON files for a dataset."""

    dataset_id: str
    files: List[ProcessedFileInfo]
