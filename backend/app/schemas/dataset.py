"""Pydantic schemas for multimodal survey dataset ingestion, validation, and quality analysis."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ValidationStatus(str, Enum):
    """Overall dataset validation status."""
    PASSED = "PASSED"
    WARNINGS = "WARNINGS"
    FAILED = "FAILED"
    CORRUPTED = "CORRUPTED"


class DataSourceType(str, Enum):
    """Classification of telemetry source authenticity."""
    REAL = "REAL"
    MOCK = "MOCK"
    HYBRID = "HYBRID"


class IntegrityCheckResult(BaseModel):
    """Manifest and cryptographic file integrity validation results."""
    manifest_valid: bool = Field(..., description="Whether manifest.json parsed and passed schema check")
    checksums_valid: bool = Field(..., description="Whether all file SHA-256 hashes match the manifest")
    record_counts_valid: bool = Field(..., description="Whether manifest counts match actual parsed rows")
    files_verified_count: int = Field(0, description="Total files successfully verified against manifest")
    files_failed_count: int = Field(0, description="Total files failing checksum or size checks")
    mismatched_files: List[str] = Field(default_factory=list, description="List of filenames with integrity failures")


class WifiQualityMetrics(BaseModel):
    """Quality and telemetry distribution metrics for Wi-Fi sweep observations."""
    total_scans: int = Field(0, description="Total unique Wi-Fi scan sweep observations")
    total_ap_records: int = Field(0, description="Total individual Access Point observation rows")
    unique_bssids_count: int = Field(0, description="Number of distinct BSSIDs observed")
    avg_aps_per_scan: float = Field(0.0, description="Mean APs visible per scan sweep")
    min_rssi_dbm: Optional[int] = Field(None, description="Weakest observed RSSI in dBm")
    max_rssi_dbm: Optional[int] = Field(None, description="Strongest observed RSSI in dBm")
    invalid_rssi_count: int = Field(0, description="Count of missing or out-of-range (> 0 or < -110) RSSI values")
    fresh_scan_count: int = Field(0, description="Count of real-time fresh sweeps")
    cached_scan_count: int = Field(0, description="Count of cached/throttled sweeps")
    duplicate_ap_count: int = Field(0, description="Count of duplicate APs within identical scan sweeps")
    has_timestamps: bool = Field(True, description="Whether valid timestamps exist for all sweeps")


class SensorQualityMetrics(BaseModel):
    """Quality and sampling rate metrics for IMU and magnetometer streams."""
    total_records: int = Field(0, description="Total sensor observation rows")
    accelerometer_count: int = Field(0, description="Count of tri-axial accelerometer events")
    gyroscope_count: int = Field(0, description="Count of tri-axial gyroscope events")
    magnetometer_count: int = Field(0, description="Count of magnetic field events")
    rotation_vector_count: int = Field(0, description="Count of rotation vector events")
    other_sensor_count: int = Field(0, description="Count of other sensor events")
    avg_sampling_freq_hz: Dict[str, float] = Field(default_factory=dict, description="Mean sampling frequency (Hz) by sensor type")
    min_interval_ms: Dict[str, float] = Field(default_factory=dict, description="Minimum sampling interval in ms")
    max_interval_ms: Dict[str, float] = Field(default_factory=dict, description="Maximum sampling interval in ms")
    missing_values_count: int = Field(0, description="Count of rows with missing x, y, or z axes")
    invalid_numeric_count: int = Field(0, description="Count of NaN or infinite numeric values")
    timestamp_gap_count: int = Field(0, description="Count of sampling gaps (> 5x mean interval)")
    real_record_count: int = Field(0, description="Count of hardware-collected sensor readings")
    mock_record_count: int = Field(0, description="Count of synthetic mock sensor readings")


class PhotoQualityMetrics(BaseModel):
    """Quality, resolution, and privacy sanitization metrics for landmark photos."""
    total_photo_records: int = Field(0, description="Total photo metadata records")
    image_files_present: int = Field(0, description="Count of actual image files present in images/")
    missing_image_files: List[str] = Field(default_factory=list, description="Photo records without matching image files")
    formats_detected: List[str] = Field(default_factory=list, description="Detected image MIME/compression formats")
    exif_sanitized_count: int = Field(0, description="Count of photos with client-reported EXIF sanitization")
    faces_detected_count: int = Field(0, description="Count of photos where faces were detected")
    face_blur_applied_count: int = Field(0, description="Count of photos with client-reported face blur")
    duplicate_hashes_count: int = Field(0, description="Count of photos with identical image content SHA-256")
    metadata_sanitization_verified: bool = Field(True, description="Whether all photo records declare sanitization")
    visual_privacy_audited: bool = Field(False, description="Whether independent visual privacy model ran")
    visual_privacy_status: str = Field("PENDING_AUDIT", description="Visual audit status (PENDING_AUDIT / NOT_PERFORMED / VERIFIED)")


class AnnotationQualityMetrics(BaseModel):
    """Quality metrics for manual surveyor ground-truth annotations."""
    total_annotations: int = Field(0, description="Total manual annotations")
    type_breakdown: Dict[str, int] = Field(default_factory=dict, description="Counts by annotation type code")
    missing_description_count: int = Field(0, description="Annotations missing description text")
    building_labels_present: int = Field(0, description="Annotations with building label")
    floor_labels_present: int = Field(0, description="Annotations with floor label")
    room_labels_present: int = Field(0, description="Annotations with room label")


class TimestampQualityMetrics(BaseModel):
    """Cross-stream clock synchronization and interval quality metrics."""
    original_timestamps_present: bool = Field(True, description="Whether raw hardware ticks are present")
    normalized_timestamps_present: bool = Field(True, description="Whether UTC normalized epoch ms are present")
    time_sources_used: List[str] = Field(default_factory=list, description="Hardware clock domains recorded")
    sync_methods_used: List[str] = Field(default_factory=list, description="Synchronization algorithms applied")
    min_reported_uncertainty_ms: Optional[float] = Field(None, description="Lowest reported uncertainty bound in ms")
    max_reported_uncertainty_ms: Optional[float] = Field(None, description="Highest reported uncertainty bound in ms")
    avg_reported_uncertainty_ms: Optional[float] = Field(None, description="Mean reported uncertainty bound in ms")
    uncertainty_provenance: str = Field("producer_reported", description="Provenance of uncertainty bound: producer_reported")
    out_of_order_count: int = Field(0, description="Count of observations with non-monotonic timestamps")
    large_gap_count: int = Field(0, description="Count of telemetry gaps exceeding 10 seconds")
    timeline_duration_seconds: float = Field(0.0, description="Total temporal span from first to last observation")


class CrossModalTemporalOverlap(BaseModel):
    """Analysis of temporal intersection and synchronization across modalities."""
    modality_time_ranges: Dict[str, Dict[str, Optional[float]]] = Field(
        default_factory=dict,
        description="Start and end epoch ms for each modality"
    )
    pairwise_overlap_seconds: Dict[str, float] = Field(
        default_factory=dict,
        description="Temporal overlap in seconds between pairs of modalities"
    )
    all_modalities_overlap_seconds: float = Field(
        0.0,
        description="Temporal overlap in seconds where all present modalities were concurrently active"
    )
    timeline_summary: str = Field(
        "",
        description="Human-readable summary of cross-modal concurrent streaming intervals"
    )


class ModalitiesQualitySummary(BaseModel):
    """Container holding individual modality quality metrics."""
    wifi: Optional[WifiQualityMetrics] = None
    sensors: Optional[SensorQualityMetrics] = None
    photos: Optional[PhotoQualityMetrics] = None
    annotations: Optional[AnnotationQualityMetrics] = None


class DatasetQualityReport(BaseModel):
    """Comprehensive machine-readable quality and validation assessment for an ingested dataset."""
    dataset_id: str = Field(..., description="Backend-assigned unique dataset identifier")
    session_id: str = Field(..., description="Original Android survey session identifier")
    session_name: str = Field(..., description="Survey session display name")
    validation_status: ValidationStatus = Field(..., description="Overall validation status")
    integrity: IntegrityCheckResult = Field(..., description="File integrity and manifest verification")
    modalities: ModalitiesQualitySummary = Field(..., description="Quality analysis across sensing streams")
    timestamp_quality: TimestampQualityMetrics = Field(..., description="Clock synchronization quality")
    cross_modal_overlap: CrossModalTemporalOverlap = Field(..., description="Temporal intersection analysis")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal data quality warnings")
    errors: List[str] = Field(default_factory=list, description="Fatal validation and integrity errors")


class DatasetSummaryResponse(BaseModel):
    """Summary representation for dataset listing and high-level queries."""
    dataset_id: str
    session_id: str
    session_name: str
    ingested_at_iso: str
    validation_status: ValidationStatus
    data_source_type: DataSourceType
    building_id: Optional[str] = None
    floor_id: Optional[str] = None
    room_id: Optional[str] = None
    modality_availability: Dict[str, bool] = Field(
        default_factory=dict,
        description="Whether wifi, sensors, photos, and annotations are populated"
    )
    observation_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Counts for wifi, imu, photo, annotation, and total observations"
    )
    file_size_bytes: int = Field(0, description="Size of preserved dataset archive in bytes")
    warnings_count: int = Field(0, description="Total quality warning count")
    errors_count: int = Field(0, description="Total fatal validation error count")


class DatasetDetailResponse(DatasetSummaryResponse):
    """Detailed dataset representation with complete session parameters and quality report."""
    created_at_ms: Optional[int] = None
    start_time_ms: Optional[int] = None
    end_time_ms: Optional[int] = None
    session_status: str = Field("COMPLETED", description="Original session lifecycle state")
    duration_seconds: float = 0.0
    clock_anchor: Optional[Dict[str, Any]] = None
    is_interrupted: bool = False
    interruption_note: Optional[str] = None
    manifest_version: str = "1.0.0"
    quality_report: Optional[DatasetQualityReport] = None


class DatasetUploadResponse(BaseModel):
    """Response returned upon successful ZIP dataset upload and ingestion."""
    dataset_id: str
    session_id: str
    session_name: str
    validation_status: ValidationStatus
    message: str
    quality_report: DatasetQualityReport


class DatasetValidationResponse(BaseModel):
    """Response returned after running on-demand validation on an existing dataset."""
    dataset_id: str
    validation_status: ValidationStatus
    warnings_count: int
    errors_count: int
    quality_report: DatasetQualityReport


class DatasetDeleteResponse(BaseModel):
    """Response returned after deleting a dataset."""
    dataset_id: str
    success: bool
    message: str
