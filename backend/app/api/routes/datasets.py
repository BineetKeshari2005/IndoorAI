"""Dataset management and ingestion API router."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

from backend.app.core.dataset_config import DatasetConfig, default_dataset_config
from backend.app.schemas.dataset import (
    DatasetDeleteResponse,
    DatasetDetailResponse,
    DatasetQualityReport,
    DatasetSummaryResponse,
    DatasetUploadResponse,
    DatasetValidationResponse,
)
from backend.app.schemas.pdr import (
    PDRRunRequest,
    PDRRunResponse,
    PDRStatusResponse,
    StepRecord,
    StepsResponse,
    TrajectoryPoint,
    TrajectoryResponse,
)
from backend.app.schemas.preprocessing import (
    PreprocessRequest,
    PreprocessResponse,
    PreprocessingStatusResponse,
    ProcessedFilesResponse,
)
from backend.app.schemas.pdr_calibration import (
    CalibrateRequest,
    CalibrationProfileResponse,
    CalibrationProfilesListResponse,
    EvaluateRequest,
    TrialListResponse,
    TrialRegisterRequest,
    TrialSummaryResponse,
)
from backend.app.schemas.wifi_survey import (
    CandidateMatchResponse,
    LocalizationPredictionResponse,
    RadioMapBuildRequest,
    RadioMapListResponse,
    RadioMapMetadataResponse,
    RadioMapResponse,
    ReferenceLocationCreate,
    ReferenceLocationResponse,
    ReferenceLocationsListResponse,
    WifiEvaluationRequest,
    WifiEvaluationResponse,
    WifiLocalizationQueryRequest,
)
from backend.app.schemas.sensor_fusion import (
    FusedStateResponse,
    FusionEvaluationRequest,
    FusionEvaluationResponse,
    FusionRunRequest,
    FusionStatusResponse,
    FusionTrajectoryResponse,
)
from backend.app.schemas.computer_vision import (
    LandmarkDetectionSchema,
    LandmarkRecordSchema,
    LandmarkVerifyRequest,
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
    VisionEvaluateRequest,
    VisionEvaluateResponse,
    VisualMatchRequest,
    VisualMatchResponse,
)
from backend.app.schemas.mapping import (
    BuildingMapSummaryResponse,
    CreateBuildingMapRequest,
    CreateConnectionRequest,
    CreateCorridorRequest,
    CreateDoorRequest,
    CreateFloorRequest,
    CreateRoomRequest,
    CreateVerticalTransitionRequest,
    ImportLandmarksRequest,
    ImportWifiReferencesRequest,
    MapMetricsResponse,
    MapValidationResponse,
    PublishVersionRequest,
)
from backend.app.services.computer_vision_service import ComputerVisionService
from backend.app.services.dataset_extraction import SecurityValidationError
from backend.app.services.dataset_ingestion import DatasetIngestionService
from backend.app.services.dataset_registry import DatasetRegistry
from backend.app.services.mapping_service import MappingService
from backend.app.services.pdr_service import PDRService
from backend.app.services.pdr_calibration_service import PDRCalibrationService
from backend.app.services.preprocessing_service import PreprocessingService
from backend.app.services.sensor_fusion_service import SensorFusionService
from backend.app.services.wifi_survey_service import WifiSurveyService
from src.core.logging import setup_logger

logger = setup_logger("DatasetsRouter")

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Global service instance dependency
_ingestion_service: Optional[DatasetIngestionService] = None
_preprocessing_service: Optional[PreprocessingService] = None
_pdr_service: Optional[PDRService] = None
_pdr_calibration_service: Optional[PDRCalibrationService] = None
_wifi_survey_service: Optional[WifiSurveyService] = None
_fusion_service: Optional[SensorFusionService] = None
_vision_service: Optional[ComputerVisionService] = None
_mapping_service: Optional[MappingService] = None


def get_ingestion_service() -> DatasetIngestionService:
    """Dependency provider for DatasetIngestionService."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = DatasetIngestionService()
    return _ingestion_service


def get_preprocessing_service() -> PreprocessingService:
    """Dependency provider for PreprocessingService."""
    global _preprocessing_service
    if _preprocessing_service is None:
        _preprocessing_service = PreprocessingService()
    return _preprocessing_service


def get_pdr_service() -> PDRService:
    """Dependency provider for PDRService."""
    global _pdr_service
    if _pdr_service is None:
        _pdr_service = PDRService()
    return _pdr_service


def get_pdr_calibration_service() -> PDRCalibrationService:
    """Dependency provider for PDRCalibrationService."""
    global _pdr_calibration_service
    if _pdr_calibration_service is None:
        _pdr_calibration_service = PDRCalibrationService()
    return _pdr_calibration_service


def get_wifi_survey_service() -> WifiSurveyService:
    """Dependency provider for WifiSurveyService (Task 23)."""
    global _wifi_survey_service
    if _wifi_survey_service is None:
        _wifi_survey_service = WifiSurveyService()
    return _wifi_survey_service


def get_sensor_fusion_service() -> SensorFusionService:
    """Dependency provider for SensorFusionService (Task 24)."""
    global _fusion_service
    if _fusion_service is None:
        _fusion_service = SensorFusionService()
    return _fusion_service


def get_computer_vision_service() -> ComputerVisionService:
    """Dependency provider for ComputerVisionService (Task 25)."""
    global _vision_service
    if _vision_service is None:
        _vision_service = ComputerVisionService()
    return _vision_service


def get_mapping_service() -> MappingService:
    """Dependency provider for MappingService (Task 26)."""
    global _mapping_service
    if _mapping_service is None:
        _mapping_service = MappingService()
    return _mapping_service


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest survey dataset ZIP archive",
    description="Accepts, validates, extracts, and catalogs an IndoorAI Android multimodal survey ZIP bundle."
)
async def upload_dataset(
    file: UploadFile = File(..., description="IndoorAI survey session ZIP export archive"),
    validate_only: bool = Query(False, description="If true, validates archive without persisting to registry"),
    service: DatasetIngestionService = Depends(get_ingestion_service)
) -> DatasetUploadResponse:
    """Ingest a survey dataset ZIP file."""
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Uploaded file must have a .zip extension."
        )

    try:
        # Read uploaded content into memory stream (fastapi UploadFile provides SpooledTemporaryFile)
        content = await file.read()
        stream = io.BytesIO(content)
        size_bytes = len(content)

        dataset_id, quality_report, entry = service.ingest_dataset_zip(
            zip_input=stream,
            original_filename=file.filename,
            reported_size=size_bytes,
            validate_only=validate_only
        )

        msg = (
            f"Dataset '{entry.get('session_name', 'Survey Session')}' ingested successfully "
            f"with validation status: {quality_report.validation_status.value}"
        )
        if validate_only:
            msg = f"Validation completed with status: {quality_report.validation_status.value} (dry run)"

        return DatasetUploadResponse(
            dataset_id=dataset_id,
            session_id=quality_report.session_id,
            session_name=quality_report.session_name,
            validation_status=quality_report.validation_status,
            message=msg,
            quality_report=quality_report
        )

    except SecurityValidationError as e:
        logger.warning("Security validation failure on upload '%s': %s", file.filename, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Security validation failed: {e}"
        )
    except Exception as e:
        logger.error("Failed to ingest dataset upload '%s': %s", file.filename, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset ingestion failed: {e}"
        )


@router.get(
    "",
    response_model=List[DatasetSummaryResponse],
    summary="List registered multimodal survey datasets",
    description="Returns high-level summaries and observation counts for cataloged survey datasets."
)
def list_datasets(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status (PASSED, WARNINGS, FAILED, CORRUPTED)"),
    source_type: Optional[str] = Query(None, description="Filter by telemetry source (REAL, MOCK, HYBRID)"),
    service: DatasetIngestionService = Depends(get_ingestion_service)
) -> List[DatasetSummaryResponse]:
    """Lists registered survey datasets."""
    datasets = service.registry.list_datasets(status=status_filter, source_type=source_type)
    return [DatasetSummaryResponse.model_validate(d) for d in datasets]


@router.get(
    "/{dataset_id}",
    response_model=DatasetDetailResponse,
    summary="Get survey dataset details",
    description="Returns full metadata, clock anchor parameters, observation counts, and quality summary."
)
def get_dataset(
    dataset_id: str,
    service: DatasetIngestionService = Depends(get_ingestion_service)
) -> DatasetDetailResponse:
    """Retrieves detailed metadata for a registered dataset."""
    dataset = service.registry.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found."
        )
    return DatasetDetailResponse.model_validate(dataset)


@router.get(
    "/{dataset_id}/quality",
    response_model=DatasetQualityReport,
    summary="Get machine-readable dataset quality report",
    description="Returns complete quality report with integrity checks, sampling frequencies, and cross-modal temporal overlap."
)
def get_dataset_quality_report(
    dataset_id: str,
    service: DatasetIngestionService = Depends(get_ingestion_service)
) -> DatasetQualityReport:
    """Retrieves full quality assessment for a dataset."""
    report = service.get_quality_report(dataset_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quality report for dataset '{dataset_id}' not found."
        )
    return report


@router.post(
    "/{dataset_id}/validate",
    response_model=DatasetValidationResponse,
    summary="Re-validate an existing dataset",
    description="Re-runs checksums, schema verification, and quality diagnostics on a registered dataset."
)
def revalidate_dataset(
    dataset_id: str,
    service: DatasetIngestionService = Depends(get_ingestion_service)
) -> DatasetValidationResponse:
    """Re-runs validation checks on a registered dataset."""
    report = service.revalidate_dataset(dataset_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found or workspace is missing."
        )
    return DatasetValidationResponse(
        dataset_id=dataset_id,
        validation_status=report.validation_status,
        warnings_count=len(report.warnings),
        errors_count=len(report.errors),
        quality_report=report
    )


@router.delete(
    "/{dataset_id}",
    response_model=DatasetDeleteResponse,
    summary="Delete a survey dataset",
    description="Permanently deletes the catalog entry, extracted observations, and raw ZIP archive."
)
def delete_dataset(
    dataset_id: str,
    service: DatasetIngestionService = Depends(get_ingestion_service)
) -> DatasetDeleteResponse:
    """Deletes a dataset and frees associated storage."""
    deleted = service.registry.delete_dataset(dataset_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found."
        )
    return DatasetDeleteResponse(
        dataset_id=dataset_id,
        success=True,
        message=f"Dataset '{dataset_id}' successfully deleted from registry and disk."
    )


@router.get(
    "/{dataset_id}/observations/{modality}",
    summary="Get observation records for a specific modality",
    description="Retrieves parsed observations (wifi, sensors, photos, or annotations) with pagination."
)
def get_dataset_observations(
    dataset_id: str,
    modality: str,
    limit: int = Query(2000, ge=1, le=10000, description="Maximum observation records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    service: DatasetIngestionService = Depends(get_ingestion_service)
):
    """Retrieves observation records for a specific modality."""
    modality_lower = modality.strip().lower()
    file_map = {
        "wifi": "wifi_observations.csv",
        "sensors": "imu_observations.csv",
        "imu": "imu_observations.csv",
        "photos": "photo_metadata.csv",
        "annotations": "annotations.csv",
    }
    if modality_lower not in file_map:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid modality '{modality}'. Allowed: {list(file_map.keys())}"
        )

    dataset = service.registry.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found."
        )

    extracted_dir = service.config.extracted_path / dataset_id
    target_csv = extracted_dir / file_map[modality_lower]
    if not target_csv.is_file():
        return {
            "dataset_id": dataset_id,
            "modality": modality_lower,
            "total_records": 0,
            "limit": limit,
            "offset": offset,
            "records": []
        }

    records = []
    try:
        import csv
        with open(target_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx < offset:
                    continue
                if len(records) >= limit:
                    break
                records.append(row)

        # Count total rows for metadata
        with open(target_csv, "r", encoding="utf-8", newline="") as f:
            total_records = sum(1 for _ in f) - 1  # Minus header
            total_records = max(0, total_records)

        return {
            "dataset_id": dataset_id,
            "modality": modality_lower,
            "total_records": total_records,
            "limit": limit,
            "offset": offset,
            "records": records
        }
    except Exception as e:
        logger.error("Failed to read observations for %s/%s: %s", dataset_id, modality_lower, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read observations: {e}"
        )


@router.get(
    "/{dataset_id}/images/{filename}",
    summary="Retrieve privacy-sanitized photo",
    description="Serves privacy-sanitized landmark photographs. Enforces EXIF sanitization check."
)
def get_dataset_image(
    dataset_id: str,
    filename: str,
    service: DatasetIngestionService = Depends(get_ingestion_service)
):
    """Safely serves confirmed sanitized landmark photographs."""
    # Path traversal protection
    clean_fn = Path(filename).name
    if clean_fn != filename or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image filename. Path climbs and slashes are prohibited."
        )

    ext = Path(filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Forbidden file extension '{ext}'."
        )

    dataset = service.registry.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found."
        )

    extracted_dir = service.config.extracted_path / dataset_id
    photo_csv = extracted_dir / "photo_metadata.csv"
    images_dir = extracted_dir / service.config.images_subfolder
    target_img = images_dir / clean_fn

    if not target_img.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photo '{clean_fn}' not found in dataset images directory."
        )

    # Privacy verification: Check photo_metadata.csv that is_exif_sanitized is TRUE
    is_confirmed_sanitized = False
    if photo_csv.is_file():
        try:
            import csv
            with open(photo_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("filename", "").strip() == clean_fn:
                        san_val = row.get("is_exif_sanitized", "").strip().lower()
                        if san_val in ("true", "1"):
                            is_confirmed_sanitized = True
                        break
        except Exception as e:
            logger.warning("Could not read photo_metadata.csv to verify sanitization: %s", e)

    if not is_confirmed_sanitized:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: Photo '{clean_fn}' is not verified as privacy-sanitized."
        )

    media_type = "image/png" if ext == ".png" else "image/jpeg"
    return FileResponse(str(target_img), media_type=media_type)


@router.post(
    "/{dataset_id}/preprocess",
    response_model=PreprocessResponse,
    status_code=status.HTTP_200_OK,
    summary="Preprocess and temporally align survey dataset",
    description="Cleans Wi-Fi, IMU, photos, and annotations, performs temporal alignment, and generates Parquet artifacts."
)
def preprocess_dataset(
    dataset_id: str,
    request_params: Optional[PreprocessRequest] = None,
    service: PreprocessingService = Depends(get_preprocessing_service)
) -> PreprocessResponse:
    """Executes multimodal preprocessing pipeline."""
    try:
        report, manifest = service.preprocess_dataset(dataset_id, request_params)
        return PreprocessResponse(
            dataset_id=dataset_id,
            status=report.status,
            message=f"Preprocessing completed with status '{report.status}'.",
            report=report.to_dict()
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except Exception as e:
        logger.error("Preprocessing failed for %s: %s", dataset_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing execution failed: {e}"
        )


@router.get(
    "/{dataset_id}/preprocessing",
    response_model=PreprocessingStatusResponse,
    summary="Get preprocessing report and status",
    description="Checks whether preprocessing has been executed and returns the latest quality report and manifest."
)
def get_preprocessing_status(
    dataset_id: str,
    service: PreprocessingService = Depends(get_preprocessing_service)
) -> PreprocessingStatusResponse:
    """Retrieves preprocessing status and report."""
    is_preprocessed, report, manifest = service.get_status(dataset_id)
    return PreprocessingStatusResponse(
        dataset_id=dataset_id,
        is_preprocessed=is_preprocessed,
        report=report,
        manifest=manifest
    )


@router.get(
    "/{dataset_id}/preprocessing/files",
    response_model=ProcessedFilesResponse,
    summary="List processed dataset Parquet and JSON files",
    description="Returns metadata, record counts, file sizes, and SHA-256 checksums of generated processed artifacts."
)
def list_processed_files(
    dataset_id: str,
    service: PreprocessingService = Depends(get_preprocessing_service)
) -> ProcessedFilesResponse:
    """Lists processed artifact files for a dataset."""
    files = service.list_processed_files(dataset_id)
    return ProcessedFilesResponse(
        dataset_id=dataset_id,
        files=files
    )


@router.post(
    "/{dataset_id}/pdr",
    response_model=PDRRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Pedestrian Dead Reckoning (PDR)",
    description="Reconstructs relative 2D pedestrian movement path from preprocessed IMU and orientation streams."
)
def run_pdr(
    dataset_id: str,
    request_params: Optional[PDRRunRequest] = None,
    service: PDRService = Depends(get_pdr_service)
) -> PDRRunResponse:
    """Executes PDR trajectory reconstruction."""
    try:
        report, manifest = service.run_pdr(dataset_id, request_params)
        return PDRRunResponse(
            dataset_id=dataset_id,
            status=report.status,
            message=f"PDR completed with status '{report.status}'.",
            report=report.to_dict()
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except Exception as e:
        logger.error("PDR execution failed for %s: %s", dataset_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDR execution failed: {e}"
        )


@router.get(
    "/{dataset_id}/pdr",
    response_model=PDRStatusResponse,
    summary="Get PDR computation status",
    description="Checks whether PDR has been executed and returns the latest quality report and manifest."
)
def get_pdr_status(
    dataset_id: str,
    service: PDRService = Depends(get_pdr_service)
) -> PDRStatusResponse:
    """Retrieves PDR status and summary report."""
    is_computed, report, manifest = service.get_status(dataset_id)
    return PDRStatusResponse(
        dataset_id=dataset_id,
        is_computed=is_computed,
        report=report,
        manifest=manifest
    )


@router.get(
    "/{dataset_id}/pdr/trajectory",
    response_model=TrajectoryResponse,
    summary="Get reconstructed relative 2D trajectory",
    description="Returns sequential relative trajectory coordinate points (rel_x, rel_y) with headings and uncertainty."
)
def get_pdr_trajectory(
    dataset_id: str,
    service: PDRService = Depends(get_pdr_service)
) -> TrajectoryResponse:
    """Retrieves 2D relative trajectory points."""
    points = service.get_trajectory(dataset_id)
    return TrajectoryResponse(
        dataset_id=dataset_id,
        total_points=len(points),
        points=points
    )


@router.get(
    "/{dataset_id}/pdr/steps",
    response_model=StepsResponse,
    summary="Get detected human step events",
    description="Returns detected step events with timestamps, peak acceleration, and estimated step lengths."
)
def get_pdr_steps(
    dataset_id: str,
    service: PDRService = Depends(get_pdr_service)
) -> StepsResponse:
    """Retrieves detected step records."""
    steps = service.get_steps(dataset_id)
    return StepsResponse(
        dataset_id=dataset_id,
        total_steps=len(steps),
        steps=steps
    )


@router.get(
    "/{dataset_id}/pdr/report",
    summary="Get comprehensive PDR report",
    description="Returns the full JSON diagnostic, movement, and uncertainty report."
)
def get_pdr_report(
    dataset_id: str,
    service: PDRService = Depends(get_pdr_service)
):
    """Retrieves full PDR report JSON."""
    is_computed, report, _ = service.get_status(dataset_id)
    if not is_computed or not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No PDR report found for dataset '{dataset_id}'."
        )
    return report


# -----------------------------------------------------------------------------
# PDR Calibration, Ground-Truth Validation & Drift Analysis Endpoints (Task 22)
# -----------------------------------------------------------------------------

@router.post(
    "/{dataset_id}/pdr/trials",
    response_model=TrialSummaryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a measured ground-truth walking trial",
    description="Registers manual survey measurements (distance, heading, steps) linked to a dataset session."
)
def register_walking_trial(
    dataset_id: str,
    trial_req: TrialRegisterRequest,
    cal_service: PDRCalibrationService = Depends(get_pdr_calibration_service)
) -> TrialSummaryResponse:
    """Registers a new measured ground-truth walking trial."""
    try:
        trial = cal_service.register_trial(dataset_id, trial_req)
        return TrialSummaryResponse(
            trial_id=trial.trial_id,
            dataset_id=trial.dataset_id,
            session_id=trial.session_id,
            participant_id=trial.participant_id,
            device_model=trial.device_model,
            phone_placement=trial.phone_placement.value,
            walking_condition=trial.walking_condition.value,
            trial_type=trial.trial_type.value,
            start_timestamp_ms=trial.start_timestamp_ms,
            end_timestamp_ms=trial.end_timestamp_ms,
            measured_distance_m=trial.ground_truth.measured_distance_m,
            measurement_method=trial.ground_truth.measurement_method,
            ground_truth_step_count=trial.ground_truth.ground_truth_step_count,
            is_synthetic=trial.is_synthetic,
            created_at_iso=trial.created_at_iso
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        logger.error("Failed to register walking trial: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to register trial: {e}")


@router.get(
    "/{dataset_id}/pdr/trials",
    response_model=TrialListResponse,
    summary="List registered walking trials",
    description="Retrieves summary metadata for all ground-truth trials associated with a dataset."
)
def list_walking_trials(
    dataset_id: str,
    cal_service: PDRCalibrationService = Depends(get_pdr_calibration_service)
) -> TrialListResponse:
    """Lists registered trials for a dataset."""
    trials = cal_service.list_trials(dataset_id=dataset_id)
    summaries = [
        TrialSummaryResponse(
            trial_id=t.trial_id,
            dataset_id=t.dataset_id,
            session_id=t.session_id,
            participant_id=t.participant_id,
            device_model=t.device_model,
            phone_placement=t.phone_placement.value,
            walking_condition=t.walking_condition.value,
            trial_type=t.trial_type.value,
            start_timestamp_ms=t.start_timestamp_ms,
            end_timestamp_ms=t.end_timestamp_ms,
            measured_distance_m=t.ground_truth.measured_distance_m,
            measurement_method=t.ground_truth.measurement_method,
            ground_truth_step_count=t.ground_truth.ground_truth_step_count,
            is_synthetic=t.is_synthetic,
            created_at_iso=t.created_at_iso
        )
        for t in trials
    ]
    return TrialListResponse(
        dataset_id=dataset_id,
        total_trials=len(summaries),
        trials=summaries
    )


@router.get(
    "/{dataset_id}/pdr/trials/{trial_id}",
    summary="Retrieve full ground-truth trial details",
    description="Returns complete trial metadata, ground-truth measurements, and waypoints."
)
def get_walking_trial(
    dataset_id: str,
    trial_id: str,
    cal_service: PDRCalibrationService = Depends(get_pdr_calibration_service)
):
    """Retrieves full trial record."""
    trial = cal_service.get_trial(trial_id)
    if not trial or trial.dataset_id != dataset_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trial '{trial_id}' not found for dataset '{dataset_id}'."
        )
    return trial.to_dict()


@router.post(
    "/{dataset_id}/pdr/calibrate",
    response_model=CalibrationProfileResponse,
    summary="Calibrate Weinberg step-length constant",
    description="Optimizes Weinberg parameter K from training trials and validates on held-out trials."
)
def calibrate_weinberg_stride(
    dataset_id: str,
    cal_req: CalibrateRequest,
    cal_service: PDRCalibrationService = Depends(get_pdr_calibration_service)
) -> CalibrationProfileResponse:
    """Executes Weinberg parameter estimation."""
    try:
        profile = cal_service.calibrate(dataset_id, cal_req)
        return CalibrationProfileResponse(
            profile_id=profile.profile_id,
            profile_name=profile.profile_name,
            profile_type=profile.profile_type.value,
            calibration_method=profile.calibration_method,
            weinberg_k=profile.weinberg_k,
            step_length_bounds=list(profile.step_length_bounds),
            device_model=profile.device_model,
            phone_placement=profile.phone_placement.value if profile.phone_placement else None,
            participant_id=profile.participant_id,
            source_trial_ids=profile.source_trial_ids,
            calibration_date_iso=profile.calibration_date_iso,
            calibration_status=profile.calibration_status.value,
            training_metrics=profile.training_metrics,
            validation_metrics=profile.validation_metrics,
            notes=profile.notes
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error("Calibration failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Calibration failed: {e}")


@router.get(
    "/{dataset_id}/pdr/calibration",
    response_model=CalibrationProfilesListResponse,
    summary="List calibration profiles",
    description="Retrieves all registered PDR calibration profiles."
)
def list_calibration_profiles(
    dataset_id: str,
    cal_service: PDRCalibrationService = Depends(get_pdr_calibration_service)
) -> CalibrationProfilesListResponse:
    """Retrieves all registered calibration profiles."""
    profiles = cal_service.list_profiles()
    resp_list = [
        CalibrationProfileResponse(
            profile_id=p.profile_id,
            profile_name=p.profile_name,
            profile_type=p.profile_type.value,
            calibration_method=p.calibration_method,
            weinberg_k=p.weinberg_k,
            step_length_bounds=list(p.step_length_bounds),
            device_model=p.device_model,
            phone_placement=p.phone_placement.value if p.phone_placement else None,
            participant_id=p.participant_id,
            source_trial_ids=p.source_trial_ids,
            calibration_date_iso=p.calibration_date_iso,
            calibration_status=p.calibration_status.value,
            training_metrics=p.training_metrics,
            validation_metrics=p.validation_metrics,
            notes=p.notes
        )
        for p in profiles
    ]
    return CalibrationProfilesListResponse(
        total_profiles=len(resp_list),
        profiles=resp_list
    )


@router.post(
    "/{dataset_id}/pdr/evaluate",
    summary="Evaluate PDR against ground-truth walking trials",
    description="Computes step detection, heading, trajectory drift, and 95% spatial confidence ellipse metrics."
)
def evaluate_pdr_trials(
    dataset_id: str,
    eval_req: EvaluateRequest,
    cal_service: PDRCalibrationService = Depends(get_pdr_calibration_service)
):
    """Evaluates PDR against specified trials."""
    try:
        reports = cal_service.evaluate(dataset_id, eval_req)
        return [r.to_dict() for r in reports]
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Evaluation failed: {e}")


@router.get(
    "/{dataset_id}/pdr/evaluation",
    summary="Get recent PDR evaluation reports",
    description="Retrieves historical evaluation reports for the dataset."
)
def get_evaluation_reports(
    dataset_id: str,
    cal_service: PDRCalibrationService = Depends(get_pdr_calibration_service)
):
    """Retrieves stored evaluation reports."""
    reports = cal_service.get_evaluation_reports(dataset_id=dataset_id)
    return reports


# =========================================================================
# Task 23: Site-Specific Wi-Fi Survey Radio Map & Localization Endpoints
# =========================================================================

@router.post(
    "/{dataset_id}/wifi-survey/locations",
    response_model=ReferenceLocationResponse,
    summary="Register a site reference location",
    description="Registers or updates a verified reference landmark location for site-specific Wi-Fi localization."
)
def register_reference_location(
    dataset_id: str,
    location_data: ReferenceLocationCreate,
    service: WifiSurveyService = Depends(get_wifi_survey_service),
) -> ReferenceLocationResponse:
    """Registers or updates a reference location."""
    try:
        ref_loc = service.register_location(site_id=location_data.site_id, loc_data=location_data)
        return ReferenceLocationResponse(**ref_loc.to_dict())
    except Exception as e:
        logger.error(f"Failed to register reference location: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "/{dataset_id}/wifi-survey/locations",
    response_model=ReferenceLocationsListResponse,
    summary="List site reference locations",
    description="Retrieves all registered reference locations for a specific physical site."
)
def list_reference_locations(
    dataset_id: str,
    site_id: str = Query(..., description="Site identifier"),
    service: WifiSurveyService = Depends(get_wifi_survey_service),
) -> ReferenceLocationsListResponse:
    """Retrieves reference locations for a site."""
    locations = service.get_locations(site_id=site_id)
    resp_list = [ReferenceLocationResponse(**loc.to_dict()) for loc in locations]
    return ReferenceLocationsListResponse(
        site_id=site_id,
        total_locations=len(resp_list),
        locations=resp_list,
    )


@router.post(
    "/{dataset_id}/wifi-survey/fingerprints/build",
    response_model=RadioMapResponse,
    summary="Compile site radio map from survey observations",
    description="Aggregates multi-sweep Wi-Fi observations into mean/median/std signal distributions and visibility rates."
)
def build_radio_map(
    dataset_id: str,
    build_req: RadioMapBuildRequest,
    service: WifiSurveyService = Depends(get_wifi_survey_service),
) -> RadioMapResponse:
    """Compiles a site radio map from dataset observations."""
    try:
        radio_map = service.build_radio_map(dataset_id=dataset_id, request=build_req)
        return RadioMapResponse(
            metadata=RadioMapMetadataResponse(**radio_map.metadata.to_dict()),
            locations={k: ReferenceLocationResponse(**v.to_dict()) for k, v in radio_map.locations.items()},
            profiles={k: v.to_dict() for k, v in radio_map.profiles.items()},
            all_bssids=radio_map.all_bssids,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to build radio map: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build radio map: {e}",
        )


@router.get(
    "/{dataset_id}/wifi-survey/radio-map",
    response_model=RadioMapResponse,
    summary="Get compiled site radio map",
    description="Retrieves full radio map metadata, reference locations, and aggregated radio profiles."
)
def get_radio_map(
    dataset_id: str,
    site_id: str = Query(..., description="Site identifier"),
    version: str = Query("v1", description="Radio map version"),
    service: WifiSurveyService = Depends(get_wifi_survey_service),
) -> RadioMapResponse:
    """Retrieves a compiled radio map."""
    radio_map = service.get_radio_map(site_id=site_id, version=version)
    if not radio_map:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Radio map for site '{site_id}' version '{version}' not found.",
        )
    return RadioMapResponse(
        metadata=RadioMapMetadataResponse(**radio_map.metadata.to_dict()),
        locations={k: ReferenceLocationResponse(**v.to_dict()) for k, v in radio_map.locations.items()},
        profiles={k: v.to_dict() for k, v in radio_map.profiles.items()},
        all_bssids=radio_map.all_bssids,
    )


@router.get(
    "/{dataset_id}/wifi-survey/radio-maps",
    response_model=RadioMapListResponse,
    summary="List compiled site radio maps",
    description="Lists metadata for all compiled radio maps on disk."
)
def list_site_radio_maps(
    dataset_id: str,
    site_id: Optional[str] = Query(None, description="Optional site filter"),
    service: WifiSurveyService = Depends(get_wifi_survey_service),
) -> RadioMapListResponse:
    """Lists compiled radio maps."""
    maps = service.list_maps(site_id=site_id)
    resp_list = [RadioMapMetadataResponse(**m) for m in maps]
    return RadioMapListResponse(
        total_radio_maps=len(resp_list),
        radio_maps=resp_list,
    )


@router.post(
    "/{dataset_id}/wifi-survey/localize",
    response_model=LocalizationPredictionResponse,
    summary="Estimate location from Wi-Fi fingerprint",
    description="Matches query AP observations against site radio map using weighted k-NN or Gaussian log-likelihood."
)
def localize_fingerprint(
    dataset_id: str,
    query_req: WifiLocalizationQueryRequest,
    service: WifiSurveyService = Depends(get_wifi_survey_service),
) -> LocalizationPredictionResponse:
    """Performs site-specific fingerprint localization."""
    try:
        pred = service.localize(request=query_req)
        return LocalizationPredictionResponse(
            predicted_building_id=pred.predicted_building_id,
            predicted_floor_id=pred.predicted_floor_id,
            predicted_x_m=pred.predicted_x_m,
            predicted_y_m=pred.predicted_y_m,
            predicted_room_id=pred.predicted_room_id,
            confidence=pred.confidence,
            algorithm=pred.algorithm,
            matched_ap_count=pred.matched_ap_count,
            total_query_aps=pred.total_query_aps,
            candidate_matches=[CandidateMatchResponse(**c.to_dict()) for c in pred.candidate_matches],
            warnings=pred.warnings,
            is_low_confidence=pred.is_low_confidence,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Localization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Localization failed: {e}",
        )


@router.post(
    "/{dataset_id}/wifi-survey/evaluate",
    response_model=WifiEvaluationResponse,
    summary="Evaluate Wi-Fi localization accuracy under session partition",
    description="Computes building, floor, location, and coordinate error metrics under strict session-level train/test separation."
)
def evaluate_wifi_localization(
    dataset_id: str,
    eval_req: WifiEvaluationRequest,
    service: WifiSurveyService = Depends(get_wifi_survey_service),
) -> WifiEvaluationResponse:
    """Runs session-partitioned offline evaluation."""
    try:
        metrics = service.evaluate(dataset_id=dataset_id, request=eval_req)
        return WifiEvaluationResponse(
            site_id=eval_req.site_id,
            algorithm=eval_req.algorithm,
            metrics=metrics.to_dict(),
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {e}",
        )


# =========================================================================
# Task 24: Wi-Fi + PDR Sensor Fusion Endpoints (Extended Kalman Filter)
# =========================================================================

@router.post(
    "/{dataset_id}/fusion/run",
    response_model=FusionTrajectoryResponse,
    summary="Run Wi-Fi + PDR Extended Kalman Filter fusion",
    description="Combines continuous PDR step displacements with site-specific Wi-Fi position observations using EKF and statistical outlier gating."
)
def run_sensor_fusion(
    dataset_id: str,
    fusion_req: FusionRunRequest,
    service: SensorFusionService = Depends(get_sensor_fusion_service),
) -> FusionTrajectoryResponse:
    """Executes EKF sensor fusion on survey dataset."""
    try:
        fused_states, summary = service.run_fusion(dataset_id=dataset_id, request=fusion_req)
        resp_trajectory = [
            FusedStateResponse(
                timestamp_epoch_ms=s.timestamp_epoch_ms,
                building_id=s.building_id,
                floor_id=s.floor_id,
                x_m=s.x_m,
                y_m=s.y_m,
                theta_deg=s.theta_deg,
                sigma_x_m=s.sigma_x_m,
                sigma_y_m=s.sigma_y_m,
                status=s.status.value,
                semi_major_axis_m=s.ellipse.semi_major_axis_m,
                semi_minor_axis_m=s.ellipse.semi_minor_axis_m,
                ellipse_angle_deg=s.ellipse.ellipse_angle_deg,
                conservative_95_radius_m=s.ellipse.conservative_95_radius_m,
                mahalanobis_dist_sq=s.mahalanobis_dist_sq,
                rejection_reason=s.rejection_reason,
            )
            for s in fused_states
        ]
        return FusionTrajectoryResponse(
            dataset_id=dataset_id,
            total_points=len(resp_trajectory),
            summary=summary.to_dict(),
            trajectory=resp_trajectory,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Sensor fusion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sensor fusion failed: {e}",
        )


@router.get(
    "/{dataset_id}/fusion/status",
    response_model=FusionStatusResponse,
    summary="Get sensor fusion status and summary",
    description="Checks whether a fused trajectory has been computed and saved for the dataset."
)
def get_fusion_status(
    dataset_id: str,
    service: SensorFusionService = Depends(get_sensor_fusion_service),
) -> FusionStatusResponse:
    """Retrieves current sensor fusion artifact status."""
    return service.get_status(dataset_id=dataset_id)


@router.get(
    "/{dataset_id}/fusion/trajectory",
    response_model=FusionTrajectoryResponse,
    summary="Get fused trajectory points",
    description="Retrieves computed EKF trajectory points with confidence ellipses and fusion statuses."
)
def get_fused_trajectory(
    dataset_id: str,
    limit: int = Query(5000, ge=1, le=20000, description="Maximum trajectory points to return"),
    service: SensorFusionService = Depends(get_sensor_fusion_service),
) -> FusionTrajectoryResponse:
    """Retrieves stored fused trajectory points."""
    status_resp = service.get_status(dataset_id=dataset_id)
    if not status_resp.is_available:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fused trajectory not found for dataset '{dataset_id}'. Run POST /{dataset_id}/fusion/run first.",
        )
    points = service.get_trajectory(dataset_id=dataset_id, limit=limit)
    resp_trajectory = []
    for p in points:
        ellipse_d = p.get("ellipse", {})
        resp_trajectory.append(
            FusedStateResponse(
                timestamp_epoch_ms=int(p.get("timestamp_epoch_ms", p.get("timestamp_ms", 0))),
                building_id=str(p.get("building_id", "BLDG-1")),
                floor_id=int(p.get("floor_id", 1)),
                x_m=float(p.get("x_m", 0.0)),
                y_m=float(p.get("y_m", 0.0)),
                theta_deg=float(p.get("theta_deg", 0.0)),
                sigma_x_m=float(p.get("sigma_x_m", 1.0)),
                sigma_y_m=float(p.get("sigma_y_m", 1.0)),
                status=str(p.get("status", "PDR_ONLY")),
                semi_major_axis_m=float(ellipse_d.get("semi_major_axis_m", p.get("semi_major_axis_m", 1.0))),
                semi_minor_axis_m=float(ellipse_d.get("semi_minor_axis_m", p.get("semi_minor_axis_m", 1.0))),
                ellipse_angle_deg=float(ellipse_d.get("ellipse_angle_deg", p.get("ellipse_angle_deg", 0.0))),
                conservative_95_radius_m=float(ellipse_d.get("conservative_95_radius_m", p.get("conservative_95_radius_m", 1.0))),
                mahalanobis_dist_sq=p.get("mahalanobis_dist_sq"),
                rejection_reason=p.get("rejection_reason"),
            )
        )
    return FusionTrajectoryResponse(
        dataset_id=dataset_id,
        total_points=len(resp_trajectory),
        summary=status_resp.summary or {},
        trajectory=resp_trajectory,
    )


@router.post(
    "/{dataset_id}/fusion/evaluate",
    response_model=FusionEvaluationResponse,
    summary="Run comparative evaluation across PDR-only, Wi-Fi-only, and Fused",
    description="Quantifies position error, endpoint error, drift per meter, outlier rejection rate, and uncertainty coverage."
)
def evaluate_sensor_fusion(
    dataset_id: str,
    eval_req: FusionEvaluationRequest,
    service: SensorFusionService = Depends(get_sensor_fusion_service),
) -> FusionEvaluationResponse:
    """Executes comparative trajectory evaluation."""
    try:
        report = service.evaluate(dataset_id=dataset_id, request=eval_req)
        return FusionEvaluationResponse(
            dataset_id=dataset_id,
            report=report.to_dict(),
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Fusion evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fusion evaluation failed: {e}",
        )


@router.get(
    "/{dataset_id}/fusion/evaluation",
    response_model=FusionEvaluationResponse,
    summary="Get recent sensor fusion evaluation report",
    description="Retrieves comparative evaluation report for dataset."
)
def get_fusion_evaluation(
    dataset_id: str,
    site_id: str = Query("SITE-DEFAULT", description="Site identifier"),
    service: SensorFusionService = Depends(get_sensor_fusion_service),
) -> FusionEvaluationResponse:
    """Retrieves evaluation report for dataset."""
    req = FusionEvaluationRequest(site_id=site_id, use_synthetic_fixture=True)
    report = service.evaluate(dataset_id=dataset_id, request=req)
    return FusionEvaluationResponse(
        dataset_id=dataset_id,
        report=report.to_dict(),
    )


# ---------------------------------------------------------------------------
# Task 25: Computer Vision-Based Indoor Landmark Detection Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/{dataset_id}/vision/analyze",
    response_model=VisionAnalyzeResponse,
    summary="Detect indoor landmarks and recognize signs across survey photos",
    description="Loads privacy-sanitized survey photos, runs landmark detection and OCR, and extracts descriptors."
)
def analyze_dataset_vision(
    dataset_id: str,
    analyze_req: Optional[VisionAnalyzeRequest] = None,
    service: ComputerVisionService = Depends(get_computer_vision_service),
) -> VisionAnalyzeResponse:
    """Runs landmark detection and OCR on survey photographs."""
    try:
        return service.analyze_dataset(dataset_id=dataset_id, request_params=analyze_req)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        logger.error(f"Vision analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision analysis failed: {e}",
        )


@router.get(
    "/{dataset_id}/vision/detections",
    response_model=List[LandmarkDetectionSchema],
    summary="Get landmark detections for a survey dataset",
    description="Retrieves computed landmark candidates with bounding boxes, confidence scores, and OCR text."
)
def get_dataset_landmark_detections(
    dataset_id: str,
    category: Optional[str] = Query(None, description="Optional landmark category filter"),
    service: ComputerVisionService = Depends(get_computer_vision_service),
) -> List[LandmarkDetectionSchema]:
    """Retrieves computed landmark detections."""
    try:
        return service.get_detections(dataset_id=dataset_id, category=category)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to retrieve landmark detections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve landmark detections: {e}",
        )


@router.get(
    "/{dataset_id}/vision/landmarks",
    response_model=List[LandmarkRecordSchema],
    summary="List registered landmarks in building landmark registry",
    description="Queries persistent landmark registry with optional building, floor, category, and verification status filters."
)
def list_registered_landmarks(
    dataset_id: str,
    building_id: Optional[str] = Query(None, description="Building ID filter"),
    floor_id: Optional[int] = Query(None, description="Floor ID filter"),
    category: Optional[str] = Query(None, description="Landmark category filter"),
    status: Optional[str] = Query(None, description="Verification status (UNVERIFIED, MANUALLY_VERIFIED, REJECTED)"),
    service: ComputerVisionService = Depends(get_computer_vision_service),
) -> List[LandmarkRecordSchema]:
    """Queries registered landmarks."""
    return service.list_landmarks(
        building_id=building_id,
        floor_id=floor_id,
        category=category,
        status=status,
    )


@router.post(
    "/{dataset_id}/vision/landmarks/verify",
    response_model=LandmarkRecordSchema,
    summary="Manually verify, update, or reject a registered landmark",
    description="Associates ground-truth coordinates, verified name, and manual verification status with a landmark record."
)
def verify_landmark_record(
    dataset_id: str,
    verify_req: LandmarkVerifyRequest,
    service: ComputerVisionService = Depends(get_computer_vision_service),
) -> LandmarkRecordSchema:
    """Manually updates or confirms a landmark record."""
    updated = service.verify_landmark(request_params=verify_req)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Landmark with ID '{verify_req.landmark_id}' not found in registry.",
        )
    return updated


@router.post(
    "/{dataset_id}/vision/match",
    response_model=VisualMatchResponse,
    summary="Match survey photo against registered indoor landmarks",
    description="Computes visual feature similarity and descriptor inliers against registered landmarks."
)
def match_survey_photo(
    dataset_id: str,
    match_req: VisualMatchRequest,
    service: ComputerVisionService = Depends(get_computer_vision_service),
) -> VisualMatchResponse:
    """Compares a photo against registered landmarks."""
    try:
        return service.match_photo(dataset_id=dataset_id, request_params=match_req)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Visual matching failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Visual matching failed: {e}",
        )


@router.post(
    "/{dataset_id}/vision/evaluate",
    response_model=VisionEvaluateResponse,
    summary="Run computer vision evaluation on detection, OCR, and matching",
    description="Calculates multi-class precision, recall, F1, mean IoU, confusion matrix, and OCR character error rate."
)
def evaluate_computer_vision(
    dataset_id: str,
    eval_req: VisionEvaluateRequest,
    service: ComputerVisionService = Depends(get_computer_vision_service),
) -> VisionEvaluateResponse:
    """Evaluates computer vision performance."""
    try:
        return service.evaluate_vision(request_params=eval_req)
    except Exception as e:
        logger.error(f"Vision evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision evaluation failed: {e}",
        )


# =====================================================================
# Task 26: Indoor Map Generation & Spatial Representation Endpoints
# =====================================================================

@router.get(
    "/{dataset_id}/maps/buildings",
    response_model=List[BuildingMapSummaryResponse],
    summary="List all indoor building maps",
    description="Returns high-level metadata summaries of all persisted building maps."
)
def list_indoor_maps(
    dataset_id: str,
    service: MappingService = Depends(get_mapping_service),
) -> List[BuildingMapSummaryResponse]:
    """Retrieves summaries of all registered building maps."""
    summaries = service.list_buildings()
    return [
        BuildingMapSummaryResponse(
            building_id=s["building_id"],
            name=s["name"],
            map_version=s.get("map_version", "1.0.0"),
            floor_count=s.get("floor_count", 0),
            total_rooms_count=s.get("total_rooms_count", 0),
            total_doors_count=s.get("total_doors_count", 0),
            updated_at=s.get("updated_at", ""),
        )
        for s in summaries
    ]


@router.get(
    "/{dataset_id}/maps/{building_id}",
    summary="Retrieve full building map definition",
    description="Returns the comprehensive hierarchical building map including all floors, rooms, corridors, doors, and transitions."
)
def get_indoor_map(
    dataset_id: str,
    building_id: str,
    service: MappingService = Depends(get_mapping_service),
) -> dict:
    """Retrieves complete building map."""
    bmap = service.get_building_map(building_id)
    if not bmap:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Building map '{building_id}' not found.",
        )
    return bmap.to_dict()


@router.post(
    "/{dataset_id}/maps/buildings",
    summary="Create or update an indoor building map",
    description="Initializes or updates root building attributes."
)
def create_or_update_building_map(
    dataset_id: str,
    req: CreateBuildingMapRequest,
    service: MappingService = Depends(get_mapping_service),
) -> dict:
    """Creates or updates a building map."""
    bmap = service.create_or_update_building(req)
    return bmap.to_dict()


@router.post(
    "/{dataset_id}/maps/floors",
    summary="Add a floor to a building map",
    description="Registers a building floor level with boundary and elevation."
)
def add_map_floor(
    dataset_id: str,
    building_id: str,
    req: CreateFloorRequest,
    service: MappingService = Depends(get_mapping_service),
) -> dict:
    """Adds a floor level to the building."""
    try:
        bmap = service.add_floor(building_id=building_id, req=req)
        return bmap.to_dict()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/{dataset_id}/maps/rooms",
    summary="Add a room polygon to a floor",
    description="Registers an enclosed room space with polygon boundary, category, and door links."
)
def add_map_room(
    dataset_id: str,
    building_id: str,
    req: CreateRoomRequest,
    service: MappingService = Depends(get_mapping_service),
) -> dict:
    """Adds a room polygon to the floor."""
    try:
        bmap = service.add_room(building_id=building_id, req=req)
        return bmap.to_dict()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/{dataset_id}/maps/corridors",
    summary="Add a corridor centerline to a floor",
    description="Registers a walkable corridor centerline trajectory and width."
)
def add_map_corridor(
    dataset_id: str,
    building_id: str,
    req: CreateCorridorRequest,
    service: MappingService = Depends(get_mapping_service),
) -> dict:
    """Adds a corridor to the floor."""
    try:
        bmap = service.add_corridor(building_id=building_id, req=req)
        return bmap.to_dict()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/{dataset_id}/maps/doors",
    summary="Add a door connecting spaces",
    description="Registers a door portal linking rooms, corridors, or exits."
)
def add_map_door(
    dataset_id: str,
    building_id: str,
    req: CreateDoorRequest,
    service: MappingService = Depends(get_mapping_service),
) -> dict:
    """Adds a door to the floor."""
    try:
        bmap = service.add_door(building_id=building_id, req=req)
        return bmap.to_dict()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/{dataset_id}/maps/vertical-transitions",
    summary="Add a vertical transition across floors",
    description="Registers a staircase or elevator shaft with landings on multiple floors."
)
def add_map_vertical_transition(
    dataset_id: str,
    building_id: str,
    req: CreateVerticalTransitionRequest,
    service: MappingService = Depends(get_mapping_service),
) -> dict:
    """Adds a staircase or elevator transition."""
    try:
        bmap = service.add_vertical_transition(building_id=building_id, req=req)
        return bmap.to_dict()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/{dataset_id}/maps/connections",
    summary="Add a navigation topological connection edge",
    description="Explicitly connects two indoor spaces or vertical landings."
)
def add_map_connection(
    dataset_id: str,
    building_id: str,
    req: CreateConnectionRequest,
    service: MappingService = Depends(get_mapping_service),
) -> dict:
    """Adds a topological connection edge."""
    try:
        bmap = service.add_connection(building_id=building_id, req=req)
        return bmap.to_dict()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/{dataset_id}/maps/{building_id}/import-wifi",
    summary="Import Wi-Fi survey reference locations",
    description="Imports Task 23 Wi-Fi fingerprint reference points and associates them with floor spaces."
)
def import_wifi_references_to_map(
    dataset_id: str,
    building_id: str,
    req: ImportWifiReferencesRequest,
    service: MappingService = Depends(get_mapping_service),
) -> dict:
    """Imports Wi-Fi reference locations into the floor map."""
    try:
        bmap = service.import_wifi_references(building_id=building_id, req=req)
        return bmap.to_dict()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/{dataset_id}/maps/{building_id}/import-landmarks",
    summary="Import verified visual landmarks",
    description="Imports Task 25 computer vision landmark detections and links them with doors and rooms."
)
def import_landmarks_to_map(
    dataset_id: str,
    building_id: str,
    req: ImportLandmarksRequest,
    service: MappingService = Depends(get_mapping_service),
) -> dict:
    """Imports visual landmarks into the floor map."""
    try:
        bmap = service.import_landmarks(building_id=building_id, req=req)
        return bmap.to_dict()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/{dataset_id}/maps/{building_id}/validate",
    response_model=MapValidationResponse,
    summary="Validate building map geometry and topological consistency",
    description="Performs geometric and topological validation, checking polygon validity, corridor lengths, door connectivity, and planar vertical transition shifts."
)
def validate_indoor_map(
    dataset_id: str,
    building_id: str,
    service: MappingService = Depends(get_mapping_service),
) -> MapValidationResponse:
    """Runs full structural validation on a building map."""
    try:
        report = service.validate_building_map(building_id)
        return MapValidationResponse(
            is_valid=report.is_valid,
            error_count=report.error_count,
            warning_count=report.warning_count,
            errors=[e.to_dict() for e in report.detailed_errors] if report.detailed_errors else [{"error_type": "ERROR", "message": e} for e in report.errors],
            warnings=[w.to_dict() for w in report.detailed_warnings] if report.detailed_warnings else [{"error_type": "WARNING", "message": w} for w in report.warnings],
            summary=report.summary,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/{dataset_id}/maps/{building_id}/evaluate",
    response_model=MapMetricsResponse,
    summary="Evaluate indoor map coverage and topological metrics",
    description="Calculates total surveyed area, graph node/edge counts, connected components, verification ratio, and accessibility ratio."
)
def evaluate_indoor_map(
    dataset_id: str,
    building_id: str,
    service: MappingService = Depends(get_mapping_service),
) -> MapMetricsResponse:
    """Evaluates quantitative metrics for a building map."""
    try:
        metrics = service.evaluate_building_map(building_id)
        return MapMetricsResponse(**metrics.to_dict())
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/{dataset_id}/maps/{building_id}/publish",
    summary="Publish an immutable version snapshot of the building map",
    description="Freezes and stores a version snapshot under versions/v{N}/."
)
def publish_indoor_map_version(
    dataset_id: str,
    building_id: str,
    req: PublishVersionRequest,
    service: MappingService = Depends(get_mapping_service),
) -> dict:
    """Publishes a numbered map release."""
    try:
        bmap = service.publish_version(building_id=building_id, version_tag=req.version_tag)
        return bmap.to_dict()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "/{dataset_id}/maps/{building_id}/versions",
    summary="List all published map version snapshots",
    description="Returns version history for a building map."
)
def list_indoor_map_versions(
    dataset_id: str,
    building_id: str,
    service: MappingService = Depends(get_mapping_service),
) -> List[dict]:
    """Retrieves list of published version snapshots."""
    return service.list_versions(building_id=building_id)


@router.get(
    "/{dataset_id}/maps/{building_id}/export/{export_format}",
    summary="Export indoor map as JSON or GeoJSON FeatureCollection",
    description="Exports the complete map or specific floor in standard JSON or RFC 7946 GeoJSON format."
)
def export_indoor_map(
    dataset_id: str,
    building_id: str,
    export_format: str,
    floor_id: Optional[int] = Query(None, description="Optional floor ID filter"),
    service: MappingService = Depends(get_mapping_service),
):
    """Exports building map in JSON or GeoJSON format."""
    try:
        result = service.export_map(building_id=building_id, export_format=export_format, floor_id=floor_id)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))







