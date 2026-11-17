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
from backend.app.services.dataset_extraction import SecurityValidationError
from backend.app.services.dataset_ingestion import DatasetIngestionService
from backend.app.services.dataset_registry import DatasetRegistry
from src.core.logging import setup_logger

logger = setup_logger("DatasetsRouter")

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Global service instance dependency
_ingestion_service: Optional[DatasetIngestionService] = None


def get_ingestion_service() -> DatasetIngestionService:
    """Dependency provider for DatasetIngestionService."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = DatasetIngestionService()
    return _ingestion_service


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

