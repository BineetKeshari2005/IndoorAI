"""Dataset management and ingestion API router."""

from __future__ import annotations

import io
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

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
