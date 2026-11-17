"""Dataset API client connecting the IndoorAI frontend dashboard to the FastAPI backend."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from src.core.logging import setup_logger

logger = setup_logger("DatasetApiClient")

DEFAULT_BACKEND_URL = os.getenv("INDOORAI_BACKEND_URL", "http://127.0.0.1:8000")


class DatasetApiClientException(Exception):
    """Custom exception raised for client-side API communication failures."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


class DatasetApiClient:
    """Robust HTTP client for interacting with IndoorAI backend survey dataset services."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_seconds: float = 30.0,
        http_client: Optional[Any] = None
    ):
        self.base_url = (base_url or DEFAULT_BACKEND_URL).rstrip("/")
        self.timeout = timeout_seconds
        self.http_client = http_client

    def _request(self, method: str, url: str, timeout: Optional[float] = None, **kwargs) -> Any:
        """Executes an HTTP request using either the injected client or a fresh httpx.Client."""
        if self.http_client is not None:
            func = getattr(self.http_client, method.lower())
            return func(url, **kwargs)
        with httpx.Client(timeout=timeout or self.timeout) as client:
            func = getattr(client, method.lower())
            return func(url, **kwargs)

    def check_health(self) -> Tuple[bool, str]:
        """Probes backend health endpoint to confirm server availability."""
        url = f"{self.base_url}/health"
        try:
            res = self._request("GET", url, timeout=3.0)
            if res.status_code == 200:
                return True, "Backend Connected (FastAPI)"
            return False, f"Backend returned HTTP {res.status_code}"
        except httpx.ConnectError:
            return False, f"Cannot connect to backend at {self.base_url} (Connection Refused)"
        except httpx.TimeoutException:
            return False, f"Connection timed out contacting {self.base_url}"
        except Exception as e:
            return False, f"Connection check failed: {e}"

    def list_datasets(
        self,
        status: Optional[str] = None,
        source_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves list of all cataloged multimodal survey datasets."""
        url = f"{self.base_url}/api/datasets"
        params: Dict[str, str] = {}
        if status:
            params["status"] = status
        if source_type:
            params["source_type"] = source_type

        try:
            res = self._request("GET", url, params=params)
            if res.status_code == 200:
                return res.json()
            raise DatasetApiClientException(
                f"Failed to list datasets: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text
            )
        except httpx.RequestError as e:
            logger.warning("Error listing datasets from %s: %s", url, e)
            raise DatasetApiClientException(f"Connection error to backend: {e}")

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Fetches detailed metadata for a single dataset."""
        url = f"{self.base_url}/api/datasets/{dataset_id}"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json()
            if res.status_code == 404:
                return None
            raise DatasetApiClientException(
                f"Failed to retrieve dataset {dataset_id}: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text
            )
        except httpx.RequestError as e:
            logger.warning("Error fetching dataset %s: %s", dataset_id, e)
            raise DatasetApiClientException(f"Connection error fetching dataset: {e}")

    def get_quality_report(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the full machine-readable quality report for a dataset."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/quality"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json()
            if res.status_code == 404:
                return None
            raise DatasetApiClientException(
                f"Failed to retrieve quality report: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text
            )
        except httpx.RequestError as e:
            logger.warning("Error fetching quality report for %s: %s", dataset_id, e)
            raise DatasetApiClientException(f"Connection error fetching quality report: {e}")

    def upload_dataset(
        self,
        file_bytes: bytes,
        filename: str = "survey_export.zip",
        validate_only: bool = False
    ) -> Tuple[bool, Dict[str, Any], str]:
        """Uploads a survey session ZIP bundle to the ingestion pipeline."""
        url = f"{self.base_url}/api/datasets/upload"
        params = {"validate_only": str(validate_only).lower()}
        files = {"file": (filename, file_bytes, "application/zip")}

        try:
            res = self._request("POST", url, params=params, files=files, timeout=max(self.timeout, 60.0))
            if res.status_code in (200, 201):
                data = res.json()
                msg = data.get("message", "Upload and ingestion successful.")
                return True, data, msg
            detail = "Upload failed"
            try:
                detail = res.json().get("detail", res.text)
            except Exception:
                detail = res.text
            return False, {}, f"HTTP {res.status_code}: {detail}"
        except httpx.RequestError as e:
            return False, {}, f"Network connection error during upload: {e}"

    def revalidate_dataset(self, dataset_id: str) -> Tuple[bool, Dict[str, Any], str]:
        """Triggers on-demand revalidation of an ingested dataset."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/validate"
        try:
            res = self._request("POST", url)
            if res.status_code == 200:
                data = res.json()
                return True, data, "Revalidation completed successfully."
            detail = res.text
            try:
                detail = res.json().get("detail", detail)
            except Exception:
                pass
            return False, {}, f"Revalidation failed: HTTP {res.status_code} ({detail})"
        except httpx.RequestError as e:
            return False, {}, f"Connection error during revalidation: {e}"

    def delete_dataset(self, dataset_id: str) -> Tuple[bool, str]:
        """Deletes a cataloged survey dataset and purges its stored files."""
        url = f"{self.base_url}/api/datasets/{dataset_id}"
        try:
            res = self._request("DELETE", url)
            if res.status_code == 200:
                return True, f"Dataset '{dataset_id}' deleted successfully."
            detail = res.text
            try:
                detail = res.json().get("detail", detail)
            except Exception:
                pass
            return False, f"Deletion failed: HTTP {res.status_code} ({detail})"
        except httpx.RequestError as e:
            return False, f"Connection error during deletion: {e}"

    def get_observations(
        self,
        dataset_id: str,
        modality: str,
        limit: int = 2000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Fetches parsed observation rows for a given modality (wifi, sensors, photos, annotations)."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/observations/{modality}"
        params = {"limit": limit, "offset": offset}
        try:
            res = self._request("GET", url, params=params)
            if res.status_code == 200:
                data = res.json()
                return data.get("records", [])
            if res.status_code == 404:
                return []
            raise DatasetApiClientException(
                f"Failed to fetch {modality} observations: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text
            )
        except httpx.RequestError as e:
            logger.warning("Error fetching %s observations for %s: %s", modality, dataset_id, e)
            raise DatasetApiClientException(f"Connection error fetching observations: {e}")

    def get_image_bytes(self, dataset_id: str, filename: str) -> Optional[bytes]:
        """Fetches binary bytes for a privacy-sanitized landmark photo."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/images/{filename}"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.content
            if res.status_code in (403, 404):
                logger.warning("Image %s retrieval denied or not found (HTTP %d)", filename, res.status_code)
                return None
            return None
        except httpx.RequestError as e:
            logger.warning("Error fetching image %s for %s: %s", filename, dataset_id, e)
            return None

    def preprocess_dataset(self, dataset_id: str, config_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Triggers preprocessing and temporal alignment for an ingested dataset."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/preprocess"
        payload = config_params or {}
        try:
            res = self._request("POST", url, json=payload, timeout=60.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"Preprocessing failed: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text
            )
        except httpx.RequestError as e:
            logger.error("Connection error triggering preprocessing for %s: %s", dataset_id, e)
            raise DatasetApiClientException(f"Failed to connect to preprocessing service: {e}")

    def get_preprocessing_status(self, dataset_id: str) -> Dict[str, Any]:
        """Fetches the latest preprocessing status and report for a dataset."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/preprocessing"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json()
            if res.status_code == 404:
                return {"dataset_id": dataset_id, "is_preprocessed": False}
            raise DatasetApiClientException(
                f"Failed to fetch preprocessing status: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text
            )
        except httpx.RequestError as e:
            logger.error("Connection error fetching preprocessing status for %s: %s", dataset_id, e)
            raise DatasetApiClientException(f"Failed to connect to backend: {e}")

    def get_processed_files(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Retrieves list of generated processed artifact files for a dataset."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/preprocessing/files"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                data = res.json()
                return data.get("files", [])
            if res.status_code == 404:
                return []
            return []
        except httpx.RequestError as e:
            logger.warning("Error fetching processed files for %s: %s", dataset_id, e)
            return []

    def run_pdr(self, dataset_id: str, config_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Triggers Pedestrian Dead Reckoning (PDR) trajectory reconstruction."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/pdr"
        payload = config_params or {}
        try:
            res = self._request("POST", url, json=payload, timeout=60.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"PDR execution failed: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text
            )
        except httpx.RequestError as e:
            logger.error("Connection error running PDR for %s: %s", dataset_id, e)
            raise DatasetApiClientException(f"Failed to connect to PDR service: {e}")

    def get_pdr_status(self, dataset_id: str) -> Dict[str, Any]:
        """Fetches the latest PDR status and summary report for a dataset."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/pdr"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json()
            if res.status_code == 404:
                return {"dataset_id": dataset_id, "is_computed": False}
            raise DatasetApiClientException(
                f"Failed to fetch PDR status: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text
            )
        except httpx.RequestError as e:
            logger.error("Connection error fetching PDR status for %s: %s", dataset_id, e)
            raise DatasetApiClientException(f"Failed to connect to backend: {e}")

    def get_pdr_trajectory(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Retrieves 2D relative trajectory coordinate points."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/pdr/trajectory"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                data = res.json()
                return data.get("points", [])
            if res.status_code == 404:
                return []
            return []
        except httpx.RequestError as e:
            logger.warning("Error fetching PDR trajectory for %s: %s", dataset_id, e)
            return []

    def get_pdr_steps(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Retrieves detected step events with kinematics."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/pdr/steps"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                data = res.json()
                return data.get("steps", [])
            if res.status_code == 404:
                return []
            return []
        except httpx.RequestError as e:
            logger.warning("Error fetching PDR steps for %s: %s", dataset_id, e)
            return []


