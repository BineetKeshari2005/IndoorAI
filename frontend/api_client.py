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

    def register_pdr_trial(self, dataset_id: str, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Registers a measured ground-truth walking trial linked to a dataset session."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/pdr/trials"
        try:
            res = self._request("POST", url, json=trial_data, timeout=30.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"Failed to register trial: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text
            )
        except httpx.RequestError as e:
            logger.error("Connection error registering trial for %s: %s", dataset_id, e)
            raise DatasetApiClientException(f"Connection error registering trial: {e}")

    def get_pdr_trials(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Fetches registered walking trials for a dataset."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/pdr/trials"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json().get("trials", [])
            return []
        except httpx.RequestError as e:
            logger.warning("Error fetching trials for %s: %s", dataset_id, e)
            return []

    def get_pdr_trial_detail(self, dataset_id: str, trial_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves full trial record and ground-truth measurements."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/pdr/trials/{trial_id}"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json()
            return None
        except httpx.RequestError as e:
            logger.warning("Error fetching trial %s detail: %s", trial_id, e)
            return None

    def calibrate_pdr(self, dataset_id: str, cal_params: Dict[str, Any]) -> Dict[str, Any]:
        """Triggers Weinberg step-length parameter estimation over selected trials."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/pdr/calibrate"
        try:
            res = self._request("POST", url, json=cal_params, timeout=60.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"Calibration failed: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text
            )
        except httpx.RequestError as e:
            logger.error("Connection error calibrating PDR for %s: %s", dataset_id, e)
            raise DatasetApiClientException(f"Connection error calibrating PDR: {e}")

    def get_pdr_calibration_profiles(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Retrieves all registered PDR calibration profiles."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/pdr/calibration"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json().get("profiles", [])
            return []
        except httpx.RequestError as e:
            logger.warning("Error fetching calibration profiles: %s", e)
            return []

    def evaluate_pdr(self, dataset_id: str, eval_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluates PDR trajectories against specified ground-truth walking trials."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/pdr/evaluate"
        try:
            res = self._request("POST", url, json=eval_params, timeout=60.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"Evaluation failed: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text
            )
        except httpx.RequestError as e:
            logger.error("Connection error evaluating PDR for %s: %s", dataset_id, e)
            raise DatasetApiClientException(f"Connection error evaluating PDR: {e}")

    def get_pdr_evaluation_reports(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Retrieves recent evaluation reports for a dataset."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/pdr/evaluation"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json()
            return []
        except httpx.RequestError as e:
            logger.warning("Error fetching evaluation reports for %s: %s", dataset_id, e)
            return []

    # -------------------------------------------------------------------------
    # Task 23: Site-Specific Wi-Fi Radio Map & Localization API Methods
    # -------------------------------------------------------------------------

    def register_wifi_survey_location(
        self, dataset_id: str, location_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Registers or updates a verified reference landmark location for a site."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/wifi-survey/locations"
        try:
            res = self._request("POST", url, json=location_data, timeout=30.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"Failed to register reference location: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text,
            )
        except httpx.RequestError as e:
            logger.error("Connection error registering reference location for %s: %s", dataset_id, e)
            raise DatasetApiClientException(f"Connection error registering reference location: {e}")

    def list_wifi_survey_locations(
        self, dataset_id: str, site_id: str
    ) -> List[Dict[str, Any]]:
        """Retrieves all registered reference locations for a specific site."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/wifi-survey/locations?site_id={site_id}"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json().get("locations", [])
            return []
        except httpx.RequestError as e:
            logger.warning("Error fetching reference locations for site %s: %s", site_id, e)
            return []

    def build_wifi_radio_map(
        self, dataset_id: str, build_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compiles a versioned site radio map from survey dataset observations."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/wifi-survey/fingerprints/build"
        try:
            res = self._request("POST", url, json=build_params, timeout=60.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"Failed to build radio map: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text,
            )
        except httpx.RequestError as e:
            logger.error("Connection error building radio map for %s: %s", dataset_id, e)
            raise DatasetApiClientException(f"Connection error building radio map: {e}")

    def get_wifi_radio_map(
        self, dataset_id: str, site_id: str, version: str = "v1"
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a compiled site radio map."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/wifi-survey/radio-map?site_id={site_id}&version={version}"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json()
            return None
        except httpx.RequestError as e:
            logger.warning("Error fetching radio map %s/%s: %s", site_id, version, e)
            return None

    def list_wifi_radio_maps(
        self, dataset_id: str, site_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Lists compiled radio maps available on disk."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/wifi-survey/radio-maps"
        if site_id:
            url += f"?site_id={site_id}"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json().get("radio_maps", [])
            return []
        except httpx.RequestError as e:
            logger.warning("Error listing radio maps: %s", e)
            return []

    def localize_wifi_fingerprint(
        self, dataset_id: str, query_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimates location from an input Wi-Fi query fingerprint."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/wifi-survey/localize"
        try:
            res = self._request("POST", url, json=query_data, timeout=30.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"Localization failed: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text,
            )
        except httpx.RequestError as e:
            logger.error("Connection error during fingerprint localization: %s", e)
            raise DatasetApiClientException(f"Connection error during localization: {e}")

    def evaluate_wifi_localization(
        self, dataset_id: str, eval_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Runs session-partitioned offline evaluation on survey dataset."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/wifi-survey/evaluate"
        try:
            res = self._request("POST", url, json=eval_params, timeout=60.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"Evaluation failed: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text,
            )
        except httpx.RequestError as e:
            logger.error("Connection error during Wi-Fi evaluation: %s", e)
            raise DatasetApiClientException(f"Connection error during evaluation: {e}")

    # -------------------------------------------------------------------------
    # Task 24: Wi-Fi + PDR Sensor Fusion API Methods (Extended Kalman Filter)
    # -------------------------------------------------------------------------

    def run_sensor_fusion(
        self, dataset_id: str, fusion_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes EKF sensor fusion combining PDR step kinematics with Wi-Fi positioning."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/fusion/run"
        try:
            res = self._request("POST", url, json=fusion_params, timeout=60.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"Fusion execution failed: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text,
            )
        except httpx.RequestError as e:
            logger.error("Connection error running sensor fusion for %s: %s", dataset_id, e)
            raise DatasetApiClientException(f"Connection error running sensor fusion: {e}")

    def get_fusion_status(self, dataset_id: str) -> Dict[str, Any]:
        """Retrieves sensor fusion artifact availability and summary."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/fusion/status"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json()
            return {"dataset_id": dataset_id, "is_available": False, "summary": None, "artifact_paths": {}}
        except httpx.RequestError as e:
            logger.warning("Error fetching fusion status for %s: %s", dataset_id, e)
            return {"dataset_id": dataset_id, "is_available": False, "summary": None, "artifact_paths": {}}

    def get_fused_trajectory(
        self, dataset_id: str, limit: int = 5000
    ) -> List[Dict[str, Any]]:
        """Retrieves fused trajectory points for visualization."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/fusion/trajectory?limit={limit}"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json().get("trajectory", [])
            return []
        except httpx.RequestError as e:
            logger.warning("Error fetching fused trajectory for %s: %s", dataset_id, e)
            return []

    def evaluate_sensor_fusion(
        self, dataset_id: str, eval_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Runs comparative trajectory evaluation comparing PDR-only, Wi-Fi-only, and Fused."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/fusion/evaluate"
        try:
            res = self._request("POST", url, json=eval_params, timeout=60.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"Fusion evaluation failed: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text,
            )
        except httpx.RequestError as e:
            logger.error("Connection error evaluating sensor fusion for %s: %s", dataset_id, e)
            raise DatasetApiClientException(f"Connection error evaluating sensor fusion: {e}")

    def get_fusion_evaluation_report(
        self, dataset_id: str, site_id: str = "SITE-DEFAULT"
    ) -> Dict[str, Any]:
        """Retrieves evaluation report for dataset."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/fusion/evaluation?site_id={site_id}"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json().get("report", {})
            return {}
        except httpx.RequestError as e:
            logger.warning("Error fetching evaluation report for %s: %s", dataset_id, e)
            return {}

    # -------------------------------------------------------------------------
    # Task 25: Computer Vision & Indoor Landmark Detection API Methods
    # -------------------------------------------------------------------------

    def analyze_vision(
        self, dataset_id: str, analyze_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Runs landmark detection and OCR on survey photographs."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/vision/analyze"
        payload = analyze_params or {}
        try:
            res = self._request("POST", url, json=payload, timeout=90.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"Vision analysis failed: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text,
            )
        except httpx.RequestError as e:
            logger.error("Connection error running vision analysis for %s: %s", dataset_id, e)
            raise DatasetApiClientException(f"Connection error running vision analysis: {e}")

    def get_vision_detections(
        self, dataset_id: str, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves computed landmark candidates for a dataset."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/vision/detections"
        if category:
            url += f"?category={category}"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json()
            return []
        except httpx.RequestError as e:
            logger.warning("Error fetching landmark detections for %s: %s", dataset_id, e)
            return []

    def list_landmarks(
        self,
        dataset_id: str,
        building_id: Optional[str] = None,
        floor_id: Optional[int] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Queries persistent landmarks in the landmark registry."""
        params = []
        if building_id:
            params.append(f"building_id={building_id}")
        if floor_id is not None:
            params.append(f"floor_id={floor_id}")
        if category:
            params.append(f"category={category}")
        if status:
            params.append(f"status={status}")
        query_str = ("?" + "&".join(params)) if params else ""
        url = f"{self.base_url}/api/datasets/{dataset_id}/vision/landmarks{query_str}"
        try:
            res = self._request("GET", url)
            if res.status_code == 200:
                return res.json()
            return []
        except httpx.RequestError as e:
            logger.warning("Error fetching landmarks: %s", e)
            return []

    def verify_landmark(
        self, dataset_id: str, verify_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Manually updates or verifies a landmark record."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/vision/landmarks/verify"
        try:
            res = self._request("POST", url, json=verify_data, timeout=30.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"Failed to verify landmark: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text,
            )
        except httpx.RequestError as e:
            logger.error("Connection error verifying landmark: %s", e)
            raise DatasetApiClientException(f"Connection error verifying landmark: {e}")

    def match_survey_photo(
        self, dataset_id: str, match_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Matches a survey photo against registered landmarks."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/vision/match"
        try:
            res = self._request("POST", url, json=match_data, timeout=30.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"Visual matching failed: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text,
            )
        except httpx.RequestError as e:
            logger.error("Connection error matching photo: %s", e)
            raise DatasetApiClientException(f"Connection error matching photo: {e}")

    def evaluate_vision(
        self, dataset_id: str, eval_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Runs evaluation for landmark detection, OCR, and matching."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/vision/evaluate"
        try:
            res = self._request("POST", url, json=eval_params, timeout=60.0)
            if res.status_code in (200, 201):
                return res.json()
            raise DatasetApiClientException(
                f"Vision evaluation failed: HTTP {res.status_code}",
                status_code=res.status_code,
                details=res.text,
            )
        except httpx.RequestError as e:
            logger.error("Connection error evaluating vision: %s", e)
            raise DatasetApiClientException(f"Connection error evaluating vision: {e}")

    # =====================================================================
    # Task 26: Indoor Map Generation & Spatial Representation
    # =====================================================================

    def list_indoor_maps(self, dataset_id: str = "default") -> List[Dict[str, Any]]:
        """Lists metadata summaries for all building maps."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/buildings"
        try:
            res = self._request("GET", url, timeout=15.0)
            if res.status_code == 200:
                return res.json()
            return []
        except Exception as e:
            logger.warning("Error fetching indoor maps: %s", e)
            return []

    def get_indoor_map(self, dataset_id: str, building_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves comprehensive building map definition."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/{building_id}"
        try:
            res = self._request("GET", url, timeout=15.0)
            if res.status_code == 200:
                return res.json()
            return None
        except Exception as e:
            logger.warning("Error fetching building map %s: %s", building_id, e)
            return None

    def create_or_update_building_map(
        self, dataset_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Creates or updates an indoor building map."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/buildings"
        res = self._request("POST", url, json=payload, timeout=20.0)
        if res.status_code in (200, 201):
            return res.json()
        raise DatasetApiClientException(f"Failed to create building map: HTTP {res.status_code}", details=res.text)

    def add_map_floor(
        self, dataset_id: str, building_id: str, floor_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adds a floor level to the building map."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/floors?building_id={building_id}"
        res = self._request("POST", url, json=floor_data, timeout=20.0)
        if res.status_code in (200, 201):
            return res.json()
        raise DatasetApiClientException(f"Failed to add floor: HTTP {res.status_code}", details=res.text)

    def add_map_room(
        self, dataset_id: str, building_id: str, room_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adds a room polygon to the building floor."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/rooms?building_id={building_id}"
        res = self._request("POST", url, json=room_data, timeout=20.0)
        if res.status_code in (200, 201):
            return res.json()
        raise DatasetApiClientException(f"Failed to add room: HTTP {res.status_code}", details=res.text)

    def add_map_corridor(
        self, dataset_id: str, building_id: str, corridor_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adds a corridor to the building floor."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/corridors?building_id={building_id}"
        res = self._request("POST", url, json=corridor_data, timeout=20.0)
        if res.status_code in (200, 201):
            return res.json()
        raise DatasetApiClientException(f"Failed to add corridor: HTTP {res.status_code}", details=res.text)

    def add_map_door(
        self, dataset_id: str, building_id: str, door_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adds a door to the building floor."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/doors?building_id={building_id}"
        res = self._request("POST", url, json=door_data, timeout=20.0)
        if res.status_code in (200, 201):
            return res.json()
        raise DatasetApiClientException(f"Failed to add door: HTTP {res.status_code}", details=res.text)

    def add_map_vertical_transition(
        self, dataset_id: str, building_id: str, vt_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adds a vertical transition (staircase / elevator)."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/vertical-transitions?building_id={building_id}"
        res = self._request("POST", url, json=vt_data, timeout=20.0)
        if res.status_code in (200, 201):
            return res.json()
        raise DatasetApiClientException(f"Failed to add vertical transition: HTTP {res.status_code}", details=res.text)

    def add_map_connection(
        self, dataset_id: str, building_id: str, conn_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adds a topological connection edge."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/connections?building_id={building_id}"
        res = self._request("POST", url, json=conn_data, timeout=20.0)
        if res.status_code in (200, 201):
            return res.json()
        raise DatasetApiClientException(f"Failed to add connection: HTTP {res.status_code}", details=res.text)

    def import_wifi_references_to_map(
        self, dataset_id: str, building_id: str, import_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Imports Wi-Fi reference points into floor map."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/{building_id}/import-wifi"
        res = self._request("POST", url, json=import_data, timeout=30.0)
        if res.status_code in (200, 201):
            return res.json()
        raise DatasetApiClientException(f"Failed to import Wi-Fi references: HTTP {res.status_code}", details=res.text)

    def import_landmarks_to_map(
        self, dataset_id: str, building_id: str, import_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Imports verified visual landmarks into floor map."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/{building_id}/import-landmarks"
        res = self._request("POST", url, json=import_data, timeout=30.0)
        if res.status_code in (200, 201):
            return res.json()
        raise DatasetApiClientException(f"Failed to import landmarks: HTTP {res.status_code}", details=res.text)

    def validate_indoor_map(self, dataset_id: str, building_id: str) -> Dict[str, Any]:
        """Runs geometric and topological validation on the building map."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/{building_id}/validate"
        res = self._request("POST", url, timeout=20.0)
        if res.status_code == 200:
            return res.json()
        raise DatasetApiClientException(f"Map validation failed: HTTP {res.status_code}", details=res.text)

    def evaluate_indoor_map(self, dataset_id: str, building_id: str) -> Dict[str, Any]:
        """Evaluates quantitative coverage and topological connectivity metrics."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/{building_id}/evaluate"
        res = self._request("POST", url, timeout=20.0)
        if res.status_code == 200:
            return res.json()
        raise DatasetApiClientException(f"Map evaluation failed: HTTP {res.status_code}", details=res.text)

    def publish_indoor_map_version(
        self, dataset_id: str, building_id: str, publish_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Publishes an immutable version release snapshot."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/{building_id}/publish"
        res = self._request("POST", url, json=publish_data or {}, timeout=20.0)
        if res.status_code in (200, 201):
            return res.json()
        raise DatasetApiClientException(f"Failed to publish map version: HTTP {res.status_code}", details=res.text)

    def list_indoor_map_versions(self, dataset_id: str, building_id: str) -> List[Dict[str, Any]]:
        """Lists version snapshot history for a building map."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/{building_id}/versions"
        try:
            res = self._request("GET", url, timeout=15.0)
            if res.status_code == 200:
                return res.json()
            return []
        except Exception as e:
            logger.warning("Error fetching map versions: %s", e)
            return []

    def export_indoor_map(
        self,
        dataset_id: str,
        building_id: str,
        export_format: str = "json",
        floor_id: Optional[int] = None,
    ) -> Any:
        """Exports building map as JSON string or GeoJSON FeatureCollection dictionary."""
        params = f"?floor_id={floor_id}" if floor_id is not None else ""
        url = f"{self.base_url}/api/datasets/{dataset_id}/maps/{building_id}/export/{export_format}{params}"
        res = self._request("GET", url, timeout=20.0)
        if res.status_code == 200:
            return res.json()
        raise DatasetApiClientException(f"Failed to export map: HTTP {res.status_code}", details=res.text)

    # -------------------------------------------------------------------------
    # Map-Constrained Multimodal Particle Filter Localization (Task 27)
    # -------------------------------------------------------------------------

    def run_map_localization(
        self,
        dataset_id: str,
        run_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Executes map-constrained particle filtering on dataset."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/map-localization/run"
        res = self._request("POST", url, json=run_params, timeout=60.0)
        if res.status_code == 200:
            return res.json()
        raise DatasetApiClientException(f"Failed to run map localization: HTTP {res.status_code}", details=res.text)

    def get_map_localization_status(self, dataset_id: str) -> Dict[str, Any]:
        """Checks whether map localization trajectory artifacts are computed."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/map-localization/status"
        res = self._request("GET", url, timeout=10.0)
        if res.status_code == 200:
            return res.json()
        return {"dataset_id": dataset_id, "is_available": False}

    def get_map_localization_trajectory(
        self,
        dataset_id: str,
        limit: int = 5000,
    ) -> Dict[str, Any]:
        """Retrieves map-constrained particle filter trajectory points and operational summary."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/map-localization/trajectory?limit={limit}"
        res = self._request("GET", url, timeout=30.0)
        if res.status_code == 200:
            return res.json()
        raise DatasetApiClientException(f"Failed to get map trajectory: HTTP {res.status_code}", details=res.text)

    def evaluate_map_localization(
        self,
        dataset_id: str,
        eval_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Executes comparative evaluation of map-constrained PF against ground truth or baseline."""
        url = f"{self.base_url}/api/datasets/{dataset_id}/map-localization/evaluate"
        res = self._request("POST", url, json=eval_params, timeout=60.0)
        if res.status_code == 200:
            return res.json()
        raise DatasetApiClientException(f"Failed to evaluate map localization: HTTP {res.status_code}", details=res.text)

    def get_map_localization_evaluation(
        self,
        dataset_id: str,
        use_synthetic_fixture: bool = True,
    ) -> Dict[str, Any]:
        """Fetches or generates default evaluation report."""
        synth_flag = "true" if use_synthetic_fixture else "false"
        url = f"{self.base_url}/api/datasets/{dataset_id}/map-localization/evaluation?use_synthetic_fixture={synth_flag}"
        res = self._request("GET", url, timeout=40.0)
        if res.status_code == 200:
            return res.json()
        raise DatasetApiClientException(f"Failed to fetch evaluation report: HTTP {res.status_code}", details=res.text)






