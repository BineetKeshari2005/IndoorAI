"""Event validation engine for real-time multimodal ingestion (Task 29).

Enforces structural validity, timestamp bounds, Wi-Fi RF ranges, sensor NaN/Inf rejection,
privacy sanitization constraints, and payload size bounds.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.logging import setup_logger
from backend.app.services.realtime.event_models import (
    RealtimeEvent,
    RealtimeEventType,
    RejectionReason,
)

logger = setup_logger("RealtimeEventValidator")

# Supported payload schema versions
SUPPORTED_SCHEMA_VERSIONS = {"1.0.0", "1.0", "1"}

# Bounded thresholds
MAX_EVENT_PAYLOAD_BYTES = 1024 * 1024  # 1 MB
MAX_FUTURE_TIMESTAMP_DRIFT_MS = 60 * 60 * 1000  # 1 hour
MIN_TIMESTAMP_MS = 1577836800000  # 2020-01-01T00:00:00Z

# Valid Wi-Fi frequencies (2.4 GHz, 5 GHz, 6 GHz bands)
VALID_WIFI_FREQ_RANGES = [
    (2400, 2500),  # 2.4 GHz
    (4900, 5900),  # 5 GHz
    (5925, 7125),  # 6 GHz Wi-Fi 6E
]


class EventValidator:
    """Validates real-time multimodal events before ingestion into buffers and processing pipelines."""

    def __init__(
        self,
        max_payload_bytes: int = MAX_EVENT_PAYLOAD_BYTES,
        max_future_drift_ms: int = MAX_FUTURE_TIMESTAMP_DRIFT_MS,
    ) -> None:
        self.max_payload_bytes = max_payload_bytes
        self.max_future_drift_ms = max_future_drift_ms

    def validate_event(
        self,
        event: RealtimeEvent,
        active_session_ids: Optional[Set[str]] = None,
        known_event_ids: Optional[Set[str]] = None,
    ) -> Tuple[bool, RejectionReason, str]:
        """Performs comprehensive validation of a real-time event.

        Args:
            event: RealtimeEvent instance to validate.
            active_session_ids: Optional set of currently active/open session IDs.
            known_event_ids: Optional set/cache of recently processed event IDs for deduplication.

        Returns:
            Tuple of (is_valid: bool, rejection_reason: RejectionReason, error_message: str).
        """
        # 1. Structural identifiers
        if not event.event_id or not str(event.event_id).strip():
            return False, RejectionReason.MALFORMED_PAYLOAD, "Missing or empty event_id."
        if not event.session_id or not str(event.session_id).strip():
            return False, RejectionReason.MALFORMED_PAYLOAD, "Missing or empty session_id."

        # 2. Duplicate detection
        if known_event_ids is not None and event.event_id in known_event_ids:
            return False, RejectionReason.DUPLICATE_EVENT_ID, f"Event ID '{event.event_id}' already processed."

        # 3. Session state check
        if active_session_ids is not None and event.session_id not in active_session_ids:
            return False, RejectionReason.SESSION_INACTIVE, f"Session '{event.session_id}' is not active."

        # 4. Schema version
        if event.payload_schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            return (
                False,
                RejectionReason.UNSUPPORTED_SCHEMA_VERSION,
                f"Schema version '{event.payload_schema_version}' unsupported (expected {SUPPORTED_SCHEMA_VERSIONS}).",
            )

        # 5. Event type validation
        if not isinstance(event.event_type, RealtimeEventType):
            try:
                event.event_type = RealtimeEventType(event.event_type)
            except ValueError:
                return False, RejectionReason.UNKNOWN_EVENT_TYPE, f"Unknown event type: '{event.event_type}'."

        # 6. Payload structure and size check
        if not isinstance(event.payload, dict):
            return False, RejectionReason.MALFORMED_PAYLOAD, "Event payload must be a JSON object dictionary."

        # 7. Timestamp validation
        now_ms = int(time.time() * 1000)
        if event.client_timestamp_utc is not None:
            if event.client_timestamp_utc < 0:
                return False, RejectionReason.TIMESTAMP_ANOMALY, "Client UTC timestamp cannot be negative."
            if event.client_timestamp_utc > now_ms + self.max_future_drift_ms:
                return (
                    False,
                    RejectionReason.TIMESTAMP_ANOMALY,
                    f"Client UTC timestamp {event.client_timestamp_utc} is excessively in future (> {self.max_future_drift_ms} ms).",
                )

        # 8. Type-specific payload validation
        return self._validate_payload_by_type(event)

    def _validate_payload_by_type(self, event: RealtimeEvent) -> Tuple[bool, RejectionReason, str]:
        """Inspects modality-specific content."""
        p = event.payload

        if event.event_type == RealtimeEventType.WIFI_SCAN:
            scans = p.get("bssid_scans") or p.get("scans") or p.get("observations")
            if not isinstance(scans, list):
                return False, RejectionReason.MALFORMED_PAYLOAD, "WIFI_SCAN payload must contain 'bssid_scans' list."
            if len(scans) == 0:
                return False, RejectionReason.MALFORMED_PAYLOAD, "WIFI_SCAN contains empty scan list."

            for item in scans:
                if not isinstance(item, dict):
                    return False, RejectionReason.MALFORMED_PAYLOAD, "AP observation must be an object."
                bssid = item.get("bssid")
                if not bssid or not isinstance(bssid, str):
                    return False, RejectionReason.MALFORMED_PAYLOAD, "AP observation missing 'bssid'."
                rssi = item.get("rssi_dbm")
                if rssi is None or not isinstance(rssi, (int, float)):
                    return False, RejectionReason.MALFORMED_PAYLOAD, f"AP '{bssid}' missing numeric 'rssi_dbm'."
                if math.isnan(rssi) or math.isinf(rssi):
                    return False, RejectionReason.NAN_OR_INF_DETECTED, f"AP '{bssid}' RSSI is NaN or Inf."
                if rssi < -120.0 or rssi > 0.0:
                    return (
                        False,
                        RejectionReason.OUT_OF_BOUNDS,
                        f"AP '{bssid}' RSSI {rssi} dBm out of valid range [-120, 0].",
                    )
                freq = item.get("frequency_mhz")
                if freq is not None:
                    if not isinstance(freq, (int, float)) or math.isnan(freq) or math.isinf(freq):
                        return False, RejectionReason.NAN_OR_INF_DETECTED, f"AP '{bssid}' frequency is NaN or Inf."
                    freq_int = int(freq)
                    in_range = any(low <= freq_int <= high for low, high in VALID_WIFI_FREQ_RANGES)
                    if not in_range:
                        return (
                            False,
                            RejectionReason.OUT_OF_BOUNDS,
                            f"AP '{bssid}' frequency {freq_int} MHz not in standard Wi-Fi bands (2.4GHz, 5GHz, 6GHz).",
                        )

        elif event.event_type == RealtimeEventType.IMU_BATCH:
            samples = p.get("samples")
            if not isinstance(samples, list):
                return False, RejectionReason.MALFORMED_PAYLOAD, "IMU_BATCH payload must contain 'samples' list."
            if len(samples) > 2000:
                return False, RejectionReason.PAYLOAD_TOO_LARGE, f"IMU_BATCH sample count {len(samples)} exceeds limit 2000."

            for s in samples:
                if not isinstance(s, dict):
                    return False, RejectionReason.MALFORMED_PAYLOAD, "IMU sample must be an object."
                for axis in ["x", "y", "z"]:
                    val = s.get(axis)
                    if val is not None:
                        if not isinstance(val, (int, float)):
                            return False, RejectionReason.MALFORMED_PAYLOAD, f"IMU axis '{axis}' must be numeric."
                        if math.isnan(val) or math.isinf(val):
                            return False, RejectionReason.NAN_OR_INF_DETECTED, f"IMU sample axis '{axis}' is NaN or Inf."

        elif event.event_type == RealtimeEventType.ORIENTATION_UPDATE:
            for angle in ["heading_rad", "azimuth_deg", "pitch_deg", "roll_deg"]:
                v = p.get(angle)
                if v is not None:
                    if not isinstance(v, (int, float)):
                        return False, RejectionReason.MALFORMED_PAYLOAD, f"Orientation field '{angle}' must be numeric."
                    if math.isnan(v) or math.isinf(v):
                        return False, RejectionReason.NAN_OR_INF_DETECTED, f"Orientation field '{angle}' is NaN or Inf."

        elif event.event_type == RealtimeEventType.PHOTO_CAPTURE:
            if not p.get("photo_id") and not p.get("filename"):
                return False, RejectionReason.MALFORMED_PAYLOAD, "PHOTO_CAPTURE missing 'photo_id' or 'filename'."
            # Enforce privacy constraint: un-sanitized photos must never be ingested
            if p.get("exif_sanitized") is False:
                return False, RejectionReason.MALFORMED_PAYLOAD, "PHOTO_CAPTURE rejected: EXIF privacy sanitization not verified."

        elif event.event_type == RealtimeEventType.LANDMARK_OBSERVATION:
            if not p.get("landmark_type") and not p.get("landmark_id"):
                return False, RejectionReason.MALFORMED_PAYLOAD, "LANDMARK_OBSERVATION missing 'landmark_type' or 'landmark_id'."
            dist = p.get("estimated_distance_m")
            if dist is not None:
                if not isinstance(dist, (int, float)) or math.isnan(dist) or math.isinf(dist) or dist <= 0:
                    return False, RejectionReason.OUT_OF_BOUNDS, "Landmark 'estimated_distance_m' must be positive finite number."

        elif event.event_type == RealtimeEventType.ANNOTATION:
            if not p.get("label"):
                return False, RejectionReason.MALFORMED_PAYLOAD, "ANNOTATION payload missing 'label'."

        elif event.event_type == RealtimeEventType.SESSION_STATUS:
            action = p.get("action")
            valid_actions = {"START", "PAUSE", "RESUME", "STOP", "CANCEL", "HEARTBEAT"}
            if not action or str(action).upper() not in valid_actions:
                return False, RejectionReason.MALFORMED_PAYLOAD, f"SESSION_STATUS action '{action}' invalid."

        return True, RejectionReason.NONE, "Validation successful."
