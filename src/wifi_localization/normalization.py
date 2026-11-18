"""BSSID, RSSI, and frequency normalization utilities for site-specific Wi-Fi localization."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.wifi_localization.models import ApObservation

# Standard 6-octet MAC address regular expression
_MAC_REGEX = re.compile(r"^([0-9a-fA-F]{2}[:-]){5}([0-9a-fA-F]{2})$")

MIN_VALID_RSSI_DBM: float = -105.0
MAX_VALID_RSSI_DBM: float = -20.0


def normalize_bssid(bssid_raw: str) -> Optional[str]:
    """Normalize a BSSID string to lowercase standard colon-separated format.

    Args:
        bssid_raw: Raw MAC/BSSID string (e.g. '00:1A:2B:3C:4D:5E' or '00-1a-2b-3c-4d-5e').

    Returns:
        Standardized BSSID 'aa:bb:cc:dd:ee:ff', or None if invalid.
    """
    if not bssid_raw or not isinstance(bssid_raw, str):
        return None
    cleaned = bssid_raw.strip().lower()
    if not _MAC_REGEX.match(cleaned):
        return None
    # Replace dashes or underscores with colons
    return cleaned.replace("-", ":").replace("_", ":")


def is_valid_rssi(rssi_dbm: float, min_rssi: float = MIN_VALID_RSSI_DBM, max_rssi: float = MAX_VALID_RSSI_DBM) -> bool:
    """Validate whether an RSSI measurement falls within realistic indoor Wi-Fi range."""
    if rssi_dbm is None:
        return False
    try:
        val = float(rssi_dbm)
        return min_rssi <= val <= max_rssi
    except (ValueError, TypeError):
        return False


def classify_channel_band(frequency_mhz: Optional[int]) -> str:
    """Classify Wi-Fi frequency into standard channel band string."""
    if frequency_mhz is None:
        return "unknown"
    try:
        f = int(frequency_mhz)
        if 2400 <= f <= 2500:
            return "2.4GHz"
        elif 4900 <= f <= 5900:
            return "5GHz"
        elif 5925 <= f <= 7125:
            return "6GHz"
        return "unknown"
    except (ValueError, TypeError):
        return "unknown"


def clean_ap_observation(obs_raw: Dict[str, Any]) -> Optional[ApObservation]:
    """Clean, normalize, and validate a raw access point observation record.

    Args:
        obs_raw: Dictionary containing bssid, rssi_dbm, and optional ssid/frequency.

    Returns:
        Cleaned ApObservation instance, or None if invalid.
    """
    bssid_raw = obs_raw.get("bssid")
    if not bssid_raw:
        return None
    normalized_bssid = normalize_bssid(str(bssid_raw))
    if not normalized_bssid:
        return None

    rssi_raw = obs_raw.get("rssi_dbm")
    if rssi_raw is None:
        rssi_raw = obs_raw.get("rssi")
    if not is_valid_rssi(rssi_raw):
        return None
    rssi_dbm = float(rssi_raw)

    freq_raw = obs_raw.get("frequency_mhz")
    if freq_raw is None:
        freq_raw = obs_raw.get("frequency")
    frequency_mhz = int(freq_raw) if freq_raw is not None else None

    channel_raw = obs_raw.get("channel")
    channel = int(channel_raw) if channel_raw is not None else None

    band = obs_raw.get("channel_band") or classify_channel_band(frequency_mhz)
    ssid = obs_raw.get("ssid")
    if ssid is not None:
        ssid = str(ssid).strip()

    return ApObservation(
        bssid=normalized_bssid,
        rssi_dbm=rssi_dbm,
        ssid=ssid,
        frequency_mhz=frequency_mhz,
        channel=channel,
        channel_band=band,
    )


def deduplicate_sweep_observations(observations: List[ApObservation]) -> List[ApObservation]:
    """Deduplicate observations within a single sweep by keeping the strongest RSSI per BSSID.

    Args:
        observations: List of cleaned ApObservation instances.

    Returns:
        Deduplicated list with at most one entry per BSSID.
    """
    best_by_bssid: Dict[str, ApObservation] = {}
    for obs in observations:
        if obs.bssid not in best_by_bssid or obs.rssi_dbm > best_by_bssid[obs.bssid].rssi_dbm:
            best_by_bssid[obs.bssid] = obs
    return list(best_by_bssid.values())
