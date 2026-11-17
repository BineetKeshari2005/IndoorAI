"""Dataset validation and multimodal quality assurance engine for Task 17 exports."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from backend.app.core.dataset_config import DatasetConfig, default_dataset_config
from backend.app.schemas.dataset import (
    AnnotationQualityMetrics,
    CrossModalTemporalOverlap,
    DatasetQualityReport,
    IntegrityCheckResult,
    ModalitiesQualitySummary,
    PhotoQualityMetrics,
    SensorQualityMetrics,
    TimestampQualityMetrics,
    ValidationStatus,
    WifiQualityMetrics,
)
from src.core.logging import setup_logger

logger = setup_logger("DatasetValidator")

# Expected CSV Column Headers (Task 17 Specification)
EXPECTED_WIFI_HEADER = [
    "observation_id", "session_id", "original_timestamp", "time_source",
    "normalized_epoch_ms", "normalized_iso", "sync_method", "uncertainty_ms",
    "is_fresh", "is_mock", "bssid", "ssid", "rssi_dbm", "frequency_mhz",
    "channel", "channel_band", "building", "floor", "room"
]

EXPECTED_IMU_HEADER = [
    "observation_id", "session_id", "original_timestamp_ns", "time_source",
    "normalized_epoch_ms", "normalized_iso", "sync_method", "uncertainty_ms",
    "sensor_type", "x", "y", "z", "magnitude", "accuracy", "sampling_rate", "is_mock"
]

EXPECTED_PHOTO_HEADER = [
    "photo_id", "session_id", "original_timestamp", "time_source",
    "normalized_epoch_ms", "normalized_iso", "sync_method", "uncertainty_ms",
    "filename", "file_path", "file_size_bytes", "image_width", "image_height",
    "building", "floor", "room", "landmark_type_code", "landmark_type_name",
    "description", "notes", "is_exif_sanitized", "faces_detected_count",
    "face_blur_applied", "face_detection_status", "is_mock"
]

EXPECTED_ANNOTATION_HEADER = [
    "annotation_id", "session_id", "original_timestamp", "time_source",
    "normalized_epoch_ms", "normalized_iso", "sync_method", "uncertainty_ms",
    "type_code", "type_name", "description", "building", "floor", "room",
    "related_observation_id", "is_mock"
]

UUID_REGEX = re.compile(
    r"^(SES-|WIFI-|IMU-|ANN-|IMG-)?[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})?$"
)


def calculate_sha256(file_path: Path) -> str:
    """Computes hexadecimal SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


class DatasetValidator:
    """Validates structure, integrity, schema, and data quality of extracted survey datasets."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or default_dataset_config

    def validate_dataset(self, extracted_dir: Path, dataset_id: str) -> DatasetQualityReport:
        """Runs the complete end-to-end validation pipeline on an extracted dataset workspace."""
        errors: List[str] = []
        warnings: List[str] = []

        extracted_dir = extracted_dir.resolve()
        session_file = extracted_dir / "session.json"
        manifest_file = extracted_dir / "manifest.json"
        wifi_file = extracted_dir / "wifi_observations.csv"
        imu_file = extracted_dir / "imu_observations.csv"
        photo_file = extracted_dir / "photo_metadata.csv"
        annotation_file = extracted_dir / "annotations.csv"
        images_dir = extracted_dir / self.config.images_subfolder

        # 1. Verify existence of required core files
        for req in self.config.required_archive_files:
            target = extracted_dir / req
            if not target.exists() or not target.is_file():
                errors.append(f"Missing required file in dataset archive: '{req}'")

        if errors:
            return DatasetQualityReport(
                dataset_id=dataset_id,
                session_id="UNKNOWN",
                session_name="Incomplete Dataset",
                validation_status=ValidationStatus.CORRUPTED,
                integrity=IntegrityCheckResult(
                    manifest_valid=False,
                    checksums_valid=False,
                    record_counts_valid=False,
                    files_verified_count=0,
                    files_failed_count=len(errors),
                    mismatched_files=errors.copy()
                ),
                modalities=ModalitiesQualitySummary(),
                timestamp_quality=TimestampQualityMetrics(
                    original_timestamps_present=False,
                    normalized_timestamps_present=False
                ),
                cross_modal_overlap=CrossModalTemporalOverlap(),
                warnings=warnings,
                errors=errors
            )

        # 2. Parse session.json and manifest.json
        session_data: Dict[str, Any] = {}
        manifest_data: Dict[str, Any] = {}

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
        except Exception as e:
            errors.append(f"Failed to parse session.json: {e}")

        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as e:
            errors.append(f"Failed to parse manifest.json: {e}")

        session_id = session_data.get("sessionId", "UNKNOWN")
        session_name = session_data.get("sessionName", "Survey Session")

        # Session ID consistency
        manifest_session_id = manifest_data.get("sessionId")
        if manifest_session_id and manifest_session_id != session_id:
            errors.append(
                f"Session ID mismatch: session.json has '{session_id}' but manifest.json has '{manifest_session_id}'"
            )

        # Export version check
        export_version = manifest_data.get("exportVersion", "")
        if export_version not in self.config.supported_export_versions:
            warnings.append(
                f"Manifest exportVersion '{export_version}' is not in supported versions {self.config.supported_export_versions}"
            )

        # Clock anchor check
        clock_anchor = session_data.get("clockAnchor")
        if not clock_anchor or not isinstance(clock_anchor, dict):
            errors.append("session.json is missing 'clockAnchor' object required for timestamp reconciliation.")
        else:
            for required_anchor_field in ("wallClockEpochMs", "bootTimeNs", "anchorIsoUtc"):
                if required_anchor_field not in clock_anchor:
                    errors.append(f"clockAnchor missing required field '{required_anchor_field}'")

        # 3. Cryptographic Manifest & File Checksums Check
        integrity_result = self._verify_manifest_checksums(extracted_dir, manifest_data)
        if not integrity_result.checksums_valid or integrity_result.files_failed_count > 0:
            errors.extend([f"Checksum mismatch on file: '{fn}'" for fn in integrity_result.mismatched_files])

        # 4. Parse and Validate Observations
        wifi_rows, wifi_errs = self._read_and_validate_csv(wifi_file, EXPECTED_WIFI_HEADER, session_id, "WIFI")
        imu_rows, imu_errs = self._read_and_validate_csv(imu_file, EXPECTED_IMU_HEADER, session_id, "IMU")
        photo_rows, photo_errs = self._read_and_validate_csv(photo_file, EXPECTED_PHOTO_HEADER, session_id, "IMG")
        annotation_rows, ann_errs = self._read_and_validate_csv(annotation_file, EXPECTED_ANNOTATION_HEADER, session_id, "ANN")

        errors.extend(wifi_errs)
        errors.extend(imu_errs)
        errors.extend(photo_errs)
        errors.extend(ann_errs)

        # 5. Manifest record counts validation
        manifest_counts = manifest_data.get("recordCounts", {})
        wifi_sweeps_count = len({r["observation_id"] for r in wifi_rows if "observation_id" in r})
        imu_count = len(imu_rows)
        photo_count = len(photo_rows)
        annotation_count = len(annotation_rows)

        expected_wifi = manifest_counts.get("wifiCount")
        expected_imu = manifest_counts.get("imuCount")
        expected_photo = manifest_counts.get("photoCount")
        expected_ann = manifest_counts.get("annotationCount")

        if expected_wifi is not None and expected_wifi != wifi_sweeps_count:
            errors.append(f"Wi-Fi sweep count mismatch: manifest expected {expected_wifi}, parsed {wifi_sweeps_count}")
        if expected_imu is not None and expected_imu != imu_count:
            errors.append(f"IMU count mismatch: manifest expected {expected_imu}, parsed {imu_count}")
        if expected_photo is not None and expected_photo != photo_count:
            errors.append(f"Photo count mismatch: manifest expected {expected_photo}, parsed {photo_count}")
        if expected_ann is not None and expected_ann != annotation_count:
            errors.append(f"Annotation count mismatch: manifest expected {expected_ann}, parsed {annotation_count}")

        # Update record_counts_valid flag
        integrity_result.record_counts_valid = (
            (expected_wifi is None or expected_wifi == wifi_sweeps_count) and
            (expected_imu is None or expected_imu == imu_count) and
            (expected_photo is None or expected_photo == photo_count) and
            (expected_ann is None or expected_ann == annotation_count)
        )

        # 6. Deep Multimodal Quality Analysis
        wifi_quality = self._analyze_wifi_quality(wifi_rows, warnings)
        sensor_quality = self._analyze_sensor_quality(imu_rows, warnings)
        photo_quality = self._analyze_photo_quality(photo_rows, images_dir, errors, warnings)
        annotation_quality = self._analyze_annotation_quality(annotation_rows, warnings)

        modalities = ModalitiesQualitySummary(
            wifi=wifi_quality,
            sensors=sensor_quality,
            photos=photo_quality,
            annotations=annotation_quality
        )

        # 7. Timestamp Synchronization & Cross-Modal Analysis
        timestamp_quality, temporal_overlap = self._analyze_timestamps_and_overlap(
            wifi_rows, imu_rows, photo_rows, annotation_rows, warnings
        )

        # 8. Determine overall validation status
        if errors:
            status = ValidationStatus.FAILED if integrity_result.checksums_valid else ValidationStatus.CORRUPTED
        elif warnings:
            status = ValidationStatus.WARNINGS
        else:
            status = ValidationStatus.PASSED

        return DatasetQualityReport(
            dataset_id=dataset_id,
            session_id=session_id,
            session_name=session_name,
            validation_status=status,
            integrity=integrity_result,
            modalities=modalities,
            timestamp_quality=timestamp_quality,
            cross_modal_overlap=temporal_overlap,
            warnings=warnings,
            errors=errors
        )

    def _verify_manifest_checksums(self, extracted_dir: Path, manifest_data: Dict[str, Any]) -> IntegrityCheckResult:
        """Verifies SHA-256 and byte sizes of all files declared in manifest.json."""
        manifest_files = manifest_data.get("files", [])
        if not manifest_files or not isinstance(manifest_files, list):
            return IntegrityCheckResult(
                manifest_valid=False,
                checksums_valid=False,
                record_counts_valid=False,
                files_verified_count=0,
                files_failed_count=1,
                mismatched_files=["manifest.json files array is empty or missing"]
            )

        verified = 0
        failed = 0
        mismatched: List[str] = []

        for entry in manifest_files:
            fn = entry.get("filename")
            expected_sha = (entry.get("sha256") or "").lower()
            expected_size = entry.get("sizeBytes")

            if not fn:
                failed += 1
                mismatched.append("Unnamed manifest entry")
                continue

            target_file = (extracted_dir / fn).resolve()
            if not target_file.is_relative_to(extracted_dir) or not target_file.is_file():
                failed += 1
                mismatched.append(f"{fn} (missing on disk)")
                continue

            actual_size = target_file.stat().st_size
            if expected_size is not None and actual_size != expected_size:
                failed += 1
                mismatched.append(f"{fn} (size mismatch: expected {expected_size}, got {actual_size})")
                continue

            actual_sha = calculate_sha256(target_file)
            if actual_sha != expected_sha:
                failed += 1
                mismatched.append(f"{fn} (hash mismatch: expected {expected_sha}, got {actual_sha})")
                continue

            verified += 1

        return IntegrityCheckResult(
            manifest_valid=True,
            checksums_valid=(failed == 0),
            record_counts_valid=True,  # Checked subsequently
            files_verified_count=verified,
            files_failed_count=failed,
            mismatched_files=mismatched
        )

    def _read_and_validate_csv(
        self,
        file_path: Path,
        expected_header: List[str],
        expected_session_id: str,
        modality_code: str
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """Reads CSV file, validates header schema, session ID referential integrity, and basic fields."""
        rows: List[Dict[str, str]] = []
        errors: List[str] = []

        if not file_path.exists():
            return rows, errors

        try:
            with open(file_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    # Empty file (even no header)
                    errors.append(f"{file_path.name} is completely empty (missing CSV header).")
                    return rows, errors

                # Header comparison
                stripped_header = [col.strip() for col in header]
                if stripped_header != expected_header:
                    missing = set(expected_header) - set(stripped_header)
                    extra = set(stripped_header) - set(expected_header)
                    errors.append(
                        f"{file_path.name} header mismatch. Missing: {sorted(missing)}, Unexpected: {sorted(extra)}"
                    )

                f.seek(0)
                dict_reader = csv.DictReader(f)
                for line_num, row in enumerate(dict_reader, start=2):
                    rows.append(row)
                    row_session_id = row.get("session_id", "").strip()
                    if expected_session_id != "UNKNOWN" and row_session_id and row_session_id != expected_session_id:
                        errors.append(
                            f"{file_path.name}:{line_num} session_id mismatch: expected '{expected_session_id}', got '{row_session_id}'"
                        )

        except Exception as e:
            errors.append(f"Failed to read CSV {file_path.name}: {e}")

        return rows, errors

    def _analyze_wifi_quality(self, rows: List[Dict[str, str]], warnings: List[str]) -> WifiQualityMetrics:
        """Computes Wi-Fi quality and coverage statistics."""
        if not rows:
            return WifiQualityMetrics()

        sweeps: Dict[str, List[Dict[str, str]]] = {}
        unique_bssids: Set[str] = set()
        rssi_values: List[int] = []
        invalid_rssi_count = 0
        fresh_count = 0
        cached_count = 0
        duplicate_aps = 0

        for r in rows:
            obs_id = r.get("observation_id", "")
            sweeps.setdefault(obs_id, []).append(r)

            bssid = r.get("bssid", "").strip().lower()
            if bssid:
                unique_bssids.add(bssid)

            rssi_str = r.get("rssi_dbm", "").strip()
            if rssi_str:
                try:
                    rssi_val = int(float(rssi_str))
                    # Wi-Fi RSSI should generally be between -110 and 0 dBm
                    if -115 <= rssi_val <= 0:
                        rssi_values.append(rssi_val)
                    else:
                        invalid_rssi_count += 1
                except ValueError:
                    invalid_rssi_count += 1

            is_fresh = r.get("is_fresh", "").lower() in ("true", "1")
            if is_fresh:
                fresh_count += 1
            else:
                cached_count += 1

        total_scans = len(sweeps)
        total_ap_records = len(rows)
        avg_aps = total_ap_records / total_scans if total_scans > 0 else 0.0

        # Check for intra-sweep duplicate BSSIDs
        for obs_id, sweep_rows in sweeps.items():
            sweep_bssids: Set[str] = set()
            for row in sweep_rows:
                b = row.get("bssid", "").strip().lower()
                if b and b in sweep_bssids:
                    duplicate_aps += 1
                elif b:
                    sweep_bssids.add(b)

        if total_scans > 0 and avg_aps < 2.0:
            warnings.append(f"Low Wi-Fi density: average visible APs per sweep is {avg_aps:.1f} (< 2.0).")

        return WifiQualityMetrics(
            total_scans=total_scans,
            total_ap_records=total_ap_records,
            unique_bssids_count=len(unique_bssids),
            avg_aps_per_scan=round(avg_aps, 2),
            min_rssi_dbm=min(rssi_values) if rssi_values else None,
            max_rssi_dbm=max(rssi_values) if rssi_values else None,
            invalid_rssi_count=invalid_rssi_count,
            fresh_scan_count=fresh_count,
            cached_scan_count=cached_count,
            duplicate_ap_count=duplicate_aps,
            has_timestamps=all("normalized_epoch_ms" in r and r["normalized_epoch_ms"] for r in rows)
        )

    def _analyze_sensor_quality(self, rows: List[Dict[str, str]], warnings: List[str]) -> SensorQualityMetrics:
        """Computes IMU sampling intervals, frequencies, and missing value checks."""
        if not rows:
            return SensorQualityMetrics()

        sensor_buckets: Dict[str, List[int]] = {}
        missing_vals = 0
        invalid_numerics = 0
        real_count = 0
        mock_count = 0

        for r in rows:
            stype = r.get("sensor_type", "UNKNOWN").upper()
            ts_str = r.get("normalized_epoch_ms") or r.get("original_timestamp_ns")
            if ts_str:
                try:
                    ts_val = int(float(ts_str))
                    # If nanoseconds, scale to ms for interval calculations
                    if ts_val > 1_000_000_000_000_000:
                        ts_val = ts_val // 1_000_000
                    sensor_buckets.setdefault(stype, []).append(ts_val)
                except ValueError:
                    pass

            # Axes check
            for axis in ("x", "y", "z"):
                val_str = r.get(axis, "").strip()
                if not val_str:
                    missing_vals += 1
                else:
                    try:
                        v = float(val_str)
                        if np.isnan(v) or np.isinf(v):
                            invalid_numerics += 1
                    except ValueError:
                        invalid_numerics += 1

            if r.get("is_mock", "").lower() in ("true", "1"):
                mock_count += 1
            else:
                real_count += 1

        avg_freqs: Dict[str, float] = {}
        min_intervals: Dict[str, float] = {}
        max_intervals: Dict[str, float] = {}
        gap_count = 0

        for stype, timestamps in sensor_buckets.items():
            if len(timestamps) > 1:
                sorted_ts = sorted(timestamps)
                deltas = np.diff(sorted_ts)
                positive_deltas = deltas[deltas > 0]
                if len(positive_deltas) > 0:
                    mean_delta = float(np.mean(positive_deltas))
                    min_intervals[stype] = float(np.min(positive_deltas))
                    max_intervals[stype] = float(np.max(positive_deltas))
                    avg_freqs[stype] = round(1000.0 / mean_delta, 1) if mean_delta > 0 else 0.0

                    # Gaps exceeding 5x mean interval
                    gaps = np.sum(positive_deltas > (5 * mean_delta))
                    gap_count += int(gaps)
                else:
                    avg_freqs[stype] = 0.0
                    min_intervals[stype] = 0.0
                    max_intervals[stype] = 0.0

        accel_cnt = len(sensor_buckets.get("ACCELEROMETER", []))
        gyro_cnt = len(sensor_buckets.get("GYROSCOPE", []))
        mag_cnt = len(sensor_buckets.get("MAGNETIC_FIELD", []))
        rot_cnt = len(sensor_buckets.get("ROTATION_VECTOR", []))
        other_cnt = len(rows) - (accel_cnt + gyro_cnt + mag_cnt + rot_cnt)

        if gap_count > 10:
            warnings.append(f"Frequent sensor sampling gaps detected ({gap_count} gaps > 5x mean interval).")

        return SensorQualityMetrics(
            total_records=len(rows),
            accelerometer_count=accel_cnt,
            gyroscope_count=gyro_cnt,
            magnetometer_count=mag_cnt,
            rotation_vector_count=rot_cnt,
            other_sensor_count=other_cnt,
            avg_sampling_freq_hz=avg_freqs,
            min_interval_ms=min_intervals,
            max_interval_ms=max_intervals,
            missing_values_count=missing_vals,
            invalid_numeric_count=invalid_numerics,
            timestamp_gap_count=gap_count,
            real_record_count=real_count,
            mock_record_count=mock_count
        )

    def _analyze_photo_quality(
        self,
        rows: List[Dict[str, str]],
        images_dir: Path,
        errors: List[str],
        warnings: List[str]
    ) -> PhotoQualityMetrics:
        """Validates referenced image files, formats, and privacy sanitization provenance."""
        if not rows:
            return PhotoQualityMetrics(
                total_photo_records=0,
                image_files_present=0,
                metadata_sanitization_verified=True,
                visual_privacy_audited=False,
                visual_privacy_status="NOT_APPLICABLE"
            )

        images_present = 0
        missing_images: List[str] = []
        formats: Set[str] = set()
        sanitized_cnt = 0
        faces_detected_cnt = 0
        face_blur_cnt = 0
        image_hashes: Dict[str, str] = {}
        duplicate_hashes = 0

        for r in rows:
            fn = r.get("filename", "").strip()
            photo_id = r.get("photo_id", "")
            img_path = images_dir / fn

            if not fn or not img_path.is_file():
                missing_images.append(fn or f"photo_id:{photo_id}")
            else:
                images_present += 1
                try:
                    with Image.open(img_path) as img:
                        formats.add(img.format or "UNKNOWN")
                    # Calculate content hash
                    h = calculate_sha256(img_path)
                    if h in image_hashes:
                        duplicate_hashes += 1
                    else:
                        image_hashes[h] = fn
                except Exception as e:
                    warnings.append(f"Failed to inspect image header of '{fn}': {e}")

            # Check privacy sanitization metadata
            is_sanitized = r.get("is_exif_sanitized", "").lower() in ("true", "1")
            if is_sanitized:
                sanitized_cnt += 1
            else:
                errors.append(f"Photo '{photo_id}' ({fn}) violates privacy policy: is_exif_sanitized is FALSE.")

            try:
                faces = int(r.get("faces_detected_count", "0"))
                faces_detected_cnt += faces
            except ValueError:
                pass

            if r.get("face_blur_applied", "").lower() in ("true", "1"):
                face_blur_cnt += 1

        if missing_images:
            errors.append(f"Dataset references {len(missing_images)} photo(s) missing from images/ directory.")

        all_metadata_sanitized = (sanitized_cnt == len(rows))

        return PhotoQualityMetrics(
            total_photo_records=len(rows),
            image_files_present=images_present,
            missing_image_files=missing_images,
            formats_detected=sorted(formats),
            exif_sanitized_count=sanitized_cnt,
            faces_detected_count=faces_detected_cnt,
            face_blur_applied_count=face_blur_cnt,
            duplicate_hashes_count=duplicate_hashes,
            metadata_sanitization_verified=all_metadata_sanitized,
            visual_privacy_audited=False,
            visual_privacy_status="PENDING_AUDIT"
        )

    def _analyze_annotation_quality(
        self,
        rows: List[Dict[str, str]],
        warnings: List[str]
    ) -> AnnotationQualityMetrics:
        """Analyzes manual ground-truth annotations and architectural labels."""
        if not rows:
            return AnnotationQualityMetrics()

        type_counts: Dict[str, int] = {}
        missing_desc = 0
        bldg_cnt = 0
        flr_cnt = 0
        rm_cnt = 0

        for r in rows:
            t = r.get("type_code", r.get("type_name", "UNKNOWN")).strip().upper()
            type_counts[t] = type_counts.get(t, 0) + 1

            if not r.get("description", "").strip():
                missing_desc += 1

            if r.get("building", "").strip():
                bldg_cnt += 1
            if r.get("floor", "").strip():
                flr_cnt += 1
            if r.get("room", "").strip():
                rm_cnt += 1

        if missing_desc > 0:
            warnings.append(f"{missing_desc} annotation(s) have empty descriptions.")

        return AnnotationQualityMetrics(
            total_annotations=len(rows),
            type_breakdown=type_counts,
            missing_description_count=missing_desc,
            building_labels_present=bldg_cnt,
            floor_labels_present=flr_cnt,
            room_labels_present=rm_cnt
        )

    def _analyze_timestamps_and_overlap(
        self,
        wifi_rows: List[Dict[str, str]],
        imu_rows: List[Dict[str, str]],
        photo_rows: List[Dict[str, str]],
        annotation_rows: List[Dict[str, str]],
        warnings: List[str]
    ) -> Tuple[TimestampQualityMetrics, CrossModalTemporalOverlap]:
        """Performs cross-stream timestamp quality audit and concurrent temporal overlap calculations."""
        time_sources: Set[str] = set()
        sync_methods: Set[str] = set()
        uncertainties: List[float] = []
        out_of_order_count = 0
        large_gaps = 0

        modality_ranges: Dict[str, Dict[str, Optional[float]]] = {}
        all_timestamps_ms: List[float] = []

        modality_streams = [
            ("wifi", wifi_rows),
            ("sensors", imu_rows),
            ("photos", photo_rows),
            ("annotations", annotation_rows)
        ]

        for mod_name, rows in modality_streams:
            if not rows:
                modality_ranges[mod_name] = {"start_ms": None, "end_ms": None}
                continue

            stream_ts: List[float] = []
            for r in rows:
                ts_src = r.get("time_source")
                if ts_src:
                    time_sources.add(ts_src.strip())

                sync = r.get("sync_method")
                if sync:
                    sync_methods.add(sync.strip())

                unc = r.get("uncertainty_ms")
                if unc:
                    try:
                        uncertainties.append(float(unc))
                    except ValueError:
                        pass

                norm_str = r.get("normalized_epoch_ms")
                if norm_str:
                    try:
                        val = float(norm_str)
                        stream_ts.append(val)
                        all_timestamps_ms.append(val)
                    except ValueError:
                        pass

            if stream_ts:
                # Check stream monotonicity
                for i in range(1, len(stream_ts)):
                    if stream_ts[i] < stream_ts[i - 1]:
                        out_of_order_count += 1
                    elif (stream_ts[i] - stream_ts[i - 1]) > 10_000:  # > 10s gap
                        large_gaps += 1

                modality_ranges[mod_name] = {
                    "start_ms": min(stream_ts),
                    "end_ms": max(stream_ts)
                }
            else:
                modality_ranges[mod_name] = {"start_ms": None, "end_ms": None}

        total_duration = 0.0
        if all_timestamps_ms:
            total_duration = round((max(all_timestamps_ms) - min(all_timestamps_ms)) / 1000.0, 2)

        # Pairwise overlap calculation
        pairwise_overlap: Dict[str, float] = {}
        present_modalities = [m for m, r in modality_ranges.items() if r["start_ms"] is not None]

        for i in range(len(present_modalities)):
            for j in range(i + 1, len(present_modalities)):
                m1, m2 = present_modalities[i], present_modalities[j]
                r1, r2 = modality_ranges[m1], modality_ranges[m2]
                overlap_start = max(r1["start_ms"], r2["start_ms"])  # type: ignore
                overlap_end = min(r1["end_ms"], r2["end_ms"])        # type: ignore
                overlap_sec = max(0.0, (overlap_end - overlap_start) / 1000.0)
                pairwise_overlap[f"{m1}_x_{m2}"] = round(overlap_sec, 2)

        # All-modality concurrent overlap
        all_mod_overlap = 0.0
        if len(present_modalities) > 1:
            starts = [modality_ranges[m]["start_ms"] for m in present_modalities]
            ends = [modality_ranges[m]["end_ms"] for m in present_modalities]
            max_start = max(starts)  # type: ignore
            min_end = min(ends)      # type: ignore
            all_mod_overlap = max(0.0, (min_end - max_start) / 1000.0)

        # Summary timeline text
        active_summaries = []
        for m in present_modalities:
            span = round((modality_ranges[m]["end_ms"] - modality_ranges[m]["start_ms"]) / 1000.0, 1)  # type: ignore
            active_summaries.append(f"{m.capitalize()} ({span}s)")
        summary_text = (
            f"Active streams: {', '.join(active_summaries)}. "
            f"Total span: {total_duration}s. "
            f"Concurrent multimodal overlap: {round(all_mod_overlap, 1)}s. "
            "Timestamps normalized via producer ClockAnchor (producer-reported)."
        )

        timestamp_metrics = TimestampQualityMetrics(
            original_timestamps_present=True,
            normalized_timestamps_present=len(all_timestamps_ms) > 0,
            time_sources_used=sorted(time_sources),
            sync_methods_used=sorted(sync_methods),
            min_reported_uncertainty_ms=min(uncertainties) if uncertainties else None,
            max_reported_uncertainty_ms=max(uncertainties) if uncertainties else None,
            avg_reported_uncertainty_ms=round(float(np.mean(uncertainties)), 2) if uncertainties else None,
            uncertainty_provenance="producer_reported",
            out_of_order_count=out_of_order_count,
            large_gap_count=large_gaps,
            timeline_duration_seconds=total_duration
        )

        temporal_overlap = CrossModalTemporalOverlap(
            modality_time_ranges=modality_ranges,
            pairwise_overlap_seconds=pairwise_overlap,
            all_modalities_overlap_seconds=round(all_mod_overlap, 2),
            timeline_summary=summary_text
        )

        return timestamp_metrics, temporal_overlap
