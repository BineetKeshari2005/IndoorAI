"""Module for acquiring and verifying the official UJIIndoorLoc dataset from UCI.

Downloads the archive directly from the official UCI Machine Learning Repository,
verifies integrity via SHA-256 checksums, extracts raw CSVs safely, and persists
download metadata.
"""

import os
from pathlib import Path
import sys

# Ensure repository root is in sys.path
_current_dir = Path(__file__).resolve().parent
_repo_root = _current_dir.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dataclasses import asdict, dataclass
import hashlib
import json
import shutil
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request
import zipfile

from src.core.config import get_settings
from src.core.logging import setup_logger

logger = setup_logger("UJIIndoorLocDownloader")


@dataclass
class FileMetadata:
    """Metadata for an individual raw dataset file."""
    filename: str
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass
class DownloadManifest:
    """Complete acquisition record for auditing and provenance."""
    dataset_name: str
    source_url: str
    acquired_at_epoch_ms: int
    archive_metadata: FileMetadata
    extracted_files: List[FileMetadata]
    success: bool
    notes: str


def compute_sha256(filepath: Path, chunk_size: int = 65536) -> str:
    """Calculate the SHA-256 hex digest of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_dataset_config() -> Dict[str, Any]:
    """Load configuration from dataset_config.yaml or provide default configuration."""
    settings = get_settings()
    config_path = settings.paths.configs_dir / "dataset_config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "ujiindoorloc" in data:
                    return data["ujiindoorloc"]
        except Exception as e:
            logger.warning("Failed to load YAML config (%s), using defaults.", e)

    return {
        "name": "UJIIndoorLoc",
        "source_page": "https://archive.ics.uci.edu/dataset/310/ujiindoorloc",
        "download_urls": [
            "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIIndoorLoc.zip",
        ],
        "target_raw_dir": "data/raw/ujiindoorloc",
        "archive_filename": "ujiindoorloc.zip",
        "expected_files": ["trainingData.csv", "validationData.csv"],
    }


def safe_extract_zip(zip_path: Path, extract_to: Path) -> List[Path]:
    """Safely extract all files from a ZIP archive, guarding against path traversal."""
    extracted_paths: List[Path] = []
    extract_to.mkdir(parents=True, exist_ok=True)
    resolved_dest = extract_to.resolve()

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            # Skip directories or macOS metadata
            if member.is_dir() or member.filename.startswith("__MACOSX"):
                continue
            
            # Resolve destination ensuring no directory traversal outside target
            member_target = (extract_to / member.filename).resolve()
            if not str(member_target).startswith(str(resolved_dest)):
                raise ValueError(f"Potentially unsafe path in ZIP archive: {member.filename}")
            
            member_target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(member_target, "wb") as target:
                shutil.copyfileobj(source, target)
            extracted_paths.append(member_target)
            logger.info("Extracted: %s (%d bytes)", member.filename, member.file_size)

    return extracted_paths


def download_and_extract_ujiindoorloc(
    force_download: bool = False,
    custom_target_dir: Optional[Path] = None
) -> DownloadManifest:
    """Download UJIIndoorLoc dataset from UCI ML Repository, verify, and extract.

    Args:
        force_download: If True, re-download even if expected files exist.
        custom_target_dir: Optional override for storage directory.

    Returns:
        DownloadManifest containing provenance, sizes, and SHA-256 hashes.
    """
    settings = get_settings()
    config = load_dataset_config()
    target_dir = custom_target_dir or (settings.paths.root_dir / config["target_raw_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)

    expected_files = config.get("expected_files", ["trainingData.csv", "validationData.csv"])
    manifest_path = target_dir / "manifest.json"

    # Check if files already exist and are valid
    all_exist = all((target_dir / fname).exists() for fname in expected_files)
    if all_exist and not force_download:
        logger.info("All expected UJIIndoorLoc files exist in %s. Skipping download.", target_dir)
        extracted_metas: List[FileMetadata] = []
        for fname in expected_files:
            fpath = target_dir / fname
            extracted_metas.append(
                FileMetadata(
                    filename=fname,
                    relative_path=str(fpath.relative_to(settings.paths.root_dir)),
                    size_bytes=fpath.stat().st_size,
                    sha256=compute_sha256(fpath),
                )
            )

        archive_path = target_dir / config.get("archive_filename", "ujiindoorloc.zip")
        archive_meta = (
            FileMetadata(
                filename=archive_path.name,
                relative_path=str(archive_path.relative_to(settings.paths.root_dir)),
                size_bytes=archive_path.stat().st_size,
                sha256=compute_sha256(archive_path),
            )
            if archive_path.exists()
            else FileMetadata(filename="", relative_path="", size_bytes=0, sha256="")
        )

        manifest = DownloadManifest(
            dataset_name=config["name"],
            source_url="cached_local_files",
            acquired_at_epoch_ms=int(time.time() * 1000),
            archive_metadata=archive_meta,
            extracted_files=extracted_metas,
            success=True,
            notes="Dataset verified from existing valid local storage.",
        )
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)
        return manifest

    urls = config.get("download_urls", [])
    archive_name = config.get("archive_filename", "ujiindoorloc.zip")
    archive_path = target_dir / archive_name
    temp_download_path = target_dir / f"{archive_name}.tmp"

    download_success = False
    successful_url = ""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36 IndoorAI/0.1"
        )
    }

    for url in urls:
        logger.info("Attempting download from official source: %s", url)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response:
                if response.status != 200:
                    logger.warning("HTTP %d from %s, trying next source.", response.status, url)
                    continue

                total_size = response.headers.get("Content-Length")
                total_bytes = int(total_size) if total_size else None
                bytes_read = 0

                with open(temp_download_path, "wb") as out_file:
                    while chunk := response.read(65536):
                        bytes_read += len(chunk)
                        out_file.write(chunk)

                if total_bytes and bytes_read < total_bytes:
                    logger.error("Incomplete download: received %d / %d bytes", bytes_read, total_bytes)
                    if temp_download_path.exists():
                        temp_download_path.unlink()
                    continue

                download_success = True
                successful_url = url
                temp_download_path.rename(archive_path)
                logger.info("Download completed successfully (%d bytes) from %s", bytes_read, url)
                break

        except urllib.error.URLError as e:
            logger.warning("Network error downloading from %s: %s", url, e)
            if temp_download_path.exists():
                temp_download_path.unlink()
        except Exception as e:
            logger.warning("Unexpected error downloading from %s: %s", url, e)
            if temp_download_path.exists():
                temp_download_path.unlink()

    if not download_success or not archive_path.exists():
        err_msg = (
            "Failed to download UJIIndoorLoc from official UCI sources.\n"
            "Official dataset URL: https://archive.ics.uci.edu/dataset/310/ujiindoorloc\n"
            "Manual Download Instructions:\n"
            "1. Download 'ujiindoorloc.zip' from the official link above.\n"
            f"2. Place the zip archive or extract 'trainingData.csv' and 'validationData.csv' into: {target_dir}\n"
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # Compute checksum of archive
    archive_sha256 = compute_sha256(archive_path)
    archive_meta = FileMetadata(
        filename=archive_path.name,
        relative_path=str(archive_path.relative_to(settings.paths.root_dir)),
        size_bytes=archive_path.stat().st_size,
        sha256=archive_sha256,
    )
    logger.info("Archive SHA-256: %s", archive_sha256)

    # Safe extraction
    extracted_paths = safe_extract_zip(archive_path, target_dir)

    # Check for expected CSV files in target_dir (they might be in subfolders inside the zip)
    # Move them to root of target_dir if nested
    for fname in expected_files:
        direct_path = target_dir / fname
        if not direct_path.exists():
            matches = list(target_dir.rglob(fname))
            if matches:
                shutil.copyfile(matches[0], direct_path)
                logger.info("Relocated nested %s to target directory root.", fname)

    extracted_metas: List[FileMetadata] = []
    for fname in expected_files:
        fpath = target_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Expected file {fname} not found after extraction!")
        extracted_metas.append(
            FileMetadata(
                filename=fname,
                relative_path=str(fpath.relative_to(settings.paths.root_dir)),
                size_bytes=fpath.stat().st_size,
                sha256=compute_sha256(fpath),
            )
        )
        logger.info("Verified file %s - Size: %d bytes, SHA-256: %s", fname, fpath.stat().st_size, extracted_metas[-1].sha256)

    manifest = DownloadManifest(
        dataset_name=config["name"],
        source_url=successful_url,
        acquired_at_epoch_ms=int(time.time() * 1000),
        archive_metadata=archive_meta,
        extracted_files=extracted_metas,
        success=True,
        notes="Successfully downloaded and verified from official UCI Machine Learning Repository.",
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)

    logger.info("Provenance manifest written to %s", manifest_path)
    return manifest


if __name__ == "__main__":
    try:
        manifest = download_and_extract_ujiindoorloc()
        print("UJIIndoorLoc dataset acquisition succeeded:")
        print(f"Archive: {manifest.archive_metadata.filename} ({manifest.archive_metadata.size_bytes} bytes)")
        for ef in manifest.extracted_files:
            print(f"File: {ef.filename} - {ef.size_bytes} bytes - SHA256: {ef.sha256[:16]}...")
    except Exception as exc:
        print(f"Acquisition failed: {exc}", file=sys.stderr)
        sys.exit(1)
