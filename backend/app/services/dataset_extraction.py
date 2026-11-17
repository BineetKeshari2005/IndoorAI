"""Secure ZIP extraction and archive validation service for multimodal datasets."""

from __future__ import annotations

import io
import os
import shutil
import stat
import zipfile
from pathlib import Path
from typing import List, Optional, Set, Tuple

from backend.app.core.dataset_config import DatasetConfig, default_dataset_config
from src.core.logging import setup_logger

logger = setup_logger("DatasetExtractor")

# Standard ZIP signature constants
ZIP_MAGIC_SIGNATURES: Tuple[bytes, ...] = (
    b"PK\x03\x04",  # Standard entry header
    b"PK\x05\x06",  # Empty archive header
    b"PK\x07\x08",  # Spanned archive header
)


class SecurityValidationError(Exception):
    """Raised when an archive violates security invariants (traversal, ZIP bomb, etc.)."""
    pass


class DatasetExtractor:
    """Handles secure inspection, validation, and extraction of uploaded survey ZIP archives."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or default_dataset_config

    def validate_zip_stream(self, file_stream: io.BufferedIOBase, reported_size: Optional[int] = None) -> None:
        """Inspects file signature and size before opening archive."""
        if reported_size is not None and reported_size > self.config.max_upload_size_bytes:
            raise SecurityValidationError(
                f"Upload size {reported_size} bytes exceeds maximum allowed {self.config.max_upload_size_bytes} bytes."
            )

        # Check signature magic bytes
        current_pos = file_stream.tell()
        magic = file_stream.read(4)
        file_stream.seek(current_pos)

        if not any(magic.startswith(sig) for sig in ZIP_MAGIC_SIGNATURES):
            raise SecurityValidationError(
                f"Invalid file signature {magic!r}. Uploaded file is not a valid ZIP archive."
            )

    def inspect_and_validate_archive(self, zip_file: zipfile.ZipFile) -> Tuple[int, List[str]]:
        """Performs deep security checks on archive table-of-contents before extraction.

        Returns:
            Tuple of (total_uncompressed_bytes, list_of_entry_names)
        """
        infolist = zip_file.infolist()

        # 1. Archive entry count check
        entry_count = len(infolist)
        if entry_count > self.config.max_archive_entries:
            raise SecurityValidationError(
                f"Archive contains {entry_count} entries, exceeding maximum of {self.config.max_archive_entries}."
            )

        total_uncompressed_bytes = 0
        seen_names: Set[str] = set()
        validated_names: List[str] = []

        for info in infolist:
            filename = info.filename

            # 2. Duplicate entry check
            normalized_name = filename.strip("/").strip("\\")
            if normalized_name in seen_names:
                raise SecurityValidationError(
                    f"Duplicate archive path detected: '{filename}'. Archive tampering suspected."
                )
            seen_names.add(normalized_name)

            # 3. Path Traversal & Symlink checks
            if filename.startswith("/") or filename.startswith("\\"):
                raise SecurityValidationError(
                    f"Path traversal detected: absolute path '{filename}' is prohibited."
                )
            if ".." in filename.replace("\\", "/").split("/"):
                raise SecurityValidationError(
                    f"Path traversal detected: directory climb '..' in '{filename}' is prohibited."
                )
            if ":" in filename:
                raise SecurityValidationError(
                    f"Path traversal detected: drive specifier or invalid colon in '{filename}'."
                )

            # Check for symlinks
            mode = info.external_attr >> 16
            if mode != 0 and stat.S_ISLNK(mode):
                raise SecurityValidationError(
                    f"Symbolic link detected in archive entry '{filename}'. Symlinks are prohibited."
                )

            # 4. ZIP Bomb / Decompression Ratio checks
            file_size = info.file_size
            compress_size = info.compress_size
            if file_size < 0:
                raise SecurityValidationError(f"Negative file size reported for '{filename}'.")

            total_uncompressed_bytes += file_size
            if total_uncompressed_bytes > self.config.max_extracted_size_bytes:
                raise SecurityValidationError(
                    f"Total uncompressed size exceeds limit of {self.config.max_extracted_size_bytes} bytes."
                )

            if compress_size > 0:
                ratio = file_size / compress_size
                if ratio > self.config.max_compression_ratio:
                    raise SecurityValidationError(
                        f"Excessive compression ratio ({ratio:.1f}:1) for entry '{filename}'. "
                        f"Potential ZIP bomb rejected."
                    )

            # 5. File Extension Whitelist checks
            if not info.is_dir() and not filename.endswith("/"):
                ext = Path(filename).suffix.lower()
                if ext not in self.config.allowed_extensions:
                    raise SecurityValidationError(
                        f"Forbidden file extension '{ext}' in '{filename}'. "
                        f"Allowed extensions: {sorted(self.config.allowed_extensions)}"
                    )

            validated_names.append(filename)

        return total_uncompressed_bytes, validated_names

    def extract_dataset(
        self,
        source_zip: Path | io.BytesIO,
        dest_dir: Path,
        reported_size: Optional[int] = None
    ) -> Path:
        """Safely validates and extracts the archive into a controlled target directory.

        Args:
            source_zip: Path to the ZIP file or BytesIO buffer.
            dest_dir: Directory where contents will be extracted.
            reported_size: Optional size in bytes for early validation.

        Returns:
            Resolved Path to destination directory.

        Raises:
            SecurityValidationError: If any security invariant is violated.
            ValueError / zipfile.BadZipFile: If archive is corrupted.
        """
        dest_dir = dest_dir.resolve()

        # Pre-flight check on stream if applicable
        if isinstance(source_zip, io.BytesIO):
            self.validate_zip_stream(source_zip, reported_size)
        elif isinstance(source_zip, Path):
            with open(source_zip, "rb") as f:
                self.validate_zip_stream(f, reported_size or source_zip.stat().st_size)

        try:
            with zipfile.ZipFile(source_zip, "r") as zf:
                # Validate table of contents before extracting any file
                _, entry_names = self.inspect_and_validate_archive(zf)

                # Prepare destination directory
                dest_dir.mkdir(parents=True, exist_ok=True)
                total_written = 0

                for info in zf.infolist():
                    target_path = (dest_dir / info.filename).resolve()

                    # Final path containment check
                    if not target_path.is_relative_to(dest_dir):
                        raise SecurityValidationError(
                            f"Extraction path escape attempt: '{target_path}' outside '{dest_dir}'"
                        )

                    if info.is_dir() or info.filename.endswith("/"):
                        target_path.mkdir(parents=True, exist_ok=True)
                        continue

                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    # Extract file stream safely with chunk size and cumulative byte counter
                    with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                        while True:
                            chunk = src.read(65536)  # 64 KB chunks
                            if not chunk:
                                break
                            total_written += len(chunk)
                            if total_written > self.config.max_extracted_size_bytes:
                                raise SecurityValidationError(
                                    f"Decompressed stream exceeded {self.config.max_extracted_size_bytes} bytes."
                                )
                            dst.write(chunk)

            logger.info("Successfully extracted %d entries into %s", len(entry_names), dest_dir)
            return dest_dir

        except Exception as e:
            # Atomic cleanup on any extraction or validation failure
            logger.warning("Extraction aborted: %s. Cleaning up %s", e, dest_dir)
            if dest_dir.exists():
                shutil.rmtree(dest_dir, ignore_errors=True)
            raise
