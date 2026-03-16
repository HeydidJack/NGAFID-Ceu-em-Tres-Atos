#!/usr/bin/env python3
"""
Dataset Auto-Setup Script for NGAFID-Ceu-em-Tres-Atos

Checks if NGAFID-Ceu-em-Tres-Atos/Datasets/ contains valid 2days/ and one_parq/ subdirectories.
If not, downloads 2days.tar.gz and all_flight.tar.gz from Zenodo (DOI: 10.5281/zenodo.6624956),
extracts them to respective folders, and cleans up archives.

Usage: python setup_dataset.py
"""

import os
import sys
import tarfile
import hashlib
from pathlib import Path
from typing import Tuple, Optional
import requests
from tqdm import tqdm

# Configuration constants - centralized for maintainability
ZENODO_RECORD = "6624956"
BASE_URL = f"https://zenodo.org/record/{ZENODO_RECORD}/files"
CURRENT_DIR = os.getcwd()
MAIN_DIR = os.path.dirname(CURRENT_DIR)
DATASET_DIR = Path(MAIN_DIR + "/Datasets")
ARCHIVES = {
    "2days.tar.gz": {
        "subdir": "2days",
        "desc": "Subset data (2 days, ~1.1GB)"
    },
    "all_flight.tar.gz": {
        "subdir": "all_flights",
        "desc": "Full flight data (~4.3GB)"
    }
}

def check_dataset_integrity() -> Tuple[bool, str]:
    """
    Verify if dataset directory structure satisfies all constraints.
    Returns: (is_valid, status_message)
    """
    if not DATASET_DIR.exists():
        return False, f"Dataset root missing: {DATASET_DIR}"

    print(DATASET_DIR)

    for archive_name, config in ARCHIVES.items():
        subdir = DATASET_DIR / config["subdir"]

        # Check directory existence
        if not subdir.exists():
            return False, f"Subdirectory missing: {subdir}"

        # Check non-empty condition (at least one file recursively)
        # Using rglob('*') to catch any file, including nested ones
        if not any(subdir.rglob('*')):
            return False, f"Subdirectory empty: {subdir}"

    return True, "Dataset structure validated"


def download_with_resume(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """
    Robust download with progress bar and resume capability.
    Handles large binary files without loading into memory.
    """
    # Header for resume (if dest exists)
    headers = {}
    if dest.exists():
        headers['Range'] = f'bytes={dest.stat().st_size}-'

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        # Determine total size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        if dest.exists():
            total_size += dest.stat().st_size

        mode = 'ab' if dest.exists() else 'wb'

        with open(dest, mode) as f, tqdm(
                desc=dest.name,
                initial=dest.stat().st_size if dest.exists() else 0,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as pbar:

            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except Exception as e:
        print(f"[ERROR] Download failed for {url}: {e}")
        return False


def extract_and_cleanup(archive_path: Path, extract_to: Path) -> bool:
    """
    Extract tar.gz archive to target directory, then remove archive.
    Uses streaming extraction to handle 4GB+ files without memory explosion.
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, 'r:gz') as tar:
            # Security check: prevent path traversal attacks in tar
            for member in tar.getmembers():
                if os.path.isabs(member.name) or ".." in member.name:
                    raise ValueError(f"Malicious tar member detected: {member.name}")

            # Extract with progress (tarfile doesn't support tqdm directly, so we iterate)
            members = tar.getmembers()
            for member in tqdm(members, desc=f"Extracting {archive_path.name}"):
                tar.extract(member, path=extract_to)

        # Cleanup to save disk space (9GB is non-trivial)
        archive_path.unlink()
        print(f"[INFO] Removed archive: {archive_path.name}")
        return True

    except Exception as e:
        print(f"[ERROR] Extraction failed for {archive_path}: {e}")
        return False


def main():
    """Main execution flow with early exit optimization."""
    print("=== NGAFID Dataset Setup ===")

    # Phase 1: Integrity check (fast path)
    is_valid, msg = check_dataset_integrity()
    if is_valid:
        print(f"[OK] {msg}")
        print("No action required. Proceed with experiments.")
        sys.exit(0)

    print(f"[MISSING] {msg}")
    print("Initializing download sequence...")

    # Phase 2: Download and extraction
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for archive_name, config in ARCHIVES.items():
        archive_url = f"{BASE_URL}/{archive_name}?download=1"
        archive_path = DATASET_DIR / archive_name
        target_subdir = DATASET_DIR / config["subdir"]

        print(f"\n[PROCESSING] {config['desc']}")

        # Skip if already extracted
        if target_subdir.exists() and any(target_subdir.iterdir()):
            print(f"  [SKIP] {target_subdir} already populated")
            if archive_path.exists():
                archive_path.unlink()  # Cleanup redundant archive
            continue

        # Download if not present or incomplete
        if not archive_path.exists() or archive_path.stat().st_size == 0:
            print(f"  [DOWNLOAD] {archive_url}")
            if not download_with_resume(archive_url, archive_path):
                sys.exit(1)

        # Extract
        print(f"  [EXTRACT] {archive_name} -> {target_subdir}")
        if not extract_and_cleanup(archive_path, target_subdir):
            sys.exit(1)

    # Phase 3: Final verification
    is_valid, msg = check_dataset_integrity()
    if is_valid:
        print("\n[SUCCESS] Dataset ready at NGAFID-Ceu-em-Tres-Atos/Datasets/")
    else:
        print(f"\n[FAILED] Validation error: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()