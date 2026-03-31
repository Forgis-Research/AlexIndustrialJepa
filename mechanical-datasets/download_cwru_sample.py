"""
Download a small sample of CWRU bearing dataset for testing the pipeline.

We'll download a few representative files:
- Normal baseline
- Inner race fault
- Outer race fault
- Ball fault
"""

import urllib.request
import os
from pathlib import Path
import sys

# CWRU download URLs (direct links to .mat files)
# These are from the official CWRU bearing data center
CWRU_BASE_URL = "https://engineering.case.edu/sites/default/files/"

# Small sample files for testing (~10-20 MB total)
SAMPLE_FILES = {
    # Normal baseline (12k sampling rate)
    "normal_0.mat": CWRU_BASE_URL + "97.mat",  # Normal baseline at 0 HP
    "normal_1.mat": CWRU_BASE_URL + "98.mat",  # Normal baseline at 1 HP

    # Inner race faults (12k sampling rate, 0.007" fault)
    "IR007_0.mat": CWRU_BASE_URL + "105.mat",  # 0 HP
    "IR007_1.mat": CWRU_BASE_URL + "106.mat",  # 1 HP

    # Outer race faults (12k sampling rate, 0.007" fault)
    "OR007@6_0.mat": CWRU_BASE_URL + "130.mat",  # 0 HP, @6:00 position
    "OR007@6_1.mat": CWRU_BASE_URL + "131.mat",  # 1 HP, @6:00 position

    # Ball faults (12k sampling rate, 0.007" fault)
    "B007_0.mat": CWRU_BASE_URL + "118.mat",  # 0 HP
    "B007_1.mat": CWRU_BASE_URL + "119.mat",  # 1 HP
}

def download_file(url, dest_path, filename):
    """Download a file with progress indication."""
    try:
        print(f"Downloading {filename}...")

        # Create destination directory if it doesn't exist
        dest_path.mkdir(parents=True, exist_ok=True)

        file_path = dest_path / filename

        # Download the file
        urllib.request.urlretrieve(url, file_path)

        # Get file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] Downloaded {filename} ({size_mb:.2f} MB)")

        return True
    except Exception as e:
        print(f"  [FAIL] Failed to download {filename}: {e}")
        return False

def main():
    # Setup paths
    base_dir = Path("C:/Users/Jonaspetersen/dev/IndustrialJEPA/mechanical-datasets")
    raw_dir = base_dir / "raw" / "cwru_sample"

    print("=" * 60)
    print("CWRU Dataset Sample Download")
    print("=" * 60)
    print(f"Destination: {raw_dir}")
    print()

    # Download files
    success_count = 0
    total_count = len(SAMPLE_FILES)

    for filename, url in SAMPLE_FILES.items():
        if download_file(url, raw_dir, filename):
            success_count += 1

    print()
    print("=" * 60)
    print(f"Download complete: {success_count}/{total_count} files successful")
    print("=" * 60)

    # Calculate total size
    if raw_dir.exists():
        total_size = sum(f.stat().st_size for f in raw_dir.glob("*.mat")) / (1024 * 1024)
        print(f"Total size: {total_size:.2f} MB")

    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
