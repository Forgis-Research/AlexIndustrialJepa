"""
Convert CWRU bearing dataset to unified schema for HuggingFace upload.

Unified Schema (bearings config):
- signal: (n_channels, signal_length) - vibration data
- n_channels: number of channels
- sampling_rate_hz: sampling rate
- dataset_source: "CWRU"
- sample_id: unique identifier
- health_state: healthy | faulty
- fault_type: healthy | inner_race | outer_race | ball
- fault_severity: "0.007in" (fault diameter)
- rpm: motor speed
- load: motor load (0, 1, 2, 3)
- load_unit: "hp"
- component_type: "bearing"
- manufacturer: "SKF"
- model: "6205-2RS"
- channel_names: ["DE", "FE", "BA"] (Drive End, Fan End, Base)
- original_file: filename
- license: "public_domain"
- split: "train" (will split later if needed)
- extra_metadata: additional dataset-specific info
"""

import scipy.io
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple

# CWRU metadata mapping
FILE_METADATA = {
    # Normal baseline
    "normal_0.mat": {
        "health_state": "healthy",
        "fault_type": "healthy",
        "fault_severity": None,
        "load": 0,
        "original_code": "097"
    },
    "normal_1.mat": {
        "health_state": "healthy",
        "fault_type": "healthy",
        "fault_severity": None,
        "load": 1,
        "original_code": "098"
    },
    # Inner race faults (0.007")
    "IR007_0.mat": {
        "health_state": "faulty",
        "fault_type": "inner_race",
        "fault_severity": "0.007in",
        "load": 0,
        "original_code": "105"
    },
    "IR007_1.mat": {
        "health_state": "faulty",
        "fault_type": "inner_race",
        "fault_severity": "0.007in",
        "load": 1,
        "original_code": "106"
    },
    # Outer race faults (0.007", @6:00 position)
    "OR007@6_0.mat": {
        "health_state": "faulty",
        "fault_type": "outer_race",
        "fault_severity": "0.007in",
        "load": 0,
        "original_code": "130",
        "fault_position": "@6:00"
    },
    "OR007@6_1.mat": {
        "health_state": "faulty",
        "fault_type": "outer_race",
        "fault_severity": "0.007in",
        "load": 1,
        "original_code": "131",
        "fault_position": "@6:00"
    },
    # Ball faults (0.007")
    "B007_0.mat": {
        "health_state": "faulty",
        "fault_type": "ball",
        "fault_severity": "0.007in",
        "load": 0,
        "original_code": "118"
    },
    "B007_1.mat": {
        "health_state": "faulty",
        "fault_type": "ball",
        "fault_severity": "0.007in",
        "load": 1,
        "original_code": "119"
    }
}

def process_cwru_file(file_path: Path, metadata: Dict) -> List[Dict]:
    """
    Process a single CWRU .mat file and convert to unified schema.

    Returns a list of samples (one per segment).
    """
    print(f"\nProcessing {file_path.name}...")

    # Load .mat file
    mat_data = scipy.io.loadmat(file_path)

    # Extract data
    code = metadata["original_code"]

    # Find variable keys
    de_key = f"X{code}_DE_time"
    fe_key = f"X{code}_FE_time"
    ba_key = f"X{code}_BA_time"
    rpm_key = f"X{code}RPM"

    # Load channels
    channels = []
    channel_names = []

    if de_key in mat_data:
        channels.append(mat_data[de_key].flatten())
        channel_names.append("DE")

    if fe_key in mat_data:
        channels.append(mat_data[fe_key].flatten())
        channel_names.append("FE")

    if ba_key in mat_data:
        channels.append(mat_data[ba_key].flatten())
        channel_names.append("BA")

    # Get RPM
    rpm = int(mat_data[rpm_key][0, 0]) if rpm_key in mat_data else None

    # Stack channels: (n_channels, signal_length)
    signal = np.array(channels, dtype=np.float32)
    n_channels, signal_length = signal.shape

    print(f"  Channels: {channel_names}")
    print(f"  Shape: {signal.shape}")
    print(f"  RPM: {rpm}")

    # Segment the signal into smaller chunks for training
    # Use 10-second windows with 50% overlap
    # CWRU is 12 kHz, so 10 seconds = 120,000 samples
    segment_length = 120000  # 10 seconds at 12 kHz
    overlap = 0.5
    step = int(segment_length * (1 - overlap))

    samples = []
    segment_idx = 0

    for start in range(0, signal_length - segment_length + 1, step):
        end = start + segment_length
        segment_signal = signal[:, start:end]

        # Create sample ID
        sample_id = f"cwru_{code}_{segment_idx:03d}"

        # Build unified schema sample
        sample = {
            # Required fields
            "signal": segment_signal.tolist(),  # Convert to list for JSON/Parquet
            "n_channels": int(n_channels),
            "sampling_rate_hz": 12000,
            "dataset_source": "CWRU",
            "sample_id": sample_id,
            "health_state": metadata["health_state"],

            # Standard fields
            "channel_names": channel_names,
            "fault_type": metadata["fault_type"],
            "fault_severity": metadata.get("fault_severity"),
            "rpm": rpm,
            "load": metadata["load"],
            "load_unit": "hp",

            # Component info
            "component_type": "bearing",
            "manufacturer": "SKF",
            "model": "6205-2RS",

            # Provenance
            "original_file": file_path.name,
            "license": "public_domain",
            "split": "train",

            # Extra metadata
            "extra_metadata": {
                "sensor_positions": {
                    "DE": "drive_end",
                    "FE": "fan_end",
                    "BA": "base"
                },
                "fault_position": metadata.get("fault_position"),
                "fault_method": "EDM",  # Electro-discharge machining
                "motor_power": "2hp",
                "original_matlab_key_prefix": f"X{code}",
                "segment_index": segment_idx,
                "segment_start_sample": int(start),
                "segment_end_sample": int(end)
            }
        }

        samples.append(sample)
        segment_idx += 1

    print(f"  Created {len(samples)} segments")

    return samples

def main():
    # Setup paths
    base_dir = Path("C:/Users/Jonaspetersen/dev/IndustrialJEPA/mechanical-datasets")
    raw_dir = base_dir / "raw" / "cwru_sample"
    processed_dir = base_dir / "processed" / "cwru_unified"

    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CWRU to Unified Schema Conversion")
    print("=" * 60)

    all_samples = []

    # Process each file
    for filename, metadata in FILE_METADATA.items():
        file_path = raw_dir / filename

        if not file_path.exists():
            print(f"\nWarning: {filename} not found, skipping...")
            continue

        samples = process_cwru_file(file_path, metadata)
        all_samples.extend(samples)

    print(f"\n{'='*60}")
    print(f"Total samples created: {len(all_samples)}")
    print('='*60)

    # Save to JSON (will convert to HF dataset later)
    output_file = processed_dir / "cwru_unified.json"
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)

    print(f"\nSaved unified dataset to: {output_file}")

    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 60)

    health_counts = {}
    fault_counts = {}
    load_counts = {}

    for sample in all_samples:
        # Count health states
        health = sample["health_state"]
        health_counts[health] = health_counts.get(health, 0) + 1

        # Count fault types
        fault = sample["fault_type"]
        fault_counts[fault] = fault_counts.get(fault, 0) + 1

        # Count loads
        load = sample["load"]
        load_counts[load] = load_counts.get(load, 0) + 1

    print(f"Health states: {dict(sorted(health_counts.items()))}")
    print(f"Fault types: {dict(sorted(fault_counts.items()))}")
    print(f"Load distribution: {dict(sorted(load_counts.items()))}")

    # Calculate total size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"\nOutput file size: {file_size_mb:.2f} MB")

    return output_file

if __name__ == "__main__":
    output_file = main()
