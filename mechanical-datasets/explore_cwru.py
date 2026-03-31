"""
Explore CWRU bearing dataset structure to understand the data format.
"""

import scipy.io
import numpy as np
from pathlib import Path
import json

def explore_mat_file(file_path):
    """Explore a single MATLAB file and extract metadata."""
    print(f"\n{'='*60}")
    print(f"File: {file_path.name}")
    print('='*60)

    # Load the .mat file
    mat_data = scipy.io.loadmat(file_path)

    # Get all variable names (excluding metadata keys)
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]

    print(f"Variables found: {data_keys}")

    file_info = {
        'filename': file_path.name,
        'variables': {}
    }

    for key in data_keys:
        data = mat_data[key]

        if isinstance(data, np.ndarray):
            print(f"\n{key}:")
            print(f"  Type: {data.dtype}")
            print(f"  Shape: {data.shape}")

            # Flatten if needed to get basic stats
            if data.size > 0:
                flat_data = data.flatten()
                print(f"  Length: {len(flat_data):,} samples")
                print(f"  Mean: {np.mean(flat_data):.6f}")
                print(f"  Std: {np.std(flat_data):.6f}")
                print(f"  Min: {np.min(flat_data):.6f}")
                print(f"  Max: {np.max(flat_data):.6f}")

                file_info['variables'][key] = {
                    'dtype': str(data.dtype),
                    'shape': data.shape,
                    'length': int(len(flat_data)),
                    'mean': float(np.mean(flat_data)),
                    'std': float(np.std(flat_data)),
                    'min': float(np.min(flat_data)),
                    'max': float(np.max(flat_data))
                }

    return file_info

def main():
    # Setup paths
    base_dir = Path("C:/Users/Jonaspetersen/dev/IndustrialJEPA/mechanical-datasets")
    raw_dir = base_dir / "raw" / "cwru_sample"

    print("=" * 60)
    print("CWRU Dataset Structure Exploration")
    print("=" * 60)

    # Get all .mat files
    mat_files = sorted(raw_dir.glob("*.mat"))
    print(f"\nFound {len(mat_files)} .mat files")

    all_info = []

    # Explore each file
    for mat_file in mat_files:
        info = explore_mat_file(mat_file)
        all_info.append(info)

    # Save exploration results
    output_file = base_dir / "cwru_exploration.json"
    with open(output_file, 'w') as f:
        json.dump(all_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Exploration results saved to: {output_file}")
    print('='*60)

    # Print summary
    print("\nSUMMARY:")
    print("-" * 60)

    # Analyze variable name patterns
    all_vars = set()
    for info in all_info:
        all_vars.update(info['variables'].keys())

    print(f"Unique variable names across all files: {sorted(all_vars)}")

    # Check sampling rates (if RPM data exists, we can infer sampling rate)
    # CWRU typically uses 12kHz or 48kHz
    print("\nNote: CWRU dataset uses:")
    print("  - 12 kHz sampling rate for most Drive End (DE) and all Fan End (FE) data")
    print("  - 48 kHz sampling rate for some Drive End (DE) data")
    print("  - Motor loads: 0, 1, 2, 3 HP")
    print("  - Bearing type: SKF 6205-2RS deep groove ball bearing")

if __name__ == "__main__":
    main()
