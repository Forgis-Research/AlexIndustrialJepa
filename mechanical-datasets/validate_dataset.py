"""
Validate the unified dataset before uploading to HuggingFace.

Checks:
- No NaN or infinite values
- Signal statistics reasonable
- All required fields present
- Consistent shapes
- Metadata complete
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

def validate_sample(sample: Dict, sample_idx: int) -> List[str]:
    """Validate a single sample and return list of issues."""
    issues = []

    # Check required fields
    required_fields = [
        "signal", "n_channels", "sampling_rate_hz", "dataset_source",
        "sample_id", "health_state"
    ]

    for field in required_fields:
        if field not in sample:
            issues.append(f"Sample {sample_idx}: Missing required field '{field}'")

    if issues:
        return issues

    # Validate signal
    signal = np.array(sample["signal"])

    # Check shape
    if len(signal.shape) != 2:
        issues.append(f"Sample {sample_idx}: Signal should be 2D, got shape {signal.shape}")
    else:
        n_channels, signal_length = signal.shape
        if n_channels != sample["n_channels"]:
            issues.append(f"Sample {sample_idx}: n_channels mismatch. "
                          f"Signal has {n_channels}, metadata says {sample['n_channels']}")

    # Check for NaN or inf
    if np.any(np.isnan(signal)):
        issues.append(f"Sample {sample_idx}: Contains NaN values")

    if np.any(np.isinf(signal)):
        issues.append(f"Sample {sample_idx}: Contains infinite values")

    # Check statistics (basic sanity checks)
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)

    # Vibration data should have mean near zero
    if abs(signal_mean) > 1.0:
        issues.append(f"Sample {sample_idx}: Unusual mean {signal_mean:.3f} (expected ~0)")

    # Should have reasonable variation
    if signal_std < 0.001:
        issues.append(f"Sample {sample_idx}: Very low std {signal_std:.6f} (possible constant signal)")

    return issues

def validate_dataset(dataset: List[Dict]) -> bool:
    """Validate entire dataset."""
    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)

    all_issues = []

    print(f"\nValidating {len(dataset)} samples...")

    for idx, sample in enumerate(dataset):
        issues = validate_sample(sample, idx)
        all_issues.extend(issues)

    if all_issues:
        print(f"\nFound {len(all_issues)} issues:")
        for issue in all_issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n[OK] All samples passed validation!")
        return True

def visualize_samples(dataset: List[Dict], output_dir: Path):
    """Create visualizations of sample signals."""
    print("\nGenerating visualizations...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize one sample from each fault type
    fault_samples = {}

    for sample in dataset:
        fault_type = sample["fault_type"]
        if fault_type not in fault_samples:
            fault_samples[fault_type] = sample

    # Create multi-panel plot
    n_faults = len(fault_samples)
    fig, axes = plt.subplots(n_faults, 1, figsize=(15, 3*n_faults))

    if n_faults == 1:
        axes = [axes]

    for idx, (fault_type, sample) in enumerate(sorted(fault_samples.items())):
        signal = np.array(sample["signal"])
        n_channels = signal.shape[0]

        # Plot first channel (DE - most informative)
        time = np.arange(signal.shape[1]) / sample["sampling_rate_hz"]

        axes[idx].plot(time, signal[0], linewidth=0.5)
        axes[idx].set_title(f"{fault_type} (Sample: {sample['sample_id']})")
        axes[idx].set_xlabel("Time (s)")
        axes[idx].set_ylabel("Acceleration (g)")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "signal_examples.png"
    plt.savefig(output_file, dpi=150)
    print(f"  Saved: {output_file}")
    plt.close()

    # Create statistics plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract statistics
    fault_types = []
    means = []
    stds = []
    n_channels_list = []

    for sample in dataset:
        signal = np.array(sample["signal"])
        fault_types.append(sample["fault_type"])
        means.append(np.mean(signal))
        stds.append(np.std(signal))
        n_channels_list.append(sample["n_channels"])

    # Plot 1: Mean by fault type
    fault_type_unique = sorted(set(fault_types))
    mean_by_fault = {ft: [] for ft in fault_type_unique}
    for ft, m in zip(fault_types, means):
        mean_by_fault[ft].append(m)

    axes[0, 0].boxplot([mean_by_fault[ft] for ft in fault_type_unique],
                        labels=fault_type_unique)
    axes[0, 0].set_title("Signal Mean by Fault Type")
    axes[0, 0].set_ylabel("Mean (g)")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Std by fault type
    std_by_fault = {ft: [] for ft in fault_type_unique}
    for ft, s in zip(fault_types, stds):
        std_by_fault[ft].append(s)

    axes[0, 1].boxplot([std_by_fault[ft] for ft in fault_type_unique],
                        labels=fault_type_unique)
    axes[0, 1].set_title("Signal Std by Fault Type")
    axes[0, 1].set_ylabel("Std (g)")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Distribution histogram
    axes[1, 0].hist(means, bins=30, alpha=0.7, label='Mean')
    axes[1, 0].set_title("Distribution of Signal Means")
    axes[1, 0].set_xlabel("Mean (g)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(stds, bins=30, alpha=0.7, label='Std', color='orange')
    axes[1, 1].set_title("Distribution of Signal Stds")
    axes[1, 1].set_xlabel("Std (g)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "statistics.png"
    plt.savefig(output_file, dpi=150)
    print(f"  Saved: {output_file}")
    plt.close()

def main():
    # Setup paths
    base_dir = Path("C:/Users/Jonaspetersen/dev/IndustrialJEPA/mechanical-datasets")
    processed_dir = base_dir / "processed" / "cwru_unified"
    validation_dir = base_dir / "validation_reports"

    dataset_file = processed_dir / "cwru_unified.json"

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} samples")

    # Validate
    validation_passed = validate_dataset(dataset)

    # Visualize
    visualize_samples(dataset, validation_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    if validation_passed:
        print("[OK] Dataset ready for upload!")
        print(f"[OK] {len(dataset)} samples")
        print(f"[OK] Visualizations saved to: {validation_dir}")
    else:
        print("[FAIL] Dataset has issues, fix before uploading")

    return validation_passed

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
