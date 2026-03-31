"""
Test loading the uploaded dataset from HuggingFace to verify it's accessible.
"""

from datasets import load_dataset
from dotenv import load_dotenv
import os
import numpy as np

# Load HF token
load_dotenv("C:/Users/Jonaspetersen/dev/IndustrialJEPA/.env")
HF_TOKEN = os.getenv("HF_TOKEN")

REPO_ID = "Forgis/Mechanical-Components"
CONFIG_NAME = "bearings"

def test_dataset_load():
    """Test loading and accessing the dataset."""
    print("=" * 60)
    print("Testing HuggingFace Dataset Load")
    print("=" * 60)

    try:
        # Load dataset
        print(f"\nLoading {REPO_ID} (config: {CONFIG_NAME})...")
        dataset = load_dataset(REPO_ID, CONFIG_NAME, split="train", token=HF_TOKEN)

        print(f"[OK] Dataset loaded successfully!")
        print(f"[OK] Number of samples: {len(dataset)}")

        # Print dataset info
        print("\nDataset Info:")
        print("-" * 60)
        print(dataset)

        # Access first sample
        print("\nFirst Sample:")
        print("-" * 60)
        sample = dataset[0]

        print(f"Sample ID: {sample['sample_id']}")
        print(f"Dataset Source: {sample['dataset_source']}")
        print(f"Health State: {sample['health_state']}")
        print(f"Fault Type: {sample['fault_type']}")
        print(f"Fault Severity: {sample['fault_severity']}")
        print(f"Component Type: {sample['component_type']}")
        print(f"Manufacturer: {sample['manufacturer']}")
        print(f"Model: {sample['model']}")
        print(f"RPM: {sample['rpm']}")
        print(f"Load: {sample['load']} {sample['load_unit']}")
        print(f"Channels: {sample['channel_names']}")
        print(f"Sampling Rate: {sample['sampling_rate_hz']} Hz")

        # Check signal
        signal = np.array(sample['signal'])
        print(f"\nSignal Shape: {signal.shape}")
        print(f"  n_channels: {sample['n_channels']}")
        print(f"  signal_length: {signal.shape[1]:,} samples")
        print(f"  duration: {signal.shape[1] / sample['sampling_rate_hz']:.2f} seconds")

        # Basic statistics
        print(f"\nSignal Statistics:")
        print(f"  Mean: {np.mean(signal):.6f}")
        print(f"  Std: {np.std(signal):.6f}")
        print(f"  Min: {np.min(signal):.6f}")
        print(f"  Max: {np.max(signal):.6f}")

        # Test all samples
        print("\nTesting All Samples:")
        print("-" * 60)

        fault_counts = {}
        health_counts = {}

        for sample in dataset:
            fault_type = sample['fault_type']
            health_state = sample['health_state']

            fault_counts[fault_type] = fault_counts.get(fault_type, 0) + 1
            health_counts[health_state] = health_counts.get(health_state, 0) + 1

        print(f"Fault Type Distribution: {dict(sorted(fault_counts.items()))}")
        print(f"Health State Distribution: {dict(sorted(health_counts.items()))}")

        print("\n" + "=" * 60)
        print("[OK] All tests passed!")
        print("=" * 60)
        print(f"\nDataset URL: https://huggingface.co/datasets/{REPO_ID}")
        print(f"Config: {CONFIG_NAME}")

        return True

    except Exception as e:
        print(f"\n[FAIL] Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_dataset_load()
    sys.exit(0 if success else 1)
