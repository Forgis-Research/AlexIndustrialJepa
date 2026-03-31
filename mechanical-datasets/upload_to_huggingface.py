"""
Upload processed dataset to HuggingFace.

Dataset: Forgis/Mechanical-Components
Config: bearings (for bearing datasets)

This script:
1. Loads the unified JSON dataset
2. Converts to HuggingFace Dataset format
3. Uploads to the specified config
4. Optionally appends to existing data
"""

import json
from pathlib import Path
from datasets import Dataset, Features, Value, Sequence
from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

# Load HF token from .env
load_dotenv("C:/Users/Jonaspetersen/dev/IndustrialJEPA/.env")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file!")

# HuggingFace repo details
REPO_ID = "Forgis/Mechanical-Components"
CONFIG_NAME = "bearings"

def create_dataset_from_json(json_file: Path) -> Dataset:
    """
    Load unified JSON and convert to HuggingFace Dataset.
    """
    print(f"Loading dataset from {json_file}...")

    with open(json_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # Define schema
    features = Features({
        # Required fields
        "signal": Sequence(Sequence(Value("float32"))),
        "n_channels": Value("int32"),
        "sampling_rate_hz": Value("int32"),
        "dataset_source": Value("string"),
        "sample_id": Value("string"),
        "health_state": Value("string"),

        # Standard fields (can be null)
        "channel_names": Sequence(Value("string")),
        "fault_type": Value("string"),
        "fault_severity": Value("string"),
        "rpm": Value("int32"),
        "load": Value("int32"),
        "load_unit": Value("string"),

        # Component info
        "component_type": Value("string"),
        "manufacturer": Value("string"),
        "model": Value("string"),

        # Provenance
        "original_file": Value("string"),
        "license": Value("string"),
        "split": Value("string"),

        # Extra metadata (JSON string)
        "extra_metadata": Value("string")
    })

    # Convert extra_metadata dict to JSON string for storage
    for sample in data:
        if "extra_metadata" in sample and isinstance(sample["extra_metadata"], dict):
            sample["extra_metadata"] = json.dumps(sample["extra_metadata"])

    # Create dataset
    dataset = Dataset.from_list(data, features=features)

    print(f"Created HuggingFace dataset with {len(dataset)} samples")

    return dataset

def upload_dataset(dataset: Dataset, append: bool = True):
    """
    Upload dataset to HuggingFace.

    Args:
        dataset: HuggingFace Dataset to upload
        append: If True, appends to existing config. If False, overwrites.
    """
    print(f"\n{'='*60}")
    print(f"Uploading to HuggingFace")
    print(f"{'='*60}")
    print(f"Repo: {REPO_ID}")
    print(f"Config: {CONFIG_NAME}")
    print(f"Samples: {len(dataset)}")
    print(f"Append mode: {append}")

    # Check if repo exists
    api = HfApi()
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
        print(f"\n[OK] Repository exists")
    except Exception as e:
        print(f"\n[INFO] Repository may not exist yet: {e}")
        print("[INFO] Will create on first upload")

    if append:
        try:
            # Try to load existing dataset
            from datasets import load_dataset
            print(f"\n[INFO] Attempting to load existing '{CONFIG_NAME}' config...")
            existing = load_dataset(REPO_ID, CONFIG_NAME, split="train", token=HF_TOKEN)
            print(f"[OK] Found existing data with {len(existing)} samples")

            # Concatenate
            from datasets import concatenate_datasets
            dataset = concatenate_datasets([existing, dataset])
            print(f"[OK] Appending: Total {len(dataset)} samples")

        except Exception as e:
            print(f"\n[INFO] No existing '{CONFIG_NAME}' config found: {e}")
            print("[INFO] Creating new config")

    # Upload
    print(f"\n[INFO] Pushing to hub...")
    dataset.push_to_hub(
        REPO_ID,
        config_name=CONFIG_NAME,
        split="train",
        token=HF_TOKEN,
        private=False  # Make it public
    )

    print(f"\n[OK] Upload complete!")
    print(f"[OK] Dataset URL: https://huggingface.co/datasets/{REPO_ID}")
    print(f"[OK] Config: {CONFIG_NAME}")

    return True

def main():
    # Setup paths
    base_dir = Path("C:/Users/Jonaspetersen/dev/IndustrialJEPA/mechanical-datasets")
    processed_dir = base_dir / "processed" / "cwru_unified"
    json_file = processed_dir / "cwru_unified.json"

    if not json_file.exists():
        print(f"Error: {json_file} not found!")
        return False

    # Create dataset
    dataset = create_dataset_from_json(json_file)

    # Print dataset info
    print("\nDataset Info:")
    print("-" * 60)
    print(dataset)
    print("\nFeatures:")
    for name, feature in dataset.features.items():
        print(f"  {name}: {feature}")

    # Ask for confirmation
    print("\n" + "=" * 60)
    response = input("Upload to HuggingFace? (yes/no): ").strip().lower()

    if response not in ["yes", "y"]:
        print("Upload cancelled.")
        return False

    # Upload
    success = upload_dataset(dataset, append=True)

    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
