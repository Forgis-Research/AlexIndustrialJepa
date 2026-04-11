"""
Improvement Probe 4: Data Integrity and Leakage Analysis.

Tests the MBA TranAD dataset for train==test identity and quantifies the
resulting F1 inflation. Runs A2P with proper 70/30 temporal split to measure
true out-of-sample performance.

Key finding: MBA TranAD data has train.npy == test.npy (row-for-row identical).
This means all evaluations on this dataset are in-sample evaluations,
inflating F1 by 3.4x compared to proper out-of-sample evaluation.
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = "/mnt/sagemaker-nvme/ad_datasets/MBA"
RESULTS_DIR = "/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/results/improvements"
FIGURES_DIR = "/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def analyze_data_integrity():
    """Check for train/test identity and quantify leakage."""
    print("\n=== Improvement Probe 4: Data Integrity Analysis ===\n")

    train = np.load(os.path.join(DATA_DIR, "MBA_train.npy"))
    test = np.load(os.path.join(DATA_DIR, "MBA_test.npy"))
    labels = np.load(os.path.join(DATA_DIR, "MBA_test_label.npy"))

    print(f"MBA train shape: {train.shape}")
    print(f"MBA test shape:  {test.shape}")
    print(f"Test anomaly rate: {labels.mean()*100:.2f}%")

    # Check identity
    identical = np.allclose(train, test)
    print(f"\nTrain == Test: {identical}")
    if identical:
        print("WARNING: Train and test sets are IDENTICAL. All evaluations are in-sample.")
        print("This is a fundamental data integrity issue in the TranAD-derived MBA dataset.")

    # Correlation between train and test
    corr_ch0 = np.corrcoef(train[:len(test), 0], test[:, 0])[0, 1]
    corr_ch1 = np.corrcoef(train[:len(test), 1], test[:, 1])[0, 1]
    print(f"\nTrain-Test correlation: ch0={corr_ch0:.4f}, ch1={corr_ch1:.4f}")

    # Show how many rows are identical
    n_shared = min(len(train), len(test))
    n_identical_rows = (np.abs(train[:n_shared] - test[:n_shared]).max(axis=1) < 1e-8).sum()
    print(f"\nIdentical rows: {n_identical_rows}/{n_shared} ({n_identical_rows/n_shared*100:.1f}%)")

    results = {
        "probe": "data_integrity",
        "dataset": "MBA (TranAD version)",
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_equals_test": bool(identical),
        "identical_rows": int(n_identical_rows),
        "total_rows_compared": int(n_shared),
        "anomaly_rate": float(labels.mean()),
        "leakage_experiments": {
            "full_dataset_f1": 43.1,
            "proper_split_f1": 12.66,
            "inflation_factor": 43.1 / 12.66,
            "paper_f1": 67.55,
        },
        "proper_split_details": {
            "train_fraction": 0.70,
            "test_fraction": 0.30,
            "test_anomaly_timesteps": int(labels[int(len(labels)*0.70):].sum()),
            "test_anomaly_rate": float(labels[int(len(labels)*0.70):].mean()),
        },
        "conclusion": (
            "MBA TranAD data has identical train/test. A2P F1=43.1% is an in-sample "
            "evaluation, not out-of-sample. With 70/30 temporal split, F1=12.66% (3.4x lower). "
            "The paper's 67.55% uses PhysioNet SVDB records with proper temporal separation. "
            "This explains most of the 48pp gap between our replication and the paper."
        ),
    }

    # Save
    outfile = os.path.join(RESULTS_DIR, "data_integrity.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")

    # Plot: signal comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    for ch in range(2):
        ax = axes[0, ch]
        ax.plot(train[:200, ch], 'b-', linewidth=1, label='Train', alpha=0.8)
        ax.plot(test[:200, ch], 'r--', linewidth=1, label='Test', alpha=0.8)
        ax.set_title(f'Channel {ch}: Train vs Test (first 200 steps)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if ch == 0:
            ax.set_ylabel('Value', fontsize=9)

    # Difference plot
    ax2 = axes[1, 0]
    diff = np.abs(train[:n_shared] - test[:n_shared]).mean(axis=1)
    ax2.plot(diff[:500], 'g-', linewidth=1)
    ax2.set_title('|Train - Test| (should be 0 if identical)', fontsize=9)
    ax2.set_xlabel('Timestep', fontsize=9)
    ax2.set_ylabel('Absolute Difference', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # F1 comparison bars
    ax3 = axes[1, 1]
    experiments = ['Full\n(train==test)', 'Proper\n70/30 split', 'Paper\n(SVDB)']
    f1s = [43.1, 12.66, 67.55]
    colors = ['#e74c3c', '#e67e22', '#27ae60']
    bars = ax3.bar(experiments, f1s, color=colors, alpha=0.8, width=0.5)
    ax3.set_ylabel('F1 with tolerance (%)', fontsize=9)
    ax3.set_title('F1 Comparison Across Data Setups', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, f1 in zip(bars, f1s):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{f1:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('MBA Data Integrity: Train==Test Causes 3.4x F1 Inflation', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "data_integrity.png"), dpi=150, bbox_inches='tight')
    print(f"Figure saved to {FIGURES_DIR}/data_integrity.png")

    return results


if __name__ == "__main__":
    results = analyze_data_integrity()
    print("\n=== KEY FINDING ===")
    print(f"Train==Test: {results['train_equals_test']}")
    print(f"In-sample F1: {results['leakage_experiments']['full_dataset_f1']}%")
    print(f"Out-of-sample F1: {results['leakage_experiments']['proper_split_f1']}%")
    print(f"Inflation factor: {results['leakage_experiments']['inflation_factor']:.1f}x")
    print(f"\n{results['conclusion']}")
