"""
Improvement Probe 5: Foundation Model Distillation Baseline.

Tests whether Chronos-Small (20M pretrained time series model) + linear head
can compete with A2P on anomaly prediction.

Key question: Does a frozen foundation model encode anomaly-relevant features?
If yes: challenges the value of A2P's architecture-specific innovations.
If no: confirms that anomaly prediction requires specialized training.

Uses proper MBA train/test split to avoid data leakage.
"""

import numpy as np
import json
import os
import sys
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/mnt/sagemaker-nvme/ad_datasets/MBA"
SPLIT_DIR = "/tmp/MBA_split"
RESULTS_DIR = "/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/results/improvements"
FIGURES_DIR = "/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def f1_with_tolerance(pred, gt, tolerance=50):
    """F1 with tolerance (A2P-style)."""
    pred = np.array(pred, dtype=int).copy()
    gt = np.array(gt, dtype=int)
    adjusted = pred.copy()
    in_seg = False
    for i in range(len(gt)):
        if gt[i] == 1 and adjusted[i] == 1 and not in_seg:
            in_seg = True
            for j in range(i, max(0, i-tolerance), -1):
                if gt[j] == 0: break
                adjusted[j] = 1
            for j in range(i, min(len(gt), i+tolerance)):
                if gt[j] == 0: break
                adjusted[j] = 1
        elif gt[i] == 0:
            in_seg = False
    p, r, f, _ = precision_recall_fscore_support(gt, adjusted, average='binary', zero_division=0)
    return f * 100, p * 100, r * 100


def extract_chronos_embeddings(signals, seq_len=100, pred_len=100, device='cuda'):
    """Extract Chronos embeddings for sliding windows."""
    from chronos import BaseChronosPipeline
    import torch

    print("Loading Chronos-Small...")
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device,
        torch_dtype=torch.float32,
    )
    pipeline.model.eval()

    n = len(signals)
    n_channels = signals.shape[1]

    # Extract embeddings for windows of length seq_len
    # For AP: we want to predict anomaly in [t, t+pred_len] given [t-seq_len, t]
    embeddings = []
    labels_list = []

    # Get test labels (for the prediction window)
    test_labels = np.load(os.path.join(SPLIT_DIR, "MBA_test_label.npy"))

    print(f"Extracting embeddings from {n - seq_len} windows...")
    batch_size = 64

    all_context = []
    for i in range(0, n - seq_len - pred_len + 1, pred_len):  # non-overlapping
        context = signals[i:i+seq_len]  # (seq_len, 2)
        all_context.append(context)

    n_windows = len(all_context)
    print(f"Total windows: {n_windows}")

    all_embeddings = []
    with torch.no_grad():
        for b_start in range(0, n_windows, batch_size):
            b_end = min(b_start + batch_size, n_windows)
            batch = all_context[b_start:b_end]  # list of (seq_len, 2)

            # Chronos takes univariate series; we process each channel
            batch_embeds = []
            for ch in range(n_channels):
                # Shape: (batch, seq_len)
                context_tensor = torch.tensor(
                    np.stack([w[:, ch] for w in batch]),
                    dtype=torch.float32
                ).to(device)
                # Get embeddings from Chronos encoder
                # Use forecast to get embedding
                quantiles, mean = pipeline.predict(
                    context=context_tensor,
                    prediction_length=1,
                    num_samples=1,
                    limit_prediction_length=False,
                )
                # Extract encoder hidden states by running encoder directly
                encoder_out = pipeline.model.encoder(
                    input_ids=None,
                    inputs_embeds=None,
                )
                batch_embeds.append(mean.squeeze(-1).cpu().numpy())

            # Concatenate channel embeddings
            combined = np.concatenate(batch_embeds, axis=-1)  # (batch, n_channels)
            all_embeddings.append(combined)

            if b_start % (batch_size * 5) == 0:
                print(f"  Processed {b_start}/{n_windows} windows")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Embeddings shape: {all_embeddings.shape}")
    return all_embeddings


def simple_chronos_probe():
    """
    Simpler version: use Chronos forecast error as anomaly score.

    For each window [t-seq_len, t], use Chronos to predict [t, t+pred_len].
    Compute MSE between Chronos forecast and actual future.
    High MSE = potentially anomalous.
    """
    from chronos import ChronosPipeline
    import torch

    print("\n=== Improvement Probe 5: Foundation Model Baseline (Chronos) ===\n")

    # Use proper split data
    if not os.path.exists(os.path.join(SPLIT_DIR, "MBA_train.npy")):
        print("Creating proper split...")
        test = np.load(os.path.join(DATA_DIR, "MBA_test.npy"))
        labels = np.load(os.path.join(DATA_DIR, "MBA_test_label.npy"))
        n = len(test)
        n_train = int(n * 0.70)
        np.save(os.path.join(SPLIT_DIR, "MBA_train.npy"), test[:n_train])
        np.save(os.path.join(SPLIT_DIR, "MBA_test.npy"), test[n_train:])
        np.save(os.path.join(SPLIT_DIR, "MBA_test_label.npy"), labels[n_train:])

    train = np.load(os.path.join(SPLIT_DIR, "MBA_train.npy"))  # (5376, 2)
    test = np.load(os.path.join(SPLIT_DIR, "MBA_test.npy"))   # (2304, 2)
    labels = np.load(os.path.join(SPLIT_DIR, "MBA_test_label.npy"))

    # Also load full test (train==test)
    test_full = np.load(os.path.join(DATA_DIR, "MBA_test.npy"))
    labels_full = np.load(os.path.join(DATA_DIR, "MBA_test_label.npy"))

    seq_len = 100
    pred_len = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Chronos-Small...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device,
        torch_dtype=torch.float32,
    )
    pipeline.model.eval()
    print(f"Model loaded")

    def get_chronos_forecast_errors(data, labels_data, seq_len=100, pred_len=100):
        """Get per-timestep anomaly scores from Chronos forecast error."""
        n = len(data)
        scores = np.zeros(n)
        n_windows = 0

        with torch.no_grad():
            for i in range(seq_len, n - pred_len + 1, pred_len):
                context = data[i-seq_len:i]  # (seq_len, 2)
                future = data[i:i+pred_len]   # (pred_len, 2)

                # Per-channel forecast
                channel_mse = []
                for ch in range(data.shape[1]):
                    # Chronos expects CPU tensor, handles device internally
                    ctx = torch.tensor(context[:, ch], dtype=torch.float32).unsqueeze(0)
                    # Chronos API: inputs first positional arg, returns (batch, n_samples, pred_len)
                    samples = pipeline.predict(
                        ctx,
                        prediction_length=pred_len,
                        num_samples=20,
                        limit_prediction_length=False,
                    )
                    # samples shape: (batch=1, n_samples=20, pred_len)
                    forecast = samples[0].mean(dim=0).cpu().numpy()  # (pred_len,)
                    mse = np.mean((forecast - future[:, ch]) ** 2)
                    channel_mse.append(mse)

                window_score = np.mean(channel_mse)
                scores[i:i+pred_len] = window_score
                n_windows += 1

        return scores

    print("\nRunning Chronos on proper split test set...")
    scores_split = get_chronos_forecast_errors(
        np.concatenate([train[-seq_len:], test]), labels, seq_len, pred_len
    )
    scores_split = scores_split[seq_len:]  # align with labels

    print(f"Scores computed: {scores_split.shape}")
    auroc_split = roc_auc_score(labels, scores_split)
    auprc_split = average_precision_score(labels, scores_split)

    print("\nRunning Chronos on full dataset (train==test)...")
    scores_full = get_chronos_forecast_errors(test_full, labels_full, seq_len, pred_len)
    scores_full_trimmed = scores_full[seq_len:]
    labels_full_trimmed = labels_full[seq_len:]

    auroc_full = roc_auc_score(labels_full_trimmed, scores_full_trimmed)
    auprc_full = average_precision_score(labels_full_trimmed, scores_full_trimmed)

    # Threshold at top-1% (same as A2P)
    thresh_split = np.percentile(scores_split, 99)
    thresh_full = np.percentile(scores_full_trimmed, 99)
    pred_split = (scores_split > thresh_split).astype(int)
    pred_full = (scores_full_trimmed > thresh_full).astype(int)

    f1_split, p_split, r_split = f1_with_tolerance(pred_split, labels, tolerance=50)
    f1_full, p_full, r_full = f1_with_tolerance(pred_full, labels_full_trimmed, tolerance=50)

    print(f"\n=== Chronos-Small Results ===")
    print(f"Proper split: AUROC={auroc_split:.4f}, AUPRC={auprc_split:.4f}, F1={f1_split:.2f}%")
    print(f"Full (train==test): AUROC={auroc_full:.4f}, AUPRC={auprc_full:.4f}, F1={f1_full:.2f}%")
    print(f"\nA2P comparison: AUROC=0.528, F1=43.1% (train==test), F1=12.66% (proper split)")

    results = {
        "probe": "chronos_baseline",
        "method": "Chronos-Small (20M, frozen) + forecast error score",
        "proper_split": {
            "auroc": float(auroc_split),
            "auprc": float(auprc_split),
            "f1_tolerance": float(f1_split),
            "precision": float(p_split),
            "recall": float(r_split),
        },
        "full_traintest": {
            "auroc": float(auroc_full),
            "auprc": float(auprc_full),
            "f1_tolerance": float(f1_full),
        },
        "a2p_comparison": {
            "auroc": 0.528,
            "f1_traintest": 43.1,
            "f1_proper_split": 12.66,
        }
    }

    outfile = os.path.join(RESULTS_DIR, "chronos_baseline.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")

    return results


if __name__ == "__main__":
    results = simple_chronos_probe()
    print("\n=== KEY FINDING ===")
    if results["proper_split"]["auroc"] > 0.528:
        print(f"Chronos AUROC ({results['proper_split']['auroc']:.3f}) > A2P AUROC (0.528)")
        print("Foundation model beats specialized system on raw discrimination!")
    else:
        print(f"Chronos AUROC ({results['proper_split']['auroc']:.3f}) <= A2P AUROC (0.528)")
        print("Foundation model does NOT beat A2P on raw discrimination")
