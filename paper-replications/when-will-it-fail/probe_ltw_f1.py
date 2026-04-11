"""
Improvement Probe 9: Lead-Time-Weighted F1 (LTW-F1).

Computes F1 with weights based on how early each anomaly is predicted.
Predictions further in advance from anomaly onset get higher weight.
This captures the "when" in "When Will It Fail?" better than standard F1.

Uses existing A2P predictions (already computed).
"""

import numpy as np
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

AP_DIR = "/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/AP"
DATA_DIR = "/mnt/sagemaker-nvme/ad_datasets/MBA"
RESULTS_DIR = "/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/results/improvements"
FIGURES_DIR = "/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def lead_time_weighted_f1(pred, gt, pred_len=100, tolerance=50):
    """
    Compute Lead-Time-Weighted F1.

    For each true positive in a predicted window, the weight is proportional to
    how early in the window the anomaly is first detected.

    A prediction at position 0 in the window (first timestep) gets weight 1.0
    A prediction at position pred_len-1 (last timestep) gets weight 0.0

    This rewards early predictions over last-minute ones.
    """
    pred = np.array(pred, dtype=int)
    gt = np.array(gt, dtype=int)
    n = len(pred)

    # Find anomaly segments in gt
    anomaly_segs = []
    in_seg = False
    seg_start = 0
    for i in range(n):
        if gt[i] == 1 and not in_seg:
            in_seg = True
            seg_start = i
        elif gt[i] == 0 and in_seg:
            in_seg = False
            anomaly_segs.append((seg_start, i-1))
    if in_seg:
        anomaly_segs.append((seg_start, n-1))

    # Standard F1
    p_std, r_std, f_std, _ = precision_recall_fscore_support(gt, pred, average='binary', zero_division=0)

    # For LTW-F1: weight each TP by (seg_end - hit_pos) / pred_len
    # where seg_end is the anomaly segment start, hit_pos is when we first predict it
    weighted_tp = 0.0
    total_weight = 0.0
    n_seg_detected = 0

    for (seg_start, seg_end) in anomaly_segs:
        seg_len = seg_end - seg_start + 1
        # Find first prediction in the window leading up to this segment
        # Look back pred_len timesteps from seg_start
        search_start = max(0, seg_start - pred_len)
        search_end = seg_end

        first_hit = None
        for i in range(search_start, search_end + 1):
            if pred[i] == 1:
                first_hit = i
                break

        if first_hit is not None:
            # Lead time: how far ahead of the anomaly start was the detection?
            lead_time = max(0, seg_start - first_hit)
            # Weight: 0 if at anomaly start, 1 if pred_len ahead
            weight = lead_time / pred_len
            weighted_tp += weight
            n_seg_detected += 1

        # Max possible weight for this segment
        total_weight += 1.0

    # LTW-F1: weighted TP / (weighted TP + FP + FN)
    fp = pred.sum() - n_seg_detected  # FP = flags that don't correspond to a segment
    fn = len(anomaly_segs) - n_seg_detected  # FN = segments not detected

    if weighted_tp + fp + fn > 0:
        ltw_f1 = weighted_tp / (weighted_tp + 0.5 * fp + 0.5 * fn) * 100
    else:
        ltw_f1 = 0.0

    return ltw_f1, f_std * 100, n_seg_detected, len(anomaly_segs)


def ltw_f1_probe():
    """Analyze LTW-F1 vs standard F1 across methods."""
    print("\n=== Improvement Probe 9: Lead-Time-Weighted F1 ===\n")

    # Load actual A2P predictions
    scores = np.load('/tmp/a2p_scores_pred.npy')
    labels = np.load('/tmp/a2p_labels_pred.npy').astype(int)
    thresh = 1.8916
    pred_len = 100

    a2p_pred = (scores > thresh).astype(int)
    print(f"N={len(labels)}, anomaly_rate={labels.mean()*100:.2f}%")
    print(f"A2P: flagged {a2p_pred.sum()} timesteps ({a2p_pred.mean()*100:.2f}%)")

    # Baseline predictions
    base_rate = labels.mean()
    np.random.seed(42)
    random_pred = (np.random.random(len(labels)) < base_rate).astype(int)

    # Perfect prediction (oracle): predict 50 steps ahead of each anomaly
    oracle_pred = np.zeros(len(labels), dtype=int)
    anomaly_positions = np.where(labels > 0)[0]
    for pos in anomaly_positions:
        # Flag 50 timesteps before the anomaly
        for lead in range(50, 0, -1):
            idx = pos - lead
            if 0 <= idx < len(labels):
                oracle_pred[idx] = 1
        oracle_pred[pos] = 1

    methods = {
        "A2P": a2p_pred,
        "Random": random_pred,
        "Oracle (50-step lead)": oracle_pred,
        "Always-0": np.zeros(len(labels), dtype=int),
    }

    results = {
        "probe": "ltw_f1",
        "n_timesteps": int(len(labels)),
        "anomaly_rate": float(base_rate),
        "pred_len": pred_len,
        "methods": {}
    }

    print(f"\n{'Method':25} {'Std F1':>10} {'LTW-F1':>10} {'LTW/Std Ratio':>15} {'Segs Det':>10}")
    print("-" * 75)

    for name, pred in methods.items():
        ltw, std, n_det, n_total = lead_time_weighted_f1(pred, labels, pred_len=pred_len)
        ratio = ltw / std if std > 0 else 0
        print(f"{name:25} {std:>10.2f} {ltw:>10.2f} {ratio:>15.2f} {n_det:>3}/{n_total:>3}")

        results["methods"][name] = {
            "std_f1": float(std),
            "ltw_f1": float(ltw),
            "ltw_std_ratio": float(ratio),
            "segments_detected": int(n_det),
            "total_segments": int(n_total),
        }

    # Save
    outfile = os.path.join(RESULTS_DIR, "ltw_f1_analysis.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")

    # Analysis
    a2p_std = results["methods"]["A2P"]["std_f1"]
    a2p_ltw = results["methods"]["A2P"]["ltw_f1"]

    print(f"\nKey finding:")
    print(f"A2P: Std F1={a2p_std:.2f}%, LTW-F1={a2p_ltw:.2f}%")
    if a2p_ltw < a2p_std * 0.5:
        print("A2P's LTW-F1 is << Std F1: predictions are NOT early (last-minute detections)")
        print("This contradicts the paper's claim of predicting 'when' anomalies will happen")
    elif a2p_ltw > a2p_std * 0.8:
        print("A2P's LTW-F1 is comparable to Std F1: predictions are reasonably early")
    else:
        print("A2P has moderate lead time in predictions")

    # Plot
    method_names = list(results["methods"].keys())
    std_f1s = [results["methods"][m]["std_f1"] for m in method_names]
    ltw_f1s = [results["methods"][m]["ltw_f1"] for m in method_names]

    x = np.arange(len(method_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, std_f1s, width, label='Standard F1', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, ltw_f1s, width, label='Lead-Time-Weighted F1', color='coral', alpha=0.8)

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_title('Standard F1 vs Lead-Time-Weighted F1\n(MBA dataset, A2P predictions)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ltw_f1_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"Figure saved to {FIGURES_DIR}/ltw_f1_comparison.png")

    return results


if __name__ == "__main__":
    results = ltw_f1_probe()
