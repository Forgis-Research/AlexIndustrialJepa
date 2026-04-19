"""
Rule 1 internal-consistency check for my v1 reproduction.

Loads the V1 Trajectory JEPA checkpoint that Phase 2 produced
(best_pretrain_L1.pt, d_model=128), trains an E2E RUL probe on FD001
for one seed, then computes:
  (a) test RMSE in the canonical last-window protocol
  (b) the prediction trajectory for every test engine across all
      cut points >= min_past

Writes a scatter of predicted-vs-true and a facet grid of per-engine
prediction trajectories. Also emits a summary JSON with the scalar
RMSE, within-sequence Spearman rho (mean over engines), and
cross-engine trajectory flatness (std of predictions across cut
points averaged over engines).

If the canonical RMSE is "good" but the within-sequence Spearman rho
is near zero or the average-std is tiny, that is the v11 near-miss
failure mode (Rule 1 violation) and must be logged as
⚠️ INTERNAL INCONSISTENCY in EXPERIMENT_LOG.md.
"""
import os
import sys
import json
import copy
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = "/home/sagemaker-user/IndustrialJEPA"
V11 = os.path.join(ROOT, "mechanical-jepa/experiments/v11")
OUT_DIR = "/home/sagemaker-user/AlexIndustrialJepa/alex-contribution/experiments/v1"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
sys.path.insert(0, V11)

from data_utils import (
    load_cmapss_subset, CMAPSSFinetuneDataset,
    collate_finetune, N_SENSORS, RUL_CAP,
)
from models import TrajectoryJEPA, RULProbe
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)


def train_e2e_probe(ckpt_path, d_model, n_layers, n_heads, d_ff,
                    train_eng, val_eng, seed=42, max_epochs=100):
    """Quick E2E fine-tune matching run_prediction_trajectories.py."""
    model = TrajectoryJEPA(n_sensors=N_SENSORS, patch_length=1,
                            d_model=d_model, n_heads=n_heads,
                            n_layers=n_layers, d_ff=d_ff)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(DEVICE)
    probe = RULProbe(d_model).to(DEVICE)

    for p in model.context_encoder.parameters():
        p.requires_grad = True
    optim = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(probe.parameters()),
        lr=1e-4,
    )
    torch.manual_seed(seed); np.random.seed(seed)

    tr_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=True)
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True,
                    collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=16, shuffle=False,
                    collate_fn=collate_finetune)

    best_v, best_ps, best_es, ni = float("inf"), None, None, 0
    for ep in range(max_epochs):
        model.train(); probe.train()
        for past, mask, rul in tr:
            past = past.to(DEVICE); mask = mask.to(DEVICE); rul = rul.to(DEVICE)
            optim.zero_grad()
            h = model.encode_past(past, mask)
            F.mse_loss(probe(h), rul).backward()
            torch.nn.utils.clip_grad_norm_(
                list(probe.parameters()) +
                list(model.context_encoder.parameters()), 1.0)
            optim.step()

        model.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past = past.to(DEVICE); mask = mask.to(DEVICE)
                pv.append(probe(model.encode_past(past, mask)).cpu().numpy())
                tv.append(rul.numpy())
        val_r = float(np.sqrt(np.mean(
            (np.concatenate(pv) * RUL_CAP - np.concatenate(tv) * RUL_CAP) ** 2)))
        if val_r < best_v:
            best_v, best_ps, best_es, ni = (
                val_r, copy.deepcopy(probe.state_dict()),
                copy.deepcopy(model.context_encoder.state_dict()), 0)
        else:
            ni += 1
            if ni >= 20:
                break

    probe.load_state_dict(best_ps)
    model.context_encoder.load_state_dict(best_es)
    model.eval(); probe.eval()
    return model, probe, best_v


def main():
    t0 = time.time()
    ckpt_path = os.path.join(V11, "best_pretrain_L1.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: no checkpoint at {ckpt_path}", flush=True)
        sys.exit(1)

    print("Loading FD001 data...", flush=True)
    data = load_cmapss_subset("FD001")

    print("Training E2E probe on V1 checkpoint (seed=42)...", flush=True)
    model, probe, best_val_r = train_e2e_probe(
        ckpt_path, d_model=128, n_layers=2, n_heads=4, d_ff=256,
        train_eng=data["train_engines"], val_eng=data["val_engines"],
        seed=42)
    print(f"  best val RMSE: {best_val_r:.2f}", flush=True)

    test_engines = data["test_engines"]
    test_rul = data["test_rul"]
    eng_ids = sorted(test_engines.keys())
    id_to_idx = {eid: idx for idx, eid in enumerate(eng_ids)}

    min_past = 10
    per_engine = {}
    last_preds, last_trues = [], []
    for eid in eng_ids:
        eng = test_engines[eid]
        T = len(eng)
        true_final_rul = float(test_rul[id_to_idx[eid]])
        x = torch.tensor(eng, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        mask = torch.ones(1, T, dtype=torch.bool, device=DEVICE)
        preds, trues = [], []
        with torch.no_grad():
            for cut in range(min_past, T + 1):
                xc = x[:, :cut, :]
                mc = mask[:, :cut]
                h = model.encode_past(xc, mc)
                p = probe(h).item() * RUL_CAP
                preds.append(p)
                rul = min(true_final_rul + (T - cut), RUL_CAP)
                trues.append(rul)
        per_engine[eid] = {
            "T": T,
            "true_final_rul": true_final_rul,
            "preds": preds,
            "trues": trues,
        }
        last_preds.append(preds[-1])
        last_trues.append(min(true_final_rul, RUL_CAP))

    last_preds = np.array(last_preds); last_trues = np.array(last_trues)
    canonical_rmse = float(
        np.sqrt(((last_preds - last_trues) ** 2).mean()))
    print(f"Canonical last-window test RMSE: {canonical_rmse:.2f}",
          flush=True)

    within_rhos = []
    within_stds = []
    for eid, e in per_engine.items():
        p = np.array(e["preds"]); t = np.array(e["trues"])
        if len(p) > 3 and p.std() > 1e-6 and t.std() > 1e-6:
            rho, _ = spearmanr(p, t)
            within_rhos.append(rho)
        within_stds.append(p.std())

    within_rhos = np.array(within_rhos)
    within_stds = np.array(within_stds)
    print(f"Within-sequence Spearman rho (pred vs true): "
          f"mean={within_rhos.mean():.3f} median={np.median(within_rhos):.3f} "
          f"n_usable={len(within_rhos)}", flush=True)
    print(f"Pred trajectory std per engine: "
          f"mean={within_stds.mean():.2f} median={np.median(within_stds):.2f} "
          f"min={within_stds.min():.2f}", flush=True)

    last_pred_std_across_engines = float(last_preds.std())
    print(f"Cross-engine std of last-window prediction: "
          f"{last_pred_std_across_engines:.2f}", flush=True)

    # --- Plot: prediction vs true scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(last_trues, last_preds, alpha=0.6)
    lim = max(last_trues.max(), last_preds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("True RUL (cycles, capped 125)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(
        f"Alex v1 (V1 E2E seed=42) canonical scatter\n"
        f"RMSE={canonical_rmse:.2f}, pred_std={last_pred_std_across_engines:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "v1_prediction_scatter.png"), dpi=120)
    plt.close(fig)

    # --- Plot: five per-engine trajectories
    lengths = sorted(eng_ids, key=lambda i: len(test_engines[i]))
    n = len(lengths)
    picks = [lengths[5], lengths[n // 4], lengths[n // 2],
             lengths[3 * n // 4], lengths[-5]]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Alex v1: per-engine RUL trajectory (V1 E2E seed=42)",
                 fontweight="bold")
    for ax, eid in zip(axes, picks):
        e = per_engine[eid]
        cycles = np.arange(min_past, min_past + len(e["preds"]))
        ax.plot(cycles, e["trues"], "k-", label="true", alpha=0.7)
        ax.plot(cycles, e["preds"], "C0-", label="pred", alpha=0.9)
        ax.set_title(f"engine {eid} T={e['T']}")
        ax.set_xlabel("cycle"); ax.set_ylabel("RUL")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "v1_prediction_trajectories.png"),
                dpi=120)
    plt.close(fig)

    summary = {
        "canonical_last_window_rmse": canonical_rmse,
        "within_sequence_spearman_rho_mean": float(within_rhos.mean()),
        "within_sequence_spearman_rho_median": float(np.median(within_rhos)),
        "n_engines_with_usable_rho": int(len(within_rhos)),
        "pred_trajectory_std_mean": float(within_stds.mean()),
        "pred_trajectory_std_median": float(np.median(within_stds)),
        "cross_engine_last_pred_std": last_pred_std_across_engines,
        "best_val_rmse": best_val_r,
        "elapsed_s": time.time() - t0,
    }
    out = os.path.join(OUT_DIR, "prediction_trajectories_v1.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("saved ->", out, flush=True)


if __name__ == "__main__":
    main()
