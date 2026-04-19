"""
Rule 1 audit, part 2: replicate `train_utils.finetune(mode='e2e')`
inline so I can keep the trained model+probe around and audit the
per-engine prediction trajectories. Compare to the buggy audit I ran
earlier via the `run_prediction_trajectories.py` recipe.
"""
import os
import sys
import json
import time
import copy
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
OUT_DIR = ("/home/sagemaker-user/AlexIndustrialJepa/alex-contribution/"
           "experiments/v1")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
sys.path.insert(0, V11)

from data_utils import (
    load_cmapss_subset, CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_finetune, collate_test, N_SENSORS, RUL_CAP)
from models import TrajectoryJEPA, RULProbe
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def finetune_inline(model, train_engines, val_engines, test_engines,
                    test_rul, seed=0, n_epochs=100,
                    early_stop_patience=20, batch_size=16):
    """Mirrors train_utils.finetune(mode='e2e', seed=seed) but keeps
    trained model+probe. Bit-exact loop except for the return value."""
    model = model.to(DEVICE)
    probe = RULProbe(model.d_model).to(DEVICE)

    for p in model.context_encoder.parameters():
        p.requires_grad = True
    optim = torch.optim.Adam(
        list(model.context_encoder.parameters())
        + list(model.predictor.parameters())
        + list(probe.parameters()),
        lr=1e-4,
    )
    torch.manual_seed(seed); np.random.seed(seed)

    tr_ds = CMAPSSFinetuneDataset(train_engines,
                                   n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    te_ds = CMAPSSTestDataset(test_engines, test_rul)
    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                    collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                    collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                    collate_fn=collate_test)

    best_v = float("inf"); best_p = None; best_e = None; ni = 0
    for ep in range(1, n_epochs + 1):
        model.train(); probe.train()
        for past, mask, rul in tr:
            past = past.to(DEVICE); mask = mask.to(DEVICE); rul = rul.to(DEVICE)
            optim.zero_grad()
            h = model.encode_past(past, mask)
            F.mse_loss(probe(h), rul).backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optim.step()

        # val: use_last_only=True => RUL normalized label per engine at T
        model.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past = past.to(DEVICE); mask = mask.to(DEVICE)
                pv.append(probe(model.encode_past(past, mask)).cpu().numpy())
                tv.append(rul.numpy())
        val_r = float(np.sqrt(np.mean(
            (np.concatenate(pv) * RUL_CAP
             - np.concatenate(tv) * RUL_CAP) ** 2)))

        if val_r < best_v:
            best_v = val_r
            best_p = copy.deepcopy(probe.state_dict())
            best_e = copy.deepcopy(model.context_encoder.state_dict())
            ni = 0
        else:
            ni += 1
            if ni >= early_stop_patience:
                break

    probe.load_state_dict(best_p)
    model.context_encoder.load_state_dict(best_e)
    model.eval(); probe.eval()

    # canonical test RMSE (same as _eval_test_rmse)
    preds_test, tgt_test = [], []
    with torch.no_grad():
        for past, mask, rul in te:
            past = past.to(DEVICE); mask = mask.to(DEVICE)
            preds_test.append(probe(model.encode_past(past, mask)).cpu().numpy())
            tgt_test.append(rul.numpy())
    preds_test = np.concatenate(preds_test) * RUL_CAP
    tgt_test = np.concatenate(tgt_test)  # raw RUL (not normalized)
    test_rmse = float(np.sqrt(np.mean((preds_test - tgt_test) ** 2)))

    return model, probe, best_v, test_rmse


def main():
    t0 = time.time()
    ckpt = os.path.join(V11, "best_pretrain_L1.pt")
    data = load_cmapss_subset("FD001")

    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=128, n_heads=4,
        n_layers=2, d_ff=256)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    print("Fine-tuning E2E with proper protocol (seed=0)...", flush=True)
    out_model, out_probe, best_val, test_rmse = finetune_inline(
        model, data["train_engines"], data["val_engines"],
        data["test_engines"], data["test_rul"], seed=0)
    print(f"  best val RMSE: {best_val:.3f}", flush=True)
    print(f"  test RMSE (_eval_test_rmse equivalent): {test_rmse:.3f}",
          flush=True)

    test_engines = data["test_engines"]
    test_rul = data["test_rul"]
    eng_ids = sorted(test_engines.keys())
    id_to_idx = {eid: idx for idx, eid in enumerate(eng_ids)}
    min_past = 10

    per_engine = {}
    last_preds, last_trues = [], []
    for eid in eng_ids:
        seq = test_engines[eid]
        T = len(seq)
        true_final = float(test_rul[id_to_idx[eid]])
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        mask = torch.zeros(1, T, dtype=torch.bool, device=DEVICE)  # True = padding; no padding here
        preds, trues = [], []
        with torch.no_grad():
            for cut in range(min_past, T + 1):
                h = out_model.encode_past(x[:, :cut, :], mask[:, :cut])
                preds.append(out_probe(h).item() * RUL_CAP)
                trues.append(min(true_final + (T - cut), RUL_CAP))
        per_engine[eid] = {"T": T, "true_final_rul": true_final,
                            "preds": preds, "trues": trues}
        last_preds.append(preds[-1])
        last_trues.append(min(true_final, RUL_CAP))

    last_preds = np.array(last_preds); last_trues = np.array(last_trues)
    canonical_recomputed = float(
        np.sqrt(((last_preds - last_trues) ** 2).mean()))

    within_rhos, within_stds = [], []
    for e in per_engine.values():
        p = np.array(e["preds"]); t = np.array(e["trues"])
        within_stds.append(p.std())
        if len(p) > 3 and p.std() > 1e-6 and t.std() > 1e-6:
            rho, _ = spearmanr(p, t)
            within_rhos.append(rho)
    within_rhos = np.array(within_rhos); within_stds = np.array(within_stds)

    print(f"Recomputed canonical RMSE: {canonical_recomputed:.2f}",
          flush=True)
    print(f"Within-sequence Spearman rho mean={within_rhos.mean():.3f} "
          f"median={np.median(within_rhos):.3f} "
          f"(n_usable={len(within_rhos)})", flush=True)
    print(f"Pred trajectory std per engine: "
          f"mean={within_stds.mean():.2f} "
          f"median={np.median(within_stds):.2f} "
          f"min={within_stds.min():.2f}", flush=True)
    print(f"Cross-engine last-pred std: {last_preds.std():.2f}",
          flush=True)

    # Plots
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(last_trues, last_preds, alpha=0.6)
    lim = max(last_trues.max(), last_preds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("True RUL"); ax.set_ylabel("Predicted RUL")
    ax.set_title(
        f"V1 E2E proper seed=0 scatter\n"
        f"RMSE={canonical_recomputed:.2f}, "
        f"within-rho={within_rhos.mean():.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR,
                             "v1_prediction_scatter_proper.png"), dpi=120)
    plt.close(fig)

    lengths = sorted(eng_ids, key=lambda i: len(test_engines[i]))
    n = len(lengths)
    picks = [lengths[5], lengths[n // 4], lengths[n // 2],
             lengths[3 * n // 4], lengths[-5]]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("V1 E2E proper seed=0: per-engine RUL trajectories",
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
    fig.savefig(os.path.join(PLOTS_DIR,
                             "v1_prediction_trajectories_proper.png"),
                dpi=120)
    plt.close(fig)

    summary = {
        "finetune_protocol": "inline clone of train_utils.finetune(mode='e2e', seed=0)",
        "canonical_last_window_rmse_via_test_loader": test_rmse,
        "canonical_rmse_recomputed_via_full_sequence": canonical_recomputed,
        "within_sequence_spearman_rho_mean": float(within_rhos.mean()),
        "within_sequence_spearman_rho_median": float(np.median(within_rhos)),
        "n_engines_with_usable_rho": int(len(within_rhos)),
        "pred_trajectory_std_mean": float(within_stds.mean()),
        "pred_trajectory_std_median": float(np.median(within_stds)),
        "cross_engine_last_pred_std": float(last_preds.std()),
        "elapsed_s": time.time() - t0,
    }
    out = os.path.join(OUT_DIR, "prediction_trajectories_v1_proper.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("saved ->", out, flush=True)


if __name__ == "__main__":
    main()
