"""
Phase 2 runner — wraps v11 Part D (pretraining) with W&B logging.

Observer-mode: does NOT change v11's code. Just sets up wandb.init() so
the run is registered, then invokes the team's run_experiments.py
main() with --parts D. Metrics from the final history are logged to
wandb as summary stats.
"""
import os
import sys
import time
import json

from dotenv import load_dotenv

ROOT = "/home/sagemaker-user/IndustrialJEPA"
V11 = os.path.join(ROOT, "mechanical-jepa/experiments/v11")

load_dotenv(os.path.join(ROOT, ".env"))

import wandb
wandb.init(
    project="industrialjepa",
    name=f"alex-v1-phase2-pretrain-{time.strftime('%Y%m%d-%H%M')}",
    tags=["v1", "reproduce-v11", "phase2", "pretrain", "FD001"],
    config={
        "alex_experiment": "v1",
        "upstream_version": "v11",
        "part": "D",
        "model": "TrajectoryJEPA V1",
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 256,
        "patch_length": 1,
        "n_pretrain_epochs": 200,
        "batch_size": 8,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "n_cuts_per_epoch": 20,
        "ema_momentum": 0.996,
        "dataset": "C-MAPSS FD001",
    },
)

sys.argv = ["run_experiments.py", "--parts", "D"]
sys.path.insert(0, V11)
os.chdir(V11)

from run_experiments import main
t0 = time.time()
main()
elapsed_min = (time.time() - t0) / 60.0
print(f"[alex-wrapper] Part D wall time: {elapsed_min:.2f} min", flush=True)

history_path = os.path.join(V11, "pretrain_history_L1.json")
diag_path = os.path.join(V11, "pretrain_diagnostics.json")
summary = {"phase2_wall_min": elapsed_min}

if os.path.exists(history_path):
    with open(history_path) as f:
        history = json.load(f)
    losses = history.get("loss", [])
    probe_rmses = history.get("probe_rmse", [])
    probe_epochs = history.get("probe_epochs", [])
    for i, (loss, pred_loss) in enumerate(zip(
            history.get("loss", []), history.get("pred_loss", []))):
        wandb.log({"epoch": i, "pretrain/loss": loss,
                   "pretrain/pred_loss": pred_loss})
    for ep, r in zip(probe_epochs, probe_rmses):
        wandb.log({"epoch": ep, "pretrain/probe_rmse": r})
    if losses:
        summary.update({
            "pretrain/initial_loss": losses[0],
            "pretrain/final_loss": losses[-1],
            "pretrain/loss_ratio": losses[-1] / max(losses[0], 1e-12),
            "pretrain/n_epochs": len(losses),
        })
    if probe_rmses:
        summary["pretrain/best_probe_rmse"] = min(probe_rmses)

if os.path.exists(diag_path):
    with open(diag_path) as f:
        diag = json.load(f)
    for k in ("pc1_rho", "max_component_rho", "shuffle_rmse",
             "embedding_std_mean", "embedding_std_min"):
        if k in diag:
            summary[f"pretrain/{k}"] = diag[k]

wandb.summary.update(summary)
print("[alex-wrapper] wandb summary:", json.dumps(summary, indent=2), flush=True)
wandb.finish()
