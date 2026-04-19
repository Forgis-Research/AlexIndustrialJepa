"""
Phase 3 runner — direct call to v11 part_e (fine-tune) + a minimal
F-equivalent (label-efficiency plot). Bypasses run_experiments.main()
because that path hits a device-mismatch bug in the else-branch of
Part D (loaded model is never .to(DEVICE)).
"""
import os
import sys
import time
import json

from dotenv import load_dotenv

ROOT = "/home/sagemaker-user/IndustrialJEPA"
V11 = os.path.join(ROOT, "mechanical-jepa/experiments/v11")
OUT_DIR = ("/home/sagemaker-user/AlexIndustrialJepa/alex-contribution/"
           "experiments/v1")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
load_dotenv(os.path.join(ROOT, ".env"))

import wandb
wandb.init(
    project="industrialjepa",
    name=f"alex-v1-phase3-finetune-{time.strftime('%Y%m%d-%H%M')}",
    tags=["v1", "reproduce-v11", "phase3", "finetune", "FD001"],
    config={
        "alex_experiment": "v1",
        "upstream_version": "v11",
        "parts": "E+F (direct call, bypassing main)",
        "n_seeds": 5,
        "budgets": [1.0, 0.5, 0.2, 0.1, 0.05],
    },
)

sys.path.insert(0, V11)
os.chdir(V11)

from data_utils import load_cmapss_subset
import run_experiments as rex
from run_experiments import part_e, save_json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Redirect all `log()` calls to my v1 log, never truncate the team's
# v11 EXPERIMENT_LOG.md.
V1_LOG = os.path.join(OUT_DIR, "logs", "v1_part_EF_from_part_e.md")
os.makedirs(os.path.dirname(V1_LOG), exist_ok=True)
with open(V1_LOG, "w") as f:
    f.write(f"# v1 Phase 3 (Part E direct) — "
            f"{time.strftime('%Y-%m-%d %H:%M')}\n\n")


def _alex_log(msg: str):
    print(msg, flush=True)
    with open(V1_LOG, "a") as f:
        f.write(msg + "\n")


rex.log = _alex_log  # patch the module-level log used by part_e
log = _alex_log

log("Loading FD001 for Phase 3 (fine-tune)...")
data = load_cmapss_subset("FD001")

# Dummy model placeholder — part_e re-creates fresh model per seed
# from the checkpoint on disk, so `model` arg is ignored by my read
# of the source but we pass None to be explicit.
t0 = time.time()
results = part_e(data, model=None, n_seeds=5)
elapsed_min = (time.time() - t0) / 60.0
log(f"\n[alex-wrapper] Part E wall time: {elapsed_min:.2f} min")

budgets = [1.0, 0.5, 0.2, 0.1, 0.05]
STAR = 10.61
AE_LSTM = 13.99

budget_pcts = [b * 100 for b in budgets]
lstm = [results["supervised_lstm"][b] for b in budgets]
frozen = [results["jepa_frozen"][b] for b in budgets]
e2e = [results["jepa_e2e"][b] for b in budgets]

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(budget_pcts, [x["mean"] for x in lstm],
            yerr=[x["std"] for x in lstm], marker="o",
            label="Supervised LSTM", capsize=4, linewidth=2)
ax.errorbar(budget_pcts, [x["mean"] for x in frozen],
            yerr=[x["std"] for x in frozen], marker="s",
            label="Traj JEPA (frozen)", capsize=4, linewidth=2)
ax.errorbar(budget_pcts, [x["mean"] for x in e2e],
            yerr=[x["std"] for x in e2e], marker="^",
            label="Traj JEPA (E2E)", capsize=4, linewidth=2)
ax.axhline(STAR, color="red", linestyle="--",
           label=f"STAR 2024 (sup SOTA): {STAR}")
ax.axhline(AE_LSTM, color="purple", linestyle=":",
           label=f"AE-LSTM SSL ref: {AE_LSTM}")
ax.axhline(20.14, color="gray", linestyle="-.",
           label="Ridge on 58 hand-feat (my baseline): 20.14")
ax.set_xscale("log")
ax.set_xlabel("Label fraction (%)")
ax.set_ylabel("Test RMSE (cycles)")
ax.set_title("v1 reproduction: FD001 label efficiency (V1 JEPA)")
ax.set_xticks(budget_pcts)
ax.set_xticklabels([f"{int(p)}%" for p in budget_pcts])
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
fig_path = os.path.join(PLOTS_DIR, "v1_label_efficiency.png")
plt.savefig(fig_path, dpi=120)
plt.close()
log(f"  Saved {fig_path}")

summary = {"phase3_wall_min": elapsed_min}
for method, bmap in results.items():
    for b, vals in bmap.items():
        summary[f"ft/{method}/{int(b*100)}/mean"] = vals["mean"]
        summary[f"ft/{method}/{int(b*100)}/std"] = vals["std"]

log("\n[alex-wrapper] Fine-tune summary (my v1 reproduction):")
for method in ("supervised_lstm", "jepa_frozen", "jepa_e2e"):
    for b in budgets:
        m = results[method][b]["mean"]
        s = results[method][b]["std"]
        log(f"  {method:<18s} @ {int(b*100):>3d}%: {m:.2f} +/- {s:.2f}")
        wandb.log({
            f"ft/{method}/mean": m, f"ft/{method}/std": s,
            "budget_pct": b * 100,
        })

wandb.summary.update(summary)
with open(os.path.join(OUT_DIR, "finetune_results_v1.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
wandb.finish()
