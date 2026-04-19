"""
Phase 3 runner — wraps v11 Parts E + F (fine-tune + plots) with W&B
logging. Consumes the checkpoint produced by Phase 2.
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
    name=f"alex-v1-phase3-finetune-{time.strftime('%Y%m%d-%H%M')}",
    tags=["v1", "reproduce-v11", "phase3", "finetune", "FD001"],
    config={
        "alex_experiment": "v1",
        "upstream_version": "v11",
        "parts": "E,F",
        "n_seeds": 5,
        "budgets": [1.0, 0.5, 0.2, 0.1, 0.05],
        "modes": ["supervised_lstm", "jepa_frozen", "jepa_e2e"],
        "dataset": "C-MAPSS FD001",
    },
)

sys.argv = ["run_experiments.py", "--parts", "E,F"]
sys.path.insert(0, V11)
os.chdir(V11)

from run_experiments import main
t0 = time.time()
main()
elapsed_min = (time.time() - t0) / 60.0
print(f"[alex-wrapper] Parts E+F wall time: {elapsed_min:.2f} min", flush=True)

results_path = os.path.join(V11, "finetune_results.json")
summary = {"phase3_wall_min": elapsed_min}
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
    for method, budget_map in results.items():
        for budget_key, vals in budget_map.items():
            budget_pct = float(budget_key) * 100
            summary[f"ft/{method}/{int(budget_pct)}/mean"] = vals.get("mean")
            summary[f"ft/{method}/{int(budget_pct)}/std"] = vals.get("std")
            wandb.log({
                f"ft/{method}/mean": vals.get("mean"),
                f"ft/{method}/std": vals.get("std"),
                "budget_pct": budget_pct,
            })

wandb.summary.update(summary)
print("[alex-wrapper] wandb summary:", json.dumps(summary, indent=2), flush=True)
wandb.finish()
