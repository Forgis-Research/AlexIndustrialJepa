---
name: STAR Replication Progress
description: Replicating Fan et al. 2024 STAR (supervised SOTA on C-MAPSS) - FD001 in progress
type: project
---

Paper: "A Two-Stage Attention-Based Hierarchical Transformer for Turbofan Engine RUL Prediction" (Fan et al., Sensors 2024)
Replication dir: `/home/sagemaker-user/IndustrialJEPA/paper-replications/star/`
GitHub: https://github.com/KU-VGI/AP (STAR architecture in models.py)

## Target Results (Table 5 of paper)

| Subset | Paper RMSE | Target EXACT (<10%) | Target GOOD (<20%) |
|--------|-----------|--------------------|--------------------|
| FD001 | 10.61 | ≤11.7 | ≤12.7 |
| FD002 | 13.47 | ≤14.8 | ≤16.2 |
| FD003 | 10.71 | ≤11.8 | ≤12.9 |
| FD004 | 15.87 | ≤17.5 | ≤19.0 |

## FD001 Results So Far (April 11, 2026)

| Seed | Test RMSE | Test Score | Status |
|------|-----------|------------|--------|
| 42 | 12.301 | 253.2 | Done |
| 123 | 12.969 | 286.7 | Done |
| 456 | 11.250 | 196.2 | Done (EXACT!) |
| 789 | TBD | TBD | Running (~15:39 ETA) |
| 1024 | TBD | TBD | Pending |
| Mean (3 seeds) | 12.17 | 245.4 | GOOD range |

**Assessment**: GOOD replication on FD001 (14.7% above paper). One seed (456) achieves EXACT (≤11.7). Variance is high across seeds.

**Why:** Likely causes: (1) slight differences in patch_length selection, (2) seed sensitivity of attention heads, (3) val split fraction (we use 15%, paper doesn't specify).

## Architecture Notes

- Model: 3.67M parameters (FD001 config)
- Config: lr=0.0002, batch_size=32, window_length=32, n_scales=3, d_model=128, n_heads=1
- Training: Max 200 epochs, patience=20 on val RMSE
- Typical: 39-54 epochs to convergence
- Wall time per seed: ~30-34 minutes on A10G

## Running State (April 11, 2026)

PID 69842 is running all 4 subsets sequentially via `run_overnight.sh`. FD001 seed 789 is in progress (ETA 15:39). After FD001 finishes all 5 seeds, the script automatically starts FD002.

**Why:** Overnight job started at 13:25 on April 11. STAR is used as the supervised SOTA reference for IndustrialJEPA V11 results.

**How to apply:** STAR RMSE=10.61 (FD001) is the target for all C-MAPSS supervised benchmarks. Our JEPA E2E=13.80 is 30% above STAR.
