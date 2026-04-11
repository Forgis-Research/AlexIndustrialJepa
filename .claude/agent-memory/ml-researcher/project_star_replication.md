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

## FD001 Results - COMPLETE (April 11, 2026)

| Seed | Test RMSE | Test Score | Status |
|------|-----------|------------|--------|
| 42 | 12.301 | 253.2 | Done (GOOD) |
| 123 | 12.969 | 286.7 | Done (ABOVE) |
| 456 | 11.250 | 196.2 | Done (EXACT!) |
| 789 | 12.095 | 252.8 | Done (GOOD) |
| 1024 | 12.315 | 278.4 | Done (GOOD) |
| **Mean (5 seeds)** | **12.186 +/- 0.553** | **253.5 +/- 31.6** | **GOOD** |

**Assessment**: GOOD replication on FD001 (14.9% above paper's 10.61). One seed (456) achieves EXACT (≤11.7). FD002 started at 15:44:20 April 11.

## FD002 Results - IN PROGRESS (April 11, 2026)

Running as of 15:44:20. Expected ~30-34 min per seed x 5 seeds = ~2.5 hours. ETA: ~18:15 April 11.

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

**How to apply:** STAR 5-seed RMSE=12.186 +/- 0.553 (FD001), paper reports 10.61. GOOD replication (14.9% above paper). Our JEPA E2E=13.80 is ~13% above our replicated STAR. FD002 currently running.
