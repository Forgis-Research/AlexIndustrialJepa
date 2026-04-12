---
name: STAR Replication Progress
description: Replicating Fan et al. 2024 STAR (supervised SOTA on C-MAPSS) - ALL 4 subsets
type: project
---

Paper: "A Two-Stage Attention-Based Hierarchical Transformer for Turbofan Engine RUL Prediction" (Fan et al., Sensors 2024)
Replication dir: `/home/sagemaker-user/IndustrialJEPA/paper-replications/star/`
GitHub: https://github.com/KU-VGI/AP (STAR architecture in models.py)

## Target Results (Table 5 of paper)

| Subset | Paper RMSE | Our Replication | Status |
|--------|-----------|-----------------|--------|
| FD001 | 10.61 | 12.19 +/- 0.55 | GOOD (+14.9%) |
| FD002 | 13.47 | 20.03 +/- 1.51 | POOR (+48.7%) |
| FD003 | 10.71 | 12.74 +/- 0.10 | GOOD (+18.9%) |
| FD004 | 15.87 | PENDING (running since 00:27 Apr 12) | - |

**Why:** Run via run_overnight.sh as the supervised SOTA reference for IndustrialJEPA comparison.

## FD001 Results - COMPLETE

5-seed RMSE = 12.186 +/- 0.553 (paper=10.61, GOOD). Best seed (456) = 11.25.

## FD002 Results - COMPLETE (April 12, poor replication)

5-seed RMSE = 20.025 +/- 1.506 (paper=13.47, 48.7% above paper).
- FD002 is the hardest multi-condition subset for STAR too
- Our JEPA FD002 frozen=26.33, E2E=24.45 - STAR replication is still better at 20.03

## FD003 Results - COMPLETE

5-seed RMSE = 12.736 +/- 0.099 (paper=10.71, GOOD). Very tight std.

## FD004 Results - IN PROGRESS

Running on GPU (PID 245063), started 00:27 Apr 12. Expected 2-4h.

## V12 Phase 2: STAR Label Efficiency Sweep

PID 243354 running phase2_star_label_sweep.py on FD001 only.
- 5 budgets [100%, 50%, 20%, 10%, 5%] x 5 seeds = 25 STAR runs on FD001
- No intermediate results yet (1h55m elapsed)
- Budget 100% expected to complete ~03:05 Apr 12
- Kill criterion: STAR@20% <= 14 RMSE kills JEPA label-efficiency pitch

## Architecture Notes

- Model: 3.67M parameters (FD001 config)
- Config: lr=0.0002, batch_size=32, window_length=32, n_scales=3, d_model=128, n_heads=1
- Training: Max 200 epochs, patience=20 on val RMSE
- Typical: 39-54 epochs to convergence
- Wall time per seed on A10G: ~28-35 min (with competing GPU processes: ~35-40 min)

**How to apply:** STAR FD001 replication=12.19+/-0.55. Our JEPA E2E=14.23+/-0.39.
The JEPA gap to STAR is ~2 RMSE at 100% labels. Label efficiency at reduced budgets
is the key differentiator - results pending from Phase 2.
