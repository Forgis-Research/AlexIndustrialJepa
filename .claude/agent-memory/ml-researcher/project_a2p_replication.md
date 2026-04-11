---
name: A2P Replication Findings (April 2026)
description: Critical findings from replicating Park et al. ICML 2025 "When Will It Fail?" - reveals A2P evaluation flaws and NeurIPS contribution
type: project
---

## Key Facts

Paper: "When Will It Fail? Anomaly to Prompt for Forecasting Future Anomalies in Time Series" (ICML 2025, Park et al.)
Code: https://github.com/KU-VGI/AP
Replication dir: `/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/`

## Critical Findings (All Confirmed, April 2026)

1. **AUROC essentially random on proper split (3-seed confirmed)**: Seed 42 (0.490), seed 1 (0.498), seed 2 (0.508). Mean=0.499 +/- 0.008. Indistinguishable from random 0.500.

2. **RANDOM SCORES BEAT A2P on ALL 3 datasets**: SVDB4: 68.10% vs 67.55%; SMD: 67.60% vs 52.07% (+15.5pp!); SVDB1: 58.91% vs 19.17% (+39.7pp). F1-tol is trivially gamed by random noise.

3. **Rolling variance (max-chan) DOMINATES A2P**: MBA SVDB4 w=50: 86.70% vs 67.55% (+19.15pp). SMD w=10: 63.95% vs 52.07% (+11.9pp). ALL window sizes beat A2P.

4. **MBA train==test data leakage (3.4x inflation)**: TranAD-derived MBA has identical train/test. With proper 70/30 split: 12.66%.

5. **F1-tolerance 8x inflation**: Raw F1=5.35%, F1-tol=43.1% (8x). Point adjustment inflates random scores 10x (6.68% -> 68.19% on SVDB4).

6. **Oracle AP AUROC = 0.347 (below random!)**: Future variance doesn't predict current anomaly labels on SVDB4. Evaluation tests detection, not prediction.

7. **Correct AP evaluation oracle = 0.720**: Define future_labels[t]=1 if anomaly in [t+100, t+150]. Oracle AUROC=0.720 (SVDB4), 0.554 (SMD), 0.692 (SVDB1). AP IS learnable with proper evaluation.

8. **SVDB1 temporal confound**: All 5 anomaly segments at t>65000 of 69120. Time index AUROC=0.954. Not a valid AP dataset.

9. **E2E training probe**: Unfreezing AAFN during joint training: +12.8pp F1 (16%->28.8%), AUROC crosses 0.5 for first time (0.490->0.507).

10. **Metric rank inversion (Spearman rho=0.000)**: F1-tol ranks A2P #1, AUROC ranks A2P last.

## 3-Seed SVDB1 Results (FINAL, April 11, 2026)

| Seed | F1-tol | AUROC |
|------|--------|-------|
| 42 | 16.06% | 0.490 |
| 1 | 22.29% | 0.498 |
| 2 | 36.41% | 0.508 |
| Mean | 24.92% ± 8.51% | 0.499 ± 0.008 |
| Paper | 67.55% | - |

## Random Score Baselines (5 seeds each)

| Dataset | Random F1-tol | A2P | Beats A2P? |
|---------|--------------|-----|------------|
| SVDB4 | 68.10% ± 0.04% | 67.55% | YES |
| SMD | 67.60% ± 0.03% | 52.07% | YES (+15.5pp) |
| SVDB1 | 58.91% ± 7.64% | 19.17% | YES (+39.7pp) |

## Oracle AP Upper Bounds (Correct Evaluation)

| Dataset | Oracle AUROC | Method |
|---------|-------------|--------|
| SVDB4 | 0.720 | Oracle future var |
| SVDB4 | 0.679 | Supervised MLP (15 features, 30 epochs) |
| SMD | 0.554 | Oracle future var |
| SMD | 0.652 | Supervised MLP (15 features, 30 epochs) |
| SVDB1 | 0.692 | Oracle future var (but confounded!) |

## Running Experiments (April 11, 2026)

- SMD A2P seed=42 full run (PID 5584): in pretraining Epoch 1. ETA: ~20+ hours.

## Result Files

All in `results/improvements/`:
- `svdb1_multiseed_final.json`: 3-seed complete
- `random_baselines.json`: Random scores on all 3 datasets
- `oracle_ap_auroc.json`: Oracle AUROC analysis
- `correct_ap_evaluation.json`: Correct AP evaluation
- `oracle_mlp_ap.json`: Oracle MLP on SVDB4
- `smd_oracle_ap.json`: Oracle MLP on SMD
- `svdb1_correct_ap.json`: Correct AP on SVDB1 (confound analysis)
- `e2e_training.json`: E2E training probe result
- `tolerance_sensitivity.json`: F1-tol plateau analysis
- `smd_max_chan_var.json`: Max-channel var on SMD

## NeurIPS Narrative

Seven-step evidence chain for "F1-tol is broken for AP":
1. F1-tolerance 8x inflation (raw 5.35% -> 43.1%)
2. A2P AUROC = 0.499 ± 0.008 (3-seed, indistinguishable from random)
3. Rolling var + random scores BEAT A2P on all datasets
4. Metric rank inversion (rho=0.000)
5. Data integrity (train==test 3.4x, seed bug)
6. Oracle future var = 0.347 (below random!) - evaluation tests detection not prediction
7. Correct AP evaluation: oracle=0.720 - task IS achievable; SVDB1 is confounded

Key contribution: propose correct AP evaluation (future_labels, AUROC/AUPRC), oracle target 0.720.
Target: JEPA-AP that achieves AUROC > 0.72 under correct evaluation.
