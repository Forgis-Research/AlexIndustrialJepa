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

## Trainable AP Models on Correct Evaluation (SVDB4, April 11, 2026)

**CRITICAL: Multi-seed validation (Probe 27) reveals single-seed results are unreliable!**

| Model | AUROC | Source | Notes |
|-------|-------|--------|-------|
| Rolling var (no training) | 0.476 | deterministic | Baseline |
| Multi-scale MLP (supervised) | 0.602 | single seed=42 | May be lucky |
| JEPA temporal pretrain + finetune | 0.619 | single seed=42 | May be lucky |
| Supervised from scratch (fixed LR) | 0.625 | single seed=42 | May be lucky |
| InfoNCE contrastive pretrain | 0.641 | single seed=42 | May be lucky |
| Large-scale pretrain (4x data) | 0.632 | single seed=42 | May be lucky |
| APTransformer (cosine LR) | 0.642 | single seed=42 | WAS LUCKY (3.2 sigma above mean!) |
| **APTransformer 3-seed TRUE** | **0.524 +/- 0.037** | **3 seeds** | **Statistically validated** |
| Oracle future var | 0.720 | deterministic | Upper bound |

**KEY INSIGHT**: APTransformer 0.642 = lucky seed 42 only. True multi-seed mean = 0.524 ± 0.037.
Gap to oracle is 0.196 (not 0.078). ALL single-seed results are unvalidated.

**SSL pretraining findings (all failed)**:
- JEPA temporal: -0.023 vs APTransformer, hurts due to normalcy prior
- InfoNCE generic: -0.001 vs APTransformer, neutral
- 4x large-scale temporal: -0.010 vs APTransformer, scale doesn't fix objective mismatch

## Running Experiments (April 11, 2026 ~16:00)

- AP-aware contrastive V2 (PID 113720): Still running (~21 min CPU time). Anomalous futures as positives.
- APTransformer 5-seed d_model=64 vs 128 (PID 126506): 5 seeds for tighter CI.
- STAR FD001: COMPLETE (5-seed RMSE=12.186 +/- 0.553, paper 10.61, +14.9%). FD002 started ~15:44.
- SMD A2P full run: KILLED after 129 min CPU (stuck at epoch 2/5, ~7.5h total = intractable).

## Result Files

All in `results/improvements/`:
- `aptransformer_multiseed.json`: 3-seed APTransformer TRUE estimate (CRITICAL - single-seed misleading)
- `pretrain_transfer_ap.json`: 4x large-scale pretrain (fails: -0.010 vs ATF)
- `contrastive_ap.json`: InfoNCE V1 (neutral: -0.001 vs ATF)
- `contrastive_ap_v2.json`: AP-aware contrastive V2 (PENDING)
- `aptransformer_5seed.json`: 5-seed d_model comparison (PENDING)
- `transformer_ap.json`: APTransformer seed=42 (0.642 - lucky single seed)
- `jepa_ap.json`: JEPA temporal pretrain + finetune (0.619 single seed)
- `ar_predictor_ap.json`: Multi-scale MLP (0.602 single seed)
- Earlier files: svdb1_multiseed_final, random_baselines, oracle_ap_auroc, correct_ap_evaluation, etc.

## NeurIPS Narrative (8-step evidence chain)

1. F1-tolerance 8x inflation (raw 5.35% -> 43.1%)
2. A2P AUROC = 0.499 ± 0.008 (3-seed, indistinguishable from random)
3. Rolling var + random scores BEAT A2P on ALL 3 datasets
4. Metric rank inversion (rho=0.000)
5. Data integrity (train==test 3.4x, seed bug)
6. Oracle future var = 0.347 (below random!) - evaluation tests detection not prediction
7. Correct AP evaluation: oracle=0.720 - task IS achievable; SVDB1 confounded
8. Trainable models: TRUE 3-seed APTransformer baseline = 0.524 ± 0.037 (single-seed 0.642 was lucky); SSL pretraining consistently fails (objective mismatch)

Key contribution: propose correct AP evaluation (future_labels, AUROC/AUPRC), oracle=0.720, show F1-tol is unreliable.
Critical new finding: seed sensitivity in AP is extreme - seed=42 is 3.2 sigma above mean.
