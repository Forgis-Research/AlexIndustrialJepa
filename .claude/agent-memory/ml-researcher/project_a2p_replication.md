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

## UPDATED FINDINGS (April 11, 2026 ~17:30) - Probes 28b, 34-38

**Corrected LR variance AUROC (full dataset, Probe 35)**:
- LR (183K seq, full dataset): AUROC=0.5929 (reliable; earlier 0.616 was on 5x smaller sample)
- Oracle (full dataset): AUROC=0.7445 (revised from 0.720)
- LR captures 38% of learnable signal (not 58% as claimed before)
- Transformer 10-seed mean = 0.5211 (Probe 28b), captures 8.6% of learnable signal

**10-seed transformer distribution (COMPLETE, Probe 28b)**:
- mean=0.5211 ± 0.0415, median=0.5176
- 1/10 seeds (10%) exceed 0.60: seed=3=0.6345 (outlier)
- LR (0.5929) is 1.73 sigma above transformer mean; 9/10 transformer seeds < LR

**Supervised transformer 100-epoch (Probe 30, 2/5 seeds)**:
- seed=42=0.6274, seed=1=0.6211, mean=0.624
- Supervised 100ep >> LR (0.593) >> Unsupervised 30ep (0.521)
- 100 epochs = much more consistent (std<0.004 vs 0.042 for 30ep)
- This is the TRUE supervised upper bound for transformer AP

**Extended features (Probe 37)**:
- 25-feature LR: test=0.6182 (barely better than 8-feat: 0.616)
- GBM: test=0.6160 (no advantage)
- AP signal is linear in variance space; feature engineering at ceiling

**SVDB1 invalid (Probe 34)**:
- All AP labels at t>94%; train split has 0 positives
- SVDB1 is INVALID for AP evaluation

**Running experiments (April 11 ~17:30)**:
- Probe 38: Deep supervised transformer (128d, 4L, 150ep, 3-seed) - testing supervised upper bound
- Probe 30: Supervised 5-seed (3/5 done) - should finish ~18:30
- Probe 33: Transformer + variance features (running) - testing if explicit variance helps
- Probe 26b: AP-aware contrastive V2 (running ~70 min) - Python-buffered, no output yet

## Result Files

All in `results/improvements/`:
- `aptransformer_multiseed.json`: 3-seed APTransformer TRUE (0.524 ± 0.037)
- `aptransformer_seed_distribution.json`: 10-seed distribution (0.5211 ± 0.0415)
- `variance_features_ap.json`: Probe 29 - LR variance AUROC=0.616 (small dataset)
- `lr_variance_validation.json`: Probe 29b - C-sweep validation
- `smd_lr_variance.json`: Probe 32 - SMD LR AUROC=0.674
- `oracle_analysis.json`: Probe 35 - LR captures 38%, oracle=0.7445
- `f1tol_analysis.json`: Probe 36 - Random beats A2P F1-tol (69.57 vs 67.55%)
- `extended_features_ap.json`: Probe 37 - 25-feat LR=0.618, GBM=0.616
- `contrastive_ap_v2.json`: AP-aware contrastive V2 (PENDING)
- `deep_supervised_ap.json`: Probe 38 deep supervised (PENDING)
- Earlier files: svdb1_multiseed_final, random_baselines, oracle_ap_auroc, etc.

## NeurIPS Narrative (8-step evidence chain)

1. F1-tolerance 8x inflation (raw 5.35% -> 43.1%)
2. A2P AUROC = 0.499 ± 0.008 (3-seed, indistinguishable from random)
3. Rolling var + random scores BEAT A2P on ALL 3 datasets (random F1-tol=69.57% vs A2P 67.55%)
4. Metric rank inversion (rho=0.000): AUROC and F1-tol give opposite rankings
5. Data integrity (train==test 3.4x, seed bug)
6. Oracle future var = 0.347 (wrong AP eval); correct oracle=0.7445 (correct eval)
7. SVDB1 invalid (temporal confound: all labels at t>94%); SVDB4 is the valid dataset
8. Correct AP eval: LR variance=0.5929, supervised transformer=0.624, oracle=0.7445
8. Trainable models: TRUE 3-seed APTransformer baseline = 0.524 ± 0.037 (single-seed 0.642 was lucky); SSL pretraining consistently fails (objective mismatch)

Key contribution: propose correct AP evaluation (future_labels, AUROC/AUPRC), oracle=0.720, show F1-tol is unreliable.
Critical new finding: seed sensitivity in AP is extreme - seed=42 is 3.2 sigma above mean.
