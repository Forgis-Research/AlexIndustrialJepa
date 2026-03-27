# Mechanical-JEPA Experiment Log

## Overview

**Goal:** Self-supervised dynamics transfer across robot embodiments using JEPA.

**Key metric:** Transfer ratio < 2.0 (5x data efficiency on new robot)

---

## Current Best

| Metric | Value | Model | Date |
|--------|-------|-------|------|
| Pretraining loss | -- | -- | -- |
| Embodiment classification | -- | -- | -- |
| 1-step forecasting MSE | -- | -- | -- |
| Transfer ratio (KUKA) | -- | -- | -- |

---

## Experiments

<!--
Format for each experiment:

## Exp N: [One-line description]

**Time**: YYYY-MM-DD HH:MM
**Phase**: Sanity / Viability / Pretraining / Classification / Forecasting / Transfer
**Hypothesis**: [What you expect]
**Change**: [What you modified]

**Setup**:
- Dataset: [names, sizes]
- Model: [config]
- Training: [epochs, batch, lr]

**Results**:
| Metric | Value | Baseline | Delta |
|--------|-------|----------|-------|
| ... | ... | ... | ... |

**Sanity checks**: ✓ passed / ⚠️ issues
**Verdict**: KEEP / REVERT / INVESTIGATE
**Insight**: [What you learned]
**Next**: [What to try]
-->

---

## Exp 0: Phase 0 — Sanity Checks + Viability Test

**Time**: 2026-03-26 17:55
**Phase**: Sanity Check + Viability
**Hypothesis**: Implementation is correct; model can learn from synthetic robot trajectories.
**Change**: First run — establishing baseline behavior.

**Setup**:
- Dataset: Synthetic robot trajectories (joint random walk with momentum)
- Sanity check: 100 episodes, seq_len=32, d_model=32, 1 layer (~10K params)
- Viability: 1000 episodes, seq_len=64, d_model=64, 2 layers, 10 epochs (209K params)

**Results — Sanity Checks**:

| Check | Result |
|-------|--------|
| 1. Data loads (no NaN, shape correct) | PASSED |
| 2. Forward pass (8,32,7) -> (8,32,32) | PASSED |
| 3. Loss computes (1.1030, finite, positive) | PASSED |
| 4. Loss decreases 10 steps: 1.035->0.827 (20.1%) | PASSED |
| 5. Gradients flow (no NaN, no Inf) | PASSED |
| 6. EMA updates target (93.74->93.44) | PASSED |
| 7. Overfits single batch 100 steps: 1.048->0.044 | PASSED |
| 8. Masking works (9/32 positions) | PASSED |
| Collapse check: variance=0.823, mean_dist=7.05 | NO COLLAPSE |

**Results — Viability Test (209K params, 10 epochs)**:

| Metric | Value |
|--------|-------|
| Train loss epoch1->epoch10 | 0.9593->0.5097 (46.9% decrease) |
| Val loss epoch1->epoch10 | 0.8856->0.4895 |
| Val/Train ratio | 0.96 (no overfitting) |
| Embedding variance | 0.626 |
| Mean pairwise distance | 8.73 |
| Cross-similarity (quality) | 0.396 |

**Sanity checks**: all 8 passed, no collapse
**Verdict**: KEEP — cleared for full pretraining
**Insight**: Model trains stably on synthetic robot data. No collapse. Loss decreases 47% in 10 epochs. Healthy generalization (val ~= train loss).
**Next**: Proceed to full pretraining with `--config small --epochs 50`
