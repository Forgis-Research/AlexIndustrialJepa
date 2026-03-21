# Continue Autoresearch Prompt

Copy-paste this to Claude on the VM:

---

## Context

I'm running cross-machine transfer learning experiments for industrial time series forecasting. The goal is to train on multiple robot datasets and transfer to a held-out robot.

**Objective**: Transfer ratio ≤ 1.5 (target MSE / avg source MSE)

## Current State

**Exp04 completed** - ablation study with combined config:
- 40 epochs, 256 hidden, 4 layers, RevIN, 3 sources (Voraus + CNC + AURSAD held-out)
- Result: Transfer ratio = **1.72** (FAILED target ≤1.5)
- Voraus MSE: 0.324, CNC MSE: 0.604, AURSAD MSE: 0.798

**Key insight**: Making the model bigger/training longer doesn't help. The problem is fundamental - cross-channel correlations are domain-specific and don't transfer between robots doing different tasks (pick-and-place vs screwdriving).

## Next Step: Run Exp05

`exp05_patchtst_transfer.py` is ready to run. It implements:

1. **Channel-independent patching (PatchTST-style)** - forecast each channel independently, avoiding cross-channel coupling
2. **Smaller stride (64 vs 128)** - more training windows
3. **Stronger regularization** - dropout=0.2, weight_decay=0.05

**Hypothesis**: Per-channel temporal dynamics are more universal across robots than cross-channel correlations which are task-specific.

## Files to Reference

- `autoresearch/experiments/exp05_patchtst_transfer.py` - next experiment
- `autoresearch/results/exp04_analysis.md` - exp04 results
- `autoresearch/CROSS_MACHINE_RESEARCH.md` - full research state
- `autoresearch/DATASET_ANALYSIS_REPORT.md` - dataset transferability analysis (score: 0.94)

## Task

1. Run exp05: `python autoresearch/experiments/exp05_patchtst_transfer.py`
2. Monitor training and analyze results
3. If transfer ratio ≤ 1.5: run multi-seed validation
4. If still failing: diagnose why and propose next iteration
5. Update research docs with findings

The training should run ~20-30 minutes. Log everything for analysis.

---
