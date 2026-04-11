# Mechanical-JEPA: Trajectory JEPA for industrial RUL

Self-supervised representation learning for Remaining Useful Life (RUL) prediction via Joint Embedding Predictive Architectures. The project started on heterogeneous bearing vibration datasets (v1-v10) and pivoted in v11 to C-MAPSS turbofan trajectories, where the current best result is the first SSL method to match the AE-LSTM public reference on FD001 without using failure-time labels at pretraining.

## Iteration history

| Version | Focus | Status | Headline |
|---------|-------|:------:|----------|
| v8 | Hybrid JEPA + handcrafted features on bearings (FEMTO, XJTU-SY) | done | Hybrid JEPA+HC +75.5% vs elapsed-time baseline |
| v9 | Data-first: compatibility analysis + TCN-Transformer | done | Diagnoses why v1-v7 JEPA collapsed; shifts to spectral features |
| v10 | Trajectory JEPA v1 (bearing trajectories, no HC ablation) | done | Establishes the trajectory-prediction objective used in v11 |
| v11 | Trajectory JEPA on C-MAPSS FD001-FD004 (current) | primary result | V2 E2E 13.80 RMSE on FD001 last-window; first SSL to match AE-LSTM 13.99 |
| v12 | Verification gate for v11 (not an improvement run) | planned overnight | Phase 0 reconciles v11 RMSE against prediction-trajectory diagnostic + feature-regressor lower bound |

See `experiments/v{N}/EXPERIMENT_LOG.md` and `experiments/v{N}/RESULTS.md` for per-version details. The current v12 plan (`experiments/v12/OVERNIGHT_PROMPT.md`) is a verification gate, not a polish run — it exists because the v11 prediction-trajectory plot shows the model outputting a near-constant ~92 cycles per engine, and that inconsistency with the 13.80 RMSE needs to be reconciled before any further experiments are meaningful.

## Architecture

```
Raw vibration (1024 samples @ 12.8kHz)
        |
  [Patch split: 16 x 64]
        |
  [Context encoder: ViT 4L, d=256]
        |
  [Frozen JEPA embeddings: 256-dim per snapshot]
        |
  [Temporal head: LSTM/TCN-Transformer]
        |
  RUL% prediction per snapshot
```

## Best Results (V8, 5 seeds)

| Method | RMSE | vs Elapsed-Time-Only |
|--------|------|---------------------|
| Elapsed time only (baseline) | 0.224 | -- |
| JEPA + LSTM | 0.189 +/-0.015 | +15.8% (p=0.010) |
| HC + LSTM | 0.177 +/-0.016 | +21.2% |
| Transformer + HC | 0.070 +/-0.006 | +68.9% |
| Hybrid JEPA+HC | 0.055 +/-0.004 | +75.5% |

Cross-dataset transfer (FEMTO->XJTU-SY, 10 seeds):
- Temporal Contrastive + LSTM: RMSE=0.227 +/-0.015 (+38.1% vs elapsed time)
- JEPA + LSTM: RMSE=0.280 +/-0.007 (+23.8%)

## Quick Start

```bash
# 1. Dataset analysis (start here -- understand the data)
python data/analysis/dataset_compatibility.py

# 2. JEPA pretraining
python pretraining/train.py --config data/configs/pretrain_compatible.yaml

# 3. RUL downstream evaluation
python downstream/rul/train.py --encoder checkpoints/jepa_best.pt

# 4. All baselines
python downstream/rul/baselines.py
```

## Directory Structure

```
mechanical-jepa/
data/
    analysis/         # Dataset compatibility analysis + plots
    loader.py         # Data loading from HuggingFace cache
    preprocessing.py  # Resample, filter, normalize, window
    registry.py       # Dataset metadata (SR, channels, domain)
    configs/          # YAML configs for source selection
pretraining/
    jepa.py           # JEPA architecture (ViT encoder + predictor + EMA)
    train.py          # Pretraining loop
    masking.py        # Masking strategies (random, block, multi-block)
downstream/
    rul/
        models.py     # LSTM, TCN-Transformer heads
        baselines.py  # All 11 baselines
        train.py      # RUL training loop
        evaluate.py   # RMSE, calibration, uncertainty metrics
analysis/
    embeddings.py     # PCA, t-SNE, Spearman correlation
    plots/            # Generated figures
notebooks/            # Quarto walkthroughs
experiments/
    v8/               # V8 results + experiment log
    v9/               # V9 results + experiment log
archive/              # Legacy scripts
```

## Key Findings

1. JEPA pretraining problem: Loss collapses to minimum at epoch 2/100.
   Root cause: 8 heterogeneous sources with incompatible spectral characteristics.
   V9 addresses this with dataset compatibility analysis.

2. JEPA is complementary: JEPA alone underperforms expert handcrafted features,
   but Hybrid JEPA+HC beats either alone by >20%.

3. Contrastive wins cross-domain: Temporal contrastive pretraining achieves
   18.8% better RMSE than JEPA for cross-dataset transfer.

4. Spectral centroid is the key feature: Max correlation with RUL is 0.585
   (spectral centroid) vs 0.144 (JEPA embedding).

## Datasets

- FEMTO (PHM 2012): 17 episodes, 25.6kHz, ball bearings
- XJTU-SY (2019): 15 episodes, 25.6kHz, ball bearings
- IMS (NASA): 4 bearings, 20.48kHz, roller bearings
- CWRU: 40 signals, 12kHz, motor bearings (classification only)
- MFPT: 23 signals, 48.8kHz (classification only)
- Paderborn: 32 signals, 64kHz (classification only)
- Ottawa: 3 episodes, 42kHz (run-to-failure)
- MAFAULDA: 1951 signals, 50kHz (centrifugal pump)

## Notebooks

All walkthroughs are Quarto (`.qmd`). Rendered `.html` outputs are published to GitHub Pages at https://forgis-research.github.io/IndustrialJEPA/ .

- `notebooks/08_rul_jepa.qmd` — v8 complete walkthrough (bearing RUL, JEPA + HC hybrid)
- `notebooks/09_v9_data_first.qmd` — v9: dataset compatibility analysis + TCN-Transformer
- `notebooks/10_v10_trajectory_jepa.qmd` — v10: Trajectory JEPA v1 on bearings
- `notebooks/11_v11_cmapss_trajectory_jepa.qmd` — v11: Trajectory JEPA on C-MAPSS (current primary result)

Old pre-Quarto notebooks (v1-v7) have been moved to `archive/old-notebooks/` for reference. Publish a walkthrough with:

```bash
cd notebooks
quarto publish gh-pages <file>.qmd --no-prompt --no-browser
```
