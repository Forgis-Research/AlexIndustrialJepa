---
name: IndustrialJEPA Project Context
description: Complete project state including V6 (CWRU→Paderborn JEPA transfer) and V7 baseline establishment on Forgis/Mechanical-Components HF dataset
type: project
---

IndustrialJEPA is a research project on self-supervised learning (JEPA) for industrial time series.

## Mechanical-JEPA V6 (Completed 2026-04-04)

### CORRECTED V6 Results (JSON-backed, 3-seed, fixed Paderborn API bug)

| Method | CWRU F1 | Paderborn F1 | Transfer Gain | Source |
|--------|---------|-------------|---------------|--------|
| CNN Supervised | 1.000 ± 0.000 | 0.987 ± 0.005 | +0.457 ± 0.020 | transfer_baselines_v6_final.json |
| **JEPA V2 (ours)** | 0.773 ± 0.018 | **0.900 ± 0.008** | **+0.371 ± 0.026** | transfer_baselines_v6_final.json |
| Transformer Supervised | 0.969 ± 0.026 | 0.673 ± 0.063 | +0.144 ± 0.044 | transfer_baselines_v6_final.json |
| Random Init | ~0.412 | 0.529 ± 0.024 | 0.000 | transfer_baselines_v6_final.json |

**KEY FINDING: JEPA@N=10 (0.735) > Transformer@N=all (0.689)** — p=0.034, d=0.92.
This is the primary publishable result using local CWRU (134 samples) and local Paderborn datasets.

### JEPA V2 Architecture (jepa_v2.py)
- Encoder: 4-layer Transformer, d=512, 4 heads, sinusoidal PE
- Input: (B, 3, 4096) → 16 patches of 256 samples
- Mask ratio: 0.625, EMA momentum=0.996, L1 loss + variance reg (lambda=0.1)
- 5 critical components verified by ablation (sine PE, high mask ratio, L1, var reg, EMA)
- Best checkpoint: `mechanical-jepa/checkpoints/jepa_v2_20260401_003619.pt`

### Local Datasets (V6)
- CWRU: `mechanical-jepa/data/bearings/` (134 samples, 4 classes, 12kHz)
- Paderborn: `datasets/data/paderborn/` (K001/KA01/KI01, 64kHz → resample to 20kHz)
- Use `create_paderborn_loaders` (NOT PaderbornDataset constructor directly)

---

## V7: Baseline Establishment (Completed 2026-04-07)

### Dataset: Forgis/Mechanical-Components (HuggingFace)
- ~12,000 samples, 9.5 GB, 16 sources, 5 component types
- **USE PARQUET DIRECTLY** — `load_dataset()` fails due to fsspec glob bug
- Local cache: `/tmp/hf_cache/bearings/` (download with hf_hub_download)
- Signal structure: `row['signal']` is array-of-arrays (n_channels, n_samples)
- Signal: `np.array(row['signal'])[0]` gives channel 0

### Key signal properties (for processing)
- FEMTO: 2560 samples at 25600Hz = 0.1s (very short)
- XJTU-SY: 32768 samples at 25600Hz = 1.28s
- CWRU (HF): 120k-485k samples at 12000Hz = 10-40s
- Paderborn (HF): 256k samples at 64000Hz = 4s
- MAFAULDA: 25600 samples at 50000Hz = 0.512s (short)
- rul_percent = 1 - episode_position exactly (definitional, not predictive)

### V7 Baseline Results (4 task families)

**Task 1: Cross-Domain Fault Classification**
- Setup: Train CWRU+MAFAULDA+SEU → Test Ottawa+Paderborn
- Best: Random Forest F1=0.193 ± 0.021
- Without MAFAULDA: RF F1=0.216 (MAFAULDA hurts — noisy source)
- In-domain: CWRU=0.725, Paderborn=0.872, Ottawa=0.828
- JEPA target: F1 > 0.30 cross-domain

**Task 2: Anomaly Detection (FEMTO)**
- Best: Kurtosis threshold AUROC=0.779 (beats IsolationForest=0.710)
- CNN Autoencoder: 0.414 (poor on 0.1s FEMTO signals)
- JEPA target: AUROC > 0.85

**Task 3: HI Forecasting (FEMTO RMS)**
- H=1: Last-value RMSE=0.351, Random Forest=0.311 (best)
- ARIMA completely fails on non-stationary degradation (RMSE=1.82)
- Kurtosis much harder than RMS to forecast
- JEPA target: RMSE < 0.25 at H=1

**Task 4: RUL Estimation (FEMTO+XJTU-SY)**
- Best real baseline: XGBoost RMSE=0.212
- Constant mean: 0.290 (strong trivial since RUL~uniform on [0,1])
- JEPA target: RMSE < 0.17

### Key Architecture Insight
- rul_percent = 1 - episode_position is definitional → oracle cheat (RMSE=0)
- MAFAULDA: all fault types have nearly identical kurtosis (2.6-2.9) → not good for classification training
- Features for anomaly detection: kurtosis alone > complex ML (physically motivated)
- Deep models don't beat feature-based with limited data (40 CWRU + 140 SEU train samples)

### Code Structure (V7 baselines)
- Baselines dir: `mechanical-jepa/baselines/`
- Data loading: `data_utils.py` (uses local `/tmp/hf_cache/bearings/`)
- Features: `features.py` (18 features: time-domain, freq-domain, envelope)
- Classification: `run_classification_baselines.py`
- Anomaly: `run_anomaly_baselines.py`
- Forecasting: `run_forecasting_baselines.py`
- In-domain: `run_within_source_clf.py`
- Results: `baselines/results/*.json`
- Notebook: `notebooks/07_baseline_establishment.ipynb`
- Figures: `notebooks/plots/fig_baseline_*.{pdf,png}`

**Why:** V7 establishes the bar that JEPA-V7 needs to beat for publication. Cross-domain classification is the primary publishable target.
**How to apply:** For any new JEPA model evaluation, compare against RF cross-domain F1=0.216 (no MAFAULDA) and Kurtosis AUROC=0.779.

---

## V8: JEPA-Based RUL% Prediction (Completed 2026-04-08)

### Task: Bearing RUL% Prediction on FEMTO (16 eps) + XJTU-SY (7 eps)

**Formulation:** RUL(t) = 1 - t/T_failure (linear), episode-based train/test split (no leakage).  
**Primary metric:** RMSE on normalized RUL (0=new, 1=failure at test snapshots).

### In-Domain Results (18 train, 5 test episodes, 5 seeds)

| Method | RMSE | vs Elapsed-time |
|--------|------|----------------|
| Elapsed-time baseline | 0.224 | baseline |
| JEPA+LSTM (ours) | 0.189 ± 0.015 | +15.8%, p=0.010 |
| Random JEPA+LSTM (ablation) | 0.221 ± 0.008 | - |
| Handcrafted+MLP | 0.085 ± 0.004 | +62.1% |
| Transformer+HC | 0.070 ± 0.006 | +68.8% |

**JEPA beats random encoder (p=0.010) but NOT handcrafted (p=0.40).**

### Cross-Domain Results (FEMTO -> XJTU-SY, 10 seeds)

| Method | RMSE | vs Elapsed |
|--------|------|-----------|
| Elapsed-time | 0.367 | baseline |
| JEPA+LSTM | 0.279 ± 0.006 | +23.9% |
| Temporal Contrastive+LSTM | 0.227 ± 0.015 | +38.1%, p<0.001 |

### Key Finding: Why Temporal Contrastive Beats JEPA Cross-Domain

Root cause: spectral centroid shift is the primary bearing degradation indicator (r=0.585 with RUL).

- JEPA PC1 correlation with spectral_centroid: **0.071** (doesn't capture it)
- Contrastive PC1 correlation with spectral_centroid: **0.856** (captures it strongly)

**Mechanism:** JEPA objective (predict masked patches) learns waveform texture. Temporal contrastive objective (adjacent=positive, distant=negative) forces learning what changes over bearing life, which is the spectral centroid. This shift is universal across FEMTO and XJTU-SY, so it transfers.

### JEPA Pretraining Instability

All JEPA configs show loss minimum at epoch 2-5, then oscillation. Root cause: heterogeneous 8-source data (FEMTO, XJTU, CWRU, IMS, MAFAULDA, Paderborn, SGV, UOC) with very different signal characteristics causes EMA target encoder to drift. Use epoch-2 checkpoint (val_loss=0.01662).

### Checkpoints (v8/)

- `jepa_v8_best.pt`: epoch=2, val=0.01662 (standard JEPA)
- `jepa_v8_contrastive_best.pt`: epoch=97 (narrow temporal contrastive, best cross-domain)
- `jepa_v8_fft_best.pt`: epoch=2, val=0.01492 (2-channel FFT, similar downstream)
- `jepa_v8_contrastive_broad_best.pt`: broader data, WORSE downstream

### Files (v8/)

- `data_pipeline.py`: loads 33,939 pretrain windows + 23 RUL episodes, episode splits
- `jepa_v8.py`: 4.0M encoder, 16 patches, embed_dim=256, `get_embeddings()` → (B,256)
- `rul_model.py`: RULLSTM, HandcraftedLSTM, CNNGRUMHAEncoder, TransformerRUL
- `rul_baselines.py`: 11 methods, 3+ seeds, JSON output
- `evaluate.py`: 4-way cross-dataset transfer evaluation
- `analyze.py`: encoder quality, PCA, trajectory plots
- `results/`: all JSON result files
- `notebooks/08_rul_jepa.ipynb`: documentation notebook

**How to apply:** For RUL tasks, temporal contrastive > JEPA for cross-domain transfer.  
For in-domain RUL, handcrafted features (spectral centroid) still beat JEPA.

---

## V9: Data-First JEPA (Completed 2026-04-09)

### Goal: Fix V8 pretraining instability via dataset compatibility analysis

### Key Results (31 episodes: 16 FEMTO + 15 XJTU-SY, 24 train / 7 test, 5 seeds)

| Method | RMSE | Notes |
|--------|------|-------|
| Elapsed time | 0.224 | trivial baseline |
| V9 JEPA+LSTM (all_8) | 0.0852 ± 0.0014 | best_epoch=2 (still early) |
| V9 JEPA+LSTM (compatible_6) | 0.0873 ± 0.0018 | best_epoch=3 |
| V9 JEPA[block masking]+LSTM | 0.0886 ± 0.0049 | -1.5% vs random |
| V9 JEPA[dual-channel]+LSTM | 0.1119 ± 0.0057 | -28.2% (overfits) |
| V9 JEPA+Probabilistic-LSTM | 0.0868 ± 0.0023 | PICP@90%=0.910 (calibrated!) |

### Root Cause of V8 Instability (CONFIRMED)

MAFAULDA centrifugal pump data had spectral centroid 173Hz vs FEMTO's 2453Hz (14x difference).
KL divergence MAFAULDA vs FEMTO = 3.04 (next worst: 1.47). Instance normalization equalizes
amplitude but NOT spectral shape. Removing MAFAULDA+MFPT shifts best_epoch 2→3 and improves
val_loss 0.0161→0.0140 but does NOT fully solve multi-source JEPA instability.

### Dataset Compatibility Protocol

Resample to 12.8kHz, check vs FEMTO reference:
1. PSD KL divergence < 2.0
2. |spectral_centroid - 2453 Hz| < 1500 Hz
3. mean kurtosis < 8.0 and std < 12.0

Compatible sources: femto, xjtu_sy, cwru, ims, paderborn, ottawa
Marginal: mfpt (kurtosis std=17)
Incompatible: mafaulda (centroid=173Hz, KL=3.04)

### Key Findings

1. V9 RMSE improvement vs V8 (0.085 vs 0.189) is driven by more training episodes (24 vs 18),
   NOT model improvements. Apples-to-apples: compatible_6 vs all_8 = 0.087 vs 0.085 (p>0.05).
2. LSTM beats TCN-Transformer in small-data regime (24 episodes). TCN-Transformer RMSE=0.140 vs LSTM 0.085.
3. Probabilistic LSTM (Gaussian NLL) gives well-calibrated uncertainty PICP@90%=0.910 at near-zero
   accuracy cost (RMSE: 0.0873 → 0.0868). Enables P(RUL<threshold) deployment decisions.
4. Deviation-from-baseline features fail for short-lifetime XJTU-SY (K=10 baseline already contaminated).
5. Dual-channel raw+FFT encoder WORSE downstream (fewer pretraining windows, overfitting).

### Files (v9/)
- `experiments/v9/EXPERIMENT_LOG.md`: 12 experiments, clean (no duplicates)
- `experiments/v9/RESULTS.md`: full results table with SOTA comparison
- `experiments/v9/run_e1_light.py`: memory-efficient pretraining (loads 16k windows to avoid OOM)
- `experiments/v9/run_downstream.py`: downstream eval + all G.1 plots
- `checkpoints/jepa_v9_compatible_6.pt`: best pretrain encoder (best_ep=3, val=0.0140)
- `checkpoints/jepa_v9_block_masking.pt`: block masking encoder (best_ep=4, val=0.0173)
- `checkpoints/jepa_v9_dual_channel.pt`: dual-channel encoder (best_ep=4, val=0.0155, n_channels=2)
- `notebooks/09_v9_data_first.qmd`: Quarto notebook with all results hardcoded

**Why:** V9 establishes dataset compatibility as a prerequisite for stable multi-source JEPA.
**How to apply:** Always run compatibility check before adding new sources. For next session,
focus on getting more training episodes (50+) to enable TCN-Transformer and reduce noise in PICP estimation.
