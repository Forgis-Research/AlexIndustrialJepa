# ETTh1 Experiment Log

Date: 2026-03-22

## Setup
- Dataset: ETTh1 (17,420 rows, 7 channels, hourly)
- Split: 8640 train / 2880 val / 2880 test (standard ETT split from Informer paper)
- Lookback: 96 timesteps
- Normalization: per-channel zero-mean unit-variance (fit on train)
- All results: 3 seeds (42, 123, 456), test set MSE/MAE

## Baselines (Task 2)

| Model | H=96 MSE | H=96 MAE | H=192 MSE | H=336 MSE | H=720 MSE |
|-------|----------|----------|-----------|-----------|-----------|
| Persistence | 1.6307 | 0.8227 | 1.6819 | 1.7086 | 1.7885 |
| Linear | **0.5714** | **0.5283** | **0.7799** | **0.9704** | 1.1948 |
| MLP (1-layer) | 0.7276 | 0.6405 | 0.9583 | 1.0381 | **1.0927** |

**Observations:**
- Linear is the strongest trivial baseline for H={96,192,336}. This is consistent with the DLinear paper's finding that a simple linear model is surprisingly competitive.
- MLP overtakes Linear only at H=720 (long horizon needs more capacity, but also more prone to overfitting at short horizons).
- Persistence is terrible (MSE ~1.63-1.79), confirming the data has meaningful dynamics.

## JEPA Experiments (Task 3)

Architecture: Patch embedding (patch_len=16) -> Transformer encoder (3 layers, d=128, 4 heads) -> Latent projection (dim=64) -> MLP predictor -> Linear decoder. EMA target encoder (momentum=0.996). ~890K trainable params at H=96.

| # | Mode | H=96 MSE | H=96 MAE | H=192 MSE | H=336 MSE | H=720 MSE | Notes |
|---|------|----------|----------|-----------|-----------|-----------|-------|
| 1 | JEPA + Supervised | 0.899 | 0.717 | 0.950 | 1.004 | 1.007 | Both losses |
| 2 | Supervised only | 0.900 | 0.736 | 0.895 | 0.963 | 1.007 | No JEPA loss |
| 3 | JEPA only | 1.282 | 0.851 | 1.284 | 1.277 | 1.275 | Decoder not trained by supervision |

**Key findings:**
1. **JEPA loss provides no benefit.** EXP 1 vs EXP 2 are statistically indistinguishable. At H=192 and H=336, supervised-only is actually slightly better.
2. **JEPA-only (EXP 3) is near-useless.** MSE ~1.28 across all horizons — the decoder receives no gradient signal from supervision, so it cannot map latent predictions to meaningful forecasts. The latent loss decreases during training but this doesn't translate to forecast quality.
3. **All transformer-based models lose badly to the trivial Linear baseline.** Linear gets 0.571 at H=96; our best transformer gets 0.899. That's 57% worse.

## Published SOTA (for reference)

| Model | H=96 MSE | H=192 MSE | H=336 MSE | H=720 MSE | Source |
|-------|----------|-----------|-----------|-----------|--------|
| PatchTST | ~0.370 | ~0.383 | ~0.396 | ~0.419 | Nie et al. 2023 |
| DLinear | ~0.375 | ~0.405 | ~0.439 | ~0.472 | Zeng et al. 2023 |
| iTransformer | ~0.386 | ~0.384 | ~0.396 | ~0.428 | Liu et al. 2024 |

## Diagnosis Experiments (Task 5)

After the vanilla JEPA failed, we diagnosed the overfitting problem with three targeted experiments:

| # | Model | Params | H=96 MSE | H=192 MSE | H=336 MSE | H=720 MSE | Notes |
|---|-------|--------|----------|-----------|-----------|-----------|-------|
| 4 | **CI-Transformer** | 105K | **0.450** | **0.502** | **0.561** | **0.673** | Channel-independent, d=64, 2 layers |
| 5 | Tiny-Transformer | 142K | 0.713 | 0.819 | 0.910 | 0.943 | d=32, 1 layer, channel-mixing |
| 6 | DLinear | 19K | 0.480 | 0.538 | 0.579 | 0.683 | Trend-seasonal decomposition |

### Key finding: Channel independence is the single biggest improvement

The channel-independent transformer (EXP 4) achieves **0.450 MSE at H=96** — a 50% improvement over our vanilla model (0.899) and 21% better than the Linear baseline (0.571). This is the PatchTST insight: with only 7 channels, cross-channel mixing provides almost no value and dramatically increases overfitting.

Comparison to the naive Linear baseline (0.571): CI-Transformer beats it at every horizon.

| Horizon | Linear | CI-Transformer | Improvement |
|---------|--------|----------------|-------------|
| 96 | 0.571 | 0.450 | -21% |
| 192 | 0.780 | 0.502 | -36% |
| 336 | 0.970 | 0.561 | -42% |
| 720 | 1.195 | 0.673 | -44% |

### DLinear performs similarly to CI-Transformer

Our DLinear implementation (0.480 at H=96) is close to CI-Transformer (0.450). Both are much better than channel-mixing models. However, our DLinear is worse than published DLinear (~0.375), suggesting there's still room for tuning.

### Tiny model doesn't help if architecture is wrong

EXP 5 (Tiny-Transformer) shrinks the model but keeps channel-mixing. It improves over vanilla (0.713 vs 0.899) but is still much worse than CI-Transformer (0.450). **The architecture design (channel-independence) matters more than model size.**

## Gap Analysis (Updated)

| Comparison | H=96 | H=192 | H=336 | H=720 |
|------------|------|-------|-------|-------|
| Our best (CI-Transformer) | **0.450** | **0.502** | **0.561** | **0.673** |
| Our DLinear | 0.480 | 0.538 | 0.579 | 0.683 |
| Our Linear baseline | 0.571 | 0.780 | 0.970 | 1.195 |
| PatchTST SOTA | 0.370 | 0.383 | 0.396 | 0.419 |
| Gap: CI-Trans vs SOTA | +22% | +31% | +42% | +61% |

We've closed the gap significantly (from 2.4x worse to 1.2-1.6x worse), but there's still meaningful distance to SOTA, especially at longer horizons.

## Published SOTA (for reference)

| Model | H=96 MSE | H=192 MSE | H=336 MSE | H=720 MSE | Source |
|-------|----------|-----------|-----------|-----------|--------|
| PatchTST | ~0.370 | ~0.383 | ~0.396 | ~0.419 | Nie et al. 2023 |
| DLinear | ~0.375 | ~0.405 | ~0.439 | ~0.472 | Zeng et al. 2023 |
| iTransformer | ~0.386 | ~0.384 | ~0.396 | ~0.428 | Liu et al. 2024 |

## Honest Assessment

### What worked
1. **Channel-independent processing** is the single most impactful change. It reduces effective parameters per-channel while giving 7x more training examples. This matches the PatchTST finding.
2. **Patch-based embeddings** with a shared transformer encoder and linear head is a clean, effective architecture.
3. **DLinear's trend-seasonal decomposition** is competitive and trains in seconds.

### What didn't work
1. **JEPA loss provides zero forecasting benefit** in any configuration. The latent-space prediction objective doesn't translate to better forecasts.
2. **Channel-mixing** is actively harmful on a 7-channel dataset with only 8640 training points. The model overfits to spurious cross-channel correlations.
3. **JEPA-only training** produces useless forecasts (MSE ~1.28, near-persistence baseline).

### Remaining gap to SOTA (~22% at H=96)
Possible causes:
1. **Missing RevIN** — Reversible instance normalization is standard in PatchTST/iTransformer
2. **No instance-wise norm at inference** — our model normalizes globally, not per-instance
3. **Head design** — PatchTST uses a flatten+linear head with careful attention masking
4. **Hyperparameter tuning** — published results are heavily tuned; ours are first-pass
5. **Data preprocessing** — potential differences in split boundaries or normalization

### What to try next
1. **Add RevIN** to CI-Transformer — likely the biggest remaining easy win
2. **Add JEPA as pretraining** — pretrain channel-independent encoder with JEPA, then fine-tune with supervised loss. This is the principled way to use JEPA.
3. **Tune hyperparameters** — patch_len, d_model, n_layers, learning rate, dropout
4. **Instance normalization** — normalize each input window independently at inference
5. **Move to larger datasets** (Weather, Electricity) where JEPA's representation learning may add value

### Bottom line
**The vanilla JEPA approach cannot beat a linear model on ETTh1.** But a channel-independent transformer achieves 0.450 MSE at H=96, beating our linear baseline (0.571) and approaching published DLinear (0.375). The gap to PatchTST SOTA (0.370) is ~22%, likely closeable with RevIN and tuning.

**For the JEPA research direction**: JEPA should be used as a *pretraining* method (self-supervised representation learning), not as a combined training objective. The next step is to pretrain a channel-independent encoder with JEPA on unlabeled data, then fine-tune for forecasting. This is where JEPA could add value — especially for transfer learning to industrial datasets where labeled data is scarce.
