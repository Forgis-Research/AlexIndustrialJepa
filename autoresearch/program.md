# IndustrialJEPA Autoresearch

## Goal

Achieve SOTA on time series anomaly detection using JEPA-style latent prediction.

## Current Best

- **val_loss: 0.028** (JEPA world model, 20 epochs)
- Beat temporal baseline (0.036) by 22%

## Target Metrics

Primary: `val_loss` (lower is better) - JEPA prediction error in latent space
Secondary: `anomaly_f1` (higher is better) - F1 on held-out anomaly detection

## Constraints

- **5-minute training runs** for fast iteration
- Single GPU (A10G, 23GB VRAM)
- Batch size: 32-128 (adjust for memory)
- Dataset: AURSAD (industrial robot anomaly detection)

## Architecture

The model predicts future robot states in latent space:

```
effort(t) ──► Encoder ──► z(t) ──► Predictor ──► z_pred(t+1)
                                       │
effort(t+1) ──► EMA Encoder ──► z_target(t+1)
                                       │
                        Loss = ||z_pred - z_target||²
```

Key insight: Physics-grounded (setpoint causes effort via F=ma).

## Ideas to Explore

### Masking Strategies
- [ ] Random masking (current)
- [ ] Block masking (contiguous time blocks)
- [ ] Multi-scale masking (different temporal resolutions)
- [ ] Causal masking (only predict future)

### Architecture Variants
- [ ] Patch sizes: 8, 16, 32, 64 timesteps
- [ ] Encoder depth: 2, 4, 6, 8 layers
- [ ] Predictor depth: 1, 2, 3 layers (should be smaller than encoder)
- [ ] Hidden dims: 128, 256, 512
- [ ] Attention heads: 4, 8, 16

### Loss Functions
- [ ] MSE (current)
- [ ] Smooth L1
- [ ] Cosine similarity
- [ ] Contrastive + predictive hybrid
- [ ] Association discrepancy (from Anomaly Transformer)

### Augmentations
- [ ] Time warping
- [ ] Magnitude scaling
- [ ] Jittering
- [ ] Frequency masking (FFT domain)

### Advanced Ideas (from literature)
- [ ] FFT branch (TimesNet style)
- [ ] Channel independence (PatchTST style)
- [ ] Multi-scale temporal encoding
- [ ] Learnable positional encoding vs sinusoidal

## File Structure

```
autoresearch/
├── program.md      # This file (human-edited instructions)
├── train.py        # Training script (agent-edited)
├── prepare.py      # Data preparation (not modified)
└── results/        # Experiment outputs
```

## How to Run

```bash
cd autoresearch
python prepare.py          # Download and prepare data (~2 min)
python train.py             # Run experiment (~5 min)
```

## Evaluation Protocol

1. Train on healthy data only (one-class)
2. Measure val_loss on healthy validation set
3. Compute anomaly scores on test set (includes faults)
4. Report: val_loss, anomaly_auroc, anomaly_f1

## Rules for Agent

1. Only modify `train.py`
2. Each experiment must complete in ~5 minutes
3. Log all metrics to stdout in format: `metric_name: value`
4. Save best model to `results/best_model.pt`
5. Try one idea at a time, measure impact
6. If val_loss improves, keep the change; otherwise revert

## Leaderboard

| Experiment | val_loss | Notes |
|------------|----------|-------|
| baseline_jepa | 0.028 | Current best, 20 epochs |
| ... | ... | ... |
