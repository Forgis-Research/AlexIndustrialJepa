# Exp04 Results: Improve Many-to-1 Forecasting Transfer

**Date:** 2026-03-21
**Status:** Transfer ratio unchanged at 1.72 — combined config (E_combined) did not help.

## Configuration (E_combined - best of all ablations)
- Epochs: 40
- Hidden dim: 256, Layers: 4
- Max Voraus episodes: 1000
- CNC: all episodes (loaded as third source)
- RevIN: enabled
- LR: 0.0001, Batch: 64, Dropout: 0.1
- Parameters: 3.27M

## Training Log
- ~12-13 it/s with 336 batches/epoch ≈ 27s/epoch
- 40 epochs completed in ~18 minutes
- CNC loaded successfully as third source
- Train loss converged: 0.712 → 0.302
- Val loss converged: 0.668 → 0.306

## Key Results
| Metric | Value |
|--------|-------|
| Source Voraus MSE | 0.324 |
| Source CNC MSE | 0.604 |
| Target AURSAD MSE | 0.798 |
| Avg Source MSE | 0.464 |
| **Transfer Ratio** | **1.72** |
| Best Val Loss | 0.306 |
| Passed (≤1.5) | No |

## Analysis
- Voraus fits well (MSE 0.324), CNC fits poorly (MSE 0.604 — tiny dataset)
- Target AURSAD MSE 0.798 is much higher than source
- Problem is **fundamental**: Voraus (pick-and-place) dynamics don't predict AURSAD (screwdriving) dynamics well
- Bigger model + more data + longer training = no improvement
- Cross-channel correlations are domain-specific and don't transfer

## Next Steps (Exp05)
1. Channel-independent patching (PatchTST-style) — each channel independently, local patterns transfer better
2. Smaller stride (more training windows)
3. Stronger regularization to prevent overfitting to source dynamics

Hypothesis: By forecasting channels independently, the model learns per-channel temporal dynamics (more universal across robots) rather than cross-channel correlations (task-specific).
