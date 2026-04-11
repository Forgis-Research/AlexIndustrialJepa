# STAR Replication Specification

## Reference

Fan et al., "STAR: Spatio-Temporal Attention-based Regression for Turbofan Remaining Useful Life Prediction", Sensors 2024.

## Data Pipeline

### Sensor Selection (14 of 21)

1-indexed sensor names: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21.

In the raw file format `[engine_id, cycle, op1, op2, op3, s1..s21]` (26 columns), 0-indexed column positions:
- s2 = col 5, s3 = col 6, s4 = col 7, s7 = col 10, s8 = col 11, s9 = col 12
- s11 = col 14, s12 = col 15, s13 = col 16, s14 = col 17, s15 = col 18
- s17 = col 20, s20 = col 23, s21 = col 24

### Normalization

Min-max per sensor on training set only: `(x - x_min) / (x_max - x_min + 1e-8)`.
Applied identically to test data using train statistics.

For FD002/FD004 (multi-condition datasets), a per-operating-condition ablation is also
implemented using KMeans with 6 clusters on the 3 op_setting columns.

### RUL Labels

Piecewise linear with cap = 125:
`rul[t] = min(T - t, 125)` for t in 0..T-1 where T is the engine's total cycle count.

### Sliding Windows

Window length per subset (see hyperparams table below), stride = 1.
Label = RUL at the LAST cycle of the window.
For test engines: ONLY the last window per engine is used.
If engine has fewer cycles than window_length: left-pad by repeating the first cycle.

### Train/Val Split

15% of training engines held out for validation (by engine, not by window).
Split is seeded deterministically per run seed.

## Architecture (STAR)

### DimensionWisePatchEmbed

Input `(B, T, D)` D=14, T=window length.
Reshape into K = T/L patches per sensor (patch_length L=4 for all subsets).
Apply shared `nn.Linear(L, d_model)`.
Add learnable positional embedding `(K, D, d_model)`.
Output `(B, K, D, d_model)`.

### TwoStageAttentionEncoder

Each encoder block:
- Stage 1 (temporal): reshape to `(B*D, K, d_model)`, standard PreNorm Transformer layer (LN->MHA->residual->LN->FFN->residual), reshape back.
- Stage 2 (sensor-wise): reshape to `(B*K, D, d_model)`, same pattern, reshape back.
- FFN hidden dim = 4 * d_model. Dropout = 0.1.

### PatchMerging

Takes x of shape `(B, K, D, d_model)`.
Pairs even/odd patches along K dim: `x[:, 0::2]` and `x[:, 1::2]`.
If K is odd, the last patch is DROPPED (truncated), not padded.
Concatenates along feature dim -> `(B, K//2, D, 2*d_model)`.
Projects with `nn.Linear(2*d_model, d_model)` -> `(B, K//2, D, d_model)`.

### TwoStageAttentionDecoder

Each decoder block:
- Self-attention two-stage (same structure as encoder on decoder sequence).
- Cross-attention two-stage: temporal cross-attn (Q from decoder, K/V from encoder features), then sensor-wise cross-attn.
- First decoder layer input: fixed sinusoidal PE matching deepest encoder scale, broadcast to batch.

### STAR Full Model

Forward pass:
1. PatchEmbed: `(B, T, D)` -> `(B, K, D, d_model)`.
2. Encoder: n_scales encoder blocks with (n_scales-1) PatchMerging blocks between them. Save output at each scale.
3. Decoder: start from sinusoidal PE at deepest scale. For scale s = n_scales-1 down to 0:
   - Decoder block (self-attn + cross-attn to encoder[s]).
   - If s > 0: upsample by repeat-interleave(2) along K dim to match scale s-1.
4. Per-scale prediction heads: flatten `(B, K_s, D, d_model)` -> MLP -> scalar.
5. Final: concat per-scale scalars -> final MLP -> single scalar in [0, 1].
6. Multiply by 125 and clamp to [0, 125] during evaluation.

### Loss

MSE on normalized RUL: `loss = MSE(pred_norm, target / 125)` where pred_norm is raw model output.

## Per-Subset Hyperparameters (Table 4)

| Subset | lr     | batch | window | n_scales | d_model | n_heads | patch_L | K  |
|--------|--------|-------|--------|----------|---------|---------|---------|-----|
| FD001  | 0.0002 | 32    | 32     | 3        | 128     | 1       | 4       | 8  |
| FD002  | 0.0002 | 64    | 64     | 4        | 64      | 4       | 4       | 16 |
| FD003  | 0.0002 | 32    | 48     | 1        | 128     | 1       | 4       | 12 |
| FD004  | 0.0002 | 64    | 64     | 4        | 256     | 4       | 4       | 16 |

## Training Protocol

- Optimizer: Adam, lr as above.
- Scheduler: CosineAnnealingLR over max_epochs.
- Early stopping: val RMSE patience = 20.
- Max epochs: 200.
- 5 seeds: [42, 123, 456, 789, 1024].

## Paper Table 5 Targets

| Subset | RMSE  | PHM Score |
|--------|-------|-----------|
| FD001  | 10.61 | 169       |
| FD002  | 13.47 | 784       |
| FD003  | 10.71 | 202       |
| FD004  | 15.87 | 1449      |

## PHM 2008 Score

```python
def compute_phm_score(d):  # d = pred - true (cycles)
    return float(np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1).sum())
```

## Deviations from Paper

- PatchMerging with odd K: we DROP the last patch (truncation). Paper does not specify.
- Decoder cross-attention: two-stage applied symmetrically to match encoder design.
- Sinusoidal PE for first decoder input: Vaswani-style sin/cos, not learned.
- Per-scale prediction: scalar per scale, concatenated and fed to final MLP.

## File Layout

```
paper-replications/star/
  models.py               - STAR architecture
  data_utils.py           - C-MAPSS data loading and preprocessing
  train_utils.py          - Training loop, eval utilities
  run_experiments.py      - Main runner: all 4 subsets x 5 seeds
  test_pipeline.py        - Quick validation (shapes, no NaN, loss decreases)
  REPLICATION_SPEC.md     - This file
  EXPERIMENT_LOG.md       - Per-experiment results
  RESULTS.md              - Final comparison table
  results/
    plots/                - rul_FDXXX.png per subset
    checkpoints/          - *.pt model files (gitignored)
    FD001_results.json    - Per-seed results
    FD002_results.json
    FD003_results.json
    FD004_results.json
```
