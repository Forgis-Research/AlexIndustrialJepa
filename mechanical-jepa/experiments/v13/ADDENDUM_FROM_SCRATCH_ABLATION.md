# V13 Addendum: From-Scratch Ablation

**Priority: HIGH — run before or alongside Hypothesis 1.**

This is the one experiment that quantifies exactly how much of the E2E
result comes from pretraining vs just having a good architecture with
enough labeled data. V11 never ran it. Without it, we cannot make any
SSL claim about the E2E number.

## The experiment

Train the **exact same** V2 transformer encoder (d=256, L=2, same param
count) + the same linear probe, with the same E2E fine-tuning protocol
(LR=1e-4, same optimizer, same early stopping, same patience, same
train/val/test splits, seed=42), on FD001 at 100% labels — but
**initialized from random weights** instead of the pretrained checkpoint.

The only difference between this run and V11 E2E is `torch.init` vs
`load_state_dict(pretrained)`. Everything else identical.

## What to report

| Configuration | RMSE | Source |
|:---|:---:|:---|
| V2 E2E from pretrained weights | 14.23 +/- 0.39 | V12 multi-seed |
| V2 E2E from random init | ??? | This experiment |
| V2 Frozen (pretrained) | 17.81 +/- 1.67 | V11 |
| Supervised LSTM (different arch) | 17.36 +/- 1.24 | V11 |
| STAR (different arch) | 12.19 +/- 0.55 | Replication |

The number we care about is the delta:

```
pretraining_contribution = from_scratch_rmse - pretrained_e2e_rmse
```

## Interpretation guide

- **Delta > 3 RMSE**: pretraining is doing real work under E2E. The
  initialization carries structure that supervised training alone cannot
  learn in the same number of epochs. Strong SSL claim survives for E2E.
- **Delta 1–3 RMSE**: pretraining helps but modestly. The paper should
  lead with frozen/H.I. results, not E2E RMSE. E2E is "pretraining as
  a warm start" — useful but not the headline.
- **Delta < 1 RMSE**: pretraining contributes negligibly to E2E. The E2E
  number is essentially supervised learning in a transformer. The ONLY
  honest SSL claims are frozen probe and H.I. recovery. The paper must
  not present E2E as evidence of SSL utility.

## Also run at low label budgets

The from-scratch ablation is most informative at low labels, where
pretraining should matter most. Run at 100%, 20%, 10%, 5%:

| Labels | Pretrained E2E | From-scratch E2E | Delta |
|:------:|:--------------:|:----------------:|:-----:|
| 100% | 14.23 | ? | ? |
| 20% | 16.54 | ? | ? |
| 10% | 18.66 | ? | ? |
| 5% | 25.33 | ? | ? |

If the delta grows as labels decrease, that's the label-efficiency pitch:
"pretraining matters most when labels are scarce." If the delta is flat
or shrinks, pretraining is just a speed-up, not a capability enabler.

## Implementation

Minimal diff from V11's fine-tuning code. Pseudocode:

```python
# From-scratch: skip checkpoint loading, initialize fresh
model = TrajectoryJEPA(d_model=256, n_layers=2, ...)  # random init
probe = RULProbe(model.d_model)

# Same E2E fine-tuning as V11
optimizer = Adam([*model.context_encoder.parameters(),
                  *model.predictor.parameters(),
                  *probe.parameters()], lr=1e-4)

# Same training loop, same eval, same seeds
```

5 seeds, same seeds as V12 multi-seed (42, 123, 456, 789, 1024).
Report mean +/- std. Should take ~1 hour total (same as one E2E sweep).

## Output

Save as `experiments/v13/from_scratch_ablation.json`:
```json
{
  "from_scratch_e2e": {"1.0": {"mean": ..., "std": ..., "all": [...]}, ...},
  "pretrained_e2e_ref": {"1.0": {"mean": 14.23, "std": 0.39}},
  "pretraining_contribution": {"1.0": ..., "0.2": ..., "0.1": ..., "0.05": ...}
}
```

---

# Addendum 2: Length-vs-Content Ablation

**Priority: HIGH — run alongside from-scratch ablation. Inference only, no training.**

V12's diagnostics (shuffle test, Spearman rho, H.I. R²) rule out constant
prediction but do NOT rule out length-only encoding. In C-MAPSS training
data, every engine runs to failure, so prefix length `t` correlates
perfectly with elapsed life. An encoder that just counts timesteps —
learning a positional encoding that leaks into `h_past` — would pass all
three V12 diagnostics without reading any sensor values.

This ablation separates "the encoder reads the sensors" from "the encoder
counts the timesteps."

## Experiment 1: Constant-input test

Feed the frozen V2 encoder sequences where every row is identical (the
first cycle repeated t times), for varying lengths t = 30, 50, 80, 110,
140, 170, 200. The sensor values are real (from any engine's first cycle)
but constant across time — so there is NO degradation signal in the
content, only length information.

If the encoder is reading sensors: `h_past` should be nearly identical
across all lengths (same content, different repetition count) and the
probe should output similar RUL regardless of t.

If the encoder is just counting: `h_past` should change systematically
with t, and the probe output should decrease with longer sequences.

Report: `predicted_rul` as a function of `t` for the constant-input case.
If pred(t=30) ≈ pred(t=200), the encoder is reading content. If
pred(t=30) >> pred(t=200) with a smooth decline, the encoder is counting.

## Experiment 2: Length-matched cross-engine swap

Pick 10 pairs of engines (A, B) with similar total length (within 10
cycles). For each pair at a shared cut point t:

- Compute `h_past_A` = encoder(engine_A[0:t])
- Compute `h_past_B` = encoder(engine_B[0:t])
- Compute cosine similarity between h_past_A and h_past_B
- Compute `probe(h_past_A)` and `probe(h_past_B)`

Same-length, different sensors. If `h_past` is dominated by length, the
two embeddings will be nearly identical (high cosine similarity, similar
probe outputs). If `h_past` encodes sensor content, they will differ
(low cosine similarity, different probe outputs reflecting each engine's
actual degradation state).

Also compare against a within-engine control: `h_past` at cycle t vs
cycle t+1 for the same engine. The between-engine difference at the same
length should be LARGER than the within-engine difference across adjacent
cycles if the encoder is reading content. If the within-engine step is
larger, the encoder is primarily tracking length.

## Experiment 3: Sensor-shuffled-in-time test (strongest)

For each test engine, randomly permute the temporal order of the sensor
rows (cycle 50's readings go to position 10, cycle 10's go to position
80, etc.) but keep the sequence length the same. Run inference.

If the encoder reads temporal degradation patterns in the sensors: the
shuffled predictions should be significantly worse (degradation signal is
destroyed). Spearman rho should collapse toward 0.

If the encoder just counts positions: the shuffled predictions should be
identical to the unshuffled ones (length is unchanged, and the encoder
ignores sensor content).

Report: rho_median and RMSE for shuffled-in-time vs original. This is
the cleanest single test because it holds length constant, holds the
set of sensor values constant, and only destroys temporal structure.

## Output

Save as `experiments/v13/length_vs_content_ablation.json`:
```json
{
  "constant_input": {
    "lengths": [30, 50, 80, 110, 140, 170, 200],
    "predicted_rul": [...],
    "verdict": "counting | reading_content"
  },
  "length_matched_swap": {
    "mean_cosine_similarity": ...,
    "mean_pred_diff_between_engines": ...,
    "mean_pred_diff_within_engine_adjacent": ...,
    "verdict": "length_dominated | content_dominated"
  },
  "temporal_shuffle": {
    "original_rho_median": ...,
    "shuffled_rho_median": ...,
    "original_rmse": ...,
    "shuffled_rmse": ...,
    "verdict": "encoder_reads_temporal_structure | encoder_ignores_content"
  }
}
```

## Interpretation

All three experiments must agree. If the encoder passes (reads content,
not length), this closes the last loophole on the SSL representation
claim and should be cited in the paper. If it fails (just counting),
the H.I. R²=0.926 result is a length artifact and the paper narrative
needs fundamental rethinking — the "strongest SSL evidence" would
actually be trivial.
