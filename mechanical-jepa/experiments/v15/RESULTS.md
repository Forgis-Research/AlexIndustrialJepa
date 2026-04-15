# V15 Results

**Session date**: 2026-04-15 to 2026-04-16
**Goal**: Multi-domain grey swan benchmark - metrics, SIGReg architecture, new datasets.

---

## Phase 0: Metrics Study

### 0a. Unified Evaluation Module

Built `mechanical-jepa/evaluation/grey_swan_metrics.py`:

| Event type | Primary metric | Secondary metric | Rationale |
|-----------|---------------|-----------------|-----------|
| RUL (regression) | RMSE | nRMSE | RMSE is universal currency; nRMSE for cross-domain |
| Anomaly detection | non-PA F1 | AUC-PR | PA inflation: +0.92 F1 demonstrated in test |
| Threshold exceedance | nRMSE | RMSE in cycles | Normalizes for cross-domain comparison |

**PA inflation confirmed**: A predictor firing one point per anomaly segment gets
PA F1 = 1.000 while non-PA F1 = 0.077. This explains why literature results look
much better than they are. We report both but use non-PA F1 as primary.

### 0b. TTE on C-MAPSS FD001 s14

- **Sensor**: s14 (corrected fan speed Nc, index 10 in 14-sensor subset)
- **Threshold**: 3-sigma from cycles 1-50 baseline
- **Exceedances**: 93/100 training engines exceed 3-sigma (feasible task)
- **First exceedance**: mean cycle ~64 (out of max ~279)
- **Ridge probe (hand features, trivial baseline)**: RMSE=32.98, nRMSE=0.118

---

## Phase 1: SIGReg + Bidirectional Architecture

### Experiment A: EP-SIGReg Isotropy Test

**Result**: FAILED. EP-SIGReg alone (100 gradient steps on anisotropic z) reduced
PC1 from 0.777 (initial) to 0.690 final. Loss decreased 0.0059 -> 0.0007. But
PC1 did not reach the target (<0.20). EP gradient alone, without prediction task
co-training, is insufficient to drive isotropy.

### Phase 1c: Loss-Probe Correlation (V15-SIGReg, 50 epochs)

- Spearman rho (loss vs probe RMSE): 0.109 (p=0.750)
- **FAILED threshold** (need rho > 0.7 for passing)
- Interpretation: V15-SIGReg has noisy probe trajectory (loss and probe RMSE
  not monotonically correlated). This is expected given the oscillating collapse
  pattern.

### V15-EMA: 3 Seeds Results

**CRITICAL INTERNAL INCONSISTENCY - LOGGED AS SUSPICIOUS**:

- Seed 42: best_probe=20.83 (probe measured at epochs 1,10,20,...200)
  - Probe trajectory: 30.82 -> 41.18 -> 28.33 -> 23.93 -> 24.47 (V-shaped collapse)
  - Recovery: probes reaching ~23-25 in epochs 150-200
  - The V-shape is direct evidence of collapse and partial recovery.

- Seed 123: best_probe=13.50 (from epoch 1!) then probe degrades to 29+
  - Epoch 1: 13.50 (best by far). Epoch 10: 26.51. Epoch 60: 26.07.
  - Recovery: 17.88 at epoch 90, trending better.
  - **The epoch-1 best=13.50 is SUSPICIOUS**: better than trained V2 (17.81)
    after just 1 epoch. This likely reflects lucky random init, not learned structure.
    Should not be reported as the effective best.
  - At epoch 90, probe is recovering to ~17-18 range.

- Seed 456: RUNNING (queued after seed 123 completes at epoch 200)

**Status**: STILL RUNNING. Final results pending.

**Architecture finding**: V15-EMA collapses because:
- Context: x_{0:t}, Target: x_{0:t+k} - they share prefix x_{0:t}
- The predictor learns to copy rather than predict future structure
- Causal V2 avoids this: context=x_{0:t}, target=x_{t+1:t+k} (no shared prefix)
- Fix: V16a = bidirectional context + causal target (no prefix sharing)

### V15-SIGReg: 3 Seeds Results

PENDING - queued after V15-EMA completes.

**V2 baseline reference (from V14)**: 17.81 +/- 1.7 (frozen probe, 5 seeds)

---

## Phase 2: Improved Cross-Sensor Encoder

NOT RUN (code complete, deferred to V16).

Code at `mechanical-jepa/experiments/v15/phase2_cross_sensor_improved.py`:
- Sensor ID embeddings (learnable)
- Sensor token dropout 20%
- Attention map extraction
- V14 reference: frozen=14.98+/-0.22

---

## Phase 3: Dataset Adapters + SMAP Anomaly

### 3a. SMAP/MSL Adapters

Implemented in `mechanical-jepa/data/smap_msl.py`:
- SMAP: 135K train / 428K test, 25 channels, anomaly_rate=12.8%
- MSL: 58K train / 74K test, 55 channels, anomaly_rate=10.5%

### 3b. SWaT Adapter

Stub in `mechanical-jepa/data/swat.py` - data requires registration.
Not available for V15. Register before V16 session.

### 3c. SMAP/MSL Pretraining + Anomaly Detection (Phase 5b)

**Pretraining**: 20 epochs, 20K samples, V15-SIGReg mode, seed=42.
- SMAP final loss: 0.0119 (decreasing, indicating learning)
- MSL final loss: similar convergence

**Anomaly detection results**:

| Dataset | non-PA F1 | PA F1 | AUC-PR | TaPR F1 | Random baseline |
|---------|-----------|-------|--------|---------|-----------------|
| SMAP    | 0.0688    | 0.625 | 0.113  | 0.229   | 0.071           |
| MSL     | 0.0787    | 0.433 | 0.116  | 0.203   | 0.071           |

**Literature comparison (SMAP)**:
- MTS-JEPA: PA F1 = 33.6%
- TS2Vec: PA F1 = 32.8%
- **Our PA F1 = 62.5% (beats literature, but is inflated)**

**CRITICAL INTERNAL INCONSISTENCY**:

- PA F1 = 62.5% looks like we beat MTS-JEPA by a large margin
- BUT non-PA F1 = 0.069 barely beats random baseline (0.071)
- Anomaly score stats: mean=0.838, std=0.039 (near-constant high scores)
- The model assigns high reconstruction error to EVERYTHING, not selectively to anomalies
- PA protocol inflates this because it counts a detection within each anomaly SEGMENT
  even if the model fires everywhere

**Conclusion**: We did NOT learn anomaly-discriminative representations.
The high PA F1 is an artifact of constant-high scoring, not anomaly detection.
More pretraining needed: 100 epochs on full 135K train set (V16 item).

**The epoch-1 best=13.50 for seed 123 (V15-EMA)** is analogous to this:
an artifact of initialization/short training, not a genuine learned result.

---

## Phase 4: Sensor Correlation Analysis

### C-MAPSS FD001

- 21 total sensors, 4 natural clusters via Ward hierarchical clustering
- 39 high-correlation pairs (|r| > 0.7)
- Near-perfect redundancy: s5-s16 (r=1.000), s9-s14 (r=0.963)
- Largest degradation correlation shifts: sensors 2, 3, 6, 7
- s9-s14 Spearman rho: 0.886 (highly correlated with degradation)

### SMAP

- 25 channels, 5 clusters, only 10 high-corr pairs
- Much more independent than C-MAPSS sensors
- More diverse sensor types -> cross-sensor attention more valuable for SMAP

### Architecture Recommendation

**For C-MAPSS (stable, correlated sensors)**: Channel-fusion (V2) sufficient.
Cross-sensor attention adds value only at 100% labels.

**For SMAP/MSL (independent, diverse channels)**: Cross-sensor attention more
appropriate - sensors don't have strong shared structure.

**Key finding**: Correlation SHIFTS during degradation (not static correlation)
are the signal. Cross-sensor attention directly models this.

---

## Phase 5: Benchmark Table

### 5a. C-MAPSS TTE (Phase 0b + new V14 probe)

| Method | RMSE (cycles) | nRMSE |
|--------|--------------|-------|
| Trivial mean | 39.54 | 0.233 |
| Hand features Ridge (3 features, trivial) | 32.98 | 0.118 |
| V14 frozen encoder probe | 37.02 | 0.218 |
| V15/V16 encoder probe | TBD | TBD |

**Finding**: V14 frozen encoder (RUL-trained) does NOT improve TTE prediction
over hand-crafted features. The encoder learned RUL trajectory structure, not
short-term threshold exceedance dynamics. TTE requires different pretraining signal.

### 5b. SMAP Anomaly Detection

| Method | non-PA F1 | PA F1 | AUC-PR |
|--------|-----------|-------|--------|
| Random baseline | 7.1% | ~33%* | - |
| V15-SIGReg (20 epochs) | 6.9% | 62.5% | 11.3% |
| TS2Vec (literature) | - | 32.8% | - |
| MTS-JEPA (literature) | - | 33.6% | - |

*Random baseline PA F1 is dataset-dependent; literature doesn't report it.

**Note**: Literature methods are evaluated differently (more pretraining, tuned thresholds).
Our 20-epoch result is preliminary. The PA F1 comparison is misleading.

---

## Sanity Check Status

| Check | Status |
|-------|--------|
| Phase 0a metrics validated | PASS |
| Phase 0b TTE feasibility verified | PASS |
| Phase 1 architecture smoke tested | PASS |
| Phase 3 SMAP adapter tested | PASS |
| Phase 4 correlation analysis | PASS |
| Phase 5a TTE V14 probe | PASS (negative result: beats mean, fails hand features) |
| Phase 1 V15-EMA seeds 42,123 | RUNNING - collapse confirmed |
| Phase 1 V15-SIGReg 3 seeds | PENDING (after V15-EMA) |
| Phase 3 SMAP+MSL pretraining | COMPLETE (insufficient pretraining) |
| Phase 2 cross-sensor improved | DEFERRED to V16 |

---

## Open Negatives to Report

1. V15-EMA collapses: bidirectional full-sequence target shares prefix with context.
2. SWaT data not available - only stub adapter.
3. Phase 2 (improved cross-sensor) not run this session.
4. EP-SIGReg uses simplified quadrature (linear grid, not Gauss-Hermite).
5. V15-SIGReg Phase 1c correlation: Spearman rho=0.11 (not significant).
6. SMAP anomaly detection: 20 epochs insufficient, non-PA F1 barely beats random.
7. V14 encoder TTE probe WORSE than hand-feature Ridge (37.02 vs 32.98).
8. V15-EMA seed 123 epoch-1 best=13.50 is suspicious (NOT a valid result).

---

## Key Methodological Contributions

1. **Grey swan evaluation framework**: honest non-PA F1 vs PA-inflated alternatives.
   Demonstrated +55 percentage point PA inflation (6.9% non-PA -> 62.5% PA F1).

2. **PA inflation demo**: Firing within one anomaly segment gives PA F1=1.0 vs
   non-PA F1=0.077. This explains why all literature anomaly detection results
   look better than they are.

3. **EP-SIGReg vectorized**: O(Q x B x M) batch implementation, 0.9ms/call.
   Enables SIGReg as drop-in regularizer during pretraining.

4. **Sensor correlation analysis**: C-MAPSS has 4 natural sensor clusters,
   s5-s16 perfectly correlated (r=1.000). Architecture recommendations based on
   correlation structure.

5. **V15-EMA collapse analysis**: Identified root cause (shared prefix) and
   solution (V16a: causal target, bidirectional context).
