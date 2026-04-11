# A2P Replication Results Table

**Paper:** When Will It Fail? Anomaly to Prompt for Forecasting Future Anomalies in Time Series  
**Venue:** ICML 2025  
**Authors:** Park et al.  
**Date:** 2026-04-10/11

---

## Table 1 Reproduction: F1 with tolerance t=50 (%)

### MBA Dataset (MIT-BIH Supraventricular Arrhythmia DB, 2 channels)

| L_out | Paper (mean +/- std) | Ours (mean +/- std) | Gap | Status |
|-------|---------------------|---------------------|-----|--------|
| 100 | 67.55 +/- 5.62 | 19.07 +/- 8.77 | -48.5 pp | BELOW TARGET |
| 200 | 47.30 +/- 6.06 | - | - | Not run |
| 400 | 36.95 +/- 6.90 | - | - | Not run |

Notes:
- Our 3 "seed" runs are NOT independent (seed hardcoded to 20462 in run.py:121)
- TranAD MBA data has identical train/test sets (data integrity issue)
- SVDB 4-record run (PID 179247) may resolve gap but still running
- Critical: AUROC=0.528 (single run) - raw scores are near-random

### SMD Dataset (Server Machine Dataset, 38 channels)

| L_out | Paper (mean +/- std) | Ours (mean +/- std) | Gap | Status |
|-------|---------------------|---------------------|-----|--------|
| 100 | 52.07 +/- 0.18 | running | - | IN PROGRESS (PID 178193, 72+ min) |
| 200 | 47.02 +/- 0.07 | - | - | Not run |
| 400 | 39.78 +/- 0.24 | - | - | Not run |

### Exathlon Dataset

| L_out | Paper (mean +/- std) | Ours | Status |
|-------|---------------------|------|--------|
| 100 | 18.64 +/- 0.16 | - | Not run |
| 200 | 18.34 +/- 0.17 | - | Not run |
| 400 | 16.57 +/- 0.26 | - | Not run |

### WADI Dataset

| L_out | Paper (mean +/- std) | Ours | Status |
|-------|---------------------|------|--------|
| 100 | 64.91 +/- 0.47 | - | Not run |
| 200 | 63.32 +/- 0.53 | - | Not run |
| 400 | 60.85 +/- 0.37 | - | Not run |

### Average (Table 1, L=100)

| Method | Avg F1 L100 | Avg F1 L200 | Avg F1 L400 |
|--------|------------|------------|------------|
| A2P (paper) | 46.84 | 43.75 | 38.54 |
| Ours | TBD | - | - |

---

## Table 2 Reproduction: Ablation Study (MBA L=100)

| Component | Paper F1 | Our F1 | Gap | Status |
|-----------|----------|--------|-----|--------|
| Full A2P | 67.55 | 19.07 | -48.5pp | Replicated (with gap) |
| - Shared Backbone | 51.53 | running | - | IN PROGRESS |
| - AAF (no noise inject) | 36.26 | running | - | IN PROGRESS |
| - APP | 60.69 | - | - | Not run |
| - Contrastive Loss | 55.67 | - | - | Not run |
| - Forecast Loss | 63.18 | - | - | Not run |

Expected direction: removing each component should reduce F1. If our ablations show the same direction (F1 drops when components removed), this validates the architecture even if absolute numbers differ.

---

## Improvement Probe Results

| Probe | Method | Metric | Value | Baseline | Significance |
|-------|--------|--------|-------|----------|-------------|
| Calibration | A2P MBA | AUROC | 0.528 | 0.500 (random) | Near-random! |
| Calibration | A2P MBA | Raw F1 | 5.35% | - | vs 43.1% with tolerance |
| Calibration | A2P MBA | Brier Skill | -0.117 | 0.0 | NEGATIVE |
| Calibration | A2P MBA | AUPRC | 0.035 | 0.029 | Marginally above random |
| Grey-Swan | A2P MBA | F1@0.1% rate | 1.8% | 0.0% | 10x collapse from 3.12% |
| Grey-Swan | A2P MBA | F1@1% rate | 11.76% | 0.0% | - |
| LTW-F1 | A2P MBA | LTW-F1 | 23.85% | Random: 14.25% | 1.67x advantage |
| LTW-F1 | A2P MBA | LTW/Std ratio | 4.46x | Random: 5.07x | Random has MORE lead time |

---

## Critical Analysis: What the Numbers Mean

### The F1 Inflation Problem

```
Raw binary F1 (no tolerance) = 5.35%
F1 with t=50 tolerance = 43.1%
Paper's F1 with t=50 = 67.55%

Inflation factor = 43.1 / 5.35 = 8.1x
```

The 50-step tolerance window gives credit for any prediction within 50 timesteps of a true anomaly. With a 100-step prediction window (`pred_len=100`) and 50-step tolerance, almost any prediction near an anomaly gets counted as a true positive.

### AUROC = 0.528: What It Means

An AUROC of 0.528 (vs 0.500 for random) means that if you randomly picked one anomaly timestep and one normal timestep, the model's score would correctly rank the anomaly higher only 52.8% of the time. This is essentially a coin flip.

The F1=43.1% is NOT because the model discriminates anomalies well. It is because:
1. The model fires occasionally (threshold at 99th percentile = ~1% flagged)
2. Anomaly segments are ~100 timesteps long
3. Any flag within 50 timesteps of a 100-timestep anomaly segment gets credit
4. Combined effect: even near-random flagging near anomaly regions produces inflated F1

### Implication for NeurIPS-Level Contribution

This replication reveals a systemic issue with AP evaluation: **the F1 with tolerance metric severely inflates apparent performance and hides the fact that models cannot actually discriminate anomalies from normals in raw score space.**

A NeurIPS-worthy contribution would be:
1. Propose AUPRC (or DR@FAR) as the primary AP metric
2. Show that AUPRC rankings of methods differ substantially from F1-tolerance rankings
3. Develop a calibrated AP model (JEPA-based) that achieves AUROC > 0.7
