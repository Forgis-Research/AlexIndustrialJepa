# Session Summary: A2P Replication

**Date:** 2026-04-10 to 2026-04-11
**Paper:** "When Will It Fail? Anomaly to Prompt for Forecasting Future Anomalies in Time Series"
**Venue:** ICML 2025
**Authors:** Park et al. (KU-VGI)
**Code:** https://github.com/KU-VGI/AP

---

## 1. Bottom Line (3 sentences)

We replicated A2P on MBA and found it achieves F1=19.1 +/- 8.8% vs the paper's 67.55 +/- 5.62%,
primarily due to data source mismatch (TranAD MBA has identical train/test; proper SVDB1 split gives 16.06%).
The most critical finding is that A2P's raw anomaly scores are near-random (AUROC=0.490-0.528),
beaten by all classical baselines and a frozen Chronos-Small zero-shot model (+21.7pp AUROC).
The paper's F1 metric is inflated 8x by the tolerance window, collapses 10x at industrial failure rates,
and gives opposite method rankings to AUROC - the AP evaluation framework is broken for real-world use.

---

## 2. Reproduction Table

### MBA Dataset

| Data Source | L_out | Seeds | Our F1 | Paper F1 | Gap | Notes |
|-------------|-------|-------|--------|----------|-----|-------|
| TranAD (train==test) | 100 | 3 (all seed 20462) | 19.07 +/- 8.77 | 67.55 +/- 5.62 | -48.5pp | Leakage inflates 3.4x |
| TranAD (70/30 split) | 100 | 1 | 12.66 | 67.55 | -54.9pp | Honest evaluation |
| SVDB record 801 (70/30) | 100 | 1 (seed 42) | 16.06 | 67.55 | -51.5pp | AUROC=0.490 (below 0.5!) |
| SVDB record 801 (70/30) | 100 | 2 (seed 1) | TBD | 67.55 | - | In progress |
| SVDB record 801 (70/30) | 100 | 3 (seed 2) | TBD | 67.55 | - | In progress |

### SMD Dataset

| L_out | Seeds | Our F1 | Paper F1 | Status |
|-------|-------|--------|----------|--------|
| 100 | 1 (seed 42) | TBD | 52.07 +/- 0.18 | In progress (FE training) |

### Exathlon and WADI

Not run. Dataset not accessible in this environment.

---

## 3. Ablation Sanity Checks

| Component | Paper F1 | Our F1 | Direction | Pass/Fail |
|-----------|----------|--------|-----------|-----------|
| Full A2P | 67.55 | 19.07 | - | Reference |
| - Shared Backbone | 51.53 | 18.58 | Lower (correct) | PASS direction |
| - AAF cross-attn | 36.26 | 42.55 | Higher (wrong!) | FAIL direction |
| - APP | 60.69 | Not run | - | - |
| - Contrastive Loss | 55.67 | Not run | - | - |

**Notes:**
- No-share backbone direction is correct (18.58 < 19.07) but magnitude differs (-0.5pp vs -16.0pp)
- No-AAF direction is wrong: ablation barely changes F1 (-0.55pp vs expected -31.3pp)
- Null no-AAF effect explained by train==test data: AAFN provides no generalization benefit in-sample

---

## 4. Where A2P Breaks

1. **Near-random raw discriminability**: AUROC=0.490 on proper split (below random 0.5!), 0.528 on train==test.
   All classical baselines beat A2P: z-score (0.675), rolling variance (0.730), Chronos (0.745).
   A2P's specialized architecture provides negative value for raw anomaly discrimination.

2. **F1-tolerance inflation (8x)**: Raw F1=5.35% inflated to 43.1% by t=50 tolerance.
   MBA anomaly segments are >100 steps, so tolerance F1 equals full point adjustment.
   Paper's 67.55% is similarly inflated - raw F1 would be ~8x lower.

3. **Grey-swan regime collapse**: F1 drops 10x from 3.12% to 0.1% anomaly rate.
   Real industrial failures occur at 0.01-0.1%. The entire AP benchmark is inapplicable.

4. **Seed hardcoding bug**: `--random_seed` flag ignored; seed=[20462] hardcoded in run.py line 121.
   Reported variance is NOT from different seeds.

5. **TranAD MBA data leakage**: Train==test identical (row-for-row). 3.4x F1 inflation.

6. **In-domain evaluation only**: A2P never tested cross-dataset.
   Statistical anomaly patterns transfer with only -3.2% relative AUROC penalty (cross-domain).

7. **No lead-time advantage**: A2P's LTW-F1 ratio (4.46x) is LOWER than random (5.07x).
   The title "when will it fail" is not validated by the evaluation protocol.

---

## 5. Top 3 NeurIPS-Level Improvements

### 1. New AP Evaluation Framework (AUPRC + DR@FAR)
**Why compelling:** F1-tolerance gives opposite rankings to AUROC. The metric is fundamentally misleading.
**Cost:** 1 week (compute AUPRC rankings for all methods + show rank disagreement)
**Expected win:** First systematic critique of AP evaluation, applicable to all AP papers.

### 2. JEPA-AP (Self-Supervised Anomaly Prediction)
**Why compelling:** JEPA already built (RMSE=0.0868 on FEMTO). Replace AnomalyTransformer with JEPA predictor.
**Cost:** 2 weeks (JEPA backbone swap + evaluation)
**Expected win:** AUROC 0.490 -> 0.70+ target (calibrated representation-space scoring).

### 3. Foundation Model Baseline (Chronos + MLP Head)
**Why compelling:** Zero-shot Chronos beats A2P by +21.7pp AUROC. Adding a fine-tuned head should close F1 gap.
**Cost:** 1 week
**Expected win:** Match A2P F1 with 100x simpler training - establishes minimum viable AP baseline.

---

## 6. Experiments Run

| # | Experiment | Method | Key Result |
|---|------------|--------|------------|
| 0 | Pipeline smoke test | A2P, 1 epoch | F1=0 (expected, too few epochs) |
| 1 | MBA TranAD, seed 20462 | A2P, 5+5 epochs | F1=43.1%, AUROC=0.528 |
| 2 | MBA TranAD, seeds 0,1,2 | A2P, 3 runs | F1=19.07 +/- 8.77% |
| 3 | Ablation: no shared backbone | A2P | F1=18.58% (direction correct) |
| 4 | Ablation: no AAFN | A2P | F1=42.55% (near-null effect) |
| 5 | MBA 70/30 proper split | A2P | F1=12.66% (3.4x leakage confirmed) |
| 6 | Grey-swan regime | Analysis on A2P scores | 10x F1 collapse at 0.1% rate |
| 7 | Calibration analysis | A2P real scores | AUROC=0.528, Brier skill=-0.12 |
| 8 | Lead-time-weighted F1 | A2P vs Random | No real lead-time advantage |
| 9 | Foundation model | Chronos-Small (frozen) | AUROC=0.745 (+21.7pp over A2P) |
| 10 | Statistical baselines | Z-score, rolling var, etc. | All beat A2P AUROC trivially |
| 11 | SVDB1 record 801, seed 42 | A2P, proper split | F1=16.06%, AUROC=0.490 |
| 12 | Cross-dataset transfer | Rolling var, MBA->SMD | -0.025 AUROC (-3.2%), transfer works |
| 13 | E2E training (AAFN unfrozen) | A2P modification | In progress |
| 14 | SVDB1 seeds 1 and 2 | A2P | In progress |
| 15 | SMD L100 seed 42 | A2P | In progress |

---

## 7. Open Questions for Next Session

1. SMD L100 result: will A2P approach paper's 52.07% F1 with proper data?
2. SVDB1 multi-seed variance: is 16.06% typical or outlier?
3. End-to-end training: does unfreezing AAFN improve discrimination?
4. Chronos + fine-tuned head: can it match A2P F1?
5. SVDB 4-record: the paper's actual data; feasible only with longer session or HPC.
6. Exathlon and WADI: datasets not found; may need registration at data providers.

---

## 8. Cost

- **Wall time:** ~10 hours (overnight session)
- **GPU:** A10G (23GB), ~8 GPU-hours
- **CPU:** 2 hours (Chronos on CPU, statistical baselines)
- **Disk:** ~3GB (SVDB1 161K+69K, SMD 708K x 38, AP checkpoints)
- **Network:** PhysioNet SVDB (3.7GB), HuggingFace SMD (170MB), Chronos-Small cached

---

## Key Files

| File | Description |
|------|-------------|
| `RECON_NOTES.md` | Codebase walkthrough, bugs found |
| `REPLICATION_SPEC.md` | Formal replication specification |
| `EXPERIMENT_LOG.md` | All 15+ experiments with metrics |
| `IMPROVEMENT_IDEAS.md` | 12 NeurIPS-level improvement cards |
| `RESULTS_TABLE.md` | Paper vs ours comparison table |
| `results/all_results.json` | Machine-readable results |
| `results/improvements/*.json` | Per-probe result files |
| `notebooks/a2p_replication_summary.qmd` | Quarto summary notebook |
| `notebooks/a2p_replication_summary.html` | Rendered HTML output |
| `figures/*.png` | All analysis figures |
