---
name: Bearing RUL SOTA Literature Review (April 2026)
description: Best published results on FEMTO/XJTU-SY RUL, standard evaluation protocol, self-supervised RUL approaches, PHM 2012 metric, single-window formulation gap
type: project
---

Focused review conducted April 2026 for positioning IndustrialJEPA against SOTA.

## Standard Evaluation Protocol (Critical to Understand)

### How Published Papers Define the RUL Regression Target

The dominant convention (used by ~90% of papers):

  RUL_normalized(t) = (T_total - t) / T_total

This yields a value in [0, 1] where 1.0 = brand new, 0.0 = failure. Some papers use a
piecewise-linear label that caps at 1.0 for the "healthy phase" before a detected
degradation onset point, then decays linearly to 0.

### What Is Actually Predicted

Almost all papers do **point-to-point (P2P) regression on the full trajectory**: given
every window in a run-to-failure trajectory, predict the normalized RUL at each timestep.
This is the standard benchmark task — the model sees sequential windows and outputs a
prediction for each one.

A minority of papers do **sequence-to-point (S2P)**: given a subsequence of windows,
predict the RUL at the last timestep.

**Neither paradigm is what IndustrialJEPA's task formulation describes.** The project
task is: given ONE short window (0.1s-1.3s), without any temporal context or ordering
information, predict RUL as a percentage. This is a truly different setup.

### End-of-Life (EOL) Threshold (FEMTO/PRONOSTIA)

The PRONOSTIA rig stops each test when vibration amplitude exceeds 20 g. This is the
EOL marker. RUL is measured backward from this point.

### Dataset Splits (FEMTO)

Three operating conditions (sub-datasets). Sub-datasets 1 and 2: 2 training runs + 5 test
runs each. Sub-dataset 3: 1 test run. Standard practice uses sub-datasets 1 and 2 for
comparison; sub-dataset 3 is rarely included due to small size.

### Input Window

The standard window size in the `rul-datasets` library is 2560 samples. At 25.6 kHz
this is exactly 0.1 seconds. Papers vary widely — many use extracted statistical features
(RMS, kurtosis, etc.) over a 10-second recording interval rather than raw samples.

---

## FEMTO/PRONOSTIA SOTA Results (Normalized RUL in [0,1])

Metric is normalized RMSE (nRMSE) and MAE, where values are on the 0-1 scale.
Lower is better.

| Method | Year | RMSE (FEMTO) | MAE (FEMTO) | Notes |
|---|---|---|---|---|
| CNN-GRU-MHA (transfer learning) | 2024 | **0.0443** | -- | Transfer learning, small sample |
| MDSCT (Multi-scale Separable Conv+Transformer) | 2024 | 0.124 | 0.100 | PMC11481647 |
| Bi-LSTM-Transformer + EMD | 2025 | 0.0563 | 0.0469 | On XJTU-SY; FEMTO ~similar |
| TCN-SA (baseline in MDSCT paper) | 2024 | 0.148 | 0.114 | |
| CVT (baseline in MDSCT paper) | 2024 | 0.131 | 0.112 | |
| DCNN (baseline) | 2024 | 0.175 | 0.145 | On XJTU-SY |

The CNN-GRU-MHA result of 0.0443 appears to be among the best reported on FEMTO as of
early 2025. The MDSCT paper (Heliyon/PMC 2024) provides the cleanest apples-to-apples
comparison table.

Note: results are NOT fully comparable across papers due to:
- Different subsets of FEMTO used (some use all 3 conditions, some just 1-2)
- Different degradation onset detection methods (affects how many windows are labeled)
- Different normalization conventions

## XJTU-SY SOTA Results

| Method | Year | RMSE (XJTU-SY) | MAE (XJTU-SY) | Notes |
|---|---|---|---|---|
| Bi-LSTM-Transformer + EMD | 2025 | 0.0563 | 0.0469 | |
| MDSCT | 2024 | 0.160 | 0.134 | |
| SAGCN-SA | 2024 | **0.170** | -- | Best in that paper |
| TCN-SA (baseline) | 2024 | 0.194 | 0.158 | |
| CVAER (envelope spectrum + VAE) | 2024 | 28.47 min | -- | Absolute time, not normalized |

The XJTU-SY dataset is less standardized than FEMTO — papers sometimes report per-bearing
results, sometimes averages over condition groups.

---

## The PHM 2012 Challenge Scoring Function

The original 2012 challenge used an **asymmetric score function** based on percent error,
not RMSE. The formula (from Nectoux et al. 2012 and widely cited):

  For each bearing i:
    Er_i = (RUL_predicted - RUL_true) / RUL_true * 100   (percent error)

    if Er_i <= 0 (early prediction):  score_i = -log(1 - Er_i/100)^C1   [penalizes less]
    if Er_i > 0  (late prediction):   score_i = log(1 + Er_i/100)^C2   [penalizes more]

  Final score = (1/N) * sum(score_i)   where N = 11 test bearings

The exact penalty coefficients C1, C2 are in the original PDF (not fully accessible via
web). Late predictions (predicting longer life than actual) are penalized more heavily,
reflecting the safety-critical nature of maintenance.

The competition winner achieved a total score of ~0.28.

Crucially: the PHM 2012 metric is NOT normalized RMSE — it is a percent-error-based
score function applied to a **single final RUL estimate** per bearing (made at a specific
fraction of the bearing's life). This is conceptually different from the whole-trajectory
RMSE that most recent papers report.

Post-2015 literature has largely abandoned the PHM score in favor of normalized
RMSE/MAE on the full trajectory, which is more interpretable and allows finer comparison.

---

## Self-Supervised / Representation Learning Approaches to RUL

### Existing Work

**DCSSL (2026)** — "A novel dual-dimensional contrastive self-supervised learning-based
framework for rolling bearing RUL prediction." Scientific Reports 2026.
Stage 1: Random cropping + timestamp masking to build positive pairs. Temporal-level
and instance-level contrastive loss. Stage 2: Fine-tune RUL regression head.
Dataset: FEMTO/PRONOSTIA. Claims to outperform supervised SOTA.
Relevance: Closest existing work to IndustrialJEPA's SSL + RUL framing.

**HNCPM (Hard Negative Contrastive Prediction Model, ~2023)** — Contrastive learning
with hard negative samples (selected by cosine similarity), GRU regression module, decoder
for reconstruction. Encoder learns representations, GRU predicts RUL.

**Contrastive SSL for incipient fault detection (Reliability Eng. 2022)** — Pretrains
encoder with contrastive objective on healthy vs degraded windows, then extracts health
indicator. Not direct RUL regression.

**Semi-supervised transfer learning (MDPI Sensors 2025)** — Anti-self-healing health
indicator + semi-supervised transfer. Some self-supervised elements but primarily
supervised RUL.

**RULSurv (2024)** — Survival analysis (CoxPH, RSF) on XJTU-SY with censoring-aware
training. Not self-supervised but novel framing with probabilistic output.
XJTU-SY C1 MAE: 12.6 minutes (Random Survival Forests). Different metric from nRMSE.

**CVAER (2024)** — Convolutional Variational Autoencoder for Regression. Uses averaged
envelope spectra (AES) as input. Trains one VAE + regression head per bearing lifecycle.
XJTU-SY best RMSE: 28.47 minutes. This is the closest thing to single-spectrum prediction:
each timestamp's AES is fed independently to predict RUL. But they still train on the
full trajectory (supervised on time-ordered windows).

### What Does NOT Exist

- JEPA + RUL: no paper exists (confirmed gap from earlier search)
- MAE pretraining specifically for bearing RUL regression (pretraining on healthy vibration,
  then fine-tuning on limited run-to-failure labels): not published
- JEPA pretrained on large multi-source vibration corpus, then fine-tuned for RUL: not published
- Any paper that explicitly treats RUL as a **position-agnostic single-window problem**

---

## The Single-Window RUL Problem: Does It Exist in Literature?

### The Direct Answer: No, Not as a Formal Formulation

No published paper defines the task as: "Given one short vibration window, with no
knowledge of where in the bearing's lifecycle this window comes from, predict RUL as a
percentage."

All existing methods require either:
1. **Sequential context**: A series of windows in temporal order, so the model can see
   the degradation trend
2. **Online position information**: The model knows which timestep in the lifecycle the
   current window corresponds to
3. **Degradation onset detection first**: The model first detects when degradation begins
   (a separate module), then predicts RUL from that onset point

### The Closest Existing Work

**CVAER (Convolutional VAER, 2024)**: Makes a RUL prediction from each individual envelope
spectrum. The model doesn't explicitly see history — it predicts from a single snapshot.
BUT the training procedure is supervised on full trajectories (the model learns the mapping
from spectrum shape to position in lifecycle), and the spectrum is time-averaged over 10s
recordings (not a raw 0.1s vibration burst). Published in PMC (Sensors/MDPI family), so
not a top-venue paper.

**DMW-Trans (IEEE 2023)**: Deep Multiscale Window-based Transformer that explicitly
argues single-window approaches fail because they miss cross-temporal information. It
proposes multiscale feature maps as a solution. This paper directly acknowledges the
single-window problem but treats it as a weakness to overcome, not a feature to exploit.

### Why the Single-Window Formulation Is Novel

The conventional wisdom (stated explicitly in DMW-Trans 2023 and the survey literature)
is that single-window RUL estimation is fundamentally limited — you need temporal context.
Our hypothesis: a sufficiently powerful pretrained representation (JEPA embeddings) can
encode enough physical information about degradation state from a single window to regress
RUL without temporal context. If true, this is both novel and practically significant:
it enables RUL estimation from a single accelerometer capture, with no historical data
and no assumption about where in the lifecycle the window sits.

---

## Top Papers to Compare Against

Priority papers to include as baselines in any IndustrialJEPA RUL paper:

### Tier 1: Must-Compare (Standard Baselines)

1. **MDSCT (Multi-scale Deep Separable Convolution Transformer, Heliyon 2024)**
   PMC11481647. Provides clean comparison table on both FEMTO and XJTU-SY.
   FEMTO nRMSE: 0.124. XJTU-SY nRMSE: 0.160.
   Why: Most recent clean benchmark with multiple baselines.

2. **CNN-GRU-MHA with Transfer Learning (Applied Sciences 2024)**
   MDPI 2076-3417/14/19/9039. nRMSE 0.0443 on FEMTO (best reported).
   Why: Current best result on FEMTO (if reproducible).

3. **DCSSL (Dual-Dimensional Contrastive SSL, Scientific Reports 2026)**
   Nature/s41598-026-38417-7. Self-supervised + RUL on FEMTO.
   Why: The only SSL-based RUL paper on FEMTO — direct methodological competitor.

### Tier 2: Important Context

4. **CVAER (Envelope Spectra + Probabilistic VAE, PMC 2024)**
   PMC11597903. XJTU-SY, 28.47 min RMSE.
   Why: Closest thing to per-spectrum (single-snapshot) prediction; establishes
   feasibility of the single-window formulation even if not framed that way.

5. **Bi-LSTM-Transformer + EMD (Applied Sciences 2025)**
   MDPI 2076-3417/15/17/9529. RMSE 0.0563 on XJTU-SY, validated on FEMTO.
   Why: Strong 2025 supervised baseline for fair comparison.

### Tier 3: Cite for Context

6. **Nectoux et al. 2012** — PRONOSTIA platform description, PHM challenge setup.
   Cite as: "Nectoux, P. et al. PRONOSTIA: An experimental platform for bearings
   accelerated degradation tests. IEEE PHM 2012 Challenge."
   HAL: hal-00719503.

7. **RULSurv (arXiv 2405.01614, 2024)** — Survival analysis framing, censoring-aware.
   Methodologically distinct; useful to cite as alternative probabilistic approach.

8. **OpenMAE (ACM IMWUT 2025)** — MAE pretraining on vibration signals.
   Closest foundation-model competitor on the pretraining side (not RUL specifically).

---

## Key Positioning Takeaways for IndustrialJEPA

### What We Would Be the First to Do

1. Apply JEPA pretraining to bearing RUL prediction (no prior work)
2. Define and solve the single-window position-agnostic RUL task (no prior formulation)
3. Show that JEPA embeddings carry sufficient degradation state information
   for single-window RUL regression

### How to Frame the Contribution

Existing narrative in the field: "Single-window RUL estimation is a known weakness —
you need temporal context to predict RUL."

Our counter-claim: "A JEPA representation learned from large multi-source vibration data
encodes physical degradation state in a single window's embedding. With no temporal
context, our model achieves competitive RUL prediction — suggesting the signal is rich
enough; the failure mode was lack of a strong enough representation."

### Target Metrics to Beat

- FEMTO nRMSE: beat 0.0443 (CNN-GRU-MHA) with no temporal context. If we get <0.10
  without temporal context vs their 0.0443 with full trajectory, the story is still strong.
- XJTU-SY nRMSE: beat 0.170 (SAGCN-SA) without temporal context.
- A meaningful result: showing that our single-window approach gets within 2x of
  trajectory-based SOTA would be a strong result justifying the new problem formulation.

**Why:** IndustrialJEPA must be positioned against these numbers so reviewers see
exactly where we stand relative to methods that use orders of magnitude more temporal
context.
**How to apply:** When writing the experiment section, explicitly state: "We compare
against Tier 1 baselines, noting that all baselines use full degradation trajectory
context while our method uses only a single window."
