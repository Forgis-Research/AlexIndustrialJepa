# NeurIPS Review: MTS-JEPA

**Simulated review following the official NeurIPS review form.**

---

## Summary

This paper proposes MTS-JEPA, an extension of the Joint-Embedding Predictive Architecture (JEPA) framework to multivariate time-series anomaly prediction. The two key design choices are: (1) a dual-resolution predictive objective where a fine-grained patch-level branch and a coarse downsampled branch jointly predict future latent states, and (2) a soft codebook bottleneck that maps continuous encoder features to a finite set of K=128 learnable prototypes via differentiable vector quantization. The authors provide theoretical analysis showing the codebook bounds representation drift (Theorem A.3) and prevents collapse (Theorem A.10). Experiments on four anomaly detection benchmarks (MSL, SMAP, SWaT, PSM) repurposed for a "prediction" setting show MTS-JEPA achieves top AUC across all datasets against 10 baselines including TS-JEPA, PatchTST, iTransformer, and PAD.

---

## Scores

| Criterion | Score |
|:----------|:-----:|
| Soundness | 6/10 |
| Significance | 5/10 |
| Novelty | 6/10 |
| Clarity | 7/10 |
| Reproducibility | 7/10 |
| **Overall** | **5/10** |
| Confidence | 4/5 |

**Verdict**: Borderline reject. Strong architecture and theory, but evaluation gaps undermine the claims.

---

## Strengths

1. **Well-motivated problem formulation.** The paper clearly distinguishes anomaly *prediction* (early warning from a preceding context window) from anomaly *detection* (scoring the current window), and makes a convincing argument that JEPA's latent-space prediction is a natural fit for this task.

2. **Principled architecture design with theoretical grounding.** The soft codebook bottleneck has a stability upper bound (Theorem A.3) showing representation drift is controlled by the codebook radius M, and a non-collapse lower bound (Theorem A.10) showing Tr(Cov(z)) > 0 under reasonable conditions. The chain through Pinsker's inequality to bridge KL divergence to L1 distance to Euclidean drift (Lemma A.2) is elegant.

3. **Comprehensive ablation study.** Table 3/8 systematically removes each component. The codebook module removal causing near-collapse (std -> 0) is one of the strongest parts of the paper and directly validates the theoretical claims.

4. **Cross-domain transfer experiment.** Table 2/7 evaluates generalization by pre-training on three datasets and evaluating on the held-out fourth. MTS-JEPA degrades less than PatchTST and TS2Vec under domain shift.

5. **Reproducibility.** Thorough appendix: full hyperparameters, algorithmic pseudocode (Algorithms 1-2), dataset statistics with exact constant-channel removal lists, downstream protocol details. Single RTX 4090 makes replication accessible.

6. **Codebook interpretability analysis.** Figures 3/4/6/7 showing differential code activation for anomalous vs. normal windows provides qualitative evidence that the discrete bottleneck learns semantically meaningful regime representations.

---

## Weaknesses

### 1. "Prediction" framing is ambiguous and potentially misleading

The window-level label (Eq. 28) defines y_{t+1} = 1 if *any* point in the target window is anomalous. With T_c = T_t = 100 and stride 100, the model observes timesteps 1-100 and predicts whether 101-200 contain anomalies. This is genuinely predictive only if the anomaly begins *after* timestep 100. On datasets like SWaT where attacks span hundreds of timesteps, many anomalous target windows will have anomalous context windows too — making this closer to detection-with-delay than true early warning.

**Fix**: Report the fraction of test windows where the context window is fully normal but the target is anomalous (true early-warning cases) and evaluate AUC on this subset separately.

### 2. Baseline fairness concerns

Detection methods (DeepSVDD, LSTM-VAE) are not designed for prediction and are expected to perform poorly. Their inclusion inflates the apparent advantage. Key missing baselines:
- Park et al. (2025) "When will it fail?" — cited but not benchmarked; most directly comparable
- Zhao et al. (2024) "Abnormality forecasting" — cited but not benchmarked
- Anomaly Transformer — listed in Section 4.2 but absent from Table 1 (unexplained)

### 3. Statistical significance unclear

Examining Table 6 closely:
- MSL AUC: MTS-JEPA 66.08 +/- 3.25 vs. TS2Vec ~comparable — confidence intervals overlap
- SMAP AUC: MTS-JEPA 65.41 vs. PatchTST 61.62 — gap of ~4 points but std is 2.06
- PSM F1: MTS-JEPA 61.61 vs. TS2Vec 48.43 — this is a convincing gap

No paired statistical tests reported. **Fix**: Paired t-tests or Wilcoxon signed-rank tests between MTS-JEPA and top-2 baselines on each dataset.

### 4. Limited dataset diversity

All four datasets are standard anomaly *detection* benchmarks, not prediction benchmarks. No physical/mechanical prognostics datasets (C-MAPSS, FEMTO bearings) where early warning is the actual use case and lead time can be measured in physical units.

### 5. Theoretical results lack empirical validation

Theorem A.3 bounds drift as a function of M, epsilon, delta — but these quantities are never measured during training. Theorem A.10 requires alpha > 0 and 2*M*epsilon < Delta_c — not verified empirically.

**Fix**: Plot M, epsilon_t, delta_t, and the resulting bound throughout training.

### 6. No training cost comparison

Dual encoders (online + EMA), dual predictors, codebook, decoder — substantially more complex than baselines. 8 loss terms with 7+ weighting hyperparameters raise tuning cost concerns. No wall-clock time, GPU memory, or FLOPs reported.

---

## Questions for the Authors

1. What fraction of "correctly predicted" anomalous windows have context windows that are themselves partially or fully anomalous?
2. Why is Anomaly Transformer listed as a baseline in Section 4.2 but absent from Table 1?
3. How sensitive are results to the downstream MLP architecture? Have you tried a linear probe?
4. Have you ablated codebook size K? Does optimal K correlate with system complexity?
5. Is M (codebook radius) regularized during training, or does it grow unboundedly (weakening the stability guarantee)?
6. What motivated the reconstruction weight annealing schedule? Does the model eventually become robust enough to fully remove it?

---

## Missing References

- Park et al. (2025) "When will it fail?" — most comparable prediction method, cited but not benchmarked
- Zhao et al. (2024) "Abnormality forecasting" — cited but not benchmarked
- TranAD (Tuli et al., 2022), USAD (Audibert et al., 2020), DCdetector (Yang et al., 2023) — strong baselines on SWaT/PSM
- TimesFM (Das et al., 2024) or other foundation models

---

## Minor Issues

- Paper title uses "MTS-JEPA" but arXiv filename says "MST-JEPA" — ensure consistency
- Table 1 reports means without std (only in appendix Table 6) — std should be in the main table
- Figure 3 is very small and hard to read at standard viewing size
- "State-of-the-art" claim should be qualified to these 4 benchmarks under this specific evaluation protocol
