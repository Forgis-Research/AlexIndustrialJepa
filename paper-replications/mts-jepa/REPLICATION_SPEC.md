# MTS-JEPA Replication Specification

**Paper**: "MTS-JEPA: Multi-Resolution Joint-Embedding Predictive Architecture for Time-Series Anomaly Prediction"
**Authors**: Yanan He, Yunshi Wen, Xin Wang, Tengfei Ma
**Affiliations**: Purdue University, Rensselaer Polytechnic Institute, Stony Brook University
**Venue**: *arXiv preprint* arXiv:2602.04643v1, February 5, 2026
**Correspondence**: tengfei.ma@stonybrook.edu

---

## Goal

Replicate Table 1 (main benchmark) from the paper: F1, AUC, Precision, Recall on all four anomaly prediction benchmarks (MSL, SMAP, SWaT, PSM). MTS-JEPA is directly relevant to our IndustrialJEPA project — it's the first multi-resolution JEPA for time-series anomaly prediction, with a soft codebook bottleneck that addresses representation collapse.

---

## Target Results (Table 1 of paper)

All metrics in percentage (%), averaged over 5 independent runs.

| Dataset | F1    | AUC   | Prec  | Rec   |
|:-------:|:-----:|:-----:|:-----:|:-----:|
| MSL     | **33.58** | **66.08** | **35.87** | **40.80** |
| SMAP    | **33.64** | **65.41** | **24.24** | **56.02** |
| SWaT    | **72.89** | **84.95** | **98.00** | **58.05** |
| PSM     | **61.61** | **77.85** | **55.01** | **72.00** |

MTS-JEPA achieves top AUC on all four datasets. Bold = best across 10 baselines.

---

## Architecture: MTS-JEPA

### Problem Setting
- **Anomaly prediction** (not detection): given context window X_t, predict whether the *next* window X_{t+1} will be anomalous.
- Input: multivariate time series x in R^{T x V}
- Partition into non-overlapping windows of length T_w = P * L (P patches of length L)
- Context window length T_c = 100, target window length T_t = 100
- Stride between consecutive pairs: 100 (non-overlapping)

### Overall Framework (Figure 2 / Figure 5)

```
Input Time Series
  |
  +-- Context window X_t --> Fine patches X_t^fine in R^{P x L x V}
  |                                |
  |                                v
  |                        Online Encoder (E_theta)
  |                                |
  |                                v
  |                        h_t in R^{P x D}  (continuous latent)
  |                                |
  |                        +-------+-------+
  |                        |               |
  |                   Codebook Q      Decoder D
  |                        |               |
  |                  p_t, z_t         X_hat_t (reconstruction)
  |                        |
  |                   +----+----+
  |                   |         |
  |              Fine Pred.  Coarse Pred.
  |                   |         |
  |          Pi_hat^fine   Pi_hat^coarse
  |
  +-- Target window X_{t+1}
        |
        +-- Fine view X_{t+1}^fine --> EMA Encoder --> EMA Codebook --> Pi_{t+1}^fine (target)
        +-- Coarse view X_{t+1}^coarse (downsampled) --> EMA Encoder --> EMA Codebook --> Pi_{t+1}^coarse (target)
```

### Component 1: Input Formulation and Multi-Scale Views (Section 3.2)

1. Apply **RevIN** (Kim et al., 2022) to normalize each window. Cache affine statistics for reconstruction loss.
2. **Fine View**: tokenize into P non-overlapping patches of length L. Result: X_t^fine in R^{P x L x V}
3. **Coarse View**: downsample by averaging every P consecutive time points. Result: X_t^coarse in R^{1 x L x V}
4. Online branch sees only X_t^fine. Target branch sees both X_{t+1}^fine and X_{t+1}^coarse.

### Component 2: Shared Encoder (Section 3.3)

- Channel-independent input formulation
- Residual CNN tokenizer followed by 6-layer Transformer backbone (8 heads, dropout 0.1)
- Projection head: 2-layer MLP (hidden 64 -> 32) mapping to latent dimension D=256
- Same encoder maps both fine-grained patches and the single coarse-grained token

### Component 3: Soft Codebook Bottleneck (Section 3.3)

- K=128 learnable prototypes {c_k} in R^D where D=256
- Temperature-scaled cosine similarity: p_{t,i,k} = softmax(<h_bar_{t,i}, c_bar_k> / tau), tau=0.1
- Output: code distribution p_{t,i} in Delta^{K-1} and soft-quantized embedding z_{t,i} = sum_k p_{t,i,k} * c_k
- Window-level code sequence: Pi_t = [p_{t,1}; ...; p_{t,P}] in R^{P x K}

### Component 4: Dual-Branch Predictor (Section 3.3)

Both are 2-layer Transformers (4 heads, hidden dim 128):

1. **Fine Predictor**: maps Pi_t to Pi_hat^fine_{t+1} in R^{P x K} (patch-level predictions)
2. **Coarse Predictor**: learnable query token q aggregates entire history via cross-attention:
   Pi_hat^coarse_{t+1} = CrossAttn(q, Pi_t) in R^{1 x K}

### Component 5: Reconstruction Decoder (Section 3.3)

- Auxiliary decoder D reconstructs input patches X_t^fine from soft-quantized embeddings z_t
- Evaluated after inverting RevIN using cached statistics
- Prevents representation collapse by anchoring latent space to signal-level semantics

### Component 6: EMA Target Branch

- Momentum-updated copy of online encoder + codebook
- Processes both X_{t+1}^fine and X_{t+1}^coarse to produce targets Pi_{t+1}^fine and Pi_{t+1}^coarse
- EMA rate rho = 0.996: xi <- rho * xi + (1 - rho) * theta
- Gradients are NOT back-propagated through EMA targets (stop-gradient)

---

## Training Objective (Section 3.4)

### Overall Loss
L = L_pred + L_code + lambda_r * L_rec

### (A) Predictive Objective (L_pred)
L_pred = lambda_f * (L_KL^fine + gamma * L_MSE^fine) + lambda_c * L_KL^coarse

Where:
- L_KL^fine = sum_i D_KL(p^fine_{t+1,i} || p_hat^fine_{t+1,i})  (patch-level KL)
- L_MSE^fine = sum_i ||z_{t+1,i} - z_hat_{t+1,i}||^2  (embedding-level MSE)
- L_KL^coarse = D_KL(p^coarse_{t+1} || p_hat^coarse_{t+1})  (window-level KL)

### (B) Codebook Objective (L_code)
L_code = lambda_emb * L_emb + lambda_com * L_com + lambda_ent^sample * L_ent^sample - lambda_ent^batch * L_ent^batch

Where:
- L_emb = sum_i ||sg(z_{t,i}) - h_{t,i}||^2  (codebook -> encoder alignment)
- L_com = sum_i ||z_{t,i} - sg(h_{t,i})||^2  (encoder -> codebook alignment)
- L_ent^sample = E[H(p)]  (minimize per-sample entropy for sharp assignments)
- L_ent^batch = H(E[p])  (maximize batch entropy for diverse code usage)

### (C) Reconstruction Objective (L_rec)
L_rec = sum_i ||X_hat_{t,i} - X_{t,i}||^2  (after RevIN denormalization)

---

## Hyperparameters (Appendix B.2)

### Architecture
| Component | Parameter | Value |
|:----------|:----------|:-----:|
| Encoder | Layers | 6 |
| Encoder | Heads | 8 |
| Encoder | Dropout | 0.1 |
| Encoder | Projection head | MLP: 64 -> 32 -> D |
| Latent dim D | | 256 |
| Codebook size K | | 128 |
| Codebook dim | | 256 |
| Soft-max temperature tau | | 0.1 |
| Patch length L | | 20 |
| Patches per window P | | 5 (since T_w = P*L = 100) |
| Predictors | Layers | 2 |
| Predictors | Heads | 4 |
| Predictors | Hidden dim | 128 |

### Training
| Parameter | Value |
|:----------|:-----:|
| Optimizer | Adam |
| Learning rate | 5e-4 |
| Weight decay | 1e-5 |
| Batch size | 128 |
| Max epochs | 100 |
| Early stopping patience | 10 (after epoch 50) |
| Gradient clipping norm | 0.5 |
| EMA decay rho | 0.996 |

### Loss Weights
| Weight | Value |
|:-------|:-----:|
| lambda_f (fine prediction) | 1.0 |
| gamma (MSE local) | 0.1 |
| lambda_c (coarse prediction) | 0.5 |
| lambda_emb (codebook alignment) | 1.0 |
| lambda_com (commitment) | 0.25 |
| lambda_ent^sample | 0.005 |
| lambda_ent^batch | 0.01 |
| lambda_r (reconstruction) | annealed 0.5 -> 0.1 linearly |

---

## Data Protocol

### Datasets (Table 5 / Appendix B.1)

| Dataset | Raw Dim | Eff. Dim | Constant Indices (Zero Var) | Train | Test | Anom (%) |
|:--------|:-------:|:--------:|:---------------------------|:-----:|:----:|:--------:|
| MSL | 55 | 33 | [1,4,8,10,18,21,22,24,25,26,30,32,34,36,37,38,40,42,44,50,51,52] | 58,317 | 73,729 | 10.5 |
| SMAP | 25 | 24 | [16] | 135,183 | 427,617 | 12.8 |
| SWaT | 51 | 40 | [4,10,11,13,15,29,31,32,43,48,50] | 495,000 | 449,919 | 12.1 |
| PSM | 25 | 25 | - | 132,481 | 87,841 | 27.8 |

### Preprocessing
1. Remove constant channels (zero variance) on training split; apply same mask to test split
2. Instance-normalize independently per window (RevIN)
3. Construct non-overlapping context-target pairs: (X_t, X_{t+1}) with stride 100
4. Each window T_w = 100, tokenized into P=5 patches of length L=20

### Train/Val/Test Split
- **Pre-training**: 9:1 train-validation split for model selection and early stopping
- **Downstream**: 6:2:2 chronological split
  - Train classifier on first 60%
  - Select threshold delta* on next 20% to maximize F1
  - Evaluate on final 20%

### Evaluation Protocol
- Window-level anomaly label: y_{t+1} = 1 if *any* point in future window is anomalous
- Metrics: Precision, Recall, F1 (with threshold), AUC (threshold-free)
- Report mean +/- std over 5 independent runs

### Downstream Adaptation (Algorithm 2)
1. Freeze pre-trained encoder and codebook; discard projection heads
2. Encode context: p_t in R^{V x P x K}
3. Variable-wise max-pooling: h_t = max_v p_t^(v) in R^{P x K}
4. Flatten h_t and pass to MLP classifier C_psi
5. Train with binary cross-entropy; select threshold on validation F1

---

## Baselines (Table 1)

10 baselines across 3 categories:
1. **Classical**: K-Means, DeepSVDD
2. **Reconstruction**: LSTM-VAE
3. **Time-series backbones**: iTransformer, TimesNet, PatchTST, TS2Vec
4. **Anomaly prediction**: PAD (Jhin et al., 2023)
5. **JEPA baseline**: TS-JEPA (Ennadir et al., 2025)

---

## Implementation Priority

1. Download MSL, SMAP, SWaT, PSM datasets
2. Write data loader: remove constant channels, construct context-target pairs, RevIN
3. Implement shared encoder (residual CNN + Transformer)
4. Implement soft codebook with temperature-scaled cosine similarity
5. Implement dual-branch predictors (fine Transformer + coarse cross-attention)
6. Implement reconstruction decoder
7. Implement EMA target network
8. Wire up full training loop with all loss terms (Algorithm 1)
9. Run test pipeline (5 epochs on smallest dataset) — verify no NaN, shapes correct
10. Pre-train on each dataset (in-domain setting)
11. Implement downstream classifier (Algorithm 2) with threshold selection
12. Evaluate on all four datasets; compare to Table 1

---

## Known Unknowns and Risks

1. **Dataset access**: MSL and SMAP are from NASA (publicly available). SWaT requires registration from iTrust Singapore. PSM is from eBay (publicly available).
2. **RevIN details**: Paper says "affine statistics are cached" for reconstruction loss. Must ensure the same normalization/denormalization as Kim et al. (2022).
3. **Residual CNN tokenizer**: Exact architecture not fully specified beyond "residual CNN tokenizer followed by Transformer." Check if this follows a specific prior work.
4. **Reconstruction weight annealing**: lambda_r goes from 0.5 to 0.1 linearly. Over how many epochs? Assume over the full 100 epochs.
5. **Downstream classifier MLP**: Architecture not specified. Standard 2-layer MLP with ReLU.
6. **Window stride during downstream**: The 6:2:2 split is chronological. Are downstream windows non-overlapping or sliding? Paper says context-target pairs are non-overlapping with stride 100.
7. **Constant channel indices**: Table 5 lists indices but 1-indexing vs 0-indexing isn't specified. Verify by computing variance on training data.
8. **GPU requirements**: Paper used single NVIDIA RTX 4090. Training times not specified.
9. **Code availability**: No code released as of the arxiv preprint date.
