# Overnight: MTS-JEPA — Replication, Critique, Extension, and Discovery

**Goal**: Four-phase research campaign. (1) Perfectly replicate MTS-JEPA. (2) Implement fixes from our NeurIPS review + benchmark our Trajectory JEPA against their results. (3) Deep literature review + brainstorming loop for groundbreaking extensions. (4) Implement, review, iterate until publication-ready.

**Agent**: ml-researcher
**Estimated duration**: 12-16 hours (extended overnight run)
**Working directory**: `C:\Users\Jonaspetersen\dev\IndustrialJEPA\paper-replications\mts-jepa`
**Target venue**: NeurIPS 2026

---

## CRITICAL: Read Before Starting

Read these files in order. Do NOT skip any — each informs the next phase.

1. `REPLICATION_SPEC.md` — complete architectural specification with all hyperparameters
2. `he2026-mts-jepa.pdf` — the original paper (ALL sections including appendix)
3. `CRITICAL_REVIEW.md` — our 9-point weakness analysis + opportunity map
4. `NEURIPS_REVIEW.md` — simulated NeurIPS review (5/10, borderline reject). This contains the exact gaps we must fill.
5. `../star/` — successful replication pattern (code organization, experiment logging)
6. `../when-will-it-fail/` — A2P replication (anomaly prediction setup, similar domain)
7. `../../mechanical-jepa/experiments/v11/` — **our current best method**: Trajectory JEPA V2, C-MAPSS FD001, RMSE 13.80 ± 0.75 (E2E), first SSL to beat public SSL reference (AE-LSTM 13.99)
8. `../../mechanical-jepa/notebooks/11_v11_cmapss_trajectory_jepa.qmd` — V11 walkthrough

---

# PHASE 1: PERFECT REPLICATION (4-5 hours)

The goal is not "close enough" — it is **exact match** of every number in Table 1. If we can't match within 5%, we don't understand the method well enough to improve it.

## Part A: Dataset Acquisition and Data Pipeline (90 min)

### A.1: Download Datasets

Check if datasets already exist under `C:\Users\Jonaspetersen\dev\IndustrialJEPA\` (any subdirectory). If not:

- **MSL & SMAP**: NASA spacecraft telemetry datasets from the Telemanom repository.
  - Source: https://github.com/khundman/telemanom or search for "SMAP MSL anomaly detection dataset"
  - MSL: 55 dimensions, 58,317 train / 73,729 test points
  - SMAP: 25 dimensions, 135,183 train / 427,617 test points
  - Both include anomaly labels as binary arrays

- **SWaT**: Secure Water Treatment testbed (iTrust, Singapore)
  - This may require registration. Check if data is already available locally.
  - 51 dimensions, 495,000 train / 449,919 test points
  - If unavailable, proceed with MSL, SMAP, PSM and note SWaT as blocked

- **PSM**: Pooled Server Metrics from eBay
  - Source: https://github.com/eBay/RANSynCoders or search for "PSM dataset anomaly detection"
  - 25 dimensions, 132,481 train / 87,841 test points

### A.2: Data Loader

```python
class AnomalyPredictionDataset:
    """
    Constructs non-overlapping context-target pairs for anomaly prediction.
    
    Given continuous multivariate time series and point-level anomaly labels:
    1. Remove constant channels (zero variance on training split)
    2. Partition into non-overlapping windows of length T_w = 100
    3. Construct pairs (X_t, X_{t+1}) where X_t is context and X_{t+1} is target
    4. Window-level label: y_{t+1} = 1 if ANY point in X_{t+1} is anomalous
    
    For pre-training: return (X_t, X_{t+1}) pairs without labels
    For downstream: return (X_t, y_{t+1}) pairs with labels
    """
```

### A.3: RevIN (Reversible Instance Normalization)

```python
class RevIN(nn.Module):
    """
    Kim et al. (ICLR 2022). Per-window, per-channel normalization.
    
    Forward: normalize with running mean/std, cache statistics
    Inverse: denormalize using cached statistics (for reconstruction loss)
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        # learnable affine parameters (optional)
    
    def forward(self, x):  # x: (B, T, V)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        # cache mean, std for inverse
        return (x - mean) / (std + eps) * gamma + beta
    
    def inverse(self, x):
        return (x - beta) / gamma * cached_std + cached_mean
```

### A.4: Multi-Scale View Construction

```python
def create_views(window):
    """
    window: (B, T_w, V) normalized window
    
    Returns:
        fine_view: (B, P, L, V) — P=5 patches of length L=20
        coarse_view: (B, 1, L, V) — average every P=5 consecutive time points
    """
```

### A.5: Constant Channel Removal

Compute variance of each channel on training data. Remove channels with zero (or near-zero) variance. Apply the same mask to test data. Verify against Table 5 of the paper:
- MSL: 55 -> 33 effective dimensions
- SMAP: 25 -> 24 effective dimensions
- SWaT: 51 -> 40 effective dimensions
- PSM: 25 -> 25 (no removal)

### A.6: Train/Val/Test Splits

**Pre-training**: Use official training set. 9:1 split for train vs validation (chronological).
**Downstream**: Use official test set. 6:2:2 chronological split:
- First 60% of test windows: train classifier
- Next 20%: select threshold delta* maximizing F1
- Final 20%: report metrics

---

## Part B: MTS-JEPA Architecture Implementation (150 min)

### B.1: Shared Encoder

```python
class MTSJEPAEncoder(nn.Module):
    """
    Channel-independent encoder.
    
    1. Residual CNN tokenizer: convert (L, 1) patches to d_model tokens
    2. 6-layer Transformer (8 heads, dropout 0.1)
    3. Projection head: 2-layer MLP (hidden 64 -> 32 -> D=256)
    
    Input: (B, P, L, V) or (B, 1, L, V)
    Process channel-independently: reshape to (B*V, P, L) -> encode -> (B*V, P, D)
    Reshape back: (B, V, P, D)
    """
```

**IMPORTANT**: The encoder is channel-independent — each variable is processed as a separate univariate series. The Transformer operates over the P=5 patch tokens (or 1 token for coarse view). Cross-variable interaction happens only at the downstream aggregation stage (variable-wise max-pooling).

### B.2: Soft Codebook

```python
class SoftCodebook(nn.Module):
    """
    K=128 learnable prototypes in R^D, D=256.
    
    For each patch representation h in R^D:
    1. L2-normalize h and all prototypes c_k
    2. Compute cosine similarities: sim_k = <h_bar, c_bar_k>
    3. Apply temperature scaling: p_k = softmax(sim / tau), tau=0.1
    4. Soft-quantized embedding: z = sum_k p_k * c_k
    
    Returns: (p, z) where p in Delta^{K-1} and z in R^D
    """
```

### B.3: Fine Predictor

```python
class FinePredictor(nn.Module):
    """
    2-layer Transformer (4 heads, hidden 128).
    
    Input: Pi_t in R^{P x K} (code sequence from codebook)
    Output: Pi_hat^fine_{t+1} in R^{P x K} (predicted fine code distributions)
    
    Also predicts z_hat_{t+1} = sum_k p_hat_k * c_k for MSE loss.
    """
```

### B.4: Coarse Predictor

```python
class CoarsePredictor(nn.Module):
    """
    Learnable query token q + cross-attention over Pi_t.
    Then 2-layer Transformer (4 heads, hidden 128).
    
    Input: Pi_t in R^{P x K}
    Output: Pi_hat^coarse_{t+1} in R^{1 x K} (single global prediction)
    
    CrossAttn(q, Pi_t): q provides query, Pi_t provides keys and values.
    """
```

### B.5: Reconstruction Decoder

```python
class ReconstructionDecoder(nn.Module):
    """
    Reconstructs input patches from soft-quantized embeddings.
    
    Input: z_t in R^{P x D}
    Output: X_hat_t in R^{P x L x V} (after RevIN inverse)
    """
```

### B.6: Full MTS-JEPA Model

```python
class MTSJEPA(nn.Module):
    def __init__(self, n_vars, d_model=256, n_codes=128, tau=0.1,
                 patch_length=20, n_patches=5, ...):
        self.encoder = MTSJEPAEncoder(...)
        self.codebook = SoftCodebook(K=128, D=256, tau=0.1)
        self.fine_predictor = FinePredictor(...)
        self.coarse_predictor = CoarsePredictor(...)
        self.decoder = ReconstructionDecoder(...)
        # EMA copies
        self.ema_encoder = copy.deepcopy(self.encoder)
        self.ema_codebook = copy.deepcopy(self.codebook)
    
    @torch.no_grad()
    def update_ema(self, rho=0.996):
        for p_online, p_ema in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            p_ema.data = rho * p_ema.data + (1 - rho) * p_online.data
        # same for codebook
    
    def forward(self, x_context, x_target):
        # Online branch: encode context, get codes, predict future
        # Target branch (no grad): encode target at both resolutions, get target codes
        # Reconstruction branch: decode context embeddings
        # Return all losses
```

---

## Part C: Training Protocol (60 min)

### C.1: Pre-training Loop (Algorithm 1)

```python
def pretrain_mtsjepa(model, train_loader, val_loader, config):
    optimizer = Adam(model.online_params(), lr=5e-4, weight_decay=1e-5)
    
    for epoch in range(100):
        # Anneal lambda_r from 0.5 to 0.1
        lambda_r = 0.5 - (0.4 * epoch / 99)
        
        for x_t, x_t1 in train_loader:
            # Forward pass (Algorithm 1, lines 4-16)
            losses = model(x_t, x_t1)
            
            # Compute total loss
            L_pred = lambda_f * (L_KL_fine + gamma * L_MSE_fine) + lambda_c * L_KL_coarse
            L_code = lambda_emb * L_emb + lambda_com * L_com + lambda_ent_sample * L_ent_sample - lambda_ent_batch * L_ent_batch
            L_rec = lambda_r * reconstruction_loss
            L_total = L_pred + L_code + L_rec
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.online_params(), max_norm=0.5)
            
            # EMA update
            model.update_ema(rho=0.996)
        
        # Early stopping after epoch 50 with patience 10
```

### C.2: Downstream Evaluation (Algorithm 2)

```python
def downstream_evaluate(pretrained_model, dataset, config):
    """
    1. Freeze encoder + codebook
    2. Encode all context windows -> code distributions p_t in R^{V x P x K}
    3. Variable-wise max-pooling: h_t = max_v p_t^(v) in R^{P x K}
    4. Flatten: h_t in R^{P*K} = R^{640}
    5. Train MLP classifier on 60% of test data
    6. Select threshold on 20% validation to maximize F1
    7. Evaluate on final 20%: report F1, AUC, Precision, Recall
    """
```

### C.3: Loss Functions

Implement each loss term separately for debugging and ablation:

```python
def kl_divergence(p_pred, p_target):
    """KL(p_target || p_pred) summed over patches."""

def embedding_mse(z_pred, z_target):
    """||z_pred - z_target||^2 summed over patches."""

def codebook_alignment(h, z):
    """Bidirectional alignment with stop-gradient."""
    L_emb = ||sg(z) - h||^2
    L_com = ||z - sg(h)||^2

def dual_entropy(p_batch):
    """
    Sample entropy: E[H(p)] — minimize for sharp assignments
    Batch entropy: H(E[p]) — maximize for diverse code usage
    """

def reconstruction_loss(x_hat, x, revin):
    """MSE after RevIN denormalization."""
```

---

## Part D: Test Pipeline (30 min)

Write `test_pipeline.py` and run BEFORE any full experiments:

1. Load smallest available dataset (likely PSM or SMAP)
2. Create 100 context-target pairs
3. Build MTS-JEPA model with paper hyperparameters
4. Forward pass: verify all output shapes
   - Encoder output: (B, V, P, D) with D=256
   - Code distributions: (B, V, P, K) with K=128
   - Fine predictions: (B, V, P, K)
   - Coarse predictions: (B, V, 1, K)
   - Reconstruction: (B, P, L, V)
5. Run 5 epochs: verify loss decreases, no NaN, no mode collapse
6. Check codebook utilization: how many of K=128 codes are used?
7. Run downstream classifier on 50 samples: verify F1/AUC computation works

**Do NOT proceed to Part E if test_pipeline.py fails.**

---

## Part E: Pre-training on All Datasets (180 min)

For each dataset (MSL, SMAP, PSM, and SWaT if available):

1. Pre-train MTS-JEPA for 100 epochs with early stopping (patience 10 after epoch 50)
2. Use 5 seeds: [42, 123, 456, 789, 1024]
3. Save best model checkpoint per seed
4. Log to EXPERIMENT_LOG.md after each run:
   - Pre-training losses (pred, code, rec, total)
   - Best validation epoch
   - Wall-clock time
   - Codebook utilization stats

If SWaT is unavailable, proceed with the other three and document the gap.

---

## Part F: Downstream Evaluation + Verification (60 min)

For each dataset and seed:

1. Load pre-trained model
2. Freeze encoder + codebook
3. Construct 6:2:2 chronological splits on the test set
4. Train MLP classifier (binary cross-entropy)
5. Select threshold on validation split (maximize F1)
6. Evaluate on test split: F1, AUC, Precision, Recall
7. Save per-seed results as JSON

### Replication Verification Gate

After completing all datasets, compare results to Table 1 **and** Table 6 (with std). Compute:
- Mean ± std across 5 seeds for each metric and dataset
- Delta vs paper for every cell in the table
- Status per dataset: EXACT (<5%), GOOD (<15%), MARGINAL (<25%), FAILED (>25%)

**CRITICAL**: If ANY dataset is FAILED status, do NOT proceed to Phase 2. Instead:
1. Diagnose the failure — compare loss curves, codebook stats, downstream classifier behavior to expectations
2. Re-read the paper for missed details (especially Appendix B)
3. Fix and re-run until at least GOOD on all datasets
4. Document what was wrong in EXPERIMENT_LOG.md

### Results JSON Format

```json
{
    "dataset": "MSL",
    "seed": 42,
    "pretrain": {
        "best_epoch": 67,
        "val_loss": 0.234,
        "wall_time_seconds": 3600,
        "codebook_utilization": 0.85,
        "codebook_perplexity": 42.3,
        "dead_codes": 12
    },
    "downstream": {
        "threshold": 0.42,
        "f1": 33.2,
        "auc": 65.8,
        "precision": 35.1,
        "recall": 40.3
    },
    "paper_target": {
        "f1": 33.58,
        "auc": 66.08,
        "precision": 35.87,
        "recall": 40.80
    },
    "delta_pct": {
        "f1": -1.1,
        "auc": -0.4
    }
}
```

---

## Part G: Replicate Ablations (90 min)

Replicate Table 3 ablations — ALL of them, not optional. These validate our understanding:

1. **w/o KL Regularization**: Remove KL terms from L_pred, keep only MSE
2. **w/o Reconstruction Decoder**: Remove decoder and L_rec entirely
3. **w/o Predictive Objective**: Remove L_pred entirely
4. **w/o Codebook Loss**: Keep codebook module but remove auxiliary losses (L_emb, L_com, entropy)
5. **w/o Codebook Module**: Remove codebook entirely, operate in continuous latent space
6. **w/o Downsampling**: Remove coarse branch, only fine predictor

Run each on 1 seed on PSM and MSL (smallest + most distinctive). Verify the key finding: **codebook module removal should cause near-collapse (std -> 0 across seeds).**

If the collapse finding does NOT replicate, this is critical information — document thoroughly.

---

# PHASE 2: FIXES, IMPROVEMENTS, AND BENCHMARKING (3-4 hours)

Now that we understand MTS-JEPA perfectly, implement the fixes identified in `CRITICAL_REVIEW.md` and `NEURIPS_REVIEW.md`, and benchmark our own method.

## Part H: Implement NeurIPS Review Fixes (120 min)

### H.1: Lead-Time Analysis (Critical Fix #1)

The NeurIPS review's top concern: is this really prediction or near-detection?

```python
def lead_time_analysis(dataset):
    """
    For every correctly-predicted anomalous window:
    1. Check if the context window X_t is itself anomalous (partial/full overlap)
    2. Compute the lead time: how many timesteps before the anomaly starts in X_{t+1}
    3. Categorize:
       - TRUE_PREDICTION: context fully normal, target anomalous (genuine early warning)
       - CONTINUATION: context already anomalous (detecting ongoing anomaly)
       - BOUNDARY: anomaly starts in last 20% of context (near-detection)
    
    Report:
    - Fraction of TRUE_PREDICTION vs CONTINUATION vs BOUNDARY
    - AUC computed ONLY on TRUE_PREDICTION subset
    - Mean lead time for true predictions
    """
```

This is the single most important diagnostic. If >50% of "predictions" are CONTINUATION, the paper's claims are significantly weaker. Document findings in `analysis/lead_time_analysis.md`.

### H.2: Fair Baseline Comparison (Critical Fix #2)

The paper includes detection methods solving a different task. Compute:
- MTS-JEPA performance on TRUE_PREDICTION subset only
- Separate table: prediction-only baselines (PAD, TS-JEPA) vs MTS-JEPA

### H.3: Statistical Significance (Critical Fix #3)

```python
def statistical_tests(our_results, paper_results):
    """
    For each dataset, for MTS-JEPA vs top-2 baselines:
    - Paired t-test across 5 seeds
    - Wilcoxon signed-rank test
    - Report p-values and effect sizes
    - Flag any dataset where p > 0.05
    """
```

### H.4: Theory Verification (Critical Fix #4)

Track the theoretical quantities during training:

```python
def track_theory_quantities(model, dataloader):
    """
    At each epoch, measure:
    1. M = max_k ||c_k||_2 (codebook radius)
    2. epsilon_t = mean KL between predictions and targets (prediction error)
    3. delta_t = mean ||p_{t+1} - p_t||_1 (target smoothness)
    4. Stability bound = M * (sqrt(2*eps_{t+1}) + delta_t + sqrt(2*eps_t))
    5. Actual drift = mean ||z_hat_{t+1} - z_hat_t||_2
    6. Tr(Cov(z)) for non-collapse verification
    7. Codebook utilization (perplexity, dead code count)
    
    Save as time series for plotting.
    """
```

Plot: (a) bound vs actual drift over training, (b) Tr(Cov(z)) over training, (c) codebook utilization over training. Save to `figures/theory_validation/`.

### H.5: Computational Analysis (Critical Fix #5)

Measure and report:
- Training wall-clock time per epoch per dataset
- Peak GPU memory usage
- Inference latency (ms per window)
- Parameter count breakdown (encoder, codebook, predictors, decoder)
- Compare to baselines if available (at minimum: TS-JEPA, PatchTST parameter counts)

---

## Part I: Benchmark Our Trajectory JEPA (90 min)

### I.1: Adapt Trajectory JEPA for Anomaly Prediction

Our current best method (V11 Trajectory JEPA, C-MAPSS FD001 RMSE 13.80) was designed for RUL prediction. Adapt it for the anomaly prediction benchmarks:

1. Read the Trajectory JEPA implementation from `../../mechanical-jepa/experiments/v11/`
2. Adapt the architecture:
   - Use Trajectory JEPA's causal Transformer encoder
   - Replace the RUL prediction head with an anomaly prediction head
   - Keep the variable-horizon prediction objective
   - Use the same pre-train -> freeze -> downstream classifier pipeline as MTS-JEPA
3. Pre-train on PSM and MSL (smallest datasets, fastest iteration)
4. Evaluate with the same 6:2:2 downstream protocol

### I.2: Cross-Method Comparison Table

```markdown
| Method | MSL F1 | MSL AUC | PSM F1 | PSM AUC | Notes |
|--------|--------|---------|--------|---------|-------|
| MTS-JEPA (paper) | 33.58 | 66.08 | 61.61 | 77.85 | |
| MTS-JEPA (ours)  | ??? | ??? | ??? | ??? | Replication |
| Trajectory JEPA  | ??? | ??? | ??? | ??? | Our V11 adapted |
| Trajectory JEPA + Codebook | ??? | ??? | ??? | ??? | Hybrid (see Part J) |
```

### I.3: What Our Method Does Better/Worse

Write a comparison analysis:
- Where does the multi-resolution help that our single-scale approach misses?
- Where does our variable-horizon prediction help that their fixed-window approach misses?
- Does our approach handle the TRUE_PREDICTION subset better? (Trajectory JEPA was designed for genuine future prediction)

---

# PHASE 3: DEEP LITERATURE REVIEW + BRAINSTORMING (2-3 hours)

## Part J: Literature Deep Dive (90 min)

### J.1: Search Strategy

Use web search to find papers on these specific topics (search each separately):

1. **"codebook time series anomaly"** — other VQ/codebook approaches for time series
2. **"JEPA time series"** — all JEPA adaptations beyond I-JEPA/V-JEPA/TS-JEPA
3. **"anomaly prediction early warning multivariate"** — the actual task (not detection)
4. **"multi-resolution self-supervised time series"** — multi-scale SSL approaches
5. **"representation collapse self-supervised time series"** — the stability problem MTS-JEPA addresses
6. **"prognostics health management JEPA"** or **"remaining useful life self-supervised"** — our specific application domain
7. **"discrete bottleneck representation learning"** — theoretical connections to information bottleneck
8. **"vector quantization variational"** time series — VQ-VAE and variants for time series
9. **"lead time prediction fault prognosis"** — proper early-warning metrics literature

For each search: record the top 3-5 most relevant papers with title, venue, year, and a one-line summary of why it matters.

### J.2: Gap Map

After the literature review, construct a gap map:

```markdown
## Gap Map: What Nobody Has Done Yet

| Gap | Status | Who's Closest | Our Advantage |
|-----|--------|---------------|---------------|
| JEPA + codebook for prognostics | OPEN | MTS-JEPA (anomaly), TS-JEPA (no codebook) | We have C-MAPSS infrastructure |
| Lead-time-aware evaluation | OPEN | Park et al. 2025 (partial) | Can define proper metric |
| Codebook = degradation regimes | OPEN | - | Physical interpretation possible on C-MAPSS |
| Multi-resolution for bearing vibration | OPEN | - | We have FEMTO/XJTU data |
| Theory-validated codebook bounds | OPEN | MTS-JEPA (theory only, no empirical) | We can validate |
| ... | ... | ... | ... |
```

Save to `analysis/gap_map.md`.

### J.3: First Brainstorming Round

Generate 10+ concrete research ideas. For each idea, write:
- **One-line pitch** (what a reviewer would remember)
- **Technical approach** (2-3 sentences)
- **Why it's novel** (what prior work doesn't do)
- **Risk level** (low/medium/high)
- **Effort** (days to prototype)

Categorize into:
- **Safe bets**: incremental but solid (publishable at a workshop)
- **Medium swings**: novel combinations (NeurIPS poster territory)
- **Moonshots**: risky but potentially groundbreaking (NeurIPS oral if it works)

### J.4: Idea Filtering

After brainstorming, apply these filters:
1. **Does it address a gap in the gap map?** If not, deprioritize.
2. **Can we prototype it in <2 days with our existing infrastructure?** If not, is the potential reward worth the investment?
3. **Does it make a NeurIPS reviewer say "that's clever"?** Not just "that's more engineering."
4. **Is there a clear experiment that proves it works or doesn't?** Unfalsifiable ideas are worthless.

Select the top 3 ideas for prototyping. Write the selection rationale.

### J.5: Second Brainstorming Round

After filtering, brainstorm AGAIN with the constraint: "What would make the top 3 ideas even stronger?" This second round often produces the real breakthroughs because the first round clears the obvious ideas.

Look specifically for:
- Theoretical connections (can we prove something about our approach?)
- Surprising combinations (what if we merged idea #1 and #3?)
- Adversarial thinking (what would a reviewer attack? Preempt it.)

Save all brainstorming to `analysis/brainstorming.md`.

---

# PHASE 4: IMPLEMENT, REVIEW, ITERATE (3-4 hours)

## Part K: Prototype Top Ideas (120 min)

### K.1: Implementation

For each of the top 3 ideas from Part J:

1. Implement a **minimal prototype** — just enough to test the core hypothesis
2. Run on PSM (fastest iteration, highest anomaly rate, clearest signal)
3. Compare to MTS-JEPA baseline from Phase 1
4. Log results in EXPERIMENT_LOG.md

### K.2: Prototype Evaluation Criteria

For each prototype, measure:
- **Does it beat MTS-JEPA on AUC?** (threshold-free, most reliable metric)
- **Does it beat MTS-JEPA on TRUE_PREDICTION AUC?** (the lead-time-corrected metric from H.1)
- **Does it introduce new failure modes?** (check for collapse, NaN, degenerate predictions)
- **Is it simpler or more complex than MTS-JEPA?** (simpler + better = publishable)

### K.3: Kill or Continue

After prototyping, for each idea:
- **KILL**: worse than baseline or marginal improvement not worth the complexity
- **CONTINUE**: clear improvement, worth full-scale experiments
- **PIVOT**: interesting signal but wrong approach, refine the idea

---

## Part L: Self-Review Loop (60 min)

### L.1: Internal Review

For the surviving idea(s), write a mini NeurIPS review of your OWN work:

```markdown
## Self-Review: [Idea Name]

### Claim
[What we claim this does]

### Evidence
[What experiments support the claim]

### Weaknesses a Reviewer Would Find
1. ...
2. ...
3. ...

### How to Address Each Weakness
1. ...
2. ...
3. ...

### Missing Experiments
- ...
```

### L.2: Fix Weaknesses

For each weakness identified in L.1:
1. Run the experiment or analysis that addresses it
2. If the weakness is real and unfixable, acknowledge it honestly
3. If the weakness reveals a deeper issue, PIVOT or KILL

### L.3: Iterate

If the self-review reveals significant weaknesses:
- Return to Part K with a refined version
- Repeat K -> L cycle until the self-review is clean

**Maximum 3 iterations.** If after 3 rounds the idea still has fatal weaknesses, document it as a negative result and move on.

---

## Part M: Full-Scale Validation (60 min)

For the best surviving idea:

1. Run on ALL available datasets (MSL, SMAP, PSM, SWaT if available)
2. Use 5 seeds: [42, 123, 456, 789, 1024]
3. Compare to MTS-JEPA (our replication) AND paper numbers
4. Compute statistical significance vs MTS-JEPA
5. Run the lead-time analysis from H.1

---

## Part N: Final Compilation (30 min)

### N.1: RESULTS.md

Update with three sections:
1. **Replication Results**: Our MTS-JEPA vs paper Table 1
2. **NeurIPS Fix Results**: Lead-time analysis, statistical tests, theory validation
3. **Extension Results**: Our improved method vs MTS-JEPA vs baselines

### N.2: SESSION_SUMMARY.md

Write a comprehensive session summary:
- **Bottom line** (3 sentences — what did we find, what's the path to NeurIPS?)
- **Replication verdict** (EXACT/GOOD/MARGINAL/FAILED per dataset)
- **Critical finding: lead-time analysis** (is MTS-JEPA really predicting or near-detecting?)
- **Theory validation** (do the stability/non-collapse bounds hold empirically?)
- **Our extension** (what idea survived, what are the numbers?)
- **Competitive landscape** (where do we stand vs MTS-JEPA, TS-JEPA, PAD, A2P?)
- **Next steps** (what must happen next for a NeurIPS submission?)
- **Key files index** (every important output file and what it contains)

### N.3: Figures

Save all figures to `figures/`:
- `replication_comparison.png` — bar chart: paper vs our replication per dataset
- `lead_time_breakdown.png` — pie chart: TRUE_PREDICTION vs CONTINUATION vs BOUNDARY
- `theory_validation.png` — subplots: M, epsilon, delta, bound vs actual drift, Tr(Cov(z))
- `codebook_utilization.png` — histogram of code usage
- `extension_vs_baseline.png` — our best idea vs MTS-JEPA vs baselines
- Training loss curves per dataset

### N.4: IMPROVEMENT_IDEAS.md

Document ALL ideas from brainstorming (Part J) with their status:
- KILLED (and why)
- PROTOTYPED (results)
- VALIDATED (full-scale results)
- DEFERRED (promising but needs more time)

This file is crucial for the next session — don't lose any ideas.

---

## Experiment Ordering

| Phase | Part | Task | Est. Time | Depends On |
|-------|------|------|:---------:|:----------:|
| 1 | A | Data pipeline | 90 min | -- |
| 1 | B | Architecture | 150 min | A |
| 1 | D | Test pipeline | 30 min | A, B |
| 1 | C | Training protocol | 60 min | D passes |
| 1 | E | Pre-train all datasets | 180 min | C |
| 1 | F | Downstream eval + verification | 60 min | E |
| 1 | G | Ablation replication | 90 min | F |
| 2 | H | NeurIPS review fixes | 120 min | F |
| 2 | I | Benchmark Trajectory JEPA | 90 min | F, H.1 |
| 3 | J | Literature review + brainstorming | 90 min | H, I |
| 4 | K | Prototype top ideas | 120 min | J |
| 4 | L | Self-review loop | 60 min | K |
| 4 | M | Full-scale validation | 60 min | L |
| 4 | N | Final compilation | 30 min | M |
| | | **Total** | **~14h** | |

---

## Anti-Patterns to Avoid

1. **Do NOT skip RevIN**. The paper relies on it for both normalization and reconstruction loss inversion. Without it, reconstruction loss won't match the input space.

2. **Do NOT use sliding windows for training pairs**. The paper specifies non-overlapping windows with stride 100. Sliding windows would leak information and inflate metrics.

3. **Do NOT back-propagate through EMA targets**. The target branch must be stop-gradient. This is critical for JEPA-style training stability.

4. **Do NOT forget channel-independence in the encoder**. Each variable is encoded separately. The Transformer operates over P=5 patch tokens per variable, NOT across variables.

5. **Do NOT use the full test set for evaluation**. The 6:2:2 chronological split means only the final 20% of test windows are used for metric computation. Using all test data will inflate scores.

6. **Do NOT confuse codebook distributions with embeddings**. The predictor operates on code distributions p in R^K (probability simplex), not on soft-quantized embeddings z in R^D. The MSE loss uses z, the KL loss uses p.

7. **Do NOT ignore the entropy regularization signs**. Sample entropy is MINIMIZED (sharp assignments), batch entropy is MAXIMIZED (diverse usage). The loss has +lambda_sample * L_sample - lambda_batch * L_batch.

8. **Do NOT commit model checkpoints or large data files**. Use `.gitignore` for `*.pt`, `*.pth`, `data/`.

9. **Do NOT proceed to Phase 2 if Phase 1 replication is FAILED**. Debug first. You don't understand the method well enough to improve it if you can't replicate it.

10. **Do NOT evaluate without threshold selection**. The F1 score depends critically on the threshold delta* selected on the validation split. Using a default 0.5 threshold will produce wrong numbers.

11. **Do NOT skip the brainstorming loop**. Phase 3 is where the NeurIPS contribution lives. Phase 1-2 are table stakes. If you run out of time, cut Phase 4 short rather than skipping Phase 3.

12. **Do NOT fall in love with your first idea**. The self-review loop (Part L) exists to kill bad ideas early. Be honest.

13. **Do NOT propose ideas that are "just engineering."** More layers, bigger models, more data augmentation — these are not NeurIPS contributions. Ideas need a conceptual insight.

---

## Success Criteria

### Phase 1: Replication
- EXACT match (<5%) on at least 2 datasets
- GOOD match (<15%) on all available datasets
- Ablation collapse finding confirmed

### Phase 2: Fixes & Benchmarking
- Lead-time analysis completed with clear breakdown
- Theory quantities tracked and plotted
- Trajectory JEPA benchmarked on at least 2 datasets

### Phase 3: Ideas
- At least 10 ideas brainstormed, at least 3 prototyped
- Gap map with at least 5 open research directions
- Clear top-1 idea selected with rationale

### Phase 4: Validation
- Top idea validated on at least 2 datasets with 5 seeds
- Statistical significance vs MTS-JEPA established
- Self-review with no remaining fatal weaknesses
- SESSION_SUMMARY.md with clear NeurIPS path

---

## Context: Our Current State

### What We Have
- **Trajectory JEPA V2** on C-MAPSS FD001: RMSE 13.80 ± 0.75 (E2E), 17.81 ± 1.67 (frozen)
- First SSL method to beat public reference (AE-LSTM 13.99) on C-MAPSS
- Strong label efficiency: wins at 100%-10% budgets
- C-MAPSS data pipeline + evaluation infrastructure already built
- STAR replication (supervised SOTA: 10.61 RMSE on FD001) available as reference
- A2P (When Will It Fail?) replication framework available

### What We're Missing
- **Codebook/discrete bottleneck**: MTS-JEPA's key innovation. Our Trajectory JEPA has no codebook.
- **Multi-resolution**: We operate at a single temporal scale. MTS-JEPA's fine+coarse dual-branch is novel.
- **Anomaly prediction evaluation**: We've only done RUL prediction. Anomaly prediction is a different (and arguably more practical) framing.
- **Collapse prevention guarantees**: No theoretical stability analysis in our work.
- **Industrial benchmarks for anomaly prediction**: C-MAPSS is RUL, not anomaly prediction. We need to bridge this gap.

### The NeurIPS Opportunity
The gap between MTS-JEPA and our Trajectory JEPA is the NeurIPS contribution:
- MTS-JEPA has the right architecture (codebook, multi-resolution) but wrong evaluation (detection benchmarks, no lead time)
- We have the right evaluation context (industrial prognostics, physical degradation) but need their architectural innovations
- **The fusion** — multi-resolution codebook JEPA evaluated on proper prognostics benchmarks with lead-time-aware metrics and interpretable degradation regimes — is the NeurIPS paper.
