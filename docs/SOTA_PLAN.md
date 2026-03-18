# JEPA-FM: A Foundation Model for Time Series

**Mission**: Build the first JEPA-based Time Series Foundation Model that beats Chronos-2 and TimesFM on GIFT-Eval.

**Compute Budget**: $500k+ (equivalent to ~10,000 H100-hours)

**Target Venue**: NeurIPS 2026 / ICML 2027 (Best Paper candidate)

---

## Why JEPA Will Win

Current SOTA (Chronos-2, TimesFM, Toto) all use **autoregressive** or **masked prediction**:
- Chronos: Tokenizes values → T5 decoder
- TimesFM: Decoder-only, next-patch prediction
- Toto: Autoregressive with Student-t mixture

**The fundamental limitation**: They predict in **observation space**, token by token.

**JEPA's advantage**: Predict in **latent space**, capturing abstract dynamics.

| Approach | Prediction | Efficiency | Abstraction |
|----------|------------|------------|-------------|
| Autoregressive | O(T) sequential | Slow | Surface patterns |
| Masked | Single pass | Fast | Local patterns |
| **JEPA** | Single pass | Fast | **Global dynamics** |

---

## The Architecture: Koopman-Hierarchical JEPA (KH-JEPA)

### Core Innovation

Combine three powerful ideas:

1. **JEPA**: Predict future latent states, not raw values
2. **Koopman Theory**: Linear dynamics in latent space enable efficient long-horizon rollouts
3. **Hierarchy**: Multi-resolution latent spaces capture different time scales

```
Input: x₁, x₂, ..., xₜ (multivariate time series)
                ↓
┌─────────────────────────────────────────────────┐
│  Hierarchical Encoder                           │
│  ┌─────────────────────────────────────────┐   │
│  │ Level 3: Hours    z³ = E³(pool(x))      │   │
│  │ Level 2: Minutes  z² = E²(pool(x))      │   │
│  │ Level 1: Seconds  z¹ = E¹(x)            │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────┐
│  Koopman Predictor (per level)                  │
│                                                 │
│  ẑₜ₊ₖ = Kᵏ · zₜ    (linear dynamics!)          │
│                                                 │
│  K = learnable Koopman matrix                   │
│  Powers of K give any horizon instantly         │
└─────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────┐
│  Coarse-to-Fine Decoder                         │
│                                                 │
│  ŷ = D(ẑ¹, ẑ², ẑ³)                             │
│                                                 │
│  Coarse levels guide, fine levels refine        │
└─────────────────────────────────────────────────┘
                ↓
Output: ŷₜ₊₁, ŷₜ₊₂, ..., ŷₜ₊ₕ (predictions)
```

### Why Koopman?

Koopman operator theory: Any nonlinear dynamical system can be represented as **linear dynamics in a lifted space**.

```
Physical space (nonlinear):    ẋ = f(x)
Koopman space (linear):        ż = Kz        where z = φ(x)
```

**Benefits**:
1. **Efficient rollout**: zₜ₊ₖ = Kᵏzₜ (matrix power, not sequential)
2. **Interpretable**: Eigenvalues of K reveal system modes
3. **Stable**: Can enforce spectral constraints
4. **Proven**: Used in fluid dynamics, robotics, power systems

### Why Hierarchy?

Different phenomena operate at different time scales:
- **Macro** (hours): Trends, seasonality, regime changes
- **Meso** (minutes): Local patterns, event responses
- **Micro** (seconds): Noise, rapid fluctuations

Current models use single-resolution. We decompose explicitly.

### Key Architectural Components

#### 1. Cross-Variate Patch Encoder
```python
# Unlike PatchTST (channel-independent), we model cross-channel
class CrossVariateEncoder(nn.Module):
    def __init__(self, d_model, n_heads):
        self.temporal_attn = Attention(d_model, n_heads)  # within channel
        self.variate_attn = Attention(d_model, n_heads)   # across channels

    def forward(self, x):  # (B, C, T, d)
        x = self.temporal_attn(x, dim=2)   # attend over time
        x = self.variate_attn(x, dim=1)    # attend over channels
        return x
```

#### 2. Learnable Koopman Operator
```python
class KoopmanPredictor(nn.Module):
    def __init__(self, latent_dim):
        # Parameterize K via eigendecomposition for stability
        self.eigenvalues = nn.Parameter(torch.randn(latent_dim))
        self.eigenvectors = nn.Parameter(torch.randn(latent_dim, latent_dim))

    def forward(self, z, horizon):
        # K = V @ diag(λ) @ V⁻¹
        # Kᵏ = V @ diag(λᵏ) @ V⁻¹
        lambdas_k = self.eigenvalues.pow(horizon)
        K_k = self.eigenvectors @ torch.diag(lambdas_k) @ self.eigenvectors.inverse()
        return z @ K_k
```

#### 3. VICReg Regularization (prevents collapse)
```python
def vicreg_loss(z):
    # Variance: prevent collapse to constant
    std = z.std(dim=0)
    var_loss = F.relu(1 - std).mean()

    # Covariance: decorrelate dimensions
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (z.shape[0] - 1)
    cov_loss = cov.pow(2).fill_diagonal_(0).mean()

    return var_loss + cov_loss
```

#### 4. Hierarchical Consistency Loss
```python
def hierarchy_loss(z_fine, z_coarse):
    # Coarse should be smoothed version of fine
    z_fine_pooled = F.avg_pool1d(z_fine, kernel_size=k)
    return F.mse_loss(z_fine_pooled, z_coarse)
```

---

## Training Strategy

### Phase 1: Architecture Validation (Week 1-2)
**Goal**: Prove KH-JEPA works on small scale

| Task | Dataset | Compute | Success Metric |
|------|---------|---------|----------------|
| Single-resolution JEPA | ETTh1 | 1 GPU-day | MSE < 0.40 |
| Add Koopman predictor | ETTh1 | 1 GPU-day | MSE < 0.38 |
| Add hierarchy | ETT-full | 2 GPU-days | Beat iTransformer |
| Add cross-variate | ETT-full | 2 GPU-days | Beat PatchTST |

### Phase 2: Foundation Pre-training (Week 3-8)
**Goal**: Train at scale on Time-300B

| Size | Parameters | Data | Compute | Target |
|------|------------|------|---------|--------|
| Small | 50M | 10B points | 500 H100-hrs | Validate scaling |
| Medium | 200M | 50B points | 2000 H100-hrs | Match Moirai |
| Large | 1B | 300B points | 5000 H100-hrs | Beat Chronos-2 |
| XL | 2.4B | 300B points | 8000 H100-hrs | **SOTA** |

**Training recipe**:
- Optimizer: AdamW, cosine LR with warmup
- Loss: Huber (robust to outliers) + VICReg
- Masking: 70% random patches (like TS-JEPA)
- Sequence length: 4096 (match Time-MoE)
- Batch size: Scale with model size

### Phase 3: Benchmark Domination (Week 9-10)
**Goal**: Top GIFT-Eval leaderboard

| Benchmark | Current SOTA | Our Target |
|-----------|--------------|------------|
| GIFT-Eval (28 datasets) | Toto (rank 5.5) | Rank < 4 |
| ETT-full | TTT (0.358) | < 0.35 |
| Electricity | Chronos-2 | Beat |
| Traffic | TimesFM | Beat |

### Phase 4: Paper & Release (Week 11-12)
- Write NeurIPS/ICML submission
- Open-source model + code
- HuggingFace model hub

---

## Novel Contributions (Paper Story)

### Primary Contribution
**First JEPA-based Time Series Foundation Model**
- Predicts in latent space, not observation space
- Single forward pass for any horizon (not autoregressive)
- Naturally handles multivariate with cross-channel attention

### Secondary Contributions

1. **Koopman JEPA Predictor**
   - Linear dynamics enable O(1) long-horizon prediction
   - Interpretable: eigenvalues reveal dominant frequencies
   - Grounded in dynamical systems theory

2. **Hierarchical Multi-Resolution**
   - Explicit decomposition of time scales
   - Coarse-to-fine prediction
   - Better long-horizon accuracy

3. **VICReg for Time Series**
   - First application to time series foundation models
   - Prevents representation collapse
   - Enables stable training at scale

### Ablation Studies
- JEPA vs Autoregressive (architecture)
- Koopman vs MLP predictor (efficiency)
- Single vs Multi-resolution (long-horizon)
- Channel-independent vs Cross-variate (multivariate)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Koopman doesn't scale | Fall back to MLP predictor |
| Hierarchy adds too much complexity | Use single-res with multi-scale patches |
| Training instability | VICReg + Huber loss + gradient clipping |
| Can't beat Chronos-2 | Target "competitive + novel architecture" story |

---

## Compute Plan

**Total budget**: $500k ≈ 10,000 H100-hours

| Phase | H100-hours | Cost |
|-------|------------|------|
| Architecture validation | 200 | $10k |
| Small-scale pre-training | 500 | $25k |
| Medium-scale pre-training | 2000 | $100k |
| Large-scale pre-training | 5000 | $250k |
| XL experiments (if needed) | 2000 | $100k |
| Buffer | 300 | $15k |

**Infrastructure**:
- AWS SageMaker with H100 instances
- Distributed training: DeepSpeed ZeRO-3 or FSDP
- Checkpointing: Every 1000 steps

---

## Timeline

```
Week 1-2:   Architecture validation on ETT (Phase 1)
Week 3:     Small-scale pre-training (50M params)
Week 4-5:   Medium-scale pre-training (200M params)
Week 6-8:   Large-scale pre-training (1B params)
Week 9-10:  GIFT-Eval benchmark + ablations
Week 11-12: Paper writing + open-source release
```

**Key Milestones**:
- Week 2: KH-JEPA beats iTransformer on ETT
- Week 5: 200M model matches Moirai on GIFT-Eval subset
- Week 8: 1B model competitive with Chronos-2
- Week 10: Top-3 on GIFT-Eval leaderboard

---

## Current Status

| Item | Status |
|------|--------|
| ETTh1 autoresearch setup | ✅ Ready |
| Basic JEPA architecture | ✅ Implemented |
| VICReg loss | 🔄 In program.md |
| Koopman predictor | ❌ Not started |
| Hierarchical encoder | ❌ Not started |
| Time-300B dataloader | ❌ Not started |
| Distributed training | ❌ Not started |

---

## Next Actions

1. **Immediate**: Run baseline on ETTh1, establish starting MSE
2. **This week**: Implement Koopman predictor, test on ETTh1
3. **Next week**: Add hierarchy, validate architecture
4. **Week 3**: Begin pre-training on Time-300B subset

---

## References

### Core Papers
- [I-JEPA](https://arxiv.org/abs/2301.08243) - Original image JEPA
- [V-JEPA](https://arxiv.org/abs/2404.07524) - Video JEPA
- [C-JEPA](https://proceedings.neurips.cc/paper_files/paper/2024/file/04a80267ad46fc730011f8760f265054-Paper-Conference.pdf) - VICReg for JEPA
- [TS-JEPA](https://arxiv.org/abs/2509.25449) - JEPA for time series

### Foundation Models
- [Chronos-2](https://arxiv.org/pdf/2510.15821) - Amazon
- [TimesFM](https://github.com/google-research/timesfm) - Google
- [Time-MoE](https://github.com/Time-MoE/Time-MoE) - ICLR 2025 Spotlight
- [Toto](https://www.datadoghq.com/blog/datadog-time-series-foundation-model/) - Datadog

### Koopman Theory
- [Deep learning for universal linear embeddings of nonlinear dynamics](https://www.nature.com/articles/s41467-018-07210-0)
- [Data-driven discovery of Koopman eigenfunctions](https://arxiv.org/abs/1712.01378)

### Datasets
- [Time-300B](https://huggingface.co/datasets/Maple728/Time-300B) - 300B time points
- [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) - Benchmark

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-18 | Target GIFT-Eval SOTA | Most competitive benchmark |
| 2026-03-18 | Koopman-Hierarchical architecture | Novel + principled + efficient |
| 2026-03-18 | $500k compute commitment | Enables foundation model scale |
| 2026-03-18 | NeurIPS 2026 target | ~May deadline, achievable timeline |
