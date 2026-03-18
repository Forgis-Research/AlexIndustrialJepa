# Autoresearch: KH-JEPA Architecture Validation

## Mission

Validate **Koopman-Hierarchical JEPA (KH-JEPA)** on ETTh1 before scaling to foundation model.

**Target**: Beat TTT (0.358 MSE) on ETTh1 horizon-96.

**Context**: This is Phase 1 of the JEPA-FM foundation model project. See `docs/SOTA_PLAN.md` for full plan.

---

## The Architecture

```
KH-JEPA = JEPA + Koopman Predictor + Hierarchy + VICReg + Cross-Variate
```

We validate each component incrementally.

---

## Phase 1: Baseline

Run basic JEPA to establish starting point:
```bash
python run.py --single
```

Record: `test_mse = ???`

**Expected**: MSE ~0.45-0.50 (unoptimized JEPA)

---

## Phase 2: Quick Wins

### 2.1 Longer Context (Critical!)
```python
SEQ_LEN = 512  # was 96, standard for foundation models
```
**Why**: TTT and all SOTA use 512. We were handicapped.

### 2.2 More Training
```python
EPOCHS = 20  # was 10
```

### 2.3 Lower Learning Rate
```python
LEARNING_RATE = 1e-4  # was 1e-3
```

---

## Phase 3: VICReg (Prevent Collapse)

**Why**: C-JEPA (NeurIPS 2024) showed EMA alone doesn't prevent collapse.

```python
def vicreg_loss(z):
    # Variance: prevent collapse to constant
    std = z.std(dim=0)
    var_loss = F.relu(1 - std).mean()

    # Covariance: decorrelate dimensions
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (z.shape[0] - 1)
    cov_loss = cov.pow(2).fill_diagonal_(0).mean()

    return 0.04 * var_loss + 0.04 * cov_loss

# Add to total loss
total_loss = mse_loss + vicreg_loss(z_pred)
```

**Expected**: More stable training, better generalization.

---

## Phase 4: Koopman Predictor (Key Innovation #1)

**Why**: Linear dynamics in latent space enable efficient long-horizon prediction.

Replace MLP predictor with Koopman operator:

```python
class KoopmanPredictor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Parameterize via eigendecomposition for stability
        self.eigenvalues = nn.Parameter(torch.randn(latent_dim) * 0.1)
        self.eigenvectors = nn.Parameter(torch.eye(latent_dim) + torch.randn(latent_dim, latent_dim) * 0.01)

    def forward(self, z, horizon=1):
        # K^horizon via eigendecomposition
        # K = V @ diag(λ) @ V^(-1)
        # K^k = V @ diag(λ^k) @ V^(-1)
        V = self.eigenvectors
        lambdas = torch.sigmoid(self.eigenvalues)  # Keep in (0,1) for stability
        lambdas_k = lambdas.pow(horizon)

        # Avoid explicit inverse - use solve
        z_transformed = torch.linalg.solve(V, z.T).T  # z @ V^(-1)
        z_scaled = z_transformed * lambdas_k
        z_pred = z_scaled @ V.T
        return z_pred
```

**Why this is novel**:
- O(1) for any horizon (vs O(H) for autoregressive)
- Interpretable: eigenvalues reveal system dynamics
- Grounded in Koopman operator theory

**Expected**: Better long-horizon accuracy, faster inference.

---

## Phase 5: Cross-Variate Attention (Key Innovation #2)

**Why**: iTransformer (ICLR 2024) showed cross-channel modeling beats channel-independent.

```python
class CrossVariateEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.variate_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])

    def forward(self, x):  # (B, C, T, d)
        B, C, T, d = x.shape

        for temp_layer, var_layer in zip(self.temporal_layers, self.variate_layers):
            # Temporal attention: within each channel
            x = x.view(B * C, T, d)
            x = temp_layer(x)
            x = x.view(B, C, T, d)

            # Variate attention: across channels at each time
            x = x.permute(0, 2, 1, 3).reshape(B * T, C, d)
            x = var_layer(x)
            x = x.view(B, T, C, d).permute(0, 2, 1, 3)

        return x
```

**Expected**: Captures sensor dependencies, beats PatchTST.

---

## Phase 6: Hierarchy (If Needed)

Only if Phase 4-5 don't reach target.

Multi-resolution latent spaces:
```python
# Level 1: Full resolution
z1 = encoder(x)

# Level 2: Downsampled 4x
x_down = F.avg_pool1d(x, kernel_size=4)
z2 = encoder(x_down)

# Level 3: Downsampled 16x
x_down2 = F.avg_pool1d(x, kernel_size=16)
z3 = encoder(x_down2)

# Predict at each level, combine
```

---

## Success Criteria

| MSE | Status | Next Action |
|-----|--------|-------------|
| > 0.45 | Baseline | Apply Phase 2 quick wins |
| 0.40-0.45 | Improving | Apply VICReg (Phase 3) |
| 0.38-0.40 | Good | Apply Koopman (Phase 4) |
| 0.358-0.38 | **Competitive** | Apply Cross-Variate (Phase 5) |
| < 0.358 | **SOTA!** | Document, move to Phase 2 of SOTA_PLAN |

---

## Rules

1. **One change at a time** - Know what works
2. **Keep what helps** - Don't revert improvements
3. **Log everything** - Results go to experiment_log.jsonl
4. **5 minutes max** - Fast iteration
5. **Don't overcomplicate** - Simple + working > complex + broken

---

## Current Best

| Run | MSE | Changes |
|-----|-----|---------|
| baseline | ??? | Initial config |

---

## Architecture Decision Tree

```
Start: Basic JEPA
    ↓
MSE > 0.45?  → Quick wins (SEQ_LEN=512, more epochs)
    ↓
MSE > 0.40?  → Add VICReg
    ↓
MSE > 0.38?  → Add Koopman predictor
    ↓
MSE > 0.358? → Add Cross-Variate attention
    ↓
MSE > 0.35?  → Add Hierarchy
    ↓
SUCCESS → Move to foundation model training
```

---

## After Validation

Once we beat 0.358 MSE:

1. Validate on full ETT suite (ETTh1, ETTh2, ETTm1, ETTm2)
2. Test all prediction horizons (96, 192, 336, 720)
3. Begin foundation model pre-training on Time-300B

See `docs/SOTA_PLAN.md` for full roadmap.

---

## References

- **Target**: TTT (0.358 MSE on ETTh1-96)
- **Baselines**: iTransformer (0.386), PatchTST (0.414)
- **C-JEPA**: VICReg prevents collapse
- **Koopman Theory**: Linear dynamics in lifted space
- **iTransformer**: Cross-variate attention
