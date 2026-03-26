# Mechanical-JEPA: Autoresearch Program

**Goal:** Build and validate a JEPA-based encoder for mechanical state-action sequences that learns transferable dynamics representations across robot embodiments.

**Success metric:** Cross-embodiment transfer ratio < 2.0 (pretrained model needs 5x less data on new robot than training from scratch).

---

## Phase 0: Rapid Sanity Check (First 2-3 Hours)

**Purpose:** Validate implementation before committing to full training. Catch bugs early.

### 0.1 Micro Dataset

```python
# Use tiny subset — just enough to verify code works
SANITY_DATA = {
    'dataset': 'maniskill',  # Cleanest, most consistent
    'n_episodes': 100,       # Not 30k
    'window_size': 32,       # Not 128
    'stride': 16,
}
# Total: ~500 windows, fits in memory
```

### 0.2 Tiny Model

```python
SANITY_MODEL = {
    'd_model': 32,      # Not 128
    'heads': 2,         # Not 4
    'layers': 1,        # Not 4
    'params': '~10K',   # Trains in seconds
}
```

### 0.3 Sanity Checks (Must ALL Pass)

| # | Check | How to Verify | Pass Criterion |
|---|-------|---------------|----------------|
| 1 | **Data loads** | Print shapes, sample values | No NaN, reasonable ranges |
| 2 | **Forward pass** | `model(batch)` runs | No errors, output shape correct |
| 3 | **Loss computes** | Print loss value | Finite, positive, ~1.0 initially |
| 4 | **Loss decreases** | Train 10 steps | Loss drops >10% |
| 5 | **Gradients flow** | Check `param.grad` | No NaN, no zeros, no exploding |
| 6 | **EMA updates** | Compare encoder vs target | Target slightly behind encoder |
| 7 | **Overfits 1 batch** | Train 100 steps on 1 batch | Loss → ~0 |
| 8 | **Masking works** | Visualize masked positions | Correct positions masked |

### 0.4 Sanity Check Script

```python
# autoresearch/experiments/sanity_check.py

def run_sanity_checks():
    print("=== SANITY CHECK ===\n")

    # 1. Data
    print("[1/8] Loading data...")
    data = load_micro_dataset()
    assert not torch.isnan(data).any(), "NaN in data!"
    print(f"  Shape: {data.shape}, Range: [{data.min():.2f}, {data.max():.2f}]")
    print("  ✓ Data loads\n")

    # 2. Forward pass
    print("[2/8] Forward pass...")
    model = MechanicalJEPA(d_model=32, heads=2, layers=1)
    batch = data[:8]
    z = model.encode(batch)
    print(f"  Input: {batch.shape} → Output: {z.shape}")
    print("  ✓ Forward pass\n")

    # 3. Loss
    print("[3/8] Loss computation...")
    loss = model.compute_loss(batch)
    assert torch.isfinite(loss), f"Loss is {loss}!"
    print(f"  Loss: {loss.item():.4f}")
    print("  ✓ Loss computes\n")

    # 4. Loss decreases
    print("[4/8] Training 10 steps...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for i in range(10):
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    assert losses[-1] < losses[0] * 0.9, "Loss didn't decrease!"
    print("  ✓ Loss decreases\n")

    # 5. Gradients
    print("[5/8] Checking gradients...")
    loss = model.compute_loss(batch)
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Bad gradient in {name}"
    print("  ✓ Gradients flow\n")

    # 6. EMA
    print("[6/8] EMA update...")
    model.ema_update(decay=0.99)
    # Target should be slightly different from encoder
    enc_norm = sum(p.norm() for p in model.encoder.parameters())
    tgt_norm = sum(p.norm() for p in model.target_encoder.parameters())
    print(f"  Encoder norm: {enc_norm:.2f}, Target norm: {tgt_norm:.2f}")
    print("  ✓ EMA updates\n")

    # 7. Overfit 1 batch
    print("[7/8] Overfitting 1 batch (100 steps)...")
    model = MechanicalJEPA(d_model=32, heads=2, layers=1)  # Fresh model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    single_batch = data[:8]
    for i in range(100):
        loss = model.compute_loss(single_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"  Final loss: {loss.item():.6f}")
    assert loss.item() < 0.1, "Can't overfit single batch!"
    print("  ✓ Overfits 1 batch\n")

    # 8. Masking
    print("[8/8] Checking masking...")
    mask = model.create_mask(seq_len=32, mask_ratio=0.3)
    masked_count = mask.sum().item()
    print(f"  Masked: {masked_count}/32 positions ({masked_count/32*100:.0f}%)")
    print("  ✓ Masking works\n")

    print("=== ALL SANITY CHECKS PASSED ===")
    return True
```

### 0.5 Quick Viability Test

After sanity checks pass, run a **30-minute viability test**:

```python
VIABILITY_TEST = {
    'dataset': 'maniskill',
    'n_episodes': 1000,      # 3% of full
    'model': 'small',        # 500K params
    'epochs': 10,            # Not 100
    'time_budget': '30 min',
}
```

**Viability criteria:**

| Check | Pass | Fail |
|-------|------|------|
| Training loss | Decreases smoothly | Stuck or NaN |
| Validation loss | Decreases (slower than train) | Increases (overfit) |
| Embeddings | Vary across samples | Collapse to constant |
| Robot classification | >50% accuracy | Random (14%) |

**If viability test fails:** Debug before scaling up.
**If viability test passes:** Proceed to full training.

### 0.6 Embedding Collapse Check

JEPA can collapse (all embeddings become identical). Check for this:

```python
def check_collapse(model, data):
    embeddings = []
    for batch in dataloader:
        z = model.encode(batch)
        embeddings.append(z)
    embeddings = torch.cat(embeddings)

    # Check variance
    var = embeddings.var(dim=0).mean()
    print(f"Embedding variance: {var:.4f}")

    # Check pairwise distances
    dists = torch.cdist(embeddings[:100], embeddings[:100])
    mean_dist = dists.mean()
    print(f"Mean pairwise distance: {mean_dist:.4f}")

    if var < 0.01 or mean_dist < 0.1:
        print("⚠️ WARNING: Possible embedding collapse!")
        return False
    return True
```

---

## Phase 0→1 Gate

**Do NOT proceed to Phase 1 until:**

- [ ] All 8 sanity checks pass
- [ ] 30-min viability test passes
- [ ] No embedding collapse
- [ ] Robot classification >50% (random=14%)

**If stuck:** Log the issue, investigate, fix before scaling.

---

## Phase 1: Data Setup (Hours 2-4)

### Datasets to Use

**Primary (Tier 1 — Rich Proprioception):**

| Dataset | tfds Name | Robot | DOF | State Dims | Episodes | Use For |
|---------|-----------|-------|-----|------------|----------|---------|
| DROID | `droid_100` | Franka Panda | 7 | 14 (joint_pos:7 + cartesian:6 + gripper:1) | 76,000 | Pretraining |
| ManiSkill | `maniskill_dataset_converted_externally_to_rlds` | Panda (sim) | 7 | 18 (joints:7 + gripper:2 + velocities:9) | 30,213 | Pretraining |
| Stanford KUKA | `stanford_kuka_multimodal_dataset_converted_externally_to_rlds` | KUKA iiwa | 7 | 27 (joint_pos:7 + joint_vel:7 + EE:13) | 3,000 | Transfer target |

**Secondary (Tier 2 — Transfer Targets):**

| Dataset | tfds Name | Robot | DOF | State Dims | Episodes | Use For |
|---------|-----------|-------|-----|------------|----------|---------|
| Berkeley UR5 | `berkeley_autolab_ur5` | UR5 | 6 | 15 | 896 | Transfer (6-DOF) |
| Berkeley FANUC | `berkeley_fanuc_manipulation` | FANUC | 6 | 13 | 415 | Transfer (industrial) |
| JACO Play | `jaco_play` | Kinova JACO | 6 | 8+6+7=21 | 976 | Transfer (6-DOF) |
| TOTO | `toto` | Franka | 7 | 7 | 902 | Transfer (same robot, diff task) |

### State Representation

**Use joint-space only (not EE) for core representation:**

```python
# Primary state vector (per timestep)
state = {
    'joint_pos': (7,),    # Joint angles q1..q7 (rad)
    'joint_vel': (7,),    # Joint velocities dq1..dq7 (rad/s) — if available
    'gripper': (1,),      # Gripper opening (normalized)
}
# Total: 7-15 dims depending on dataset
```

**Normalization:**
- Per-channel z-score normalization (mean=0, std=1)
- Compute stats on training set only
- Clip outliers to ±5 std

**Temporal:**
- Resample all to 10 Hz (downsample KUKA/ManiSkill from 20Hz, upsample Fractal from 3Hz)
- Window size: 128 timesteps (12.8 seconds)
- Stride: 64 timesteps (50% overlap)

### Action Representation

**Critical finding: All OXE actions are end-effector space, not joint space.**

**Approach: Treat actions as auxiliary conditioning signal**

```python
# Raw action from data (EE delta)
action_raw = [dx, dy, dz, drx, dry, drz, gripper]  # 6-7 dims

# Encoding strategy
action_embed = nn.Sequential(
    nn.Linear(action_dim, d_model),
    nn.LayerNorm(d_model),
    nn.GELU(),
    nn.Linear(d_model, d_model)
)(action_raw)
```

**Action handling options to ablate:**

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A: Raw EE | Use EE deltas directly | Simple | Doesn't match joint state |
| B: State delta | `action = state_{t+1} - state_t` | Aligned with state | Leaks future |
| C: No action | Unconditional prediction | Simplest baseline | Can't do planning |
| D: Action tokens | Separate action encoder per dataset | Handles heterogeneity | Complex |

**Start with Option A, ablate C as baseline.**

---

## Phase 2: Architecture (Hours 2-4)

### Mechanical-JEPA Architecture

```
                    ┌─────────────────┐
   state_t ────────►│  State Encoder  │────► z_t ───┐
                    │   (Transformer) │             │
                    └─────────────────┘             ▼
                                              ┌───────────┐
                    ┌─────────────────┐       │           │
action_{t:t+k} ────►│  Action Encoder │──────►│ Predictor │────► ẑ_{t+k}
                    │     (MLP)       │       │           │
                    └─────────────────┘       └───────────┘
                                                    │
                                                    ▼ L2 loss
                    ┌─────────────────┐             │
 state_{t+k} ──────►│ Target Encoder  │────► z_{t+k}
                    │   (EMA copy)    │
                    └─────────────────┘
```

### Model Configurations

| Config | d_model | heads | layers | params | Use |
|--------|---------|-------|--------|--------|-----|
| Tiny | 64 | 4 | 2 | ~100K | Debug |
| Small | 128 | 4 | 4 | ~500K | Ablations |
| Base | 256 | 8 | 6 | ~2M | Main experiments |
| Large | 512 | 8 | 8 | ~8M | Final if needed |

**Start with Small, scale up if it works.**

### Masking Strategy

**Temporal block masking (predict future from past):**

```python
# Input: state sequence [t-T, ..., t]
# Mask: [t+1, ..., t+k] (predict next k steps)
# k = prediction horizon (start with 10, ablate 1, 5, 20, 50)

mask_ratio = 0.3  # 30% of timesteps masked
block_size = 10   # Mask in contiguous blocks
```

**Alternative: Random timestep masking (ablate)**

### Training Hyperparameters

```python
config = {
    'batch_size': 256,
    'lr': 1e-4,
    'weight_decay': 0.01,
    'epochs': 100,
    'warmup_steps': 1000,
    'ema_decay': 0.996,
    'optimizer': 'AdamW',
    'scheduler': 'cosine',
}
```

---

## Phase 3: Pretraining (Hours 4-8)

### Pretraining Protocol

**Dataset mix:**
```python
pretrain_mix = {
    'droid': 0.6,      # 60% — largest, Franka
    'maniskill': 0.3,  # 30% — sim, clean
    'kuka': 0.1,       # 10% — real, rich
}
```

**Training loop:**
```
for epoch in range(100):
    for batch in dataloader:
        state_seq, action_seq = batch

        # Encode context (past states)
        z_context = encoder(state_seq[:, :T])

        # Encode actions
        a_embed = action_encoder(action_seq[:, T:T+k])

        # Predict future latents
        z_pred = predictor(z_context, a_embed)

        # Target (EMA encoder)
        with torch.no_grad():
            z_target = target_encoder(state_seq[:, T:T+k])

        # Loss
        loss = F.mse_loss(z_pred, z_target)

        # Update
        optimizer.step()
        ema_update(target_encoder, encoder)
```

### Checkpointing

- Save every 10 epochs
- Save best by validation loss
- Log: loss, learning rate, gradient norm, EMA decay

---

## Phase 4: Evaluation — Classification (Hours 8-12)

### Sanity Check: Embodiment Classification

**Task:** Given random 128-step window, predict which robot it came from.

**Protocol:**
1. Freeze pretrained encoder
2. Train linear probe on encoder outputs
3. Compare: random init encoder vs pretrained

**Expected results:**

| Encoder | Accuracy | Interpretation |
|---------|----------|----------------|
| Random features | ~15% | Chance (7 robots) |
| Raw state (no encoder) | ~85% | Signatures in raw data |
| Pretrained encoder | 70-90% | Should be comparable to raw |

**If pretrained < raw:** Encoder losing information (bad)
**If pretrained > raw:** Encoder learning useful features

### Task Classification

**Task:** Predict task category from trajectory.

**Categories (derived from language instructions):**
```python
TASK_CATEGORIES = {
    'pick': ['pick', 'grasp', 'grab'],
    'place': ['place', 'put', 'drop'],
    'pour': ['pour', 'scoop'],
    'push': ['push', 'slide'],
    'insert': ['insert', 'peg'],
    'press': ['press', 'button'],
    'sweep': ['sweep', 'wipe'],
}
```

**Protocol:**
1. Map language instructions to categories via keyword matching
2. Linear probe on frozen encoder
3. Metric: Macro F1

### Gripper State Prediction

**Task:** Predict gripper open/closed from state embedding.

**Protocol:**
1. Threshold gripper channel to get binary labels
2. Linear probe on encoder
3. Metric: Binary accuracy, AUROC

---

## Phase 5: Evaluation — Forecasting (Hours 12-18)

### Next-State Prediction

**Task:** Given (state_t, action_t), predict state_{t+1}.

**Protocol:**
1. Decoder: Linear layer from z_t to state_{t+1}
2. Train decoder only (freeze encoder) vs train both
3. Metric: MSE per joint, RMSE aggregate

**Baselines:**

| Baseline | Description |
|----------|-------------|
| Copy-last | Predict state_t (persistence) |
| Linear | Linear regression on (state, action) |
| MLP | 2-layer MLP on (state, action) |
| Transformer (scratch) | Same arch, no pretraining |

### Multi-Step Rollout

**Task:** Given (state_t, action_{t:t+H}), predict state_{t+1:t+H+1}.

**Horizons:** H = 10, 20, 50

**Metrics:**
- Cumulative MSE
- Divergence time (when error > threshold)
- Per-joint error curves

### Unconditional Forecasting

**Task:** Given state_{t-20:t}, predict state_{t+1:t+10} (no actions).

**Purpose:** Baseline for action-conditioned prediction. If action-conditioned isn't much better, actions aren't helping.

---

## Phase 6: Evaluation — Transfer (Hours 18-24)

### Cross-Embodiment Transfer Protocol

```
Source: Franka (DROID + ManiSkill pretrained)
Targets: KUKA, UR5, FANUC, JACO, TOTO

For each target:
    1. Zero-shot: Evaluate pretrained encoder directly
    2. Linear probe: Train linear decoder only
    3. Fine-tune 10%: Fine-tune encoder on 10% of target data
    4. Fine-tune 100%: Fine-tune on all target data
    5. From scratch: Train from random init on target
```

**Key metric: Transfer ratio**

```python
transfer_ratio = MSE_target / MSE_source

# Interpretation:
# < 1.5: Excellent transfer
# 1.5-2.5: Good transfer
# 2.5-4.0: Moderate transfer
# > 4.0: Poor transfer
```

### Few-Shot Learning

**Task:** Learn new robot with minimal data.

**Protocol:**
```python
for n_shots in [10, 50, 100, 500]:
    # Sample n_shots episodes from target robot
    # Fine-tune pretrained encoder
    # Evaluate on held-out target test set
    # Compare to training from scratch with same data
```

**Success criterion:** Pretrained + 10 shots ≈ Scratch + 100 shots (10x efficiency)

### DOF Mismatch Handling

**For 6-DOF robots (UR5, FANUC, JACO):**

| Strategy | Description | Test |
|----------|-------------|------|
| Zero-pad | Pad joint 7 with zeros | Default |
| Project | Learn 6→7 linear projection | Ablate |
| EE-space | Use 6D EE representation for all | Ablate |

---

## Baselines to Beat

### Forecasting Baselines

| # | Baseline | Description | Expected MSE |
|---|----------|-------------|--------------|
| 1 | Copy-last | state_{t+1} = state_t | 0.05-0.10 |
| 2 | Linear | Linear(concat(state, action)) | 0.03-0.06 |
| 3 | MLP | 2-layer MLP | 0.02-0.04 |
| 4 | Transformer-scratch | Same arch, random init | 0.015-0.03 |
| 5 | **Mechanical-JEPA** | Pretrained | **< 0.015** |

### Transfer Baselines

| # | Baseline | Description | Expected Ratio |
|---|----------|-------------|----------------|
| 1 | From scratch | Train on target only | 1.0 (reference) |
| 2 | ImageNet-style | Pretrain on source, fine-tune | 1.5-2.5 |
| 3 | Frozen encoder | Pretrain, freeze, linear probe | 2.0-4.0 |
| 4 | **Mechanical-JEPA** | Pretrain, fine-tune 10% | **< 2.0** |

---

## Experiment Schedule

### Session 1: Sanity & Viability (2-3 hours, interactive)

**Goal:** Validate implementation before overnight run.

| Step | Time | Task | Gate |
|------|------|------|------|
| 1 | 15 min | Download 100 ManiSkill episodes | Data loads, no NaN |
| 2 | 15 min | Implement tiny model | Forward pass works |
| 3 | 15 min | Run 8 sanity checks | All pass |
| 4 | 30 min | Viability test (1k episodes, 10 epochs) | Loss decreases |
| 5 | 15 min | Collapse check + robot classification | >50% acc, no collapse |
| 6 | 30 min | Debug any issues | All gates pass |

**Output:** Green light for overnight run, or list of bugs to fix.

### Night 1: Foundation (Overnight, 8-12 hours)

**Only start if Session 1 passed all gates.**

| Hour | Task | Deliverable |
|------|------|-------------|
| 0-2 | Download full data (DROID, ManiSkill, KUKA) | ~100k episodes |
| 2-4 | Preprocess & verify | Clean tensors, no issues |
| 4-8 | Pretrain (Small model, 50 epochs) | Checkpoint every 10 epochs |
| 8-10 | Embodiment classification | Accuracy numbers |
| 10-12 | Forecasting eval (1-step) | MSE vs baselines |

**Morning checkpoint:** Review logs, verify no collapse, results make sense.

### Session 2: Analysis & Decision (1-2 hours, interactive)

| Step | Task | Decision |
|------|------|----------|
| 1 | Review overnight logs | Any red flags? |
| 2 | Check classification results | >70% embodiment acc? |
| 3 | Check forecasting MSE | Beats linear baseline? |
| 4 | Decide: scale up or debug | Go/No-go for Night 2 |

### Night 2: Scale Up (Overnight, 8-12 hours)

**Only if Session 2 gives go-ahead.**

| Hour | Task | Deliverable |
|------|------|-------------|
| 0-4 | Pretrain Base model (100 epochs) | Final checkpoint |
| 4-8 | Multi-step rollout evaluation | Rollout curves |
| 8-12 | Cross-embodiment transfer (KUKA, UR5) | Transfer ratios |

### Session 3: Final Analysis (1-2 hours, interactive)

| Step | Task | Output |
|------|------|--------|
| 1 | Compile all results | Tables, figures |
| 2 | Statistical significance | p-values (5 seeds) |
| 3 | Write up findings | Draft results section |
| 4 | Decide: publish or iterate | Next steps |

### Night 3: Polish or Iterate

**If results are good:** Few-shot experiments, ablations, more baselines
**If results are weak:** Debug, try different architecture/hyperparams

---

## Logging Format

```markdown
## Exp N: [One-line description]

**Time**: YYYY-MM-DD HH:MM
**Phase**: Pretraining / Classification / Forecasting / Transfer
**Hypothesis**: [What you expect]
**Change**: [What you modified]

**Setup**:
- Dataset: [names]
- Model: [config]
- Training: [epochs, batch, lr]

**Results**:
| Metric | Value | Baseline | Delta |
|--------|-------|----------|-------|
| ... | ... | ... | ... |

**Sanity checks**: ✓ passed / ⚠️ issues
**Verdict**: KEEP / REVERT / INVESTIGATE
**Insight**: [What you learned]
**Next**: [What to try]
```

---

## Red Flags to Watch For

| Flag | Meaning | Action |
|------|---------|--------|
| Loss doesn't decrease | Bug or LR too high | Check gradients, lower LR |
| Loss = 0 | Predicting input | Check masking |
| Embodiment acc = 100% | Encoder memorizing robot ID | Add noise, check for leakage |
| Embodiment acc = chance | Encoder not learning | Check architecture |
| Transfer ratio > 5 | Transfer not working | Try different alignment |
| Pretrained ≈ scratch | Pretraining not helping | More data, longer training |

---

## Files to Create

```
autoresearch/
├── mechanical_jepa_program.md    # This file
├── EXPERIMENT_LOG.md             # Results
├── LESSONS_LEARNED.md            # Reusable insights
└── experiments/
    ├── download_data.py          # Data prep
    ├── preprocess.py             # Normalization, windowing
    ├── model.py                  # Architecture
    ├── pretrain.py               # Pretraining loop
    ├── eval_classification.py    # Classification evals
    ├── eval_forecasting.py       # Forecasting evals
    └── eval_transfer.py          # Transfer evals
```

---

## Quick Start

```bash
# 1. Download data (start with small sample)
python datasets/downloaders/download_oxe_proprio.py --datasets droid,maniskill,kuka --n-episodes 100

# 2. Preprocess
python autoresearch/experiments/preprocess.py --output data/processed/

# 3. Pretrain (small model first)
python autoresearch/experiments/pretrain.py --config small --epochs 10

# 4. Evaluate
python autoresearch/experiments/eval_classification.py --checkpoint best.pt
```

---

## Success Criteria for Paper

| Claim | Evidence Needed |
|-------|-----------------|
| "JEPA works for mechanical sequences" | Pretraining loss converges, beats scratch |
| "Learns meaningful representations" | Classification probes work |
| "Enables cross-embodiment transfer" | Transfer ratio < 2.0 on 2+ robots |
| "Reduces data requirements" | 5-10x efficiency in few-shot |
| "Actions improve prediction" | Conditioned < Unconditional |

**Minimum viable paper:** Claims 1-3 with statistical significance (p < 0.05, 5 seeds).

**Strong paper:** All 5 claims + ablations + comparison to TD-MPC2/Octo.
