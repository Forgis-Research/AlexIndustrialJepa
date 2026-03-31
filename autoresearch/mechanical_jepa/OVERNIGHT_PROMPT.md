# Overnight Autoresearch Prompt: Mechanical-JEPA v2

## Fix Predictor Collapse & Achieve Real Cross-Dataset Transfer

Use this prompt with the ml-researcher agent for overnight autonomous research.

---

## Prompt

```
Run autoresearch overnight on the Mechanical-JEPA bearing fault detection project.

## Context

You are continuing research on JEPA (Joint Embedding Predictive Architecture) for
industrial bearing fault detection. This is analogous to Brain-JEPA (NeurIPS 2024
Spotlight) but applied to vibration signals instead of fMRI.

**Working directory:** `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa`

### Where We Stand

**The good:**
- CWRU linear probe: 80.4% +/- 2.6% (3 seeds), MLP probe: 96.1%
- Cross-dataset CWRU->IMS transfer is statistically significant (p=0.00003)
- Best config: embed_dim=512, 100 epochs, mask_ratio=0.5, mean-pool

**The problem:**
- The PREDICTOR HAS COLLAPSED. Diagnostic scripts (`quick_diagnose.py`,
  `diagnose_jepa.py`) show the predictor outputs nearly identical embeddings
  regardless of which position it's asked to predict. It ignores positional
  information entirely.
- This means the pretraining objective is partially broken: the model minimizes
  loss by predicting an average patch embedding, not position-specific content.
- The encoder still learns useful features (because it must provide good context),
  but we're leaving massive performance on the table.
- Cross-dataset transfer to IMS is only +2-4%, and FFT baseline hits 100% on
  binary IMS tasks.

**Root cause hypothesis:**
The predictor (2-layer transformer, predictor_dim=embed_dim//2) with learnable
positional embeddings is too weak or too homogeneous to differentiate positions.
The mask tokens all start identical, get similar positional offsets (small learned
pos embeddings), pass through only 2 transformer layers, and collapse to similar
outputs. The loss still decreases because predicting the mean is a valid (but lazy)
minimum.

### Architecture Reference

The predictor in `src/models/jepa.py` (JEPAPredictor, line ~208):
- input_proj: Linear(embed_dim -> predictor_dim)  [predictor_dim = embed_dim//2]
- mask_token: single learnable token, expanded for all mask positions
- pos_embed: learnable (1, n_patches, predictor_dim), trunc_normal_(std=0.02)
- 2 transformer blocks (predictor_dim, 4 heads, mlp_ratio=4, dropout=0.1)
- output_proj: Linear(predictor_dim -> embed_dim)

The encoder (JEPAEncoder, line ~110):
- PatchEmbed1D: Linear(n_channels * patch_size -> embed_dim)
- Learnable pos_embed + CLS token
- 4 transformer blocks

Loss: MSE on L2-normalized predictions vs targets (cosine-normalized MSE).

## FILES TO READ FIRST

Read ALL of these before running any experiment:

1. `autoresearch/mechanical_jepa/EXPERIMENT_LOG.md` -- all prior results (15+ experiments)
2. `autoresearch/mechanical_jepa/LESSONS_LEARNED.md` -- critical insights
3. `autoresearch/mechanical_jepa/program.md` -- original research plan
4. `mechanical-jepa/src/models/jepa.py` -- FULL model code, understand every line
5. `mechanical-jepa/src/models/jepa_enhanced.py` -- enhanced masking strategies
6. `mechanical-jepa/train.py` -- training loop, loss, optimizer, LR schedule
7. `mechanical-jepa/src/data/bearing_dataset.py` -- data pipeline
8. `mechanical-jepa/ims_transfer.py` -- cross-dataset transfer evaluation
9. `mechanical-jepa/quick_diagnose.py` -- the diagnostic that found the collapse
10. `mechanical-jepa/diagnose_jepa.py` -- full diagnostic with real data

## YOUR MISSION

Fix the predictor collapse, prove it's fixed with diagnostics, and demonstrate
materially improved cross-dataset transfer. Follow the cycle:

    Literature Review -> Sharp Experiments -> Validate & Analyze -> Repeat

### =======================================================================
### ROUND 1: UNDERSTAND THE PROBLEM (Literature + Diagnostics)
### =======================================================================

#### 1A. Deep Literature Review

Search the web thoroughly for:

1. **I-JEPA (Assran et al., CVPR 2023)** -- the original image JEPA
   - How does their predictor work? What depth, what positional encoding?
   - Do they report predictor collapse? What prevents it?
   - Key: they use a NARROW predictor (small dim) but SUFFICIENT depth

2. **Brain-JEPA (NeurIPS 2024 Spotlight)**
   - Their predictor architecture for time series
   - Spatiotemporal masking strategies (Cross-ROI, Cross-Time, Double-Cross)
   - How they handle positional encoding (Brain Gradient + sinusoidal)

3. **V-JEPA (Bardes et al., 2024)** -- video JEPA
   - Predictor design for spatiotemporal data
   - Any collapse prevention mechanisms

4. **"Predictor collapse" in JEPA literature**
   - Search for: "JEPA predictor collapse", "mask prediction collapse",
     "representation collapse self-supervised"
   - VICReg, Barlow Twins, DINO -- how do they prevent collapse?
   - What specific mechanisms (variance regularization, contrastive loss,
     asymmetric architectures) are known to work?

5. **JEPA for time series / vibration / industrial**
   - Any existing work applying JEPA to 1D time series?
   - TS2Vec, TNC, TS-TCC -- how do other time series SSL methods work?
   - What do they do differently from masked prediction?

6. **Sinusoidal vs learnable positional encodings**
   - When does each work better?
   - Frequency-based positional encoding for time series
   - Rotary positional embeddings (RoPE) -- applicable here?

Document key findings. Extract specific architectural choices and hyperparameters
from papers. Note what I-JEPA's predictor depth/dim/pos-encoding actually is.

#### 1B. Diagnostic Baseline

Before changing anything, establish the collapse baseline:

```bash
cd /home/sagemaker-user/IndustrialJEPA/mechanical-jepa

# Find best existing checkpoint
ls -la checkpoints/

# Run quick diagnostic on best checkpoint
python quick_diagnose.py --checkpoint checkpoints/<best_checkpoint>.pt

# Run full diagnostic if data is available
python diagnose_jepa.py --checkpoint checkpoints/<best_checkpoint>.pt
```

Record these numbers precisely. They are the "before" measurements:
- Predictor position variance (currently near 0)
- Prediction spread ratio (pred_std / target_std)
- Per-position cosine similarity pattern
- Encoder diversity metrics (should be healthy)

### =======================================================================
### ROUND 2: FIX THE PREDICTOR (Sharp PoC Experiments)
### =======================================================================

The goal is SHORT, TARGETED experiments. Train for 30 epochs max initially.
Always use `--seed 42` for comparability. ALWAYS use wandb (do NOT pass --no-wandb).

#### 2A. Modify the Predictor Architecture

Edit `src/models/jepa.py` to support these variations. Add CLI flags to `train.py`
so experiments are easy to run without editing code each time.

**Experiment 2A-1: Sinusoidal positional encoding in predictor**
- Replace learnable pos_embed with fixed sinusoidal encoding
- Hypothesis: Learnable pos embeddings collapse to similar values during training.
  Sinusoidal encoding provides guaranteed position discrimination.
- Implementation: Standard sine/cosine positional encoding from "Attention is All
  You Need", scaled to predictor_dim.

**Experiment 2A-2: Increase predictor depth (2 -> 4 layers)**
- Hypothesis: 2 layers is insufficient for the predictor to learn position-dependent
  transformations. More depth gives it capacity to differentiate positions.
- Keep predictor_dim the same (embed_dim//2).

**Experiment 2A-3: Separate mask tokens per position**
- Instead of one shared mask_token expanded to all positions, use N separate
  learnable tokens (one per patch position).
- Hypothesis: Shared mask token + weak pos encoding = collapse. Distinct tokens
  per position guarantee initial diversity.

**Experiment 2A-4: Variance regularization on predictions**
- Add a variance term to the loss: penalize low variance across predicted positions.
- Implementation: `var_loss = max(0, threshold - predictions.var(dim=1).mean())`
- Total loss = MSE_loss + lambda * var_loss
- Hypothesis: Explicitly prevents the collapse shortcut.

**Experiment 2A-5: Combined fix (sinusoidal + deeper + variance reg)**
- Combine the most promising individual fixes.

For each experiment:
1. Train 30 epochs, seed 42, embed_dim=512, wandb enabled
2. Run quick_diagnose.py on the checkpoint IMMEDIATELY after training
3. Check: Did predictor position variance increase? Did spread ratio improve?
4. Record linear probe accuracy AND diagnostic metrics
5. Log to EXPERIMENT_LOG.md

#### 2B. Quick Decision Gate

After 2A experiments (should take ~1-2 hours total):
- Which fix(es) improved predictor position variance the most?
- Which fix(es) improved linear probe accuracy?
- Did any fix make things worse?
- Select the best 1-2 approaches for deeper investigation.

### =======================================================================
### ROUND 3: VALIDATE THE FIX (Depth + Multi-Seed)
### =======================================================================

#### 3A. Scale Up the Best Fix

Take the winning approach from Round 2 and:

1. **Train 100 epochs** (the known optimal duration for CWRU)
2. **3-seed validation** (seeds 42, 123, 456)
3. **Run full diagnostics** on each checkpoint
4. **Compare to old best** (80.4% +/- 2.6% linear, 96.1% MLP)

```bash
# Example (adjust flags based on Round 2 winners)
python train.py --epochs 100 --seed 42 --embed-dim 512 --predictor-pos sinusoidal
python train.py --epochs 100 --seed 123 --embed-dim 512 --predictor-pos sinusoidal
python train.py --epochs 100 --seed 456 --embed-dim 512 --predictor-pos sinusoidal
```

#### 3B. Critical Analysis

For each trained model, answer:
- Is the predictor collapse actually fixed? (Check diagnostics, not just accuracy)
- Does linear probe accuracy improve, or only MLP probe?
- If linear probe improved: the features are better organized (more linearly separable)
- If only MLP improved: features are richer but still tangled
- Does the gap between linear and MLP probe shrink? (It should, if collapse is fixed)
- What does the per-class breakdown look like? Does outer_race (the hardest class) improve?

#### 3C. Ablation: What Actually Mattered?

If the combined fix won, ablate:
- Remove sinusoidal pos encoding, keep rest -> how much does accuracy drop?
- Remove variance reg, keep rest -> how much drops?
- Remove depth increase, keep rest -> how much drops?

This tells us the MECHANISM, not just the result.

### =======================================================================
### ROUND 4: CROSS-DATASET TRANSFER (The Real Test)
### =======================================================================

This is where the rubber meets the road. Prior transfer was +2-4% on IMS.
With a fixed predictor, can we do materially better?

#### 4A. CWRU -> IMS Transfer

Using the best fixed-predictor checkpoint:

```bash
python ims_transfer.py --checkpoint checkpoints/<best_fixed>.pt --seeds 42,123,456
```

Test on:
- IMS binary (healthy vs failure) -- Test 1 and Test 2
- IMS 3-class (healthy / degrading / failure)
- Compare: old JEPA (collapsed predictor) vs new JEPA (fixed) vs random init
- Compare: JEPA vs FFT baseline (the FFT baseline gets 100% -- can we close the gap?)

#### 4B. IMS Self-Pretrain with Fixed Predictor

Also pretrain on IMS itself with the fixed predictor architecture:

```bash
python ims_pretrain.py --epochs 50 --seed 42 --predictor-pos sinusoidal  # adjust flags
```

This gives the new upper bound. Compare:
- Old upper bound (collapsed predictor, IMS self-pretrain): 73.2% +/- 1.1%
- New upper bound (fixed predictor, IMS self-pretrain): ???
- New cross-dataset (fixed predictor, CWRU pretrain): ???

#### 4C. Few-Shot Transfer

If transfer improves, test data efficiency:
- How many IMS labeled samples does JEPA need to match random init with full data?
- Test N = 20, 50, 100, 200 labeled samples
- This is the compelling story: "JEPA needs 50 labeled samples to match 200 from scratch"

### =======================================================================
### ROUND 5: SECOND LITERATURE PASS + ADVANCED EXPERIMENTS
### =======================================================================

Based on Round 3-4 results, do a SECOND targeted literature review:

#### 5A. If Transfer Improved Significantly

Search for:
- "Foundation models for predictive maintenance"
- "Cross-domain transfer learning bearing fault detection"
- How do our numbers compare to published cross-dataset results?
- What's the SOTA for CWRU classification? (Usually 95-99% supervised)
- Can we claim anything novel about self-supervised transfer?

Then try:
- **Multi-task fine-tuning**: fault type + severity simultaneously
- **Temporal block masking** (from jepa_enhanced.py) instead of random masking
- **Cross-channel masking**: mask entire channels, predict from remaining
- **Patch size ablation with fixed predictor**: 128, 256, 512

#### 5B. If Transfer Did NOT Improve Significantly

The predictor collapse may not be the bottleneck. Search for:
- "Domain gap vibration signals" -- is the CWRU-IMS gap too large?
- "Sampling rate invariant features" -- 12kHz vs 20kHz problem
- "Domain adaptation self-supervised learning"

Then try:
- **Resampling**: Resample IMS from 20kHz to 12kHz before transfer
- **Frequency-domain pretraining**: Pretrain on FFT/STFT patches instead of raw signal
- **Multi-scale patches**: Use multiple patch sizes simultaneously
- **Adversarial domain adaptation**: Add domain discriminator during fine-tuning

#### 5C. Regardless of Outcome

Try one "wild card" experiment:
- **Contrastive + predictive hybrid loss**: Add InfoNCE alongside MSE
  - Predictions should be closer to their target than to other targets
  - This directly prevents collapse without needing variance regularization
- **Rotary positional embeddings (RoPE)** in the predictor
  - Better relative position encoding for sequential data

### =======================================================================
### ROUND 6: FINAL ANALYSIS & DOCUMENTATION
### =======================================================================

#### 6A. Update Jupyter Notebook

Update `notebooks/03_results_analysis.ipynb` with:

1. **New section: "Predictor Collapse & Fix"**
   - Before/after diagnostic visualizations
   - PCA of predictions vs targets (collapsed vs fixed)
   - Per-position cosine similarity (collapsed vs fixed)

2. **Updated results table**
   - All configurations: old JEPA, fixed JEPA, random init, FFT baseline
   - CWRU and IMS results side by side
   - 3-seed mean +/- std for all

3. **t-SNE visualizations**
   - CWRU embeddings: old vs fixed JEPA
   - IMS embeddings: CWRU-pretrained (fixed) vs random init
   - Color by fault type / degradation stage

4. **Confusion matrices**
   - Best fixed-predictor model on CWRU (4-class)
   - Best fixed-predictor model on IMS (binary + 3-class)

5. **The story**: Clear narrative from problem -> diagnosis -> fix -> result

#### 6B. Update Documentation

- EXPERIMENT_LOG.md: All new experiments with full details
- LESSONS_LEARNED.md: New insights about predictor collapse and fix
- README.md: Update with new best results if improved

#### 6C. Final Commit

Commit all changes with descriptive message summarizing findings.

## GLOBAL RULES

### Experiment Discipline
- **ALWAYS use wandb** (never pass --no-wandb). Project: 'mechanical-jepa'
- **Log EVERY experiment** to EXPERIMENT_LOG.md, even failures
- **3+ seeds** for any claim. 1 seed for initial exploration only.
- **Never tune on test set**
- **Run diagnostics after every training run** (quick_diagnose.py at minimum)
- **Short experiments first** (30 epochs), scale up only what works

### Code Changes
- Add CLI flags for new features (don't hardcode experimental changes)
- Keep backward compatibility (old checkpoints should still load)
- Comment WHY, not WHAT

### Commit Protocol
- Commit after each round completes
- Commit message format: "Exp N-M: [brief description of finding]"
- Push after each round

### Self-Criticism Checklist (ask yourself after every experiment)
- [ ] Is this improvement real or could it be noise? (Check error bars)
- [ ] Am I comparing fairly? (Same seeds, same epochs, same eval protocol)
- [ ] Could a simpler baseline explain this? (FFT, random features, etc.)
- [ ] Does the diagnostic actually show the collapse is fixed?
- [ ] Am I fooling myself with the MLP probe? (It can fit anything)
- [ ] Would this result survive peer review?

### Stopping Conditions

Stop and write final summary when:
1. All 6 rounds complete
2. You achieve >85% CWRU linear probe (fixed predictor should unlock this)
3. You achieve >5% transfer gain on IMS (double the current +2-4%)
4. You've been running for 8+ hours
5. You hit an irrecoverable error

### Expected Timeline (rough)
- Round 1 (Literature + Diagnostics): 30-60 min
- Round 2 (PoC experiments): 60-90 min
- Round 3 (Validation): 60-90 min
- Round 4 (Transfer): 60-90 min
- Round 5 (Advanced): 60-90 min
- Round 6 (Documentation): 30-60 min
- Total: 5-8 hours

Good luck. Be rigorous. Be honest. Fix this predictor.
```

---

## Pre-Flight Checklist

Before starting the overnight run, verify:

- [ ] Dataset downloaded: `ls mechanical-jepa/data/bearings/bearing_episodes.parquet`
- [ ] IMS data available: `ls mechanical-jepa/data/bearings/ims/`
- [ ] Smoke test passes: `cd mechanical-jepa && python train.py --epochs 2 --no-wandb`
- [ ] Diagnostics work: `python quick_diagnose.py --checkpoint checkpoints/<latest>.pt`
- [ ] WandB authenticated: `python -c "import wandb; print(wandb.api.api_key[:8])"`
- [ ] Git is clean: `git status`
- [ ] GPU available: `python -c "import torch; print(torch.cuda.is_available())"`

## How to Launch

```bash
# In Claude Code, run:
# Use the ml-researcher agent. Run autoresearch overnight on the Mechanical-JEPA
# project. Read autoresearch/mechanical_jepa/OVERNIGHT_PROMPT.md for full instructions.
```

## Expected Outcomes

After a successful overnight run:

1. **Predictor collapse diagnosed and fixed** with before/after diagnostics
2. **CWRU accuracy improved** beyond 80.4% linear probe baseline
3. **IMS transfer improved** beyond +2-4% (target: +5-10%)
4. **Ablation results** showing which fix components matter
5. **Updated notebook** with full analysis and visualizations
6. **All experiments on wandb** for inspection
