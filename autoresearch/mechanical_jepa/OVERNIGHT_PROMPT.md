# Overnight Autoresearch Prompt: Mechanical-JEPA

Use this prompt with the ml-researcher agent for overnight autonomous research.

---

## Prompt

```
Run autoresearch overnight on the Mechanical-JEPA bearing fault detection project.

## Context

You are continuing research on JEPA (Joint Embedding Predictive Architecture) for industrial bearing fault detection. This is analogous to Brain-JEPA (NeurIPS 2024 Spotlight) but applied to vibration signals instead of fMRI.

**Current status:**
- Latest run achieves 66.4% test accuracy (vs ~30% random init baseline)
- This shows learning (+36% improvement)
- BUT: Only tested within CWRU dataset (same test rig, similar bearings)
- **This is NOT strong transferability** — true transfer = train on one dataset, test on another

**Working directory:** `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa` (SageMaker)

**Critical insight:** Within-dataset accuracy is easy. Cross-dataset transfer is the real test.

## Your Mission

### Phase 1: Validate Current Results (First Priority)
1. Run multi-seed validation (seeds 42, 123, 456) at 30 epochs
2. Report mean ± std for test accuracy
3. Confirm the +19.8% improvement is consistent

### Phase 2: TRUE Cross-Dataset Transfer (CRITICAL)
This is the real test of transferability:

**Experiment: Train CWRU → Test IMS**
1. Pretrain JEPA on CWRU data only
2. Freeze encoder
3. Train linear probe on small IMS subset
4. Test on held-out IMS data

**Why this matters:**
- CWRU and IMS are different test rigs, different bearings, different failure modes
- If JEPA learns generic fault signatures, it should transfer
- If it only memorizes CWRU patterns, it will fail

**Implementation:** Modify train.py or create transfer_eval.py:
```python
# Pseudocode
pretrain(data='cwru')  # JEPA pretraining
probe_train(data='ims', split='train')  # Few-shot on IMS
evaluate(data='ims', split='test')  # Cross-dataset test
```

**Success metric:** IMS test accuracy > random (25%) with CWRU-pretrained encoder

### Phase 3: Systematic Improvements (if Phase 2 works)
Follow the experiment plan in `autoresearch/mechanical_jepa/program.md`:
- Training duration experiments (50, 100, 200 epochs)
- Architecture variations (encoder depth, embed dim)
- Masking strategy (0.3, 0.5, 0.7 mask ratios)
- Patch size (128, 256, 512)

For each experiment:
1. Form a clear hypothesis
2. Run with at least 1 seed initially
3. Log results to EXPERIMENT_LOG.md
4. Run 3 seeds only for promising results
5. Use common sense to interpret results, do they really show transferability? Feel free to also tune and even adapt the architecture as needed. Thoroughly do deep research and compare to the Brain-JEPA paper, but also consider the different modality (time series instead of images) and smaller dataset size.

### Phase 4: Analysis & Documentation
Once you have strong results:
1. Create/update `notebooks/03_results_analysis.ipynb` with:
   - Clear explanation of JEPA for bearing fault detection
   - Some very foundational but telling plots (time series forecasted vs actual, classification based on the architecture, etc...). What helps a human understand what's going on and also sanity check the results.
   - All baseline comparisons in a table
   - Best result with 3+ seeds (mean ± std)
   - t-SNE visualization by fault type
   - Confusion matrix
   - Conclusions

2. Update LESSONS_LEARNED.md with new insights

### Phase 5: FINAL REPORT (Required)
Create `notebooks/04_final_report.ipynb` — a concise, useful summary:

**Structure (keep it tight):**
```
1. Executive Summary (3-5 bullet points)
   - What worked, what didn't, key numbers

2. Method (1 paragraph + 1 diagram)
   - JEPA architecture for bearing signals

3. Results Table
   | Experiment | CWRU Acc | IMS Acc | Notes |
   |------------|----------|---------|-------|
   | Random init | 30% | - | baseline |
   | JEPA 30ep | 66% | ? | ... |

4. Key Visualizations (max 4 plots)
   - t-SNE: Do faults cluster?
   - Confusion matrix: What gets confused?
   - Loss curve: Did training converge?
   - Transfer plot: CWRU vs IMS embeddings

5. Honest Assessment
   - What ACTUALLY transfers vs what doesn't
   - Limitations and caveats
   - What would need to change for production use

6. Next Steps (3 concrete items)
```

**Rules for the report:**
- No fluff, no filler — every sentence must be useful
- Include actual numbers, not vague claims
- Show failure cases, not just successes
- Be honest about limitations
- A reader should understand the full story in 5 minutes

## Commands

```bash
# Quick test
python train.py --epochs 5 --no-wandb

# Standard training
python train.py --epochs 30 --seed 42

# Variations
python train.py --epochs 100 --seed 42
python train.py --encoder-depth 6 --seed 42
python train.py --mask-ratio 0.7 --seed 42
```

## Success Criteria

**MUST achieve:**
- [ ] Multi-seed validation complete (3 seeds) on CWRU
- [ ] Cross-dataset transfer attempted (CWRU → IMS)
- [ ] Results documented in EXPERIMENT_LOG.md

**TRUE SUCCESS (what we really want):**
- [ ] Cross-dataset transfer works: CWRU-pretrained encoder improves IMS classification
- [ ] Transfer gap < 20%: IMS accuracy within 20% of CWRU accuracy

**STRETCH goals:**
- [ ] Test accuracy > 70% on CWRU
- [ ] IMS accuracy > 40% with CWRU pretraining
- [ ] Clear t-SNE clustering by fault type (across datasets)
- [ ] Jupyter notebook with full analysis

## Anti-Patterns to Avoid

1. **Never skip logging** — Every experiment goes in EXPERIMENT_LOG.md
2. **Never claim with 1 seed** — 3+ seeds for any conclusion
3. **Never tune on test set** — Only use test for final evaluation
4. **Never ignore failures** — Negative results are information
5. **Never assume — always verify** — Before reporting any result as "true" or "working", critically self-check:
   - Could this be a bug? (data leakage, wrong split, mislabeled data)
   - Could this be random chance? (run multiple seeds)
   - Does this make physical sense? (bearings, fault signatures)
   - What would disprove this claim? (try to falsify your own results)

## Critical Self-Checking Protocol

Before claiming any result:
1. **Sanity check the numbers** — Is 90% accuracy suspicious? Is 25% just random guessing?
2. **Check for data leakage** — Same bearing in train and test? Overlapping windows?
3. **Verify the baseline** — Is random init actually random? Re-run to confirm.
4. **Look at predictions** — Do per-class accuracies make sense? Any class at 0% or 100%?
5. **Visualize embeddings** — Do t-SNE clusters match labels or something else (bearing ID)?
6. **Question everything** — If it seems too good, it probably is. Investigate.

## Stopping Conditions

Stop and summarize when:
1. All Phase 1-4 experiments complete
2. You achieve cross-dataset transfer (CWRU→IMS works)
3. You've run out of promising ideas
4. You hit an irrecoverable error

---

## LATE-STAGE FALLBACK IDEAS (Only if nothing else works)

If cross-dataset transfer fails and you've exhausted standard approaches, consider these:

### Fallback A: Synthetic Data Generation
Generate synthetic bearing vibration data to augment training:
- Use physics-based models (bearing fault frequencies)
- Add realistic noise patterns
- Create synthetic faults with known signatures
- Test if synthetic pretraining helps real-world transfer

Research: Look into bearing fault simulation methods in literature.

### Fallback B: TabPFN + Brain-JEPA Fusion (Experimental)
A speculative but potentially powerful idea:

**Concept:** Combine TabPFN's in-context learning with JEPA's self-supervised representations for industrial time series.

**Why it might work:**
- TabPFN excels at few-shot tabular prediction
- Brain-JEPA learns transferable features from masked prediction
- Industrial data is often tabular (sensor readings over time)
- Could enable zero/few-shot fault detection on new machine types

**Implementation sketch:**
1. Use JEPA encoder to extract time series features
2. Treat extracted features as tabular data
3. Use TabPFN for few-shot classification on new domains
4. Or: Train a TabPFN-style transformer on bearing features

**This is highly experimental** — only explore if primary approaches fail and you have time.
The goal: A foundation model for industrial machinery that works across different component types (bearings → gears → motors → pumps).

Reference: Check `TabPFN/` folder in the repo for TabPFN code.

## Files to Read First

1. `autoresearch/mechanical_jepa/CLAUDE.md` — Quick start guide
2. `autoresearch/mechanical_jepa/program.md` — Full research plan
3. `autoresearch/mechanical_jepa/EXPERIMENT_LOG.md` — Current results
4. `mechanical-jepa/train.py` — Main training script

## Commit Protocol

- Commit after each successful experiment batch
- Push every 5 experiments or after major findings
- Use descriptive commit messages

Good luck! Maximize the utility of this overnight run.
```

---

## How to Use

Copy the prompt above and use it with the ml-researcher agent:

```bash
# In Claude Code
> Use ml-researcher agent with the prompt from OVERNIGHT_PROMPT.md
```

Or launch directly:
```
Run autoresearch overnight on the Mechanical-JEPA bearing fault detection project. [paste full prompt]
```

---

## Pre-Flight Checklist

Before starting the overnight run, verify:

- [ ] Dataset downloaded: `python data/bearings/prepare_bearing_dataset.py --verify`
- [ ] Smoke test passes: `python train.py --epochs 5 --no-wandb`
- [ ] WandB configured (or use `--no-wandb`)
- [ ] Git is clean: `git status`
- [ ] EXPERIMENT_LOG.md is readable

---

## Expected Outcomes

After a successful overnight run, you should have:

1. **Multi-seed validation** — Mean ± std for baseline config
2. **Best configuration** — From systematic exploration
3. **Ablation results** — Understanding of what matters
4. **Jupyter notebook** — Clear presentation of findings
5. **Updated documentation** — All insights captured

---
