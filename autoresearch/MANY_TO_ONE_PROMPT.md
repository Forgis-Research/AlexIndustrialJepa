# Many-to-1 Transfer Learning: Overnight Research Prompt

## The Core Insight

**1-to-1 transfer is nearly impossible. Many-to-1 is tractable.**

Training on multiple source machines forces the model to learn domain-invariant features.

---

## CRITICAL RULES

1. DO NOT STOP until objectives are achieved
2. Log EVERY experiment (success AND failure)
3. Do deep web research when stuck (WebSearch tool)
4. Make ONE change at a time, validate, then proceed
5. NO QUESTIONS - make decisions based on data and research
6. CREATE FIGURES for every significant result
7. CRITICALLY evaluate your own results - run 3 seeds minimum
8. This is MANY-to-1, not 1-to-1. Use multiple source datasets.

---

## Research Tracks

### Track 1: Bearing Transfer (Validation - Start Here)

**Why**: Clean data, same signal types, established baselines. Validates approach.

**Datasets (via PHMD library)**:
- CWRU (161 units, vibration)
- PHM 2012 (17 bearings, vibration)
- XJTU-SY (15 bearings, vibration)
- Paderborn/PU (different rig, vibration)

**Protocol**:
```
Train: CWRU + PHM2012 + XJTU-SY
Test:  Paderborn (zero-shot)
```

**Success Criteria**:
| Metric | Target |
|--------|--------|
| Diagnosis Accuracy | ≥ 80% |
| Transfer Ratio (RUL) | ≤ 2.0 |

### Track 2: Robot Manipulator Transfer (Novel Contribution)

**Why**: Our main research goal. More challenging but impactful.

**Datasets**:
1. AURSAD (UR3e) - HuggingFace
2. Voraus-AD (Yu-Cobot) - HuggingFace
3. UR3 CobotOps (UR3) - UCI
4. NIST UR5 (UR5) - NIST
5. Robot Execution Failures (PUMA) - UCI

**Protocol**:
```
Train: 4 robot datasets
Test:  1 held-out robot (rotate through all)
```

**Success Criteria**:
| Metric | Target |
|--------|--------|
| Forecasting Transfer Ratio | ≤ 2.5 |
| Anomaly AUC (avg over held-out) | ≥ 0.60 |

---

## Context Files

1. `autoresearch/MULTI_SOURCE_DATASETS.md` - Dataset inventory
2. `autoresearch/EXPERIMENT_LOG.md` - All experiments
3. `analysis/cross_machine/CROSS_EMBODIMENT_TRANSFER.md` - Theory

---

## Quick Start Commands

### Install Dependencies
```bash
pip install phmd ucimlrepo
```

### Test PHMD Library
```python
from phmd import datasets

# List available datasets
print(datasets.list_datasets())

# Load CWRU
cwru = datasets.Dataset("CWRU")
print(cwru.describe())
data = cwru.load()
```

### Load UCI Robot Datasets
```python
from ucimlrepo import fetch_ucirepo

# UR3 CobotOps
ur3 = fetch_ucirepo(id=963)
print(ur3.data.features.shape)

# Robot Execution Failures
failures = fetch_ucirepo(id=138)
print(failures.data.features.shape)
```

---

## Experiment Protocol

### Before Each Experiment
```bash
echo "## Experiment $(date +%Y%m%d_%H%M)" >> autoresearch/EXPERIMENT_LOG.md
echo "HYPOTHESIS: [what you expect]" >> autoresearch/EXPERIMENT_LOG.md
echo "SOURCES: [which datasets]" >> autoresearch/EXPERIMENT_LOG.md
echo "TARGET: [held-out dataset]" >> autoresearch/EXPERIMENT_LOG.md
echo "METRIC: [number to beat]" >> autoresearch/EXPERIMENT_LOG.md
```

### After Each Experiment
```bash
echo "RESULT: [actual metric]" >> autoresearch/EXPERIMENT_LOG.md
echo "PASS/FAIL: [status]" >> autoresearch/EXPERIMENT_LOG.md
echo "INSIGHT: [what we learned]" >> autoresearch/EXPERIMENT_LOG.md
echo "---" >> autoresearch/EXPERIMENT_LOG.md
```

---

## Research Loop

```
WHILE objectives_not_achieved:

    1. DIAGNOSE current state
       - Read EXPERIMENT_LOG.md
       - Identify what's failing

    2. RESEARCH (if stuck > 3 attempts)
       - WebSearch: "multi-source domain adaptation time series"
       - WebSearch: "bearing fault diagnosis transfer learning 2024"
       - Log insights

    3. HYPOTHESIZE
       - Form specific hypothesis
       - Define success metric

    4. IMPLEMENT
       - ONE change only
       - Track 1 first (bearings), then Track 2 (robots)

    5. VALIDATE
       - Run with 3 seeds
       - Report mean ± std
       - Create figure

    6. UPDATE
       - Log results
       - If objective achieved, document

    7. REPEAT
```

---

## Ideas to Try (Prioritized)

### For Many-to-1 Transfer
1. [ ] Simple baseline: train on concatenated data
2. [ ] Domain mixing: random sampling from all sources
3. [ ] Domain-adversarial training (DANN)
4. [ ] Multi-task learning with domain ID
5. [ ] Gradient blending (weight domains by difficulty)
6. [ ] Meta-learning (MAML-style)

### For Signal Alignment
1. [ ] Standardize each signal to mean=0, std=1
2. [ ] Semantic grouping (position/velocity/effort)
3. [ ] Learned signal embeddings
4. [ ] RevIN per-domain normalization

### Architecture Ideas
1. [ ] Channel-independent processing (PatchTST style)
2. [ ] Domain tokens (like [CLS] for each source)
3. [ ] Attention over signal semantics
4. [ ] Shared encoder + domain-specific heads

---

## Figure Requirements

For EVERY significant result, save:

```python
import matplotlib.pyplot as plt

# Save figure
fig.savefig('autoresearch/figures/exp_XX_description.png', dpi=150, bbox_inches='tight')

# Create documentation
with open('autoresearch/figures/exp_XX_description.md', 'w') as f:
    f.write(f"""# {title}

![](./exp_XX_description.png)

## What it shows
{description}

## Key observations
- Observation 1
- Observation 2

## Implications
{implications}
""")
```

### Required Figures
1. [ ] Source domain distributions (per dataset)
2. [ ] t-SNE/UMAP of learned representations (colored by domain)
3. [ ] Transfer matrix (each source → each target)
4. [ ] Learning curves (with error bands)
5. [ ] ROC curves (all held-out targets overlaid)

---

## Success Declaration

When BOTH tracks achieve targets:

1. Update `OBJECTIVES_STATUS.md`
2. Create `SUCCESS_REPORT.md`:
   - What worked
   - What didn't
   - Key insights
   - Reproducible commands
3. Commit all changes
4. THEN you may stop

---

## START NOW

1. Install phmd and ucimlrepo
2. Verify bearing datasets load correctly
3. Run simple baseline on bearings (Track 1)
4. Once bearings work, move to robots (Track 2)
5. DO NOT STOP until success

Go.
