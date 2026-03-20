# Many-to-1 Transfer Learning Research State

Last updated: 2025-03-20

## Core Insight

**1-to-1 transfer is nearly impossible. Many-to-1 is tractable.**

Training on multiple source machines forces the model to learn domain-invariant features.

---

## Current Status

### Track 1: Bearing Transfer (Validation)
| Metric | Target | Current Best | Status |
|--------|--------|--------------|--------|
| Diagnosis Accuracy | ≥ 80% | - | ⏳ Not started |
| RUL Transfer Ratio | ≤ 2.0 | - | ⏳ Not started |

### Track 2: Robot Transfer (Novel)
| Metric | Target | Current Best | Status |
|--------|--------|--------------|--------|
| Avg Anomaly AUC | ≥ 0.60 | - | ⏳ Not started |
| Avg Forecast Ratio | ≤ 2.5 | - | ⏳ Not started |

---

## Datasets

### Track 1: Bearings (via PHMD library)

| Dataset | Units | Conditions | Signals | Task |
|---------|-------|------------|---------|------|
| CWRU | 161 | 4 loads | Vibration | Diagnosis |
| PHM 2012 | 17 | 3 conditions | Vibration | RUL |
| XJTU-SY | 15 | 2 conditions | Vibration | RUL |
| Paderborn (target) | Multiple | 3 speeds | Vibration | Diagnosis |

**Advantage**: Same signal type across all datasets.

### Track 2: Robot Manipulators

| Dataset | Robot | Joints | Signals | Source |
|---------|-------|--------|---------|--------|
| AURSAD | UR3e | 6 | pos, vel, torque | HuggingFace |
| Voraus-AD | Yu-Cobot | 6 | pos, vel, voltage | HuggingFace |
| UR3 CobotOps | UR3 | 6 | current, temp, speed | UCI |
| NIST UR5 | UR5 | 6 | pos, vel, current | NIST |
| Robot Failures | PUMA 560 | 6 | force, torque | UCI |

**Challenge**: Different signal types require semantic alignment.

---

## Working Hypotheses

### H1: Multi-Source Training Enables Generalization
- Training on N-1 sources forces learning of domain-invariant features
- Status: ⏳ Untested

### H2: Signal Semantics Matter More Than Exact Values
- Normalized effort (torque/current/voltage) should transfer
- Status: ⏳ Untested

### H3: Bearings Are Easier Than Robots
- Same signal type (vibration) should transfer well
- Validates approach before harder robot problem
- Status: ⏳ Untested

---

## Failed Approaches (DO NOT REPEAT)

| Approach | Why it Failed | Date |
|----------|---------------|------|
| 1-to-1 AURSAD→Voraus | Model overfits to source-specific features | 2025-03 |

---

## Promising Directions

| Direction | Evidence | Priority |
|-----------|----------|----------|
| Many-to-1 training | RT-X, Open X-Embodiment | HIGH |
| Semantic signal embedding | CHARM paper | MEDIUM |
| Domain-adversarial training | DANN literature | MEDIUM |

---

## Key Papers & Insights

### RT-X / Open X-Embodiment (Google, 2023)
- Key idea: Co-train on 22+ robot types
- Result: 50%+ improvement over single-source
- Insight: **Many sources → better generalization**

### CHARM (C3 AI, May 2025)
- URL: https://arxiv.org/abs/2505.14543
- Key idea: Semantic signal embeddings via LLM
- Result: SOTA on cross-machine transfer
- Applicable: For signal alignment in Track 2

### Domain Generalization Literature
- Key idea: Train on N-1 domains, test on held-out
- Standard setup for transfer learning
- Applicable: Our exact protocol

---

## Research Queue

### Phase 1: Setup (TODAY)
- [x] Install phmd, ucimlrepo
- [ ] Run 00_setup_datasets.py
- [ ] Validate all datasets load

### Phase 2: Track 1 - Bearings
- [ ] Run 01_bearing_baseline.py
- [ ] Analyze results
- [ ] Iterate if accuracy < 80%

### Phase 3: Track 2 - Robots
- [ ] Create unified robot data loader
- [ ] Run Leave-One-Robot-Out experiments
- [ ] Iterate if metrics not met

---

## Architecture Notes

### Current: Simple CNN Encoder + Classifier
```
Input (N, 1, seq_len)
  → Conv1D stack
  → AdaptiveAvgPool
  → Linear projection (embedding)
  → Classification head
```

### Ideas for Improvement
- [ ] Add domain tokens
- [ ] Domain-adversarial training (DANN)
- [ ] RevIN per-domain normalization
- [ ] Multi-task learning with domain ID
- [ ] Gradient blending

---

## Session Notes

### Session: 2025-03-20

**Goal**: Reformulate research from 1-to-1 (impossible) to many-to-1 (tractable)

**Progress**:
1. Deep web research on available datasets
2. Found 5+ robot datasets, 4+ bearing datasets
3. Created MULTI_SOURCE_DATASETS.md inventory
4. Created MANY_TO_ONE_PROMPT.md for overnight research
5. Updated OBJECTIVES_STATUS.md with new targets
6. Created experiment scripts:
   - 00_setup_datasets.py
   - 01_bearing_baseline.py

**Key Insight**: 1-to-1 transfer is nearly impossible because model can't distinguish domain-specific vs universal features. Many-to-1 forces learning invariant representations.

**Next**: Run setup script on VM, start overnight research.
