# Many-to-1 Transfer Learning Objectives Status

## Core Principle

**1-to-1 transfer is nearly impossible. Many-to-1 is tractable.**

---

## TRACK 1: Bearing Transfer (Validation)

**Goal**: Validate approach on clean, established benchmark

**Protocol**: Train on CWRU + PHM2012 + XJTU-SY → Test on Paderborn

| Metric | Target | Current Best | Achieved |
|--------|--------|--------------|----------|
| Diagnosis Accuracy | ≥ 80% | - | ❌ |
| RUL Transfer Ratio | ≤ 2.0 | - | ❌ |

**Approach that worked**: (fill when achieved)

**Command to reproduce**:
```bash
# (fill when achieved)
```

---

## TRACK 2: Robot Manipulator Transfer (Novel)

**Goal**: Many-to-1 transfer across robot types

**Protocol**: Train on 4 robots → Test on held-out robot (Leave-One-Robot-Out)

| Robot (Held-Out) | Forecast Ratio | Anomaly AUC | Status |
|------------------|----------------|-------------|--------|
| UR3e (AURSAD) | - | - | ❌ |
| Yu-Cobot (Voraus) | - | - | ❌ |
| UR3 (CobotOps) | - | - | ❌ |
| UR5 (NIST) | - | - | ❌ |
| PUMA (Failures) | - | - | ❌ |

**Aggregate Targets**:
| Metric | Target | Current Best | Achieved |
|--------|--------|--------------|----------|
| Avg Forecast Ratio | ≤ 2.5 | - | ❌ |
| Avg Anomaly AUC | ≥ 0.60 | - | ❌ |

**Approach that worked**: (fill when achieved)

---

## BOTH OBJECTIVES ACHIEVED?

**Status**: ❌ NO

**When achieved, update to**:
```
**Status**: ✅ YES

**Date**: [date]
**Total Experiments**: [N]
**Key Insight**: [what made it work]
```

---

## Key Datasets

| Dataset | Robot | Signals | Source |
|---------|-------|---------|--------|
| AURSAD | UR3e | pos, vel, torque | HuggingFace |
| Voraus-AD | Yu-Cobot | pos, vel, voltage | HuggingFace |
| UR3 CobotOps | UR3 | current, temp, speed | UCI |
| NIST UR5 | UR5 | pos, vel, current | NIST |
| Robot Failures | PUMA 560 | force, torque | UCI |
| CWRU | Bearing | vibration | PHMD |
| PHM 2012 | Bearing | vibration | PHMD |
| XJTU-SY | Bearing | vibration | PHMD |
| Paderborn | Bearing | vibration | Direct |

---

## Reproducibility

Full commands to reproduce:

```bash
# 1. Setup
cd ~/IndustrialJEPA
pip install phmd ucimlrepo

# 2. Track 1: Bearing Transfer
python scripts/bearing_transfer.py

# 3. Track 2: Robot Transfer
python scripts/robot_transfer.py --leave-out voraus
```

---

## Files

- `MANY_TO_ONE_PROMPT.md` - Main research prompt
- `MULTI_SOURCE_DATASETS.md` - Dataset inventory
- `EXPERIMENT_LOG.md` - All experiments
- `figures/` - Visualizations
