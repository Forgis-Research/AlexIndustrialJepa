# Many-to-1 Transfer Experiment Log

## Core Principle

**1-to-1 transfer is nearly impossible. Many-to-1 is tractable.**

---

## Objectives Status

### Track 1: Bearings (Validation)
| Metric | Target | Best Result | Achieved? |
|--------|--------|-------------|-----------|
| Diagnosis Accuracy | ≥ 80% | - | ❌ |
| RUL Transfer Ratio | ≤ 2.0 | - | ❌ |

### Track 2: Robots (Novel)
| Metric | Target | Best Result | Achieved? |
|--------|--------|-------------|-----------|
| Avg Forecast Ratio | ≤ 2.5 | - | ❌ |
| Avg Anomaly AUC | ≥ 0.60 | - | ❌ |

---

## Experiment Table

| # | Time | Track | Sources | Target | Metric | Result | Pass/Fail |
|---|------|-------|---------|--------|--------|--------|-----------|
| 1 | - | - | - | - | - | - | - |

---

## Detailed Experiment Log

### Template

```markdown
### Experiment #N: [Title]

**Date**: [AUTO]
**Track**: [1-Bearings / 2-Robots]
**Sources**: [list datasets used for training]
**Target**: [held-out dataset]

**Hypothesis**: [What you expect and why]
**Approach**: [What you're doing differently]

**Command**:
\`\`\`bash
[exact command]
\`\`\`

**Result**:
\`\`\`
[paste output]
\`\`\`

**Metrics**:
- Accuracy/AUC: [value]
- Transfer Ratio: [value]
- Seeds: [list 3 values] → mean ± std

**Pass/Fail**: [PASS/FAIL]
**Figure**: `figures/exp_N_description.png`

**Learnings**: [What did we learn?]
**Next**: [What to try next]

---
```

---

## Summary Statistics

| Category | Track 1 | Track 2 | Total |
|----------|---------|---------|-------|
| Total | 0 | 0 | 0 |
| Pass | 0 | 0 | 0 |
| Fail | 0 | 0 | 0 |

---

## Research Pauses

| Time | Topic | Key Finding |
|------|-------|-------------|
| - | - | - |

---

## Negative Results (Important!)

| Track | Approach | Why It Failed | Evidence |
|-------|----------|---------------|----------|
| - | - | - | - |

---

## Success Milestones

### Track 1: Bearings
- [ ] PHMD library installed and working
- [ ] CWRU dataset loads correctly
- [ ] PHM2012 dataset loads correctly
- [ ] XJTU-SY dataset loads correctly
- [ ] Paderborn dataset loads correctly
- [ ] First baseline (train all, test Paderborn)
- [ ] Accuracy > 70%
- [ ] **Accuracy ≥ 80%** (OBJECTIVE)
- [ ] Transfer ratio < 2.5
- [ ] **Transfer ratio ≤ 2.0** (OBJECTIVE)

### Track 2: Robots
- [ ] UR3 CobotOps downloaded
- [ ] NIST UR5 downloaded
- [ ] Robot Failures downloaded
- [ ] Unified data loader created
- [ ] First Leave-One-Out experiment
- [ ] AUC > 0.50 (better than random)
- [ ] AUC > 0.55
- [ ] **Avg AUC ≥ 0.60** (OBJECTIVE)
- [ ] Forecast ratio < 3.0
- [ ] **Avg Forecast ratio ≤ 2.5** (OBJECTIVE)

### Both Tracks Complete
- [ ] **TRACK 1 COMPLETE**
- [ ] **TRACK 2 COMPLETE**
- [ ] SUCCESS_REPORT.md written
- [ ] All changes committed

---

## Key Insights Discovered

(Add insights as you discover them)

1. ...

---

## Final Notes

(Fill when objectives achieved)

### What Worked

### What Didn't Work

### Key Insights

### Reproducible Commands
