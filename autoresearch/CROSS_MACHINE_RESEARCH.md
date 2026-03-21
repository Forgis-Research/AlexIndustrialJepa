# Many-to-1 Transfer Learning Research State

Last updated: 2026-03-20 22:15

## Current Status

| Objective | Target | Current Best | Status |
|-----------|--------|--------------|--------|
| Many-to-1 Forecasting Transfer | Ratio ≤ 1.5 | 1.14 (1-to-1) | ⏳ Reformulating |
| Cross-Machine Anomaly Detection | AUC ≥ 0.70 | 0.53 (1-to-1) | ⏳ Reformulating |

### Track 1: Bearing Transfer (Validation)
| Metric | Target | Current Best | Status |
|--------|--------|--------------|--------|
| Diagnosis Accuracy | ≥ 80% | - | ⏳ Not started |
| RUL Transfer Ratio | ≤ 2.0 | - | ⏳ Not started |

### Track 2: Robot Transfer (Novel)
| Metric | Target | Current Best | Status |
|--------|--------|--------------|--------|
| Avg Anomaly AUC | ≥ 0.60 | 0.53 | ⏳ Reformulating |
| Avg Forecast Ratio | ≤ 2.5 | 1.14 | ⏳ Reformulating |

---

## PIVOT: Many-to-1 Transfer (from 1-to-1)

### Rationale
1-to-1 transfer (AURSAD→Voraus) showed:
- Forecasting works well (ratio 1.14) but limited by single-source generalization
- Anomaly detection fails because anomaly signatures differ across robots
- Many-to-1 (train on N-1, test on held-out) is more tractable and realistic

### Available Sources in FactoryNet

| Dataset | Robot/Machine | DOF | Episodes | Rows | Effort Signal | Task |
|---------|--------------|-----|----------|------|---------------|------|
| AURSAD | UR3e | 6 | 4094 | 6.2M | voltage | Screwdriving |
| Voraus | Yu-Cobot | 6 | 2122 | 11.6M | voltage | Pick-and-place |
| CNC | UMich CNC | 4 | 18 | 25K | current/voltage | Milling |

### Track 1: Bearings (via PHMD library)

| Dataset | Units | Conditions | Signals | Task |
|---------|-------|------------|---------|------|
| CWRU | 161 | 4 loads | Vibration | Diagnosis |
| PHM 2012 | 17 | 3 conditions | Vibration | RUL |
| XJTU-SY | 15 | 2 conditions | Vibration | RUL |
| Paderborn (target) | Multiple | 3 speeds | Vibration | Diagnosis |

### Track 2: Robots

| Dataset | Robot | Joints | Signals | Source |
|---------|-------|--------|---------|--------|
| AURSAD | UR3e | 6 | pos, vel, torque | HuggingFace |
| Voraus-AD | Yu-Cobot | 6 | pos, vel, voltage | HuggingFace |
| UR3 CobotOps | UR3 | 6 | current, temp, speed | UCI |
| NIST UR5 | UR5 | 6 | pos, vel, current | NIST |
| Robot Failures | PUMA 560 | 6 | force, torque | UCI |

### Shared Signal Space
All sources share:
- `setpoint_pos_0..N`: Joint position setpoints
- `setpoint_vel_0..N`: Joint velocity setpoints
- `effort_voltage_0..N`: Motor voltage (effort)

This enables training across sources in a unified representation.

---

## Experimental Plan

### Phase 1: Multi-Source Forecasting
- Leave-one-out: train on 2 sources, test on held-out
- Measure transfer ratio for each held-out source
- Compare 1-to-1 vs many-to-1

### Phase 2: Multi-Source Anomaly Detection
- Train on healthy data from N-1 sources
- Test anomaly detection on held-out source
- Multi-source provides more diverse "normal" patterns

### Phase 3: Scale and Improve
- Add RevIN normalization for domain adaptation
- Channel-independent processing
- External datasets if needed

---

## Key Findings (from 1-to-1 experiments)

### What Works
- Forecasting transfer ratio 0.9-1.14 with episode normalization
- Same signal space (voltage) across datasets
- Sorted DataFrame with iloc-based indexing for correct data loading

### What Doesn't Work
- Setpoint→effort prediction error for anomaly detection (AUC ~0.5)
- Episode normalization for anomaly detection (erases signal)
- Raw (no normalization) for cross-domain (incomparable scales)

### Critical Bug Fixes Applied
1. `is_anomaly` logic now works for non-AURSAD datasets (removed tightening phase check)
2. DataFrame sorted by episode_id for contiguous iloc access
3. iloc-based windowing instead of loc-based

---

## Failed Approaches (DO NOT REPEAT)

| Approach | Why it Failed | Date |
|----------|---------------|------|
| Setpoint→effort prediction + episode norm | Episode norm erases anomaly signal | 2026-03-20 |
| Setpoint→effort prediction + no norm | Scales incomparable across robots | 2026-03-20 |
| Setpoint→effort prediction + global norm | Voraus anomalies not visible in voltage stats | 2026-03-20 |
| 1-to-1 AURSAD→Voraus anomaly detection | Anomaly signatures fundamentally different | 2026-03-20 |

---

## Promising Directions

| Direction | Evidence | Priority |
|-----------|----------|----------|
| Many-to-1 forecasting | 1-to-1 already works (ratio 1.14) | HIGH |
| RevIN normalization | Handles distribution shift per-instance | HIGH |
| JEPA temporal dynamics | May capture anomaly temporal patterns | MEDIUM |
| Channel-independent processing | PatchTST-style, handles heterogeneous DOF | MEDIUM |
| External datasets (CMAPSS) | More source diversity | LOW |

---

## Key Papers & Insights

### RT-X / Open X-Embodiment (Google, 2023)
- Key idea: Co-train on 22+ robot types
- Result: 50%+ improvement over single-source
- Insight: **Many sources → better generalization**

### CHARM (C3 AI, May 2025)
- Key idea: Semantic signal embeddings via LLM
- Result: SOTA on cross-machine transfer
- Applicable: For signal alignment in Track 2

---

## Research Queue

1. [x] Fix data loading bugs (iloc, anomaly labels)
2. [x] Run 1-to-1 baseline experiments
3. [x] Diagnose anomaly detection failure
4. [ ] Build multi-source training framework
5. [ ] Run leave-one-out experiments
6. [ ] Add RevIN normalization
7. [ ] Attempt multi-source anomaly detection
8. [ ] Create comprehensive figures and report

## Session Log

### Session: 2025-03-20 (Evening)
**Goal**: Reformulate research from 1-to-1 (impossible) to many-to-1 (tractable)

**Progress**:
1. Deep web research on available datasets
2. Found 5+ robot datasets, 4+ bearing datasets
3. Created MULTI_SOURCE_DATASETS.md inventory
4. Created MANY_TO_ONE_PROMPT.md for overnight research
5. Updated OBJECTIVES_STATUS.md with new targets
6. Created experiment scripts (00_setup_datasets.py, 01_bearing_baseline.py)

**Key Insight**: 1-to-1 transfer is nearly impossible because model can't distinguish domain-specific vs universal features. Many-to-1 forces learning invariant representations.
