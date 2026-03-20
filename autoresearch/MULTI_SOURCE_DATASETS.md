# Multi-Source Dataset Inventory for Many-to-1 Transfer Learning

## Core Insight

**1-to-1 transfer is nearly impossible. Many-to-1 transfer is tractable.**

Training on multiple source machines/datasets forces the model to learn domain-invariant features, enabling generalization to held-out machines.

---

## Candidate Dataset Groups

### Group A: Robot Manipulators (Best Match for Our Work)

| Dataset | Robot | Joints | Signals | Samples | Task | Download |
|---------|-------|--------|---------|---------|------|----------|
| **AURSAD** | UR3e | 6 | pos, vel, torque, current | ~1000 eps | Anomaly | HuggingFace |
| **Voraus-AD** | Yu-Cobot | 6 | pos, vel, voltage | ~600 eps | Anomaly | HuggingFace |
| **UR3 CobotOps** | UR3 | 6 | current, temp, speed | 7409 | Fault | [UCI](https://archive.ics.uci.edu/dataset/963/ur3+cobotops) |
| **NIST UR5** | UR5 | 6 | pos, vel, current, temp | ~50 runs | Degradation | [NIST](https://data.nist.gov/pdr/lps/754A77D9DA1E771AE0532457068179851962) |
| **KUKA KR300** | KUKA KR300 | 6 | pos, vel | Full traj | Dynamics | [NonlinearBenchmark](https://www.nonlinearbenchmark.org/benchmarks/industrial-robot) |
| **Robot Failures** | PUMA 560 | 6 | force, torque (F/T) | 463 | Failure | [UCI](https://archive.ics.uci.edu/dataset/138/robot+execution+failures) |
| **REASSEMBLE** | Franka | 7 | pos, vel, effort, F/T | 4551 demos | Assembly | [Paper](https://arxiv.org/abs/2502.05086) |

**Signal Overlap Analysis:**
- Position: ALL ✓
- Velocity: ALL ✓
- Effort (torque/current/voltage): 5/7 ✓
- Temperature: 2/7
- Force/Torque (external): 2/7

### Group B: Bearings (Clean Benchmark - PHMD Library)

| Dataset | Units | Conditions | Signals | Task | Access |
|---------|-------|------------|---------|------|--------|
| **CWRU** | 161 | 4 loads | Vibration | Diagnosis | phmd |
| **PU (Paderborn)** | Multiple | 3 speeds | Vibration | Diagnosis | Direct |
| **PHM 2012** | 17 | 3 conditions | Vibration | RUL | phmd |
| **XJTU-SY** | 15 | 2 conditions | Vibration | RUL | phmd |

**Advantage**: All have same signal type (vibration), different physical machines.

### Group C: Industrial Machines (Mixed)

| Dataset | Machine | Signals | Task | Access |
|---------|---------|---------|------|--------|
| **SCANIA** | Trucks | Histograms, counters | Failure | [SND](https://snd.se/en/catalogue/dataset/2024-34) |
| **C-MAPSS** | Turbofan | 21 sensors | RUL | Kaggle |
| **CNC Milling** | CNC | Vibration, current | Tool wear | Nature |
| **Hydraulic** | Hydraulic | Pressure, temp | Fault | PHMD |

---

## Recommended Strategy: Two-Track Approach

### Track 1: Robot Manipulators (Novel Contribution)

**Sources (train on 4-5):**
1. AURSAD (UR3e)
2. UR3 CobotOps (UR3)
3. NIST UR5 (UR5)
4. KUKA KR300 (KUKA)
5. Robot Failures (PUMA 560)

**Target (hold out 1):**
- Voraus-AD (Yu-Cobot) → Test generalization

**Signal Alignment:**
```
Unified representation:
- 6 joint positions (normalized)
- 6 joint velocities (normalized)
- 6 joint efforts (torque OR current OR voltage → unified)
```

### Track 2: Bearings (Validation on Clean Data)

**Sources (train on 3):**
1. CWRU
2. PHM 2012
3. XJTU-SY

**Target (hold out 1):**
- Paderborn University (PU)

**Advantage**: Same signal type, clean separation, established baselines.

---

## Signal Normalization Strategy

### Problem
Different robots have different:
- Joint limits (radians)
- Torque ranges (Nm)
- Current ranges (A)
- Sampling rates (Hz)

### Solution: Semantic Normalization

```python
class SemanticSignalNormalizer:
    """
    Normalize signals to [-1, 1] based on semantic meaning.
    """

    POSITION_RANGE = (-2*pi, 2*pi)  # radians
    VELOCITY_RANGE = (-pi, pi)       # rad/s (typical)
    EFFORT_RANGE = (0, 1)            # normalized effort

    def normalize_joint_position(self, pos, joint_limits):
        return 2 * (pos - joint_limits[0]) / (joint_limits[1] - joint_limits[0]) - 1

    def normalize_effort(self, effort, effort_type, robot_specs):
        """Convert torque/current/voltage to normalized effort."""
        if effort_type == 'torque':
            return effort / robot_specs.max_torque
        elif effort_type == 'current':
            return effort / robot_specs.max_current
        elif effort_type == 'voltage':
            return effort / robot_specs.max_voltage
```

---

## Data Loading Plan

### Phase 1: Quick Start (Use PHMD)

```python
from phmd import datasets

# Load bearing datasets
cwru = datasets.Dataset("CWRU").load()
phm12 = datasets.Dataset("PHM12").load()  # Check exact name
xjtu = datasets.Dataset("XJTU-SY").load()

# Train on CWRU + PHM12, test on XJTU
```

### Phase 2: Robot Datasets (Custom Loaders)

```python
# 1. AURSAD/Voraus - already have in FactoryNetDataset

# 2. UR3 CobotOps - from UCI
from ucimlrepo import fetch_ucirepo
ur3_cobotops = fetch_ucirepo(id=963)

# 3. NIST UR5 - download CSV
# wget from data.nist.gov

# 4. Robot Failures - from UCI
robot_failures = fetch_ucirepo(id=138)

# 5. KUKA - from nonlinearbenchmark.org
```

---

## Evaluation Protocol

### Leave-One-Robot-Out Cross-Validation

```
For each robot R in [UR3e, UR3, UR5, KUKA, PUMA, Yu-Cobot]:
    Train on: All robots except R
    Test on: R (zero-shot)

    Metrics:
    - Forecasting: MSE, MAE, transfer ratio
    - Anomaly: ROC-AUC, PR-AUC
```

### Success Criteria (Revised)

| Metric | 1-to-1 Target | Many-to-1 Target |
|--------|---------------|------------------|
| Transfer Ratio | ≤ 1.5 | ≤ 2.0 |
| Anomaly AUC | ≥ 0.70 | ≥ 0.65 |

Many-to-1 with weaker per-target performance but consistent across ALL held-out robots = success.

---

## Implementation Priority

1. **Immediate**: Set up bearing datasets via PHMD (validation)
2. **Day 1**: Download and preprocess robot datasets
3. **Day 2**: Implement unified data loader with semantic normalization
4. **Day 3+**: Run many-to-1 experiments

---

## Key References

- [PHMD Library](https://github.com/dasolma/phmd) - 59 PHM datasets unified
- [Awesome Industrial Datasets](https://github.com/jonathanwvd/awesome-industrial-datasets)
- [UCI UR3 CobotOps](https://archive.ics.uci.edu/dataset/963/ur3+cobotops)
- [NIST UR5 Degradation](https://www.nist.gov/el/intelligent-systems-division-73500/degradation-measurement-robot-arm-position-accuracy)
- [KUKA KR300 Benchmark](https://www.nonlinearbenchmark.org/benchmarks/industrial-robot)
