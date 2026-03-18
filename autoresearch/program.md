# Autoresearch: KH-JEPA World Model Validation

## Mission

Validate **KH-JEPA** as a world model for physical time series.

**Goal**: Learn transferable dynamics, NOT minimize forecasting MSE.

**Key Metrics**:
1. Long-horizon stability (degradation ratio < 3x at H=720)
2. Cross-distribution transfer (h1→h2 ratio < 1.5)
3. Cross-task transfer (loosening→tightening)
4. MSE sanity check (< 0.45 on ETTh1-96, NOT optimization target)

---

## Quick Start

```bash
# 1. Prepare data
python prepare.py

# 2. Run training (default: jepa recipe)
python train.py

# 3. Check results
cat results/latest_results.json
```

---

## Recipes (3 Configurations)

| Recipe | JEPA | Koopman | SIGReg | Use For |
|--------|------|---------|--------|---------|
| `baseline` | No | No | No | MSE comparison |
| `jepa` | Yes | No | Yes | **Recommended** |
| `koopman` | Yes | Yes | Yes | Ablation |

Edit `RECIPE = "jepa"` in train.py to switch.

---

## Evaluation Tiers

### Tier 1: Sanity (Every Epoch)
- MSE < 0.45 (broken if > 0.50)
- z_std > 0.1 (collapsed if < 0.1)
- K_eigval < 1.0 (unstable if > 1.0)

### Tier 2: Cross-Distribution (After Tier 1)
- Train on ETTh1, test on ETTh2
- Transfer ratio < 1.5 = success
- Runs automatically after training

### Tier 3: Long-Horizon
```bash
# Test horizons 96, 192, 336, 720
python train.py --eval-long-horizon
```
- Degradation ratio = MSE_720 / MSE_96
- Good world model: ratio < 3.0
- Autoregressive: ratio >> 3.0

### Tier 4: Cross-Task (AURSAD)
```bash
python scripts/cross_task_transfer.py
```
- Train on loosening, test on tightening
- Same robot, related tasks

### Tier 5: Cross-Machine (Hardest)
```bash
python scripts/cross_machine_transfer.py
```
- AURSAD (UR3e) → voraus-AD (Yu-Cobot)
- Different robots, different tasks, different sensors
- Expect limited zero-shot, good few-shot

---

## Decision Tree

```
Run baseline recipe
    ↓
MSE > 0.50?  → Config broken, debug
    ↓
MSE > 0.45?  → Check z_std, learning rate
    ↓
MSE < 0.45?  → Tier 1 PASS, run Tier 2
    ↓
Transfer ratio > 1.5?  → JEPA not learning dynamics
    ↓
Transfer ratio < 1.5?  → Tier 2 PASS, run Tier 3
    ↓
Degradation ratio > 3.0?  → Koopman not helping
    ↓
Degradation ratio < 3.0?  → SUCCESS, run full evaluation
```

---

## Key Insight

We are NOT competing with Chronos/TimesFM on short-horizon MSE.

We ARE building a world model that:
1. Learns physics (transferable across machines)
2. Stays stable over long horizons (Koopman)
3. Detects anomalies (prediction residuals)

MSE is a sanity check, not the goal.

---

## Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script (edit RECIPE here) |
| `prepare.py` | Data preparation (ETTh1/h2) |
| `run.py` | Experiment runner |
| `benchmarks.py` | SOTA comparison numbers |

---

## Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| ETTh1-96 MSE | < 0.45 | Sanity check |
| h1→h2 transfer | < 1.5x | Core capability |
| H=720 degradation | < 3.0x | World model proof |
| AURSAD anomaly AUC | > 0.80 | Practical value |

---

## After Validation

If all tiers pass:
1. Run full benchmark suite (ETTh1, h2, m1, m2)
2. Run AURSAD/voraus-AD anomaly detection
3. Write up results for paper
4. Consider scaling to larger datasets

---

## References

- **SIGReg**: LeJEPA (Balestriero & LeCun, 2024) - provable collapse prevention
- **Koopman**: Linear dynamics in lifted space for physical systems
- **JEPA**: Joint Embedding Predictive Architecture (LeCun, 2023)
