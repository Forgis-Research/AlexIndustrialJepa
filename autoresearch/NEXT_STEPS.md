# Next Research Push: Physics-Informed Forecasting for Mechanical Systems

**Goal**: SOTA performance on mechanical dynamical systems time series forecasting via physics-informed channel grouping.

**Core Claim**: Encoding physical structure (which sensors measure the same component) outperforms learning channel relationships from data.

---

## Validated So Far

| Result | Evidence |
|--------|----------|
| Role-Trans beats CI-Trans on C-MAPSS | RMSE 12.17 vs 13.82, p=0.005 |
| Physics grouping improves transfer | Transfer ratio 4.42 vs 6.16 |
| Architecture > Pretraining | JEPA variants underperformed |

---

## Three-Tier Validation (from VALIDATION_RADICAL.md)

### Tier 1: Double Pendulum (Synthetic)
- **Why**: Ground truth physics, 2 groups (mass_1, mass_2)
- **Data**: Generate or use [Mendeley Dataset](https://data.mendeley.com/datasets/7yd2ntbh3w/1) (1000 FPS, pre-extracted angles)
- **Task**: Predict next 10-50 timesteps
- **Transfer**: Train m1/m2=1.0 → Test m1/m2=0.5

### Tier 2: KAIST Motor (Real Mechanical)
- **Why**: 4 sensor groups (vibration, acoustic, thermal, electrical)
- **Data**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340923001671)
- **Task**: Fault diagnosis or forecasting
- **Transfer**: Cross-load (0Nm → 2Nm)

### Tier 3: Weather (Large Physical)
- **Why**: GIFT-Eval benchmark, 4 groups (temp, pressure, humidity, wind)
- **Data**: GIFT-Eval / Jena Climate
- **Task**: Temperature forecast
- **Baseline**: iTransformer 0.174 MSE

---

## Immediate Actions

```bash
# 1. Double pendulum data (Tier 1)
python experiments/generate_pendulum.py --n_trajectories 1000

# 2. Run Role-Trans on pendulum
python experiments/role_transformer_pendulum.py --groups physics

# 3. Download KAIST (Tier 2)
# From ScienceDirect supplementary materials

# 4. Run on ETT/Weather (Tier 3)
python experiments/role_transformer_ett.py --groups voltage_levels
```

---

## Active Experiments

| File | Purpose |
|------|---------|
| `role_transformer_cmapss.py` | Final C-MAPSS architecture |
| `cmapss_cross_fault.py` | Transfer experiments |
| `slot_concept_v2.py` | Learned slot attention |
| `jepa_etth1_v2.py` | JEPA baseline |
| `baselines_etth1.py` | iTransformer etc. |

---

## Success Criteria

| Tier | Metric | Baseline | Target |
|------|--------|----------|--------|
| 1 | MSE (10-step) | LSTM | <LSTM |
| 2 | Transfer Ratio | CI-Trans | <5.0 |
| 3 | MSE | iTransformer 0.174 | <0.174 |

Beat 2/3 tiers = paper. Beat 3/3 + transfer = strong paper.
