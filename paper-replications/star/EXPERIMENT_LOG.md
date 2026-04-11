# STAR Replication Experiment Log

Paper: "STAR: Spatio-Temporal Attention-based Regression for Turbofan RUL Prediction"
Fan et al., Sensors 2024
Replication started: 2026-04-11

---

## Context

- Hardware: NVIDIA A10G (23 GB), but shared with 6-7 other concurrent processes
- Effective training speed: ~33 min/seed for FD001 (3.7M params, 15K windows)
- Estimated total runtime: FD001 ~2.75h, FD002 ~2.5h, FD003 ~1.7h, FD004 ~5-7.5h
- Session started: 2026-04-11T12:06 UTC

---

## Log


### 2026-04-11T12:39:12 | FD001 seed=42

- **Hyperparams**: lr=0.0002,bs=32,w=32,scales=3,dm=128,nh=1
- **Parameters**: 3,676,992
- **Epochs run**: 39 (best epoch 19)
- **Best val RMSE**: 13.150
- **Test RMSE**: 12.301 (paper target: 10.61)
- **Test Score**: 253.2 (paper target: 169)
- **Wall time**: 1980s
- **Notes**: none


### 2026-04-11T13:55:52 | FD001 seed=42

- **Hyperparams**: lr=0.0002,bs=32,w=32,scales=3,dm=128,nh=1
- **Parameters**: 3,676,992
- **Epochs run**: 39 (best epoch 19)
- **Best val RMSE**: 13.150
- **Test RMSE**: 12.301 (paper target: 10.61)
- **Test Score**: 253.2 (paper target: 169)
- **Wall time**: 1832s
- **Notes**: none


### 2026-04-11T14:29:30 | FD001 seed=123

- **Hyperparams**: lr=0.0002,bs=32,w=32,scales=3,dm=128,nh=1
- **Parameters**: 3,676,992
- **Epochs run**: 54 (best epoch 34)
- **Best val RMSE**: 11.396
- **Test RMSE**: 12.969 (paper target: 10.61)
- **Test Score**: 286.7 (paper target: 169)
- **Wall time**: 2018s
- **Notes**: none


### 2026-04-11T15:00:44 | FD001 seed=456

- **Hyperparams**: lr=0.0002,bs=32,w=32,scales=3,dm=128,nh=1
- **Parameters**: 3,676,992
- **Epochs run**: 41 (best epoch 21)
- **Best val RMSE**: 11.316
- **Test RMSE**: 11.250 (paper target: 10.61)
- **Test Score**: 196.2 (paper target: 169)
- **Wall time**: 1874s
- **Notes**: none


### 2026-04-11T15:27:45 | FD001 seed=789

- **Hyperparams**: lr=0.0002,bs=32,w=32,scales=3,dm=128,nh=1
- **Parameters**: 3,676,992
- **Epochs run**: 61 (best epoch 41)
- **Best val RMSE**: 12.140
- **Test RMSE**: 12.095 (paper target: 10.61)
- **Test Score**: 252.8 (paper target: 169)
- **Wall time**: 1621s
- **Notes**: none


### 2026-04-11T15:44:17 | FD001 seed=1024

- **Hyperparams**: lr=0.0002,bs=32,w=32,scales=3,dm=128,nh=1
- **Parameters**: 3,676,992
- **Epochs run**: 33 (best epoch 13)
- **Best val RMSE**: 11.002
- **Test RMSE**: 12.315 (paper target: 10.61)
- **Test Score**: 278.4 (paper target: 169)
- **Wall time**: 992s
- **Notes**: none

---

## FD001 Summary (5-seed, COMPLETE)

| Seed | Test RMSE | Test Score | Best Val RMSE | Epochs | Status |
|------|-----------|------------|---------------|--------|--------|
| 42 | 12.301 | 253.2 | 13.150 | 39 | GOOD |
| 123 | 12.969 | 286.7 | 11.396 | 54 | ABOVE |
| 456 | 11.250 | 196.2 | 11.316 | 41 | EXACT |
| 789 | 12.095 | 252.8 | 12.140 | 61 | GOOD |
| 1024 | 12.315 | 278.4 | 11.002 | 33 | GOOD |
| **Mean** | **12.186 +/- 0.553** | **253.5 +/- 31.6** | - | - | **GOOD** |
| Paper | 10.61 | 169 | - | - | - |
| Gap | +14.9% | +50% | - | - | GOOD |

**Assessment**: GOOD replication. 4/5 seeds in GOOD range (<=12.7), 1/5 EXACT (<=11.7). Mean 14.9% above paper. Score gap 50% (score is exponentially scaled, so larger gap is expected with RMSE gap). FD002 started 2026-04-11T15:44:20.

---

