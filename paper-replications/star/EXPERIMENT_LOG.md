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

