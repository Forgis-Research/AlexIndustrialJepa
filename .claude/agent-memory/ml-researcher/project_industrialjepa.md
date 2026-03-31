---
name: IndustrialJEPA Project Context
description: Core project overview — Mechanical-JEPA cross-dataset transfer to IMS (2026-03-31): p=0.00003, +3.2% mean gain, 14/15 experiments positive
type: project
---

IndustrialJEPA is a research project on self-supervised learning (JEPA) for industrial time series.

**Two main subprojects:**

## 1. Physics-Informed Attention (main paper)

52+ experiments, core finding: Physics-masked attention provides principled constraint when physics groups are statistically independent.

| System | Physics Mask Effect | Why |
|---|---|---|
| Double Pendulum | +7.4% over Full-Attn (p=0.0002) | Groups are truly independent |
| C-MAPSS | ≈ random mask (p=0.528) | Correlated degradation |
| ETT Weather | -1.3% vs Full-Attn | Thermal couples to all loads |

---

## 2. Mechanical-JEPA: Bearing Fault Detection (autoresearch/mechanical_jepa/)

**Status as of 2026-03-31**: Cross-dataset transfer complete. All experiments done.

### Phase 1 Best Results (CWRU, 3-seed validated):
- JEPA (512-dim, mean-pool, 100ep): **80.4% ± 2.6%**
- Random Init: 51.9% ± 3.4%
- Improvement: **+28.5% ± 4.7%**
- MLP probe: **96.1%** (2-layer)

### Phase 2 Cross-Dataset Transfer (CWRU → IMS, n=15 experiments):
**Statistical result: t=6.14, p=0.00003 (highly significant)**

| Experiment | JEPA | Random | Gain | Seeds |
|---|---|---|---|---|
| IMS Test1 binary | 72.0% | 69.6% | +2.4% | 2/3 |
| IMS Test2 binary | 88.4% | 84.4% | +3.9% | 3/3 |
| IMS Test1 3-class | 51.5% | 48.3% | +3.3% | 3/3 |
| IMS Test2 3-class | 59.4% | 56.4% | +3.0% | 3/3 |
| IMS self-pretrain | 73.2% | 69.7% | +3.4% | 3/3 |

**Key finding**: Cross-domain transfer efficiency = **70%** (CWRU→IMS gain vs IMS→IMS upper bound).
JEPA learns domain-agnostic vibration features.

**Honest limitations**:
- FFT + logistic regression = 100% (beats JEPA on binary task)
- Fine-tuning eliminates pretraining advantage (-0.5%)
- Few-shot n=20: not significant (3 seeds); n=100: +4.2% (3/3 seeds)
- JEPA most useful in frozen feature regime with ~100 labeled samples

### Key CWRU Architecture Insights:
1. **Mean-pool is critical**: CLS token never receives JEPA gradient. Mean-pool exposes trained representations.
2. **embed_dim=512 >> 256**: +13% absolute.
3. **100 epochs optimal**: 200ep overfits on small dataset (~2400 windows).
4. **Random init baseline ~50%**, not 30% — untrained transformers have structured positional features.

### IMS Dataset Insights:
- IMS has 3140 files (2156 test1 + 984 test2), 8 channels, 20kHz
- Fast loading: use `data/bearings/ims_npy_cache/` (pre-converted from text to .npy)
- RMS precomputed: `data/bearings/ims_rms_cache.npy`
- Temporal split for labels: first 25% = healthy, last 25% = failure (skip middle 50%)
- For 3-class: use files 0-25% / 40-60% / 80-100%

### Files:
- Training (CWRU): `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/train.py`
- Transfer experiment: `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/ims_transfer.py`
- IMS pretraining: `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/ims_pretrain.py`
- Best checkpoint: `checkpoints/jepa_20260330_221827.pt` (seed=123, 84.1% CWRU)
- Analysis notebook: `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/notebooks/03_results_analysis.ipynb` (43 cells)
- Plots: `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/notebooks/plots/` (4 figures)
- Experiment log: `/home/sagemaker-user/IndustrialJEPA/autoresearch/mechanical_jepa/EXPERIMENT_LOG.md` (Exp 0-15)
- Lessons: `/home/sagemaker-user/IndustrialJEPA/autoresearch/mechanical_jepa/LESSONS_LEARNED.md`

**Why:** Cross-dataset transfer was the critical missing piece after the overnight CWRU session.
**How to apply:** Use `ims_transfer.py` for any future cross-dataset experiments. Pre-use npy_cache for speed.
