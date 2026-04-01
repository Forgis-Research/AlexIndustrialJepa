---
name: IndustrialJEPA Project Context
description: Mechanical-JEPA V2: predictor collapse fixed; IMS transfer +8.8% (3.7x improvement over V1); cross-domain beats self-pretrain (142% efficiency)
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

## 2. Mechanical-JEPA: Bearing Fault Detection

**Status as of 2026-04-01**: V2 complete. Predictor collapse fixed. Major IMS transfer improvement.

### V2 Architecture (src/models/jepa_v2.py)

Key changes from V1:
- Sinusoidal positional encoding in predictor (fixed learnable collapse)
- Mask ratio 0.5 → 0.625 (fewer context patches forces position use)
- L1 loss instead of MSE (less incentive for safe mean prediction)
- Variance regularization lambda=0.1 (direct collapse penalty)
- Predictor depth 2 → 4

Files:
- `mechanical-jepa/src/models/jepa_v2.py` — V2 model
- `mechanical-jepa/train_v2.py` — training script with all CLI flags
- `mechanical-jepa/train_spectral.py` — spectral input experiments

### V1 Results (CWRU, 3-seed):
- JEPA mean-pool linear probe: **80.4% ± 2.6%**
- MLP probe: **96.1%**
- IMS transfer: +2.4% ± 2.9% (Test 1), +3.9% (Test 2)

### V2 Results (CWRU, 3-seed validated):
| Metric | V1 | V2 | Delta |
|--------|----|----|-------|
| CWRU linear probe | 80.4% ± 2.6% | **82.1% ± 5.4%** | +1.7% |
| CWRU best (seed 123) | 84.1% | **89.7%** | +5.6% |
| IMS Test1 transfer | +2.4% ± 2.9% | **+8.8% ± 0.7%** | 3.7x |
| IMS 3-class transfer | +3.3% ± 1.3% | **+7.6% ± 1.8%** | 2.3x |
| IMS self-pretrain gain | +3.4% | **+6.2% ± 1.7%** | 1.8x |
| Transfer efficiency | 70% | **142%** | 2x |
| Predictor collapse | Yes | **No** | Fixed |

### Key V2 Insights:

1. **High mask ratio is the PRIMARY lever** — forces predictor to use position info (can't average context)
2. **Sinusoidal pos encoding = largest individual contribution** (+4.3% at 30ep ablation)
3. **CWRU pretraining beats IMS self-pretrain** (142% efficiency) — fault-diverse CWRU creates richer representations
4. **Transfer boundary**: works at ≤2x sampling rate ratio; fails at 5.3x (CWRU→Paderborn)
5. **Spectral inputs (dual=raw+FFT)**: 91-95% CWRU (high variance) but zero IMS transfer gain (sampling mismatch)
6. **Few-shot**: V2 provides 6-12% gain at ALL N values vs V1 only at N≈100

### Transfer Boundary (sampling rate):
- CWRU 12kHz → IMS 20kHz: +8.8% (works, 1.7x ratio)
- CWRU 12kHz → Paderborn 64kHz: -1.4% (fails, 5.3x ratio)
- Rule: transfer works when target sampling rate is within ~2x of pretrain

### IMS Dataset Insights:
- IMS has 3140 files (2156 test1 + 984 test2), 8 channels, 20kHz
- Fast loading: use `data/bearings/ims_npy_cache/` (npy cache of raw text files)
- Raw IMS files deleted to save space; npy cache is the source of truth now
- Symlink: `data/bearings/raw/ims → data/bearings/ims_npy_cache` (code expects raw/ims/)
- RMS precomputed: `data/bearings/ims_rms_cache.npy`
- Temporal split: first 25% = healthy, last 25% = failure (skip middle)

### Disk Space Notes:
- Raw IMS files (6.2GB) deleted; npy cache (1.7GB) is sufficient
- Symlink at `data/bearings/raw/ims` → npy cache enables backward compat
- ims_transfer.py and ims_pretrain.py both handle .npy files natively (patched)
- Checkpoints: jepa_v2_20260401_003619.pt (seed=123, V2 best, 89.7%) and jepa_v2_dual_20260401.pt (dual-domain)
- Wandb local logs deleted to save space; metrics are in wandb cloud

### Files:
- Training V1 (CWRU): `mechanical-jepa/train.py`
- Training V2: `mechanical-jepa/train_v2.py`
- Spectral: `mechanical-jepa/train_spectral.py`
- Transfer: `mechanical-jepa/ims_transfer.py`
- IMS pretraining: `mechanical-jepa/ims_pretrain.py`
- Best V2 ckpt: `mechanical-jepa/checkpoints/jepa_v2_20260401_003619.pt`
- Experiment log: `autoresearch/mechanical_jepa/EXPERIMENT_LOG.md` (Exp 0-23)
- Lessons: `autoresearch/mechanical_jepa/LESSONS_LEARNED.md`
- HuggingFace: `Forgis/Mechanical-Components` (bearings+gearboxes, not yet downloaded)

**Why:** V2 was needed to fix predictor collapse that was leaving performance on the table.
**How to apply:** Use `train_v2.py` for future experiments. Check `ims_transfer.py` for cross-dataset eval.
