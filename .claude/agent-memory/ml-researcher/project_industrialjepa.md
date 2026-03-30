---
name: IndustrialJEPA Project Context
description: Core project overview — multiple JEPA experiments; Mechanical-JEPA overnight session (2026-03-30) achieves 80.4% bearing fault detection (3-seed validated)
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

**Status as of 2026-03-30**: Overnight session complete. All Phase 1-3 goals achieved.

### Best Result (3-seed validated):
- JEPA (512-dim, mean-pool, 100ep): **80.4% ± 2.6%**
- Random Init: 51.9% ± 3.4%
- Improvement: **+28.5% ± 4.7%** (min across seeds: +22.7%)
- MLP probe: **96.1%** (1 seed, 2-layer)

### Key Findings:
1. **Mean-pool is critical**: CLS token never receives JEPA gradient directly. Mean-pool over patch tokens exposes the actually-trained representations. 96.1% with MLP probe vs ~84% with CLS.
2. **embed_dim=512 >> 256**: +13% absolute improvement. Bigger width more important than depth.
3. **100 epochs optimal**: 200ep overfits on small CWRU dataset (~2400 windows total).
4. **Random init baseline is ~50%, not ~30%**: Untrained transformers have structured positional features.
5. **Per-class difficulty**: Healthy/Ball ~100%, Inner race 50-80%, Outer race 0-55% (hardest).

### Files:
- Training: `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/train.py`
- Model: `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/src/models/jepa.py`
- Data: CWRU dataset, split by bearing_id (not window), stratified
- Checkpoints: `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/checkpoints/`
- Analysis notebook: `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/notebooks/03_results_analysis.ipynb`
- Figures: `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/figures/` (11 figures)
- Experiment log: `/home/sagemaker-user/IndustrialJEPA/autoresearch/mechanical_jepa/EXPERIMENT_LOG.md`
- Lessons: `/home/sagemaker-user/IndustrialJEPA/autoresearch/mechanical_jepa/LESSONS_LEARNED.md`

### Success Criteria Status:
- [PASS] Multi-seed validation (3 seeds, all >22.7% improvement)
- [PASS] Test accuracy > 60% (best: 84.1%)
- [PASS] t-SNE clustering by fault type (notebook generated)
- [PASS] Confusion matrix analysis
- [PASS] All figures and analysis notebook complete

**Why:** Overnight autoresearch run 2026-03-30. Analogous to Brain-JEPA (NeurIPS 2024) for vibration signals.
**How to apply:** Use mean-pool for any future JEPA evaluation. The embed_dim=512, 100ep config is the production baseline.
