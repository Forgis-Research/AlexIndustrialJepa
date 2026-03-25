# Executive Summary (Updated)

**Date**: 2026-03-25 (updated from 2026-03-23)
**Total experiments**: 42+
**Key result**: Grouped architecture beats CI-Trans by 34.6% on transfer (p=0.002, 10 seeds)

---

## The Real Finding

**The contribution is the grouped architecture, not physics-specific grouping.**

After 42+ experiments across 3 tiers (pendulum, turbofan, weather), the honest story is:

### What We Proved
1. **Grouped architecture >> Channel-independent**: 34.6% better transfer on C-MAPSS (p=0.002, Cohen's d=1.43, 9/10 seeds)
2. **Consistent across domains**: Physics grouping beats CI on 3/3 tiers (21% pendulum, 27% C-MAPSS, 4.9% weather)
3. **Effect scales with task difficulty**: Grouped advantage grows from 5.4% → 9.6% as weather forecast horizon increases 96→720
4. **Variance reduction**: Physics-masked attention has remarkably low variance (±0.00008 vs CI's ±0.0005 on weather)

### What We Disproved (Critical Honest Finding)
1. **Physics grouping ≈ random grouping**: Ablation shows random groups match or beat physics (FD002: random avg 53.01 vs physics 56.98, p=0.278 ns)
2. **The benefit is architectural, not knowledge-based**: ANY grouping provides the regularization benefit
3. **Full-Attention often wins**: Unconstrained attention matches or beats physics grouping on 2/3 tiers
4. **RoleTrans underperforms on weather**: The mean-pooling bottleneck loses information for non-component-based systems

### What Failed (Don't Pursue)
- JEPA pretraining (-33% on transfer)
- Contrastive pretraining (-11%)
- Slot-based concept discovery (slots collapse)
- Patch embeddings (no benefit on short sequences)

---

## Revised Paper Narrative

### OLD narrative (invalidated by ablation):
> "Physics-informed channel grouping captures true physical structure, enabling better transfer"

### NEW narrative (supported by evidence):
> "Grouped 2D architecture (shared within-group encoder + cross-group attention) provides strong regularization for cross-condition transfer. The grouping structure itself doesn't need to match physics — any reasonable partition improves over channel-independent processing. Physics groups offer interpretability and variance reduction but not performance advantage."

---

## Cross-Tier Results

| Tier | System | Grouped vs CI | Grouped vs Full-Attn | Best Overall |
|------|--------|--------------|---------------------|-------------|
| 1. Pendulum | Synthetic, 4ch | **+21%** | Comparable | PhysicsGrouped |
| 2. C-MAPSS | Turbofan, 14ch | **+27%** (p=0.002) | -17% (loses) | Full-Attn |
| 3. Weather | Climate, 14ch | **+4.9%** | -0.7% (close) | Full-Attn |

### Multi-Horizon Weather (Tier 3 Extended)

| Horizon | CI-Trans | Full-Attn | Role-Trans | RT vs CI |
|---------|----------|-----------|------------|----------|
| H=96 | 0.4570 | 0.4267 | 0.4323 | 5.4% |
| H=336 | 0.5608 | 0.5203 | 0.5248 | 6.4% |
| H=720 | 0.6214 | 0.5435 | 0.5620 | 9.6% |

### Grouping Ablation (C-MAPSS FD001→FD002)

| Condition | FD002 RMSE | vs CI-Trans (78.44) |
|-----------|-----------|---------------------|
| random_1 | 50.87 | -35% |
| random_2 | 52.92 | -33% |
| random_0 | 55.23 | -30% |
| physics | 56.98 | -27% |
| wrong | 61.99 | -21% |

---

## Where This Paper Can Still Be Strong

1. **The 2D treatment itself**: Temporal within-channel + Spatial across-channel is a clean, general architecture
2. **Variance reduction**: Physics grouping gives the most stable results (even if not best average)
3. **Scaling with horizon**: The grouped advantage grows with task difficulty
4. **Negative results**: Honest reporting of what doesn't work (JEPA, physics specificity) is valued
5. **Practical guidance**: "Group your sensors ANY way you want — it helps"

---

## Recommended Paper Structure

**Title**: "Grouped 2D Architecture for Cross-Condition Transfer in Multivariate Time Series"
(Drop "physics-informed" — the ablation doesn't support it)

**Abstract**: We show that grouping channels with shared within-group encoders and cross-group attention improves transfer by 34.6% over channel-independent processing (p=0.002). Surprisingly, the specific grouping assignment (physics-based, random, or deliberately wrong) matters less than the architectural pattern itself. We validate across synthetic (double pendulum), mechanical (turbofan), and meteorological (weather) systems.

**Sections**:
1. Introduction: CI paradox + our 2D treatment
2. Method: Grouped architecture (RoleTrans + PhysMask variants)
3. Experiments: 3 tiers, 3+ seeds, transfer + in-domain
4. **Ablation**: Random vs physics vs wrong grouping (key finding)
5. Multi-horizon: Effect scales with difficulty
6. Negative results: JEPA, slot attention, physics specificity
7. Discussion: When and why grouping helps

---

## Risk Assessment

### Strengths
- p=0.002 on 10-seed C-MAPSS transfer (robust)
- Consistent across 3 different domains
- Honest about what doesn't work

### Weaknesses
- Full-Attention beats grouped on 2/3 tiers → hard to argue for the *specific* architecture
- Physics grouping doesn't beat random → core novelty claim is weakened
- All datasets are relatively small (14-21 channels)

### Mitigations
- Frame as "understanding what matters in grouped architectures" (ablation study)
- Emphasize variance reduction and interpretability as practical benefits
- Test on larger-scale datasets (100+ channels) where full attention is expensive

---

## Session Statistics

| Metric | Value |
|--------|-------|
| Total experiments | 42+ |
| Papers reviewed | 35+ |
| Seeds tested (key result) | 10 |
| Tiers validated | 3 |
| Grouping conditions ablated | 5 |
| Horizons tested | 3 (96, 336, 720) |
| Statistical significance | p=0.002 |
| Git commits | 12+ |
