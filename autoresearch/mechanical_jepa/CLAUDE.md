# Mechanical-JEPA

This folder contains the autoresearch setup for Mechanical-JEPA: self-supervised dynamics transfer across robot embodiments.

## Quick Start

```bash
# 1. Run sanity checks first (REQUIRED)
python experiments/sanity_check.py

# 2. If sanity checks pass, run viability test
python experiments/viability_test.py

# 3. If viability passes, start overnight pretraining
python experiments/pretrain.py --config small --epochs 50
```

## Key Files

| File | Purpose |
|------|---------|
| `program.md` | Full research plan — read this first |
| `EXPERIMENT_LOG.md` | Results log — update after each experiment |
| `LESSONS_LEARNED.md` | Reusable insights — update when you learn something |
| `OXE_SUMMARY.md` | Dataset reference — which datasets, what fields |
| `experiments/` | Python scripts for each phase |

## Core Hypothesis

A JEPA encoder pretrained on robot proprioceptive sequences (joint positions, velocities) learns transferable dynamics representations. This enables:

1. **Cross-embodiment transfer:** Pretrain on Franka → transfer to KUKA, UR5
2. **Few-shot adaptation:** 10x less data needed on new robot
3. **Action-conditioned prediction:** Better forecasting with actions than without

## Datasets

**Pretraining (7-DOF):**
- DROID: 76k episodes, Franka Panda
- ManiSkill: 30k episodes, Panda (sim)
- Stanford KUKA: 3k episodes, KUKA iiwa

**Transfer targets:**
- UR5, FANUC, JACO (6-DOF)
- TOTO (same Franka, different tasks)

## Success Criteria

| Metric | Target |
|--------|--------|
| Pretraining loss | Converges, no collapse |
| Embodiment classification | >70% (random=14%) |
| Transfer ratio | <2.0 (lower is better) |
| Few-shot efficiency | 5-10x less data |

## Anti-Patterns to Avoid

1. **Don't skip sanity checks** — bugs waste overnight compute
2. **Don't scale up too fast** — validate small, then grow
3. **Don't ignore collapse** — check embedding variance regularly
4. **Don't trust single seeds** — 3+ seeds for any claim

## Overnight Protocol

Before leaving overnight:
1. Verify sanity checks passed
2. Verify viability test passed
3. Set up checkpointing (every 10 epochs)
4. Enable logging to file
5. Set reasonable time limit

Check in morning:
1. Training converged? (loss curve)
2. No collapse? (embedding variance)
3. Checkpoints saved?
4. Any errors in logs?
