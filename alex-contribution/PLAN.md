# Plan

**Deadline**: NeurIPS 2026 (~2026-05-01)
**Status**: scoping

## Goal
Use JEPA to predict anomalies in time series. Beat MTS-JEPA; introduce a new eval protocol.

## Contribution (MVP, 3 claims)
1. **Dual metric for TSAD**: F1 (did it fire) + MAE-given-true (how early). New protocol, not just a model.
2. **JEPA baseline** on the new protocol — reuse v11 Trajectory JEPA where possible.
3. **Geometric extension**: one concrete invariance (candidate: channel-permutation à la iTransformer), ablated.

## Non-goals
- Left-truncation angle (dropped).
- New architecture from scratch.
- Beating supervised SOTA on every dataset.

## MVP paper shape
- 4 datasets (match MTS-JEPA + TSB-AD)
- 3 competitors: MTS-JEPA, a Brain-JEPA-style TS JEPA, one strong supervised
- Dual metric reported, one ablation of the geometric invariance
