# V12 Experiment Log

Session: 2026-04-12
Goal: Verify that V11's 13.80 RMSE measures what we think it measures.

## Session Start

**Time**: 2026-04-12 (overnight)
**Setup**: V12 scripts created. Launching Phase 2 in background, then Phase 0.2b, then Phase 0.1/0.2.

---

## Phase 2 Launch (T+0)
STAR label-efficiency sweep FD001 launched in background.
Script: `phase2_star_label_sweep.py`
Expected runtime: ~6 hours for all 25 runs (5 budgets x 5 seeds)

---

## Phase 0.2b - Engine Summary Regressor (T+0, highest priority)
Script: `phase0_2b_engine_regressor.py`
Expected runtime: ~20 minutes

---

## Phase 0.1/0.2 - V2 E2E Reconstruction + Trajectory Diagnostics
Script: `phase0_diagnostics.py`
Expected runtime: ~60 minutes (fine-tune + inference at every cycle for 100 engines)

---
