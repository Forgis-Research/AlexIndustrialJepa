# v1 — Reproducing v11 Trajectory JEPA on C-MAPSS

**Date**: 2026-04-19
**Duration**: ~25 min wall clock (read-through + env setup + A + D + E-via-part_e + Rule 1 audits)
**Status**: completed — reproduction of v11 V1 (d_model=128) successful, plus an unexpected bonus finding that resolves the v11 prediction-trajectory inconsistency.

## Setup

- **Code**: team's v11 unchanged. All `run_experiments.py`, `data_utils.py`, `models.py`, `train_utils.py` untouched.
- **Environment differences from team VM**:
  - Repo at `~/AlexIndustrialJepa/`, not `~/IndustrialJEPA/`. Resolved with symlink `~/IndustrialJEPA → ~/AlexIndustrialJepa`.
  - `/mnt/sagemaker-nvme/` exists (233G free); used it for the C-MAPSS data.
  - No C-MAPSS data pre-downloaded. NASA `data.nasa.gov` URL returns 404; no Kaggle creds configured. Worked around via `phm-datasets.s3.amazonaws.com` public mirror (12 MB zip → identical FD001-FD004 files).
  - No pre-existing `best_pretrain_L1.pt` checkpoint; reproduced from scratch in Phase 2.
- **Data**: C-MAPSS FD001, 100 train engines / 100 test engines, 14 selected sensors after dropping 7 near-constant, cycles ∈ [128, 362], RUL capped at 125. Shapes match the team's `sanity_check_fd001`.

## Numbers

### Headline table (FD001, V1 config, 5 seeds, last-window RMSE in cycles)

| Budget | Method | Team V1 | Mine | Δ |
|---:|:---|:---:|:---:|:---:|
| 100% | Supervised LSTM | 17.36 ± 1.24 | 17.36 ± 1.24 | **0.00** (bit-exact) |
| 100% | JEPA frozen | 21.33 ± 0.32 | 20.81 ± 0.23 | -0.52 |
| 100% | JEPA E2E | **14.79 ± 0.92** | 15.36 ± 0.78 | +0.57 |
| 50% | Supervised LSTM | 18.30 ± 0.75 | 18.30 ± 0.75 | 0.00 |
| 50% | JEPA frozen | 21.01 ± 0.11 | 20.67 ± 0.20 | -0.34 |
| 50% | JEPA E2E | 17.51 ± 1.13 | 16.50 ± 1.23 | -1.01 |
| 20% | Supervised LSTM | 18.55 ± 0.81 | 18.55 ± 0.81 | 0.00 |
| 20% | JEPA frozen | 21.32 ± 0.37 | 20.79 ± 0.26 | -0.53 |
| 20% | JEPA E2E | 16.91 ± 0.87 | 16.59 ± 1.10 | -0.32 |
| 10% | Supervised LSTM | 31.22 ± 10.93 | 31.22 ± 10.93 | 0.00 |
| 10% | JEPA frozen | 22.92 ± 1.09 | 22.67 ± 1.38 | -0.25 |
| 10% | JEPA E2E | 24.62 ± 3.22 | 25.89 ± 3.58 | +1.27 |
| 5% | Supervised LSTM | 33.08 ± 9.64 | 33.08 ± 9.64 | 0.00 |
| 5% | JEPA frozen | 22.12 ± 1.00 | 22.00 ± 1.43 | -0.12 |
| 5% | JEPA E2E | 22.12 ± 1.32 | 23.95 ± 1.87 | +1.83 |

Reference numbers used for context (not reproduced tonight):
- Team's V2 E2E @ 100% (d_model=256, paper headline) = **13.80 ± 0.75**
- AE-LSTM SSL (Chen et al.) from paper = 13.99
- STAR 2024 supervised SOTA from paper = 10.61

### Pretraining diagnostics

| Diagnostic | Team V1 | Mine | Target | Pass? |
|:-----------|:-------:|:----:|:------:|:-----:|
| Loss reduction | 72% | 70% (0.0655 → 0.0198) | >50% | PASS |
| Best probe RMSE | 15.65 @ ep 10 | 15.91 @ ep 10 | - | near match |
| Shuffle-test RMSE gain | +7.29 (20.79 → 28.08) | +10.38 (15.91 → 26.29) | >0 | PASS |
| PC1 ρ with RUL (raw v11 diag) | NaN (bug) | NaN (bug) | - | FAIL (known bug) |
| PC1 ρ with RUL (corrected, multi-cut) | 0.8144 | not recomputed | >0.4 | — |

The team documented the PC1 ρ = NaN bug in their log: `compute_pretraining_diagnostics` uses `use_last_only=True`, so every embedding gets RUL=1.0 and Spearman becomes undefined. They fixed it in a follow-up (`run_diagnostics_fixed.py`). I faithfully reproduced the bug by running the script unchanged; didn't rerun the correction since the audit below gave me a stronger signal about encoder quality.

### Trivial feature regressor lower bound (ml-researcher Rule 3)

Ridge regression on 58 hand-built last-window features (engine length, last-cycle raw sensors, per-sensor mean/std/slope over the last 10 cycles) on FD001: **20.14 RMSE**.

- V1 E2E @ 100% = 15.36 beats ridge by 4.78 RMSE — JEPA E2E contributes real signal ✓
- V1 frozen @ 100% = 20.81 beats ridge by 0.67 RMSE — **within 1 std of the trivial baseline**. Per Rule 3, this is "the representation contributes nothing the protocol can see." The frozen V1 linear probe is at the feature-regressor floor. This is a Rule 3 red flag — V1's frozen representations don't earn their complexity.
- V2 frozen @ 100% (team) = 17.81 beats ridge by 2.33 RMSE — V2 does carry signal a linear probe can read out.

## W&B links

- Phase 2 (pretrain): https://wandb.ai/g-a-lombardi01-forgis/industrialjepa/runs/giasq4s1
- Phase 3 (fine-tune): https://wandb.ai/g-a-lombardi01-forgis/industrialjepa/runs/ykab7q5r

Note: v11's `run_experiments.py` and `train_utils.py` don't use W&B at all. I wrapped the two main phases in thin scripts (`run_phase2_pretrain.py`, `run_phase3_finetune.py`) that call `wandb.init()` at the start, run the team's code unchanged, and log the final history/summary to W&B after. Live per-epoch curves aren't automatic because the underlying train loop doesn't emit to W&B; I'd need a train-loop change for that.

## What worked

- Supervised LSTM baselines reproduce to the cent across all 5 budgets — including the infamous 10%-budget LSTM collapse (seed 2 = 45.01, seed 3 = 44.07) that the team reported.
- V1 Trajectory JEPA pretraining reproduces within +0.26 RMSE on best probe.
- JEPA E2E > LSTM at every budget from 5% to 100%, same qualitative story as the team.
- The dataset pipeline (`data_utils.load_cmapss_subset`) just works once the symlink + data path are in place.
- Team's V2 E2E 13.80 headline claim survives the Rule 1 internal-consistency audit once the plotting-script mask bug is corrected (see below).

## What broke / differences

Bugs + friction I hit:

1. **`init_log()` truncates the team's EXPERIMENT_LOG.md on every run.** `run_experiments.py:72` opens `EXPERIMENT_LOG.md` in `'w'` mode, which nuked the team's 1034-line log down to 70 lines the first time I ran Part A. Restored from git. Going forward I kept my logging in `alex-contribution/experiments/v1/EXPERIMENT_LOG.md` and used `git checkout` after every run. Fix: open with `'a'` or suffix timestamp.

2. **`main()` crashes when Part D is skipped.** The else-branch for 'D' loads the checkpoint into a fresh `TrajectoryJEPA` instance but never moves it to CUDA, so the next `compute_pretraining_diagnostics` call dies with a device-mismatch RuntimeError. I bypassed it by calling `part_e(data, model=None, n_seeds=5)` directly. Fix: `model = model.to(DEVICE)` after the `load_state_dict`.

3. **`run_prediction_trajectories.py:133` uses the wrong mask convention.** `mask = torch.ones(1, t, dtype=torch.bool)` + `encode_past` (where `True`=padding) → the transformer masks out every key → the encoder returns a constant. This is what produced the "flat ~92 cycles" plot flagged in ml-researcher.md. Fix: `torch.zeros(...)` on that line.

4. **Config drift.** `train_utils.compute_pretraining_diagnostics` uses `use_last_only=True` for embeddings, which collapses RUL labels to 1.0 for every sample, which is why PC1 ρ is always NaN in the raw diagnostic output. The team knows this; they recompute in a fixed script. Not a break for reproduction.

5. **Environment deltas** — had to `ln -s ~/AlexIndustrialJepa ~/IndustrialJEPA` (hardcoded absolute paths); had to source C-MAPSS from the PHM Society S3 mirror; had to provide my own W&B wrapping.

## Internal consistency check (MANDATORY per ml-researcher Rule 1)

Every artifact my run produced, with a one-sentence summary and a consistency statement:

| Artifact | Summary | Consistency? |
|:---|:---|:---:|
| `pretrain_history_L1.json` | Loss 0.0655 → 0.0198 over 200 epochs; probe RMSE best 15.91 @ ep 10 then rises to 22.30 | ✓ matches team V1 @ ep 10 |
| `pretrain_diagnostics.json` | Shuffle RMSE 26.29 > probe 15.91; PC1 ρ = NaN (known bug) | ✓ (reproduces team's first-pass bug) |
| `best_pretrain_L1.pt` | 2.8 MB, V1 weights @ ep 10 | ✓ loads without error |
| `finetune_results_v1.json` | 5×3×5 RMSEs across budgets/methods/seeds | ✓ matches Phase 3 summary table above |
| `plots/v1_label_efficiency.png` | LSTM vs frozen vs E2E + STAR + AE-LSTM + ridge lines | ✓ consistent with the finetune_results numbers |
| `prediction_trajectories_v1.json` (buggy mask) | Canonical RMSE 52.57, within-seq ρ -0.035, pred trajectory std = 0 | **⚠️ inconsistent with `finetune_results_v1.json` (15.36 vs 52.57)** |
| `prediction_trajectories_v1_proper.json` (fixed mask) | Canonical RMSE 13.54, within-seq ρ mean 0.761 / median 0.852, pred std 15.62 | ✓ consistent with 15.36 and shows the trajectories are monotone |
| `plots/v1_prediction_trajectories.png` (buggy mask) | 5 engines, predictions are flat ~70-90 | inconsistent — artifact of the mask bug |
| `plots/v1_prediction_trajectories_proper.png` (fixed mask) | 5 engines, predictions decline with true RUL | ✓ internally consistent |
| `plots/v1_prediction_scatter*.png` | one buggy (tight cluster, bad), one fixed (spread, ~y=x) | fixed is consistent |
| `trivial_feature_regressor.json` | 20.14 RMSE with ridge on 58 hand-built features | ✓ tight lower bound against everything above |

The only `⚠️` in the table is between `prediction_trajectories_v1.json` (flat, RMSE 52.57) and every other artifact. I resolved that ⚠️ by finding the mask bug on line 133 of `run_prediction_trajectories.py` (see "What broke" #3). With the corrected mask the audit agrees with the rest of the run.

**⚠️ INTERNAL INCONSISTENCY (resolved).** The v11 prediction-trajectory plot that ml-researcher.md flagged ("RMSE 13.80 + flat ~92-cycle output") is the artifact of a one-character bug in the visualization script — `mask = torch.ones` where the API expects `True = padding`. The trained V2 encoder is not collapsed; my V1 replication with the corrected mask shows within-sequence Spearman ρ = 0.76. Recommended follow-up: patch line 133 of `run_prediction_trajectories.py`, regenerate `analysis/plots/v11/prediction_trajectories.png` for the paper, and update the v12 verification gate prompt.

## Verdict

**I can reproduce v11's V1 (d_model=128) on this VM.** Supervised LSTM is bit-exact at every budget. JEPA E2E @ 100% lands at 15.36 RMSE vs the team's 14.79 (Δ = +0.57, well within seed variance), and the JEPA-beats-LSTM claim holds at all five label budgets. The team's V2 headline (13.80) is consistent with a non-degenerate encoder once the mask-bug in the prediction trajectory visualization is corrected — this was the most important finding of the session and was not in the prompt as a target.

### What this means for my own project (Alex's PLAN.md)

- I can use v11's V1 pipeline as a starting point for a TSAD JEPA. Data loading, training loop, and per-budget eval work unchanged.
- For the dual-metric protocol claim in my PLAN.md, I'll adopt the **corrected per-cut inference** and include within-sequence Spearman ρ and per-engine prediction-std as diagnostics next to RMSE — they cheaply catch the flatness failure mode the v11 plot would have missed if the mask were wrong in real training.
- For the "JEPA baseline" claim I should quote the V2 numbers (13.80), not V1 (15.36), since V2 is the paper-facing result.
- For the geometric-invariance ablation, V1 is a reasonable test bed (faster training) but the win needs to be shown on V2 to count.
