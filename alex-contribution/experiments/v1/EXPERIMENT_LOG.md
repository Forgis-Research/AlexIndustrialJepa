# v1 — Reproduce v11 Trajectory JEPA on C-MAPSS

Alex's first overnight run. Observer mode.

## Mandatory reads — done

- [x] `alex-contribution/PLAN.md` — NeurIPS 2026 scope: dual metric + JEPA baseline + one geometric invariance
- [x] `mechanical-jepa/experiments/v11/EXPERIMENT_LOG.md` — v11 ran D+C+E+F+G, plus follow-up exps (V2/V3, FD002-4, MLP probe, PHM)
- [x] `mechanical-jepa/experiments/v11/RESULTS.md` — headline V2 E2E @ 100% = 13.80 RMSE on FD001 (beats AE-LSTM SSL 13.99)
- [x] `mechanical-jepa/experiments/v11/run_experiments.py` — A=dataset, B=pipeline, C=arch, D=pretrain, E=finetune (5 budgets, 5 seeds), F=plots, G=FD002, H=RESULTS
- [x] `mechanical-jepa/README.md` — notes v12 verification gate: v11 13.80 sits next to a flat ~92-cycle prediction trajectory (unresolved inconsistency)
- [x] `.claude/agents/ml-researcher.md` §Critical Evaluation Lens — 5-min sanity, internal consistency audit, trivial feature regressor lower bound, shuffle/ablation, warned v11 case
- [x] `archive/autoresearch/mechanical_jepa/OVERNIGHT_PROMPT_V5.md` §Pre-Flight Checklist — GPU/disk/NVMe/wandb/data/ckpt/git-clean

## Phase 0 — Environment setup

Start: 2026-04-19 19:07 UTC

| Step | Result |
|---|---|
| Symlink `~/IndustrialJEPA` → `~/AlexIndustrialJepa` | created, resolves to v11 files |
| `/mnt/sagemaker-nvme` available | 233G, 232G free |
| `nvidia-smi` | A10G, 23GB VRAM, driver 580, CUDA 13 |
| `torch.cuda.is_available()` | True, device name NVIDIA A10G |
| Requirements (torch/numpy/pandas/scipy/sklearn/matplotlib/wandb/dotenv/pyarrow) | all present, no `pip install` needed |
| `/home/sagemaker-user` disk | 10G total, 7.8G free — checkpoints must go to NVMe |
| C-MAPSS data | downloaded from `phm-datasets.s3.amazonaws.com` mirror (NASA 404'd, no Kaggle creds) |
| Data path | `/mnt/sagemaker-nvme/cmapss_data/6. Turbofan Engine Degradation Simulation Data Set/` → symlinked into `datasets/data/cmapss` |
| `load_raw('FD001')` | train (20631,26), test (13096,26), RUL (100,), 100 engines — ✓ |
| W&B API key | 86 chars loaded from `.env` |

End: 2026-04-19 19:12 UTC. Elapsed: ~5 min (under 45-min budget).

Deviations from the overnight prompt:
- Skipped `pip install -r requirements.txt` — everything was already resolvable.
- Skipped `bash setup_vm.sh` — that script targets bearing datasets (CWRU/IMS), not CMAPSS.
- Used the PHM Society S3 mirror for CMAPSS since NASA direct URL 404'd and no Kaggle creds are set up.
- No existing `best_pretrain_L1.pt` was found in `experiments/v11/` despite it being referenced in the original log, so Phase 2 (pretraining) must produce one from scratch.

## Phase 1 — Dry-run Part A

Start: 2026-04-19 19:12 UTC. End: 2026-04-19 19:13 UTC. Elapsed: ~25s (under 30-min budget).

Command: `python run_experiments.py --parts A 2>&1 | tee logs/v1_part_A.log` from `mechanical-jepa/experiments/v11/`.

Numbers (Part A.1 inventory — matches expected C-MAPSS shapes):
- FD001: 100/100 train/test, cycles [128,362] mean 206.3, 1 op condition
- FD002: 260/259 train/test, cycles [128,378] mean 206.8, 6 op conditions
- FD003: 100/100 train/test, cycles [145,525] mean 247.2, 1 op condition
- FD004: 249/248 train/test, cycles [128,543] mean 246.0, 6 op conditions
- FD001 sanity checks PASSED
- Selected sensor mean |rho| with cycle position = 0.523 (vs 0.016 for dropped sensors) — sensor selection is sane
- Top-5 sensors: s11, s4, s12, s7, s15

Artifacts produced (in `mechanical-jepa/experiments/v11/data_analysis/`):
- CMAPSS_ANALYSIS.md, inventory.md, sensor_informativeness.md
- episode_length_distributions.png, degradation_trajectories.png, rul_distribution.png
- sensor_informativeness_fd001.png, cross_subset_comparison.png
- operating_conditions_fd002.png, operating_conditions_fd004.png
- per_condition_sensor_stats.png, per_condition_sensor_stats_fd002.png, per_condition_sensor_stats_fd004.png

**Behavioural quirk discovered in the team's script** — `init_log()` in `run_experiments.py:72` opens `EXPERIMENT_LOG.md` in `'w'` mode, which truncates the team's entire run history every time `run_experiments.py` is invoked. It wiped the 1034-line v11 log down to 70 lines. I restored it from `git checkout HEAD --`. My strategy going forward: keep logging my work in `alex-contribution/experiments/v1/EXPERIMENT_LOG.md`, and after each `run_experiments.py` call restore the team's log file from git so their history isn't destroyed.

**Red flag about the end-of-run summary** — when Part A finishes, the script's `main()` falls through to the "V11 SESSION COMPLETE" summary and prints final results (e.g. "Traj JEPA E2E @ 100%: 14.79 +/- 0.92"). Those numbers are NOT from my run; they are loaded from the pre-existing `finetune_results.json` in the v11 directory, which dates from the team's 2026-04-10 session. They are stale artifacts, not a reproduction yet. I am explicitly flagging this so I do not accidentally report them as mine in Phase 4.

No errors. Proceeding to Phase 2.

## Phase 2 — Pretraining (Part D)

Start: 2026-04-19 19:15 UTC. End: 2026-04-19 19:23 UTC. Wall: 7.22 min (under 3-4h budget).

Config (V1, matching team): d_model=128, n_layers=2, n_heads=4, d_ff=256, patch_length=1, ema=0.996, params=366,336 — identical to the `Total trainable parameters: 366,336` line in the team's original v11 log.

Command: `python alex-contribution/experiments/v1/run_phase2_pretrain.py` (thin W&B wrapper around `run_experiments.py --parts D`).

W&B run: https://wandb.ai/g-a-lombardi01-forgis/industrialjepa/runs/giasq4s1

Pretraining curve (epochs at probe checkpoints):

| Ep | loss | pred | var | probe RMSE |
|---:|---:|---:|---:|---:|
| 1 | 0.0655 | 0.0572 | 0.8314 | 27.00 |
| 10 | 0.0262 | 0.0215 | 0.4711 | **15.91** (best) |
| 50 | 0.0267 | 0.0234 | 0.3354 | 20.23 |
| 100 | 0.0237 | 0.0204 | 0.3239 | 22.38 |
| 150 | 0.0210 | 0.0177 | 0.3216 | 26.76 |
| 200 | 0.0198 | 0.0166 | 0.3230 | 22.30 |

Key numbers:
- Final loss 0.0198 / initial 0.0655 = ratio 0.302 — 70% reduction (**PASS >50%**)
- Best probe RMSE: **15.91 at ep 10** vs team V1 @ ep 10 = 15.65 — delta +0.26, within seed noise
- Shuffle RMSE: 26.29 vs probe 15.91 — shuffle is worse, so the encoder has a temporal signal (**PASS**)
- Early-converging probe (same pattern as team): best probe is at ep 10 then worsens. Pretraining objective decouples from downstream RUL. This is not a bug — it is the v11 phenomenon already logged in the team's RESULTS.md.

**⚠️ PC1 rho = NaN** — same diagnostic bug the team hit in their first V1 pass. `compute_pretraining_diagnostics` in `train_utils.py` uses `use_last_only=True`, which collapses every embedding's RUL label to 1.0 (the normalized "full life"), making Spearman rho undefined. The team fixed this in a follow-up (`run_diagnostics_fixed.py`) by computing embeddings at multiple cut points. My Phase 2 run reproduces the bug faithfully; I am NOT treating PC1 rho = NaN as a training failure. Will compute corrected PC1 rho in Phase 3 alongside the prediction-trajectory audit.

Sanity checks (ml-researcher 5-min):
- Baseline: trivial ridge regressor (58 hand-built features, my `trivial_feature_regressor.py`) hits **20.14 RMSE** on FD001 canonical last-window. My V1 pretrain probe best = 15.91 beats it by 4.2 RMSE. JEPA contributes signal the ridge can't see. ✓
- Direction: loss monotone-ish decreasing, probe best at early epoch. Matches v11. ✓
- Magnitude: within 0.3 RMSE of team's V1. ✓
- Leakage: using team's `load_cmapss_subset('FD001')` unchanged, normalizer fit on train only, seed=42. ✓
- Implementation: checkpoint written at ep 10 as expected, wall time reasonable. ✓

Proceeding to Phase 3 (fine-tune Parts E+F).

## Phase 3 — Fine-tune (direct `part_e` call)

Start: 2026-04-19 19:26 UTC. End: 2026-04-19 19:32 UTC. Wall: 6.66 min (well under 1-hour budget).

Command: `python -u alex-contribution/experiments/v1/run_phase3_finetune.py`. Calls `run_experiments.part_e(data, model=None, n_seeds=5)` directly and then renders a label-efficiency plot. I bypassed `run_experiments.main()` because the `else`-branch of Part D (when D is not in parts) has a latent bug: it loads the checkpoint into a fresh `TrajectoryJEPA` model but never calls `.to(DEVICE)`, so the subsequent `compute_pretraining_diagnostics` call crashes with `RuntimeError: Expected all tensors to be on the same device`. `part_e` itself is fine — `finetune()` in `train_utils.py:262` does `model = model.to(DEVICE)` explicitly, so calling `part_e` directly is a clean, minimal workaround.

W&B run: https://wandb.ai/g-a-lombardi01-forgis/industrialjepa/runs/ykab7q5r

### Per-seed fine-tune numbers (my v1 reproduction on FD001)

| Budget | Supervised LSTM | JEPA frozen | JEPA E2E |
|---:|:---:|:---:|:---:|
| 100% | 17.36 ± 1.24 | 20.81 ± 0.23 | 15.36 ± 0.78 |
| 50% | 18.30 ± 0.75 | 20.67 ± 0.20 | 16.50 ± 1.23 |
| 20% | 18.55 ± 0.81 | 20.79 ± 0.26 | 16.59 ± 1.10 |
| 10% | 31.22 ± 10.93 | 22.67 ± 1.38 | 25.89 ± 3.58 |
| 5% | 33.08 ± 9.64 | 22.00 ± 1.43 | 23.95 ± 1.87 |

### Comparison to team's V1 (from their EXPERIMENT_LOG.md)

| Budget | Method | Mine | Team V1 | Δ |
|---:|:---|:---:|:---:|:---:|
| 100% | LSTM | 17.36 ± 1.24 | 17.36 ± 1.24 | 0.00 (bit-exact) |
| 50% | LSTM | 18.30 ± 0.75 | 18.30 ± 0.75 | 0.00 (bit-exact) |
| 20% | LSTM | 18.55 ± 0.81 | 18.55 ± 0.81 | 0.00 (bit-exact) |
| 10% | LSTM | 31.22 ± 10.93 | 31.22 ± 10.93 | 0.00 (bit-exact) |
| 5% | LSTM | 33.08 ± 9.64 | 33.08 ± 9.64 | 0.00 (bit-exact) |
| 100% | frozen | 20.81 ± 0.23 | 21.33 ± 0.32 | **-0.52** |
| 50% | frozen | 20.67 ± 0.20 | 21.01 ± 0.11 | **-0.34** |
| 20% | frozen | 20.79 ± 0.26 | 21.32 ± 0.37 | **-0.53** |
| 10% | frozen | 22.67 ± 1.38 | 22.92 ± 1.09 | -0.25 |
| 5% | frozen | 22.00 ± 1.43 | 22.12 ± 1.00 | -0.12 |
| 100% | E2E | 15.36 ± 0.78 | 14.79 ± 0.92 | +0.57 |
| 50% | E2E | 16.50 ± 1.23 | 17.51 ± 1.13 | **-1.01** |
| 20% | E2E | 16.59 ± 1.10 | 16.91 ± 0.87 | -0.32 |
| 10% | E2E | 25.89 ± 3.58 | 24.62 ± 3.22 | +1.27 |
| 5% | E2E | 23.95 ± 1.87 | 22.12 ± 1.32 | +1.83 |

Observations:
- **Supervised LSTM reproduces bit-exactly at all budgets** — the LSTM trains from scratch per seed, no JEPA checkpoint dependency; identical PyTorch/CUDA seeds give the same numbers to 2 decimal places.
- **JEPA frozen and E2E diverge slightly from the team** — 0.1 to 1.8 RMSE. This is expected. My Phase 2 produced a NEW V1 checkpoint, not their checkpoint; there is floating-point non-determinism in CUDA matmuls even with fixed seeds, so the two pretrained models are similar but not identical. The per-seed deltas are within seed-to-seed variance (team's std = 0.32 to 5.13 RMSE depending on budget).
- **Main qualitative claims reproduce**: JEPA E2E beats LSTM at all five budgets (Δ = 2.0, 1.8, 2.0, 5.3, 9.1 RMSE); frozen JEPA variance stays tight (≤1.5) while LSTM variance explodes at low budgets (9-11); the 10%-budget LSTM collapses on seeds 2 and 3 (45.01, 44.07) exactly as the team reported.

### Sanity check against primary v11 claim

The paper-ready headline is V2 E2E @ 100% = 13.80 RMSE (team). My V1 E2E @ 100% = 15.36 is 1.56 RMSE above that, because V1 (d_model=128, 366K params) is a weaker model than V2 (d_model=256, 1.26M params). The V2 pretrain is in `run_v2_pretrain.py`, a separate 22-minute job I am NOT running tonight — the overnight prompt scopes me to v11's main `run_experiments.py` flow (V1).

### Rule 3 cross-check against the trivial feature regressor

Ridge on 58 hand-built last-window features (FD001): 20.14 RMSE.
- My V1 E2E @ 100% = 15.36 — beats the ridge floor by **4.78 RMSE**. E2E clearly contributes signal the ridge cannot see.
- My V1 frozen @ 100% = 20.81 — only **0.67 RMSE above** the trivial feature regressor. Within 1 std of the lower bound. Per Rule 3, "your representation contributes nothing the protocol can see" — the frozen V1 linear probe is at the feature-regressor floor. This is a Rule 3 red flag and will be called out in RESULTS.md.

Main v1 artifacts saved:
- `alex-contribution/experiments/v1/finetune_results_v1.json` (5×3×5 RMSEs + per-seed lists)
- `alex-contribution/experiments/v1/plots/v1_label_efficiency.png` (my reproduction + ridge baseline line)

Launching Rule 1 internal-consistency audit now (prediction trajectories + within-sequence Spearman ρ on my V1 E2E seed=42 model).

## Rule 1 internal-consistency audit — the v11 "flat ~92 cycles" mystery

ml-researcher.md §"Why this section exists (cautionary case)" describes v11's headline RMSE (13.80) sitting next to a prediction-trajectory figure that showed the model emitting a near-constant ~92 cycles across every test engine and every cycle position, and says "the internal inconsistency between them was the most important finding of v11 and it was not logged." My Phase 3 E2E @ 100% (15.36 RMSE) only works because part_e's canonical protocol uses the same `_eval_test_rmse` that the team used — so if there is a flatness inconsistency, it lives here too. I tried to reproduce it.

### Audit run 1 — replicating the team's `run_prediction_trajectories.py`

Script: `alex-contribution/experiments/v1/run_prediction_trajectories_v1.py` (near-1:1 copy of team's `run_prediction_trajectories.py` except ckpt=V1 not V2, my seed=42, adapted to return per-engine summary stats and a proper scatter + trajectory plot).

Result:
- Canonical last-window test RMSE **52.57** (not 15.36!)
- Within-sequence Spearman ρ mean = -0.035, median = -0.107 (no correlation)
- Per-engine prediction std = **1e-6 mean, 0.0 median** — i.e. model emits an identical value at every cut point
- Cross-engine std of the final prediction = 5.99 (different engines get different constants)

This is the exact flatness signature the team saw. I had reproduced the v11 near-miss artifact.

### Audit run 2 — the `finetune()` protocol (matching part_e)

To check whether the flatness was in the trained model or in the *recipe* used for the trajectory plot, I wrote `run_prediction_trajectories_proper.py`: an inline clone of `train_utils.finetune(mode='e2e', seed=0)` (bit-exact loop, same optimizer/seeds/data/early-stop) that keeps the trained model+probe around, then computes per-engine trajectories with the audit's own inference loop.

Result:
- Canonical via `_eval_test_rmse`: 14.48 (sanity: part_e seed 0 = 15.62, close)
- Recomputed via per-engine inference: **49.65** — still flat!
- Within-sequence Spearman ρ mean = -0.064, median = 0.020 — still flat
- Per-engine prediction std = 0.0 median — still flat

So `_eval_test_rmse` sees "14.48" but my per-engine inference sees "49.65 and flat" on the **same trained model**. They disagree by 35 RMSE. That *is* an internal inconsistency — but now between my two inference pipelines, not between the model and reality.

### Root cause — a one-character bug in the plotting script

The `encode_past` method of `ContextEncoder` (models.py:161) documents `key_padding_mask: (B, T) bool, True = padding position`. Team's `run_prediction_trajectories.py:133` builds the mask for per-engine inference as:

```python
mask = torch.ones(1, t, dtype=torch.bool).to(DEVICE)
```

That is "every input is padding". The transformer's attention is then masked out on every key, so the encoder's output is independent of the input — it falls through to a constant driven only by position embeddings and bias. Every prediction becomes the same scalar. The canonical `_eval_test_rmse` does NOT have this bug because it goes through `collate_test` (`past_mask[i, T:] = True` only for actually-padded positions).

I repeated Audit run 2 after flipping the mask to `torch.zeros(1, T, dtype=torch.bool)` (no padding anywhere — correct for a single un-padded sequence). Results:
- Canonical via `_eval_test_rmse`: 14.48 (unchanged)
- Recomputed via per-engine inference: **13.54**
- Within-sequence Spearman ρ mean = **0.761**, median = **0.852** — strong, correct monotone signal
- Per-engine prediction std = 15.62 mean, 12.28 median — clearly non-flat
- Cross-engine std of final prediction = 42.09 — wide spread, matching RUL distribution

### Interpretation

Two separate facts:

1. **The trained V1 E2E encoder is not collapsed.** With a correct per-sequence mask it produces monotone, differentiated, per-engine RUL trajectories (within-sequence ρ ≈ 0.85).
2. **`run_prediction_trajectories.py` (team's script) has a mask bug** that makes every trajectory plot look flat regardless of what the encoder learned. This is the mechanism that produced the v11 "flat ~92 cycles" artifact flagged in ml-researcher.md. Fix: change `torch.ones(1, t, dtype=torch.bool)` → `torch.zeros(1, t, dtype=torch.bool)` on line 133 (and similarly anywhere else the unpadded single-sequence mask is constructed).

The inconsistency between the 13.80 RMSE and the flat trajectory plot is **real**, but the correct reconciliation is "the RMSE is trustworthy; the plot was generated by a buggy visualization script". Not the other way around.

I am logging this as:

**⚠️ INTERNAL INCONSISTENCY (resolved) — v11's prediction_trajectories.png is the output of a one-character mask bug in run_prediction_trajectories.py:133. The trained model itself is not collapsed. A patched visualization (my run_prediction_trajectories_proper.py) shows per-engine trajectories that are monotone in true RUL (Spearman ρ = 0.76). The v11 headline 13.80 RMSE is consistent with a functioning encoder once the visualization is fixed. Recommended follow-up: patch the script and regenerate analysis/plots/v11/prediction_trajectories.png.**

Artifacts:
- `prediction_trajectories_v1.json` — audit run 1 (team's protocol, flat)
- `prediction_trajectories_v1_proper.json` — audit run 2 (fixed mask, non-flat, ρ=0.761)
- `plots/v1_prediction_trajectories.png`, `plots/v1_prediction_scatter.png` — buggy-mask version
- `plots/v1_prediction_trajectories_proper.png`, `plots/v1_prediction_scatter_proper.png` — fixed-mask version

Proceeding to Phase 4 (write RESULTS.md).
