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
