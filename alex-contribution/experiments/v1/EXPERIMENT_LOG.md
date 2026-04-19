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
