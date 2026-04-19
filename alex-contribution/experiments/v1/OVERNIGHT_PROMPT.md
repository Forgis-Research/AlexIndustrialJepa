# Overnight Run v1 — Reproduce v11 Trajectory JEPA on C-MAPSS

**Goal**: Get the team's existing v11 experiment running end-to-end on my VM. This is not new research — this is me (Alex) learning the team's pipeline by watching it execute. Observer mode, not inventor mode.

**What I want in the morning**: a working run of (a subset of) `mechanical-jepa/experiments/v11/run_experiments.py`, with its EXPERIMENT_LOG.md updated, RESULTS.md committed, W&B curves visible, and an honest summary of what worked and what broke.

**Budget**: 8 hours hard cap. Prefer to finish in 4–6.

**Working directory**: `~/AlexIndustrialJepa/mechanical-jepa/experiments/v11/`

---

## Read first (in order, ~20 min — don't skip)

1. `alex-contribution/PLAN.md` — my project scope (NOT v11's scope; just so you know what I'm doing)
2. `mechanical-jepa/experiments/v11/EXPERIMENT_LOG.md` — what v11 did originally
3. `mechanical-jepa/experiments/v11/RESULTS.md` — the numbers they reported
4. `mechanical-jepa/experiments/v11/run_experiments.py` — structure and parts (A..G). Don't read the whole 65k lines; skim headers.
5. `mechanical-jepa/README.md` — project conventions
6. `.claude/agents/ml-researcher.md` §Critical Evaluation Lens — the rigor rules (MANDATORY)
7. `archive/autoresearch/mechanical_jepa/OVERNIGHT_PROMPT_V5.md` §Pre-Flight Checklist — what they check before launching

Log a line to `EXPERIMENT_LOG.md` (in my v1 dir) confirming each of the above is read.

---

## Phase 0 — Environment setup (budget: 45 min)

Known differences from the team's original VM:
- Repo is at `~/AlexIndustrialJepa/` (not `~/IndustrialJEPA/`)
- `/mnt/sagemaker-nvme/` may not exist on this image
- C-MAPSS data is not yet on disk
- Only `wandb` and `python-dotenv` installed so far; requirements.txt has more

Tasks:
1. **Fix the path issue**: v11's `run_experiments.py` has hardcoded absolute paths like `/home/sagemaker-user/IndustrialJEPA/...`. Simplest fix: create a symlink `ln -s ~/AlexIndustrialJepa ~/IndustrialJEPA` so the old paths resolve. Do this first. Verify with `ls ~/IndustrialJEPA/mechanical-jepa/experiments/v11/`.
2. **Install v11 requirements**: `cd ~/AlexIndustrialJepa/mechanical-jepa && pip install -r requirements.txt`. Capture any failures.
3. **Check `/mnt/sagemaker-nvme/`**: if it doesn't exist, find the actual local-scratch path on this image (`df -h` to find large local disks; likely `/opt/ml/` or similar). Document what you find.
4. **Download C-MAPSS**: check if `datasets/data/cmapss/` exists. If not, download it. Sources to try in order:
   - The repo's own downloader (`datasets/downloaders/` if present)
   - `bash mechanical-jepa/setup_vm.sh` (may handle it)
   - Search for "NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set" — official source
5. **GPU sanity**: `nvidia-smi` should show A10G. `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.
6. **W&B login**: should already be set from `.env`. Verify with a 1-line test.

Commit at the end of Phase 0:
```bash
git add alex-contribution/experiments/v1/
git commit -m "v1 phase 0: environment set up for v11 reproduction"
git push origin main
```

**Kill criteria for Phase 0**:
- C-MAPSS data can't be obtained → stop, write FAILURE.md explaining why, commit, exit. Do not fabricate data. Do not proceed.
- `pip install` fails on critical deps (torch, wandb) → stop, write FAILURE.md.

---

## Phase 1 — Dry-run Part A (budget: 30 min)

Goal: prove `run_experiments.py --parts A` runs to completion *something*. Part A is the lightest and fastest (data loading + analysis, no full pretraining). This tells us whether the environment is correctly wired.

1. `cd ~/AlexIndustrialJepa/mechanical-jepa/experiments/v11/`
2. `python run_experiments.py --parts A 2>&1 | tee v1_part_A.log`
3. Capture the run time and any warnings/errors.
4. Log outcome to `alex-contribution/experiments/v1/EXPERIMENT_LOG.md`:
   - What ran
   - Runtime
   - What files it produced (under `mechanical-jepa/experiments/v11/`)
   - Any errors (even non-fatal)

**Kill criteria**:
- Part A fails with an error → debug for up to 15 min (check paths, imports, missing data files). If still broken, stop Phase 1, record the error in detail, and proceed to Phase 4 (writing results) with a "v11 could not be reproduced" verdict.

---

## Phase 2 — Pretraining (budget: 3–4 hours)

If Phase 1 succeeded, run a real pretrain.

1. `python run_experiments.py --parts B 2>&1 | tee v1_part_B.log` (or whichever part triggers pretraining in v11 — check run_experiments.py structure)
2. W&B should show a live run with pretraining loss curves.
3. Check every 30 min (via stdout tail): is loss decreasing? Is the probe RMSE tracking as expected?
4. On completion: capture final pretrain loss, probe RMSE, checkpoint path.

**Kill criteria**:
- Loss diverges (NaN) → stop, log, move to Phase 4.
- Probe RMSE doesn't change over 100+ epochs → likely representation collapse. Stop, log, move to Phase 4.
- Disk fills → stop, clean, restart OR move on.
- W&B stops logging → diagnose, don't silently continue.

---

## Phase 3 — Fine-tune / Eval (budget: 1 hour)

If Phase 2 completed with a valid checkpoint, run fine-tune:

1. `python run_experiments.py --parts C` (or whichever parts do fine-tune; check script)
2. Capture RMSE across seeds + budgets.
3. Compare to `RESULTS.md`'s reported numbers (v11 E2E = 13.80 on FD001). My number doesn't have to match — but document the delta.

---

## Phase 4 — Write RESULTS.md (budget: 30 min)

File: `alex-contribution/experiments/v1/RESULTS.md`

Template:

```markdown
# v1 — Reproducing v11 Trajectory JEPA on C-MAPSS

**Date**: <ISO>
**Duration**: <total hours>
**Status**: <completed | partial | failed>

## Setup
- Code: team's v11 unchanged
- Environment differences from team VM: <list>
- Data: C-MAPSS FD001 <shape stats>

## Numbers (what I got)
| Metric | Team's number | Mine | Delta | Notes |
|---|---|---|---|---|
| Pretrain final loss | <from their RESULTS.md> | <mine> | | |
| Probe RMSE | | | | |
| E2E FD001 RMSE (best) | 13.80 | | | |

## W&B links
<paste URLs to my runs>

## What worked
<list>

## What broke / differences
<honest list — env issues, path fixes, data differences, etc.>

## Internal consistency check (MANDATORY)
Per ml-researcher.md Rule 1: list every artifact the run produced. Verify each one's story matches every other one.
- Prediction trajectory plot: <exists? what does it show?>
- Loss curve: <monotonic? plateau?>
- Probe RMSE curve: <tracks or not?>

## Verdict
One sentence: "I can / cannot reproduce v11 on this VM because <reason>."
```

---

## Phase 5 — Commit + push + stop (5 min)

```bash
git add alex-contribution/experiments/v1/
git add mechanical-jepa/experiments/v11/   # if any new artifacts generated
git commit -m "v1: reproduce v11 Trajectory JEPA — <1-line verdict>"
git push origin main
```

Final line in EXPERIMENT_LOG.md: `"v1 complete — <verdict>. Exiting."`

Exit the session. Do NOT start new experiments. Do NOT try to beat v11's numbers.

---

## Rigor rules (MANDATORY — from .claude/agents/ml-researcher.md)

Before logging any positive result, apply the **5-minute sanity check** (baseline / direction / magnitude / leakage / implementation). Every phase.

If you detect cross-artifact inconsistency (e.g. good RMSE + flat prediction trajectory → the famous v11 near-miss documented in ml-researcher.md), **do not silently drop the inconvenient artifact**. The inconsistency is the finding. Log as `⚠️ INTERNAL INCONSISTENCY`.

Run the **trivial feature regressor** sanity check if time allows (Rule 3). Not required if time is short, but preferred.

---

## Git + logging discipline

- Commit at the end of EACH phase (not just at end)
- Push every 1-2 commits
- Never amend, never force push
- Every phase logged in EXPERIMENT_LOG.md with start/end/outcome

---

## Stopping conditions (any one triggers immediate stop + log + push)

- Total elapsed > 8h
- Phase 0 fails (env/data)
- Training diverges (NaN/inf)
- Disk fills (>95%)
- Any error repeats 3 times without progress
- VM shows signs of imminent crash (OOM, thermal throttle, etc.)

On any stop: write what happened to RESULTS.md, commit, push. **Silent failures are unacceptable.** Alex wants to know what broke as much as what worked.
