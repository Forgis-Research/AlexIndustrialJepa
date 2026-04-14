# V14 Results

Session theme: paper polish, architectural experiments, honest reframing, theory.
Not an RMSE-chase session.

## Phase 1: Paper H.I. reframe (DONE)

- H.I. R² = 0.926 demoted from headline to representation diagnostic.
- Added equation H.I.(t) = RUL(t) / R_max making the deterministic equivalence
  with capped RUL explicit.
- New paragraph explains why R² is numerically higher than frozen RMSE (same
  encoder, two protocols): (i) ~40% healthy-plateau cycles trivially predictable
  inflate R², (ii) Ridge on in-distribution val engines vs last-window on
  canonical test set.
- Abstract and contributions reordered: from-scratch ablation (+8.8/+15.6 RMSE)
  and 5% STAR crossover are now the headlines.
- Conclusion rewritten to match.

## Phase 2: Full-sequence prediction experiment (DONE - POSITIVE)

Hypothesis: target encoder sees x_{1:t+k} (full trajectory) instead of just
x_{t+1:t+k}. Context encoder and predictor unchanged.

Pretraining: best probe RMSE on val = **14.10** at ep 50 (150-epoch budget,
early-stopped at ep 100 with patience 10 probe checks).

**Fine-tuning results (3 seeds, 100% labels):**

| Mode   | V14 full-seq   | V2 baseline  | Delta   |
|:-------|:---------------|:-------------|:--------|
| Frozen | **15.70 ± 0.21** | 17.81 ± 1.7  | **-2.11** |
| E2E    | 14.32 ± 0.64   | 14.23 ± 0.39 | +0.09   |

Per-seed frozen: [15.66, 15.46, 15.97].
Per-seed E2E:    [14.70, 14.83, 13.42].

**Verdict**: POSITIVE. Full-sequence improves frozen by -2.1 RMSE with
**8x smaller seed std** (0.21 vs 1.7). E2E is essentially tied with V2.
Kill criterion (frozen > 18.5) NOT triggered. Change is kept as the default
V14 target-encoder configuration.

This closes the frozen-probe gap to STAR at 100% labels from 5.6 RMSE
(V2: 17.81 vs STAR 12.19) to 3.5 RMSE (15.70 vs 12.19) - a 37% reduction
in the frozen-probe gap - while keeping the 5% STAR crossover intact
(frozen 21.53 still beats STAR 24.55).

Output files:
- `phase2_full_sequence.py` (script, 370 lines)
- `best_pretrain_full_sequence.pt` (checkpoint)
- `full_sequence_prediction.json` (all seed-level numbers)
- `phase2_output.log`, `phase2_stdout.log`

Output files:
- `phase2_full_sequence.py` (script)
- `best_pretrain_full_sequence.pt` (checkpoint)
- `full_sequence_prediction.json` (seed-level numbers; pending full run)
- `phase2_output.log`, `phase2_stdout.log`

## Phase 3: Cross-sensor attention (iTransformer) (DEFERRED to V15)

Not started this session - prioritized theory, SSL audit, and paper polish
instead. Script sketch documented in V15 plan below.

## Phase 4: C-MAPSS data analysis plots (DONE)

Three publication-quality figures in `analysis/plots/v14/` (PNG) and
`notebooks/plots/` (PDF), also copied to `paper-neurips/figures/v14_*.pdf`:

- **fig1_cmapss_overview**: representative engine (len 199), extreme engine
  (len 362), capped-RUL distribution with plateau fraction (~40%).
- **fig2_method_schematic**: trajectory prediction task illustration; capped
  RUL label with healthy vs degradation phases.
- **fig3_label_efficiency_and_from_scratch**: label-efficiency curve annotated
  with the 5% STAR crossover; from-scratch ablation with pretraining-contribution
  shading (+8.8/+14.5/+15.6/+8.0 deltas).

## Phase 5: Paper review and update (DONE)

- Main results table: added STAR replication at 50/20/10/5 budgets, added
  From-scratch row, explained bold-per-budget excludes single-seed entries.
- Added `fig:v14_label_fromscratch` reference. Removed old
  `fig:label_efficiency` figure (subsumed).
- Section 5.3 (label efficiency) rewritten with the crossover narrative and
  an honest AE-LSTM comparability paragraph (single-seed, best-of-28-configs).
- Section 5.4 (from-scratch ablation) added.
- Section 6 (theoretical rationale) added with SFA bias, information-theoretic
  sketch, and the frozen-vs-E2E label-gradient-bias argument.
- Length-vs-content realized from plannedc to concrete paragraph.
- Removed stale \plannedc{} and \todo{} tags for realized experiments.

## Phase 5b: SSL comparison audit (DONE)

`experiments/v14/ssl_comparison_audit.md`. Key findings:

- No prior JEPA-style method on C-MAPSS RUL: Trajectory JEPA is the first.
- AE-LSTM 13.99 is best-of-28-configurations, not a multi-seed mean. We do
  not claim to beat it; within statistical noise given no reported variance.
- STAR (paper) 10.61 is fully supervised - belongs in Supervised row, not
  SSL comparison. Our 5-seed replication 12.19 ± 0.55 is 14.9% above their
  single-run number.
- MTS-JEPA, TS-JEPA, DCSSL report no C-MAPSS RUL numbers.
- Our competitive advantage: only method reporting multi-seed mean ± std.
- AE-LSTM head-to-head replication recommended for V15.

## Phase 5c: MTS-JEPA comparison (DONE)

`experiments/v14/mtsjepa_comparison.md`. 16-dimension architectural diff.

Immediate V14-feasible action surfaced: per-cycle prediction-error anomaly
score on existing V13 checkpoints (zero-shot anomaly detector for free).

V15 candidates: dual-resolution predictor, codebook regularization (deferred
pending batch-size sensitivity), cross-domain FD001+FD003 → FD004.

## Phase 6: Theory draft (DONE)

`experiments/v14/theory_draft.md` - 203-line theoretical sketch. Three arguments:

1. **Slow feature bias**: L1 prediction loss rewards low-innovation features
   (degradation) and penalizes high-innovation features (noise). Connection
   to Wiskott & Sejnowski 2002.

2. **Information-theoretic view**: under assumptions (A1) x = f(HI, noise),
   (A2) smooth HI dynamics, the MI between past and future is dominated by
   the slow HI component. JEPA L1 loss shares CPC's incentive structure.

3. **Frozen > E2E tracking**: ~40% of RUL labels sit at the cap. MSE gradient
   under this distribution biases E2E toward plateau calibration at the cost
   of rank preservation. Frozen is uncontaminated by this bias, which is why
   ρ_frozen (0.856) > ρ_E2E (0.804).

Compact version folded into paper Section 6.

## Phase 7: Quarto notebook (DONE)

`notebooks/14_v14_analysis.qmd` - self-contained, code-fold, theme cosmo.
Sections: TL;DR, dataset analysis, Phase 2 experiment, MTS-JEPA comparison,
SSL audit, theory, paper structure diff, V15 open directions.

## Commits and push cadence

~10 commits made, pushed after each (no more than 1 unpushed at a time).

## V15 open directions

1. **AE-LSTM head-to-head replication** on our pipeline. Closes the
   comparability question. Architecture ~100 lines PyTorch.
2. **Prediction-error anomaly score** on existing V13 checkpoints. Half-day,
   high narrative value (zero-shot anomaly detector).
3. **Cross-sensor attention (iTransformer)** - V14 Phase 3 deferred.
4. **Dual-resolution predictor** from MTS-JEPA (fine + coarse branches).
5. **Codebook regularization** with careful batch-size handling.
6. **Cross-domain pretraining** FD001+FD003 → FD004.
7. **Formalize the SFA connection** - verification on synthetic slow-vs-fast
   signals, promote the theoretical sketch to a formal proposition.

## Success criterion check

> "By morning: paper honestly reframed, two architectural experiments have
> run, C-MAPSS dataset illustrated with publication-quality plots, a
> theoretical argument for WHY trajectory prediction learns degradation
> exists in draft form, and everything is in a Quarto notebook."

- Paper reframed: YES
- Architectural experiments: Phase 2 (full-sequence) completed with positive
  initial result; Phase 3 (cross-sensor) deferred to V15.
- Publication-quality plots: 3 figures, PNG + PDF.
- Theory draft: 203 lines in experiments/v14/theory_draft.md, compact
  version in paper Section 6.
- Quarto notebook: notebooks/14_v14_analysis.qmd, self-contained.
- Committed and pushed: YES (~10 commits).
