# V16 Overnight Paper Improvement - Review History

Session start: 2026-04-16 evening.
Note: Sub-agent spawning (`neurips-reviewer`) was not available in this orchestrator harness;
reviews were executed by the orchestrator assuming four distinct reviewer personas, serialized
but with the emphasis splits mandated by SESSION_PROMPT_V16.md (A=rigor, B=story, C=figures,
D=related work).

## Iter 1 (2026-04-16 evening)

- Reviewer A (empirical rigor): score=5/10, key issues:
  - Abstract "+15.6 at 10%" disagrees with table arithmetic (should be +16.93 / +16.9).
  - Empty `---/---` SIGReg row in ablation table reads like a bug.
  - V2 frozen 17.81 protocol (test vs val) is not footnoted.
- Reviewer B (story & framing): score=6/10, key issues:
  - "Generality gap" claim overstated given SWaT/TTE are empty in benchmark.
  - Causal inductive bias finding (V2 vs V16b) is under-promoted - buried in ablations.
  - "Multi-domain" in contributions oversells two domains.
- Reviewer C (figures & presentation): score=5/10, key issues:
  - Only 4 figures (thin for NeurIPS); cross-machine transfer result has no visual.
  - Benchmark table caption is 11 lines of dense symbol-legend - should move to table notes.
  - Table captions miss protocol details (test vs val, seeds).
- Reviewer D (related work & positioning): score=6/10, key issues:
  - Missing foundation-model paragraph (Moment/Chronos/TimesFM).
  - Anomaly detection para covers only 2 methods; misses TranAD, OmniAnomaly, USAD.
  - LeJEPA/SIGReg not in related work despite being used in method.
- Mean: 5.5/10
- Figures: kept=4, reworked=0, added=1 (fig_cross_machine.pdf, cross-machine grouped bar chart
  from phase4_cross_machine_results.json; passes §10 self-check with 14/14 items).
- Writer actions:
  - Fixed abstract numbers: "+15.6 at 10%" -> "+16.9 at 10%"; rounded-to-spec consistency.
  - Rewrote abstract SMAP sentence to include non-PA F1 ~7% caveat.
  - Filled SIGReg ablation row with caveated numbers ("9.2 ± 1.5 (val, biased) / not run").
  - Un-bolded V2 frozen, bolded cross-sensor as actual best frozen; bolded V2 E2E (best E2E).
  - Added "causal inductive bias matters more than complexity" as contribution item.
  - Added cross-machine figure + discussion in §5.5 Bidirectional context paragraph.
  - Added "SIGReg on V2 (speculative)" blue paragraph in §5.5 ablations.
  - Reframed "generality gap" sentence in §4 to match actual 2-domain / 2-event evidence.
  - Added blue SWaT protocol forward-looking note to benchmark table + post-table prose.
  - Added 100-epoch SMAP note (with honest diagnostic that normal>anomaly score mean pre-crash).
  - Added "Foundation models for time series" related-work paragraph (Moment/Chronos/TimesFM/TimesNet).
  - Expanded anomaly detection related-work (TranAD, OmniAnomaly, USAD).
  - Added LeJEPA/SIGReg reference to SSL-for-TS paragraph.
  - Added FD002 val/test gap (15.35 -> 26.07) to limitation #4.
  - Split tab:benchmark caption + added MSL anomaly column (43.3 PA-F1) separately from SMAP.
  - Added 3 bib entries: wu2023timesnet, tuli2022tranad, su2019omnianomaly.
- Compile: PASS (15 pages, 470KB; bibtex resolved all new citations; minor underfull vbox
  warnings only, no undefined refs, no undefined citations).

## Iter 2 (2026-04-16 evening)

- Reviewer A (empirical rigor): score=6.5/10, key issues:
  - Cross-machine delta was "+4 to +11" but minimum delta is FD003 +6.3. Fix to "+6 to +11".
  - V2 std missing from bidi comparison ("V2's 14.23" should be "14.23 ± 0.4").
  - Tab:main_results caption doesn't say the scratch row has 5 seeds.
- Reviewer B (story): score=7/10, key issues:
  - Abstract too long (9 sentences, ~280 words); NeurIPS target 150-250.
  - Contribution list at 6 items is dilutive; merge to 4.
  - Causal-inductive-bias mentioned 5 times across paper; reduce to 2-3.
- Reviewer C (figures): score=7/10, key issues:
  - "Source: phase4_cross_machine_results.json" in fig caption is reviewer-facing; remove.
  - Redundant SWaT blue table-row + blue post-table paragraph; drop the paragraph.
  - Missing V16b vs V2 label-efficiency figure (medium priority).
- Reviewer D (related work): score=7.5/10, key issues:
  - Missing RmGPT / OpenMAE references (flagged in memory as key competitors).
  - Limitation #3 frames "gap to SOTA" negatively when scarce-label wins is the main claim.
- Mean: 7.0/10 (three reviewers at ≥7; reviewer A at 6.5 remains the bottleneck).
- Figures: kept=5, reworked=0, added=1 (fig_label_efficiency_v16b.pdf: V2 vs V16b line
  chart with error ribbons, 5 label budgets; passes §10 14/14).
- Writer actions:
  - Rewrote abstract: 1 paragraph, ~220 words, promoted causal-bias finding inline.
  - Collapsed contributions from 6 to 4 items; merged method+multi-domain and
    pretraining+crossover.
  - Fixed "+4 to +11" to "+6 to +11" in contribution 2 + §5.5 paragraph.
  - Added V2 std to bidi comparison (14.23 ± 0.4).
  - Added fig_label_efficiency_v16b figure in §5.5 with caption naming the divergence.
  - Removed "Source: ...json" from fig_cross_machine caption.
  - Removed redundant post-benchmark-table SWaT paragraph (kept table row + caption note).
  - Added RmGPT + OpenMAE to foundation-models paragraph.
  - Reframed limitation #3 as a positive: "Our advantage appears specifically in the
    scarce-label regime...The gap is the price of generality, not evidence against it."
  - Tightened conclusion: causal-bias finding now one sentence instead of one paragraph.
- Compile: PASS (15 pages, 474KB; all new citations resolved; no undefined refs).

