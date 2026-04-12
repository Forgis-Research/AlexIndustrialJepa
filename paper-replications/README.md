# Paper Replications

Self-contained replications of papers we care about for the IndustrialJEPA project.
Each subfolder targets a single paper and contains its spec, code, results, and notes.

## Replications

| Folder | Paper | Venue | Status |
|--------|-------|:-----:|:------:|
| [`star/`](star/) | Fan et al. "STAR: A Simple Transformer with Adaptive Residuals for Remaining Useful Life Prediction" | ICASSP 2024 | FD001 done, FD002 done, FD003/4 in progress |
| [`cnn-gru-mha/`](cnn-gru-mha/) | Yu et al. "Remaining Useful Life of the Rolling Bearings Prediction Method Based on Transfer Learning Integrated with CNN-GRU-MHA" | Appl. Sci. 2024 | done |
| [`dcssl/`](dcssl/) | Shen et al. "A novel dual-dimensional contrastive self-supervised learning-based framework for rolling bearing RUL prediction" | Sci. Rep. (Nature) 2026 | done |
| [`when-will-it-fail/`](when-will-it-fail/) | Park et al. "When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series" | ICML 2025 | probes 1-141 complete, paper figures rendered |
| [`mts-jepa/`](mts-jepa/) | He et al. "MTS-JEPA: Multi-Resolution Joint-Embedding Predictive Architecture for Time-Series Anomaly Prediction" | arXiv 2026 | spec written, overnight prompt ready |

`star/` is the primary supervised baseline for the `mechanical-jepa/` v11/v12 C-MAPSS work. The other three were scouted as reference methods during the bearing-RUL phase.

## Convention

Each replication folder contains:

- `REPLICATION_SPEC.md` - what we are trying to match and why
- `EXPERIMENT_LOG.md` - chronological log of runs, decisions, fixes
- `results/` - structured JSON + markdown tables
- `figures/` - rendered plots
- `notebooks/` - Quarto `.qmd` summary walkthrough
- the paper's PDF at the root
- a `SESSION_SUMMARY.md` at the end of each autonomous session
