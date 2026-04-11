# IndustrialJEPA

Self-supervised representation learning for industrial remaining-useful-life (RUL) prediction. The project trains a Joint Embedding Predictive Architecture (JEPA) on sensor trajectories from run-to-failure experiments, then evaluates the learned representation on canonical RUL benchmarks — currently rolling bearings and NASA C-MAPSS turbofan engines. Target venue: NeurIPS 2026.

## Repository layout

```
IndustrialJEPA/
├── mechanical-jepa/        Primary research: Trajectory JEPA for industrial RUL
│   ├── experiments/        v8, v9, v10, v11 (current), v12 (verification gate)
│   ├── src/, pretraining/, downstream/, data/, analysis/, figures/
│   ├── notebooks/          Quarto walkthroughs (08-11), published to gh-pages
│   └── archive/            Pre-v8 scripts and old notebooks
├── paper-replications/     Baselines and reference-paper replications
│   ├── star/               STAR (Fan et al. 2024) - supervised RUL baseline
│   ├── dcssl/              DCSSL (Shen et al. 2026) - SSL RUL baseline
│   ├── cnn-gru-mha/        Yu et al. 2024 - transfer-learning baseline
│   └── when-will-it-fail/  Park et al. 2025 (ICML) - A2P replication
├── paper-neurips/          NeurIPS 2026 submission draft (paper.tex + figures)
├── mechanical-datasets/    Unified dataset curation for bearing data
├── .claude/                Agent definitions and project-scoped memory
└── archive/                Pre-pivot research directions (kept for reference)
```

Each subdirectory under `mechanical-jepa/experiments/`, `paper-replications/`, and `.claude/agent-memory/ml-researcher/` is self-contained and has its own `EXPERIMENT_LOG.md` + `RESULTS.md`. The root README only orients; the real work lives in those directories.

## Current status

- **Primary result**: `mechanical-jepa/experiments/v11/` — Trajectory JEPA on C-MAPSS FD001, V2 E2E at 13.80 RMSE on the canonical last-window protocol (matches the public SSL reference AE-LSTM at 13.99; gap to supervised STAR at 10.61).
- **Under verification**: `mechanical-jepa/experiments/v12/OVERNIGHT_PROMPT.md` — v12 is a verification gate on v11. The prediction-trajectory diagnostic and a hand-engineered feature-regressor lower bound must both confirm that v11 is tracking degradation before any further experiments run. See the OVERNIGHT_PROMPT for the full rationale.
- **Baselines**: STAR (FD001 done, FD002 seeds complete, FD003/4 in progress), DCSSL (complete), CNN-GRU-MHA (complete), A2P/when-will-it-fail (probes 1–141, publication-ready figures in `paper-replications/when-will-it-fail/figures/`).

## Published walkthroughs

The four most recent Quarto walkthroughs are rendered and deployed to GitHub Pages:

- **v8**: https://forgis-research.github.io/IndustrialJEPA/08_rul_jepa.html
- **v9**: https://forgis-research.github.io/IndustrialJEPA/09_v9_data_first.html
- **v10**: https://forgis-research.github.io/IndustrialJEPA/10_v10_trajectory_jepa.html
- **v11**: https://forgis-research.github.io/IndustrialJEPA/11_v11_cmapss_trajectory_jepa.html

Sources live at `mechanical-jepa/notebooks/*.qmd`. Publish with `quarto publish gh-pages <file>.qmd` from the `mechanical-jepa/notebooks/` directory.

## Running experiments

Each live subproject manages its own environment. See:

- `mechanical-jepa/README.md` — JEPA pretraining + RUL downstream
- `mechanical-jepa/experiments/v11/run_experiments.py` — current v11 entry point
- `paper-replications/*/README.md` + `REPLICATION_SPEC.md` — individual replication setup

The root `pyproject.toml` contains only shared tooling config (ruff, black, isort). It does not define a package to install — each subproject has its own `requirements.txt`.

## Archive

`archive/` contains pre-pivot research from the earlier "industrial robotics" scope (AURSAD, voraus, world-model experiments, foundation-model probes, lit review, early paper draft). Everything under `archive/` is frozen and kept in git for history; no active code depends on it.

## License

MIT (see individual files for copyright headers).
