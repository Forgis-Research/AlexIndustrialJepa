# Competitors

Fill in as we read.

| Model | Task | Datasets | Metric | F1 | MAE-to-event | Notes |
|---|---|---|---|---|---|---|
| MTS-JEPA | ? verify | 4 | ? | | | Primary target. Published ~2 months ago. |
| Brain-JEPA | fMRI classification/prognosis | brain | acc/AUC | | | Spatiotemporal masking + gradient positioning |
| LeJEPA | image SSL | ImageNet | probing acc | n/a | n/a | We borrow SIGReg |
| iTransformer | forecasting | ETT, Weather | MSE/MAE | | | Channel-as-token → possible GDL hook |
| PatchTST | forecasting | ETT | MSE/MAE | | | Patching baseline |
| TimesFM | forecasting | many | MAE | | | Foundation model ref |
| Chronos-2 | forecasting | many | MAE | | | Foundation model ref |

## Dataset candidates
- TSB-AD (benchmark backbone)
- MSL, SMAP (NASA)
- SMD (server machine)
- SWaT, WADI (industrial control)
- C-MAPSS (from v11, home turf)
