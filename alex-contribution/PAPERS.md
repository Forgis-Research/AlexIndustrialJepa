# Papers

All read. Ranked by how much we take from each.

## Primary inspirations
| Paper | Take from it | Notes |
|---|---|---|
| **LeJEPA** | Mathematical rigour. SIGReg / isotropic Gaussian / collapse prevention. | Images, not TS, but the framework is the gold standard. Our theoretical spine. |
| **MTS-JEPA** | Main target to beat. Task definition, dataset splits, eval baselines. | Published ~2 months ago. |
| **Brain-JEPA** | JEPA-for-TS mechanics: spatiotemporal masking, positional encoding for heterogeneous channels. | Closest architectural precedent. |
| **iTransformer** | Channel-as-token → geometric hook (permutation invariance across channels). | Our GDL angle anchors here. |

## Secondary / reference
| Paper | Role |
|---|---|
| PatchTST | Patching baseline |
| TimesFM | Decoder-only foundation model reference |
| Chronos-2 | Universal forecasting reference |
| TabPFN v2 | Reference only |
| OracleAD | TSAD-specific comparator |

## Open per-paper questions
- **MTS-JEPA**: task type (AD vs forecasting), exact datasets, eval protocol
- **Brain-JEPA**: what does the spatiotemporal masking buy, transferable to industrial TS?
- **iTransformer**: is channel-permutation invariance actually enforced, or just emergent?
- **LeJEPA**: does SIGReg still hold under non-i.i.d. temporal samples?

Notes live in `notes/<paper>.md` as we deep-dive.
