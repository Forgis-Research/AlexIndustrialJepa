# Mechanical Component Vibration Dataset Collection

**Goal**: Build a unified, comprehensive dataset of mechanical vibration signals for training Mechanical-JEPA.

## Target Components
- **Bearings**: CWRU, IMS, Paderborn, XJTU-SY, MFPT, FEMTO
- **Gearboxes**: PHM 2009, MCC5-THU, OEDI, SEU
- **Motors/Pumps**: NLN-EMP, PU Motor Current, ESPset

## Directory Structure
```
mechanical-datasets/
├── raw/                    # Original downloads (gitignored)
├── processed/              # Unified format
│   ├── bearings/
│   ├── gearboxes/
│   └── motors/
├── scripts/                # Download & processing scripts
├── metadata/               # Dataset documentation
├── analysis/               # EDA notebooks
└── hf_upload/              # HuggingFace upload scripts
```

## Quick Start

```bash
# Download CWRU dataset
python scripts/download_cwru.py

# Process to unified format (after download)
python scripts/process_cwru.py
```

## Data Sources

| Dataset | Component | Faults | Size | Link |
|---------|-----------|--------|------|------|
| CWRU | Bearing | Artificial | ~500MB | [Link](https://engineering.case.edu/bearingdatacenter) |
| IMS | Bearing | Run-to-failure | ~6GB | [Link](https://data.nasa.gov/dataset/ims-bearings) |
| Paderborn | Bearing | Artificial+Natural | ~20GB | [Link](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/) |
| PHM 2009 | Gearbox | Mixed | ~1GB | [Link](https://phmsociety.org/public-data-sets/) |

## Unified Schema

See `OVERNIGHT_TASK.md` for detailed metadata schema.

## HuggingFace Dataset

Target: `forgis/mechanical-vibration-unified`

## License

Each source dataset has its own license - see individual metadata files.
