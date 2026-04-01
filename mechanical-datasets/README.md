# Mechanical Component Vibration Dataset Collection

Unified, multi-source mechanical vibration dataset for training Mechanical-JEPA.

**HuggingFace Dataset**: https://huggingface.co/datasets/Forgis/Mechanical-Components

## Two-Level Schema

We use a **source_metadata** config (one row per source) linked to **per-sample** configs via `source_id` foreign key. This avoids duplicating constant metadata (sampling rate, component info) across thousands of samples.

```python
from datasets import load_dataset

bearings = load_dataset("Forgis/Mechanical-Components", "bearings", split="train")
gearboxes = load_dataset("Forgis/Mechanical-Components", "gearboxes", split="train")
sources = load_dataset("Forgis/Mechanical-Components", "source_metadata", split="train")

# Join
sources_dict = {s["source_id"]: s for s in sources}
sample = bearings[0]
source = sources_dict[sample["source_id"]]
print(f"Sampling rate: {source['sampling_rate_hz']} Hz")
```

## Current Contents

### source_metadata (10 entries)
One row per source dataset with constant properties.

### bearings (~6,900 samples from 7 sources)

| Source | Samples | Fault Types | Episodes | Transitions | Modalities |
|--------|---------|-------------|----------|-------------|------------|
| CWRU | 40 | healthy, inner_race, outer_race, ball | No | No | vibration |
| MFPT | 20 | healthy, inner_race, outer_race | No | No | vibration |
| FEMTO | 3,569 | healthy, degrading | Yes (RUL) | No | vibration, temperature |
| Mendeley | 280 | healthy, inner_race, outer_race, ball | Yes | **Yes (279)** | vibration |
| XJTU-SY | 1,370 | healthy, degrading | Yes (RUL) | No | vibration |
| IMS | 1,256 | healthy, degrading | Yes (RUL) | No | vibration |
| Paderborn | 384 | healthy, inner_race, outer_race, compound | No | No | vibration, current |

### gearboxes (1,085 samples from 3 sources)

| Source | Samples | Fault Types | Episodes | Transitions | Modalities |
|--------|---------|-------------|----------|-------------|------------|
| OEDI | 20 | healthy, gear_crack | No | No | vibration |
| PHM 2009 | 109 | unknown (challenge data) | No | No | vibration, tachometer |
| MCC5-THU | 956 | 8 types incl. compound | Yes | **Yes (133)** | vibration |

### Key Features
- **412 transition samples** for action-conditioning (speed/load changes)
- **Episode/RUL fields** for prognostics (FEMTO, XJTU-SY, IMS)
- **Multi-modal**: vibration + current (Paderborn), vibration + tachometer (PHM 2009)
- **4 operating condition ranges**: 680-2460 RPM, 0-6000 lbs load

## Per-Sample Schema

```python
{
    "source_id": "cwru",            # FK to source_metadata
    "sample_id": "cwru_105",
    "signal": [[0.1, 0.2, ...]],    # (n_channels, signal_length)
    "n_channels": 2,
    "channel_names": ["DE_accel", "FE_accel"],
    "channel_modalities": ["vibration", "vibration"],
    "health_state": "faulty",       # healthy | faulty | degrading
    "fault_type": "inner_race",
    "fault_severity": None,         # 0-1 continuous for prognostics
    "rpm": 1750,
    "load": 2.0,
    "load_unit": "hp",
    "episode_id": None,             # For run-to-failure datasets
    "episode_position": None,       # 0.0 (start) to 1.0 (failure)
    "rul_percent": None,            # Remaining useful life
    "is_transition": False,         # Speed/load change in this window
    "transition_type": None,        # ramp_speed | ramp_load
}
```

## Directory Structure

```
mechanical-datasets/
├── datasets_inventory.md        # 10 datasets with download links
├── dataset_temporal_analysis.md # Episode/transition analysis
├── OVERNIGHT_TASK.md            # Full schema specification
├── progress.log                 # Overnight curation session log
├── scripts/
│   ├── download_cwru.py         # CWRU download script
│   └── verify_setup.py          # Environment verification
└── validation_reports/          # Dataset validation plots
```

## Cross-Component Transfer Potential

All components share fundamental vibrational physics:
- **Impact-resonance-decay transients** (universal across localized faults)
- **Amplitude modulation patterns** (bearings: load zone, gears: mesh engagement)
- **Universal degradation trajectory** (rising RMS, peaking kurtosis, rising entropy)

Gearbox AM/FM patterns are especially valuable for bearing JEPA pretraining.

## Citations

- CWRU: Case Western Reserve University Bearing Data Center
- MFPT: Society for Machinery Failure Prevention Technology
- FEMTO: Nectoux et al., IEEE PHM 2012
- Mendeley: Bearing Fault Dataset Under Varying Speed Conditions
- XJTU-SY: Wang et al., IEEE Trans. Instrumentation and Measurement, 2020
- IMS: Lee et al., IMS/University of Cincinnati, NASA Prognostics Data Repository
- Paderborn: Lessmeier et al., KAt-DataCenter, Paderborn University (CC BY-NC 4.0)
- OEDI: U.S. Department of Energy, Open Energy Data Initiative
- PHM 2009: PHM Society 2009 Data Challenge
- MCC5-THU: Liu et al., Tsinghua University / MCC5 Group Shanghai
