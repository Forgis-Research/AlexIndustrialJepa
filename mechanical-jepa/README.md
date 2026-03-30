# Mechanical-JEPA

Self-supervised learning for mechanical systems using Joint Embedding Predictive Architectures.

## Overview

This project explores whether JEPA-style pretraining can learn useful representations from mechanical sensor data. We focus on **bearing fault detection** - a domain where:

1. **Labels are physics-governed** - Fault type is physical damage, not policy-dependent
2. **Detection is non-trivial** - Requires learning spectral signatures, not simple thresholds
3. **Multi-modal sensors enable physics-aware masking** - 4 distinct physics groups

### Why Not Robots?

Rigid-body robot proprioception is too simple for JEPA:
- Dynamics are nearly linear (forward kinematics = matrix multiplication)
- Contact detection is trivial from force sensors, or policy-dependent (gripper state)
- Previous experiments showed linear regression beats JEPA by 64x

## Datasets: 4 Bearing Fault Sources

| Dataset | Channels | Fault Types | Sampling | Size | Download |
|---------|----------|-------------|----------|------|----------|
| **Paderborn** | 8 (multimodal) | Healthy, OR, IR, Combined | 64 kHz | 33 bearings | Auto (needs RAR tool) |
| **CWRU** | 2-3 | Healthy, OR, IR, Ball | 12 kHz | 40 files, 6M samples | Auto |
| **IMS** | 4-8 | Run-to-failure | 20 kHz | 3 test runs | Manual (Kaggle) |
| **XJTU-SY** | 2 | Progressive degradation | 25.6 kHz | 15 bearings | Manual (IEEE) |

### Physics Groups (Paderborn)

| Group | Channels | Modality |
|-------|----------|----------|
| 1 | a1, a2 | Radial vibration |
| 2 | a3, v1 | Axial vibration |
| 3 | temp1, torque | Thermal/mechanical |
| 4 | ia, ib | Motor current |

## Project Structure

```
mechanical-jepa/
├── README.md                           # This file
├── data/
│   └── bearings/                       # All 4 datasets
│       ├── README.md                   # Dataset documentation
│       ├── prepare_bearing_dataset.py  # Download & processing
│       └── raw/                        # Downloaded data (gitignored)
├── notebooks/
│   └── 01_bearing_faults_analysis.ipynb  # Data exploration
└── src/                                # Model implementations (TBD)
```

## Quick Start

### 1. Set up environment

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

### 2. Prepare datasets

```bash
cd mechanical-jepa/data/bearings

# Download samples from Paderborn + CWRU
python prepare_bearing_dataset.py --download --sample --dataset all

# Process into unified format
python prepare_bearing_dataset.py --process

# Verify
python prepare_bearing_dataset.py --verify
```

### 3. Explore data

```bash
jupyter notebook notebooks/01_bearing_faults_analysis.ipynb
```

## Comparison to Brain-JEPA

[Brain-JEPA](https://arxiv.org/abs/2409.19407) (NeurIPS 2024 Spotlight) applies JEPA to fMRI data for brain disorder diagnosis. Our bearing fault detection task shares key structural similarities.

### Scale

| Metric | Brain-JEPA (fMRI) | Bearing Faults |
|--------|-------------------|----------------|
| Pretraining units | 40,162 subjects | 33-100+ bearings |
| Spatial features | 450 ROIs | 2-8 channels |
| Timepoints/episode | 160 (~5 min) | 100K-500K (~seconds) |
| Total tokens | **~2.3B** | **~100M+** (all datasets) |

### Temporal Resolution

| | Brain-JEPA | Bearings |
|-|------------|----------|
| Sampling rate | **0.5 Hz** (TR=2s) | **12-64 kHz** |
| Dynamics | Slow hemodynamics | Fast mechanical vibration |
| Window duration | ~5 minutes | ~0.1-0.3 seconds |

fMRI is **~100,000× slower** than vibration data. Brain-JEPA compensates with more subjects; bearings compensate with temporal density.

### Tasks: Classification + Regression

| Task Type | Brain-JEPA | Bearings |
|-----------|------------|----------|
| **Classification** | Sex, Disease (NC/MCI, Amyloid) | Fault type (4-class) |
| **Regression** | Age, Personality traits | RUL, Severity |
| **Prognosis** | MCI → AD progression | Degradation trajectory |

### SOTA Benchmarks

**Brain-JEPA** (targets to match):

| Task | Accuracy/Performance |
|------|---------------------|
| Sex classification | 88% |
| Age prediction | MSE = 0.50 |
| NC/MCI diagnosis | 77% |
| Cross-ethnic generalization | 66% |

**CWRU Bearing** (⚠️ [data leakage concerns](https://www.sciencedirect.com/science/article/abs/pii/S0888327021010499)):

| Setting | Reported Accuracy |
|---------|-------------------|
| Same-domain (clean) | 99%+ |
| Noisy (-6dB SNR) | 92-96% |
| Cross-load | ~95% |

**Paderborn** (more realistic):

| Setting | Accuracy |
|---------|----------|
| Same-domain (artificial faults) | 97-99% |
| **Cross-domain (artificial → real)** | **72%** |

### Why This Comparison Matters

| Aspect | Brain-JEPA | Bearing Faults |
|--------|------------|----------------|
| Physics groups | Brain regions (450 ROIs) | Sensor modalities (4 groups) |
| Masking strategy | Spatiotemporal + gradient | Physics-aware (by modality) |
| Generalization test | Cross-ethnic cohorts | Artificial → real faults |
| Hard benchmark | 66% (Asian cohort) | 72% (cross-domain) |

Both domains benefit from **physics-aware masking** over random masking, and face similar **domain shift** challenges in realistic evaluation.

### Targets for Mechanical-JEPA

| Task | Target | Rationale |
|------|--------|-----------|
| 4-class fault (cross-load) | >90% | Match supervised baselines |
| Cross-domain (artificial→real) | **>75%** | Beat current 72% SOTA |
| Few-shot (10 labels) | >80% | Demonstrate transfer |
| Physics masking vs random | +5% | Validate approach |

## References

### JEPA Methods
- [Brain-JEPA](https://arxiv.org/abs/2409.19407) - Brain dynamics foundation model (NeurIPS 2024 Spotlight)
- [I-JEPA](https://arxiv.org/abs/2301.08243) - Image JEPA from Meta AI

### Bearing Datasets
- [Paderborn Bearing Dataset](https://groups.uni-paderborn.de/kat/BearingDataCenter/) - 8-channel multimodal
- [CWRU Bearing Data](https://engineering.case.edu/bearingdatacenter/) - Most cited benchmark
- [NASA IMS Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) - Run-to-failure

### Benchmark Analysis
- [CWRU Data Leakage Analysis](https://www.sciencedirect.com/science/article/abs/pii/S0888327021010499) - Realistic evaluation methodology
- [CWRU Multi-label Benchmarking](https://arxiv.org/html/2407.14625v1) - Deep learning comparison (2024)
- [Paderborn Cross-domain Evaluation](https://arxiv.org/html/2509.22267) - Artificial vs real faults
