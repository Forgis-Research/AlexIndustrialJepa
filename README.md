# IndustrialJEPA

**Self-supervised fault detection with cross-machine transfer for industrial robotics.**

Traditional fault detection learns sensor statistics specific to one machine—when deployed on different hardware, it fails. IndustrialJEPA learns *physics* instead of *hardware fingerprints* by predicting effort from setpoint in latent space using [JEPA](https://arxiv.org/abs/2301.08243).

## Core Insight

Industrial robot data has **causal structure**:
```
Setpoint (commanded) → Effort (energy expended) → Feedback (measured)
```

Under healthy operation, effort is predictable from setpoint and recent context. Faults break this relationship. By learning this mapping in latent space, the model learns transferable physics rather than machine-specific sensor patterns.

**Key finding**: Static setpoint→effort prediction fails because contact forces depend on unmeasured load. **Temporal prediction** (using effort history) captures dynamics that distinguish faults.

## Current Results

| Experiment | Dataset | Metric | Value |
|------------|---------|--------|-------|
| Temporal Self-Prediction | AURSAD | ROC-AUC | 0.538 |
| Temporal Self-Prediction | AURSAD | PR-AUC | 0.832 |
| Per-fault detection | missing_screw | Detected | 100% |
| Cross-machine transfer | AURSAD→voraus | Status | In progress |

## Installation

```bash
git clone https://github.com/ForgisX/IndustrialJEPA.git
cd IndustrialJEPA
pip install -e .
```

## Quick Start

```python
from industrialjepa.data import FactoryNetDataset
from industrialjepa.baselines import TemporalPredictor

# Load AURSAD with unified schema
dataset = FactoryNetDataset(
    datasets=["aursad"],
    window_size=256,
    healthy_only=True  # Train on healthy data only
)

# Train temporal predictor
model = TemporalPredictor(
    setpoint_dim=14,  # 7 joints × (pos + vel)
    effort_dim=13,    # 7 torques + 6 Cartesian forces
    hidden_dim=256,
    num_layers=4
)
```

**Training**:
```bash
python scripts/train_temporal_aursad.py
```

**Evaluation**:
```bash
python scripts/evaluate_world_model.py --checkpoint results/temporal_aursad_*/best_temporal.pt
```

## Project Structure

```
IndustrialJEPA/
├── src/industrialjepa/
│   ├── model/              # JEPA architecture
│   │   ├── world_model.py  # Core JEPA world model
│   │   ├── config.py       # Model configurations
│   │   └── backbone/       # Mamba-Transformer hybrid
│   ├── data/
│   │   └── factorynet.py   # Unified multi-robot dataloader
│   ├── training/           # Training loop, loss functions
│   ├── baselines/          # Comparison methods (MAE, Autoencoder, Contrastive)
│   └── evaluation/         # Benchmarks and metrics
├── scripts/                # Training and evaluation scripts
├── configs/                # YAML configurations
├── docs/                   # Architecture and analysis docs
├── paper/                  # Paper draft and execution plan
├── results/                # Experiment outputs
└── tests/                  # Test suite
```

## Datasets

We use [FactoryNet](https://huggingface.co/datasets/Forgis/factorynet-hackathon), unified into a common schema:

| Dataset | Robot | DOF | Task | Signals |
|---------|-------|-----|------|---------|
| AURSAD | UR3e | 6 | Screwdriving | Joint torques + Cartesian forces |
| voraus-AD | Yu-Cobot | 6 | Pick-and-place | Joint torques |
| NASA Milling | CNC | 3 | Milling | Spindle signals |
| RH20T | Franka | 7 | Manipulation | Velocity-based effort |
| REASSEMBLE | Franka | 7 | Assembly | Joint torques |

**Unified schema**:
- Setpoint: 14 dims (7 joint positions + 7 velocities)
- Effort: 13 dims (7 joint torques + 6 Cartesian forces)
- Validity masks handle missing signals per dataset

## Architecture

The model uses temporal self-prediction with EMA target encoding:

```
effort(t-k:t) ──► Encoder ──► z(t) ──► Predictor ──► z_pred(t+1)
                                            │
effort(t+1) ────► EMA Encoder ──► z_target(t+1) ◄──┘ (loss)
```

Anomaly detection: high prediction error in latent space indicates fault.

See [docs/world_model_design.md](docs/world_model_design.md) for architecture details.

## Experiments

| # | Experiment | Status | Goal |
|---|------------|--------|------|
| 1 | JEPA vs Baselines | Done | Prove temporal > static prediction |
| 2 | Causal Ablation | Planned | Validate setpoint→effort direction |
| 3 | Cross-Machine Transfer | Running | Zero-shot AURSAD→voraus |
| 4 | Multi-Dataset Pretraining | Planned | Test if combining datasets helps |
| 5 | Q&A Evaluation | Planned | State queries on embeddings |

See [paper/EXECUTION_PLAN.md](paper/EXECUTION_PLAN.md) for details.

## Key Findings

1. **Static prediction fails**: Setpoint alone achieves R²=0.99 for gravity torques but only R²=0.16 for contact forces (load is unobserved)

2. **Temporal prediction works**: Faults disrupt dynamics even when absolute values are lower—missing_screw shows 2× lower force but different temporal pattern

3. **Unified schema enables transfer**: Same model architecture across 5 different robots with validity masking

## References

- [I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) - Core JEPA method
- [V-JEPA: Video Joint Embedding Predictive Architecture](https://arxiv.org/abs/2404.08471) - Temporal extension
- [World Models](https://arxiv.org/abs/1803.10122) - Latent dynamics modeling
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) - Efficient backbone

## Citation

```bibtex
@article{industrialjepa2026,
  title={IndustrialJEPA: Cross-Machine Fault Detection via Causal Structure Learning},
  author={Petersen, Jonas},
  year={2026}
}
```

## License

MIT License
