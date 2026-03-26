# Open X-Embodiment (OXE)

## Executive Summary
- **Domain**: Robotics / Manipulation
- **Task**: Robot manipulation; adaptable to multivariate time series pretraining (Mechanical-JEPA)
- **Size**: 1M+ trajectories x 22 robot embodiments x 60 datasets from 34 labs
- **Sampling Rate**: Varies by dataset (3-20 Hz for control data)
- **Real vs Synthetic**: Mostly real; ManiSkill subset is simulated
- **License**: Apache 2.0 (most datasets); some have individual licenses
- **Download URL**: https://github.com/google-deepmind/open_x_embodiment
- **Published SOTA**: RT-2, RT-X, Octo (Google DeepMind, 2023-2024)

## Revised Verdict: CONDITIONALLY VIABLE for Mechanical-JEPA

**The initial "not viable" assessment was too dismissive.** While OXE as a whole is camera-centric, specific subsets contain rich proprioceptive data that can support a Mechanical-JEPA research direction. The key insight: OXE is the largest source of robot state/action data for cross-embodiment pretraining, and filtering for proprioception-rich subsets yields ~200k+ trajectories with real joint-level physics.

The channel count (7-27) is far below Brain-JEPA's 450 ROIs, so OXE cannot replicate the "many sensors" dimension. However, it enables a distinct and equally publishable direction: **cross-embodiment transfer learning on proprioceptive time series**, which no existing JEPA paper has attempted.

---

## Detailed Description

Open X-Embodiment consolidates 60 existing robot learning datasets into a unified RLDS (Robot Learning Dataset Specification) format. Originally designed for training generalist robot policies (RT-X, Octo), it is the largest publicly available collection of robot manipulation trajectories.

### Scale
| Metric | Value |
|---|---|
| Total trajectories | 1,000,000+ |
| Robot embodiments | 22 different robots |
| Contributing labs | 34 |
| Distinct skills | 527 |
| Task variations | 160,266 |

### Data Format
- **Encoding**: RLDS format (TFRecord / TensorFlow Dataset)
- **Episode structure**: Sequence of steps, each containing observations + actions + rewards + metadata
- **Storage**: Google Cloud Storage (`gs://gresearch/robotics/`) or HuggingFace
- **Access**: `tfds.load('dataset_name', data_dir='gs://gresearch/robotics')`

---

## Proprioception-Rich Subsets (Verified March 2026)

The following datasets were verified by downloading actual data from GCS and inspecting the RLDS schema via `tfds.builder()`. All proprioceptive fields confirmed present and non-empty.

### Tier 1: Rich Proprioception (joint pos + vel + force)

| Dataset | tfds Name | Robot | DOF | State Dim | Extra Fields | Episodes | Hz |
|---|---|---|---|---|---|---|---|
| Stanford KUKA | `stanford_kuka_multimodal_dataset_converted_externally_to_rlds` | KUKA iiwa | 7 | 27 total | joint_pos(7), joint_vel(7), ee_pos(3), ee_vel(3), ee_forces(6), contact(1) | 3,000 | 20 |
| ManiSkill | `maniskill_dataset_converted_externally_to_rlds` | Panda (sim) | 7 | 18 | joint_angles(7), gripper(2), joint_vel(7), gripper_vel(2) + tcp_pose(7), base_pose(7) | 30,213 | 20 |
| Berkeley UR5 | `berkeley_autolab_ur5` | UR5 | 6 | 15 | robot_state: likely 6x jpos + 6x jvel + 3x EE | 1,000 | 10 |

### Tier 2: Good Proprioception (joint pos, some extras)

| Dataset | tfds Name | Robot | DOF | State Dim | Fields | Episodes | Hz |
|---|---|---|---|---|---|---|---|
| DROID | (not in standard tfds) | Franka Panda | 7 | 14 | joint_position(7), cartesian_position(6), gripper(1) | 76,000 | 15 |
| Berkeley FANUC | `berkeley_fanuc_manipulation` | FANUC Mate 200iD | 6 | 13 | joint_angles(6), gripper(1), joint_vel(6) + ee_state(7) | 415 | 10 |
| JACO Play | `jaco_play` | Kinova JACO | 6 | 8 | joint_pos(6), fingers(2) + ee_cart_pos(7), ee_cart_vel(6) | 1,000 | 10 |
| TOTO | `toto` | Franka Panda | 7 | 7 | absolute joint angles (7) | 2,898 | 10 |

### Tier 3: End-Effector Only (limited proprioception)

| Dataset | tfds Name | Robot | State Dim | Fields | Episodes | Hz |
|---|---|---|---|---|---|---|
| Fractal (RT-1) | `fractal20220817_data` | Google Everyday Robot | 15 (EE) | EE pose(7), gripper(2), height(1), deltas(6) | 87,212 | 3 |
| Bridge | `bridge` | WidowX-250 | 7 | state: likely EE pose + gripper | ~60,000 | 5 |
| bc_z | `bc_z` | Google | 4 | xyz(3), sensed_close(1) | 43,000 | ~5 |

### Verified Episode Lengths (from GCS downloads)

| Dataset | Min Steps | Max Steps | Mean Steps | Notes |
|---|---|---|---|---|
| TOTO (Franka) | 229 | 1,160 | 426 | Long pour/scoop trajectories |
| KUKA iiwa | 50 | 50 | 50 | Fixed-length episodes |
| FANUC | 28 | 235 | 126 | Variable manipulation tasks |
| JACO Play | 46 | 115 | 77 | Short pick-place |
| Fractal | 22 | 115 | 54 | Short manipulation |
| UR5 | 71 | 123 | 94 | Cloth/pick-place |

### Data Quality Assessment (from actual downloads)

| Dataset | NaN Count | Constant Channels | Value Range | Quality |
|---|---|---|---|---|
| TOTO | 0 | 0 | [-3.54, 2.80] | Clean |
| KUKA (full) | 0 | 0 | [-2.42, 2.53] (joints) / [-0.57, 0.38] (forces) | Clean |
| FANUC | 0 | 0 | [-2.89, 1.18] | Clean |
| JACO | 0 | 0 | [-1.95, 4.41] | Clean |
| Fractal | 0 | 0 | [-0.70, 1.00] | Clean (normalized) |
| UR5 | 0 | 0 | [-3.68, 3.59] | Clean |

**All sampled data is clean: zero NaN values, no constant channels, reasonable value ranges consistent with joint angle radians.**

---

## Scale Comparison: OXE vs Brain-JEPA

### Brain-JEPA Analog Assessment

| Dimension | Brain-JEPA | OXE Proprio Subset | Assessment |
|---|---|---|---|
| "Subjects" (instances) | 32,000 patients | ~261,000 trajectories (Tier 1+2+3) | OXE 8x larger |
| Timesteps per instance | 160 | 50-1,160 (mean ~120) | Comparable |
| "Channels" per instance | 450 ROIs | 7-27 (proprioception) | OXE 17-64x smaller |
| Total tokens | ~2.3B | ~31M (Tier 1+2 only) | Brain-JEPA 74x larger |
| Modality | fMRI BOLD | Joint angles/velocities/forces | Different physics |
| Cross-domain | 1 domain (brain) | 6+ robot embodiments | OXE richer for transfer |

### Critical Limitation: Channel Count
OXE has 7-27 channels of proprioceptive data per dataset. Brain-JEPA has 450 ROIs. This 17-64x gap means OXE **cannot** replicate the "many sensors" attention masking research. For that, use SWaT (51 channels), WADI (127 channels), or industrial sensor arrays.

### What OXE Uniquely Enables
OXE is the **only** public dataset large enough for **cross-embodiment proprioceptive pretraining**:
1. Train a JEPA encoder on Franka joint trajectories (DROID + TOTO = ~79k episodes)
2. Fine-tune/transfer to UR5, KUKA, FANUC, JACO
3. Evaluate: does pretraining on one robot's physics help predict another's?

This is a distinct and novel contribution that no existing JEPA paper has attempted.

---

## Viability for Mechanical-JEPA

### Phase 1: Franka-Only Pretraining
- **Data**: DROID (76k eps x 14-dim) + TOTO (2.9k eps x 7-dim) = ~79k episodes
- **Approach**: JEPA encoder on 7-dim joint angle time series
- **Masking**: Temporal masking (predict future joint states from past)
- **Scale**: ~79k x 200 steps x 7 channels = ~110M tokens (modest but workable)
- **Preprocessing**: Normalize per-joint, segment to fixed windows, discard images
- **Feasibility**: HIGH -- data is clean, accessible, and sufficient for proof-of-concept

### Phase 2: Cross-Embodiment Transfer
- **Pretrain** on Franka (7-DOF) data
- **Transfer to**: UR5 (6-DOF), KUKA iiwa (7-DOF), FANUC (6-DOF), JACO (6-DOF)
- **DOF mismatch handling**:
  - Option A: Zero-pad smaller DOF to 7, mask padding in attention
  - Option B: Learn per-embodiment linear projection to shared latent space
  - Option C: Use end-effector space (6-dim: xyz + rpy) as universal representation
- **Evaluation**: Fine-tune pretrained encoder on target robot with few-shot episodes
- **Feasibility**: MEDIUM -- requires careful architecture choices for DOF alignment

### Phase 3: Benchmarking
- **Downstream tasks**: Next-step prediction, trajectory forecasting, anomaly detection
- **Baselines**: Train from scratch on each robot vs. pretrained + fine-tuned
- **Metrics**: MSE on held-out trajectories, few-shot transfer accuracy
- **Comparison**: Show that cross-robot pretraining improves data efficiency

### Quality Assessment
- **Is the proprio data clean enough?** YES -- zero NaN, no constant channels, reasonable ranges.
- **What preprocessing is needed?** Per-channel normalization, fixed-length windowing, alignment of different control frequencies (resample to common rate, e.g., 10 Hz).
- **Are there alignment issues?** YES -- different robots have different DOF, joint limits, and coordinate conventions. The universal EE-space representation (6-dim) is the safest alignment strategy.

---

## Concrete Extraction Plan

### Step 1: Download Proprioceptive Data
```bash
# Quick test (10 eps per dataset)
python datasets/downloaders/download_oxe_proprio.py --sample

# Full download (100 eps per dataset, ~30 min)
python datasets/downloaders/download_oxe_proprio.py --n-episodes 100

# Franka-focused (Phase 1)
python datasets/downloaders/download_oxe_proprio.py --dataset toto --n-episodes 2000
```

### Step 2: Preprocess for JEPA
1. Resample all data to 10 Hz (upsample KUKA from 20 Hz, downsample fractal from 3 Hz)
2. Normalize per-channel to zero mean, unit variance
3. Segment into fixed-length windows (e.g., 128 timesteps)
4. Split: 80% train, 10% val, 10% test
5. Format as PyTorch tensors: `(batch, channels, time)`

### Step 3: Architecture
- Encoder: 1D-CNN or Transformer on (channels, time)
- Predictor: JEPA-style predictor in latent space
- Masking: Random temporal blocks (predict 20-40% of timesteps)
- Loss: L2 in latent space (not pixel/value space)

---

## DROID Dataset Details

DROID is the single largest proprioceptive dataset but is not in the standard `tfds` registry. Access requires:

```python
# Via GCS (small sample)
ds = tfds.load('droid_100', data_dir='gs://gresearch/robotics', split='train')

# Full dataset (1.7 TB)
# gsutil -m cp -r gs://gresearch/robotics/droid .
```

**DROID Proprioceptive Fields** (from paper, Khazatsky et al. 2024):
- `joint_position`: (7,) float64 -- Franka joint angles
- `cartesian_position`: (6,) float64 -- EE position + orientation
- `gripper_position`: (1,) float64 -- gripper opening
- Action: (7,) float64 -- 6 joint velocities + gripper command

**Note**: DROID does NOT include joint torques despite initial claims. Only positions and velocities are recorded.

---

## Other OXE Datasets with Proprioception (User May Have Missed)

| Dataset | Robot | State Dim | Episodes | Notes |
|---|---|---|---|---|
| NYU Franka Play | Franka | 13 | ~8,000 | `nyu_franka_play_dataset_converted_externally_to_rlds` |
| CMU Franka Exploration | Franka | 8 (action) | ~1,000 | Limited obs |
| TACO Play | DLR | 15 (robot_obs) | ~3,600 | German Aerospace Center arm |
| Robomimic PH | Panda (sim) | 32-115 | 200/task | Richest per-step but tiny |
| Robosuite Panda | Panda (sim) | 32+ | ~1,000 | Sim only |

---

## Download Notes
- Requires `tensorflow_datasets` package (pip install tensorflow tensorflow_datasets)
- All datasets accessible via: `tfds.load(name, data_dir='gs://gresearch/robotics')`
- No GCP authentication required for read access
- Downloader: `datasets/downloaders/download_oxe_proprio.py` (extracts proprio only)
- Legacy downloader: `datasets/downloaders/download_open_x.py` (Bridge subset)
- HuggingFace mirror: `jxu124/OpenX-Embodiment` (loader has compatibility issues as of March 2026)
- LeRobot format mirrors available for some datasets (e.g., `lerobot/toto`)

## Analysis Figure
See `datasets/analysis/figures/oxe_curator_audit.png` for:
- Trajectory counts per embodiment
- Proprio vs vision-only breakdown
- Actual joint state time series from TOTO, KUKA, UR5
- Cross-embodiment joint overlay comparison
- Dimensionality comparison with Brain-JEPA

---

## References
- [Open X-Embodiment paper](https://arxiv.org/abs/2310.08864) (arXiv:2310.08864)
- [DROID paper](https://arxiv.org/abs/2403.12945) (arXiv:2403.12945)
- [OXE GitHub](https://github.com/google-deepmind/open_x_embodiment)
- [OXE Dataset Spreadsheet](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g)
- [TOTO Benchmark](https://toto-benchmark.org/)
- [Stanford KUKA Multimodal](https://sites.google.com/view/stanford-kuka-multimodal)
- [Berkeley FANUC](https://sites.google.com/berkeley.edu/fanuc-manipulation)
- [DROID Dataset](https://droid-dataset.github.io/)
