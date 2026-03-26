# Mechanical-JEPA: Lessons Learned

Reusable insights from experiments. Update as you learn.

---

## Data

### OXE Datasets

- **ManiSkill** is cleanest — consistent episodes, good action-state correlation (0.473)
- **DROID** is largest (76k) but actions are EE-space, not joint-space
- **KUKA** has force data (`ee_forces_continuous`) — unique for contact prediction
- **Fractal** has NaN in `orientation_box` field — skip it
- All actions are EE deltas, never joint commands

### Preprocessing

- Resample to 10 Hz (KUKA/ManiSkill are 20 Hz, Fractal is 3 Hz)
- Z-score normalize per channel on training set
- Clip outliers to ±5 std before normalizing

---

## Architecture

### JEPA-Specific

- EMA decay 0.996 is standard; lower (0.99) for faster adaptation
- Predictor should be smaller than encoder (2 layers vs 4-6)
- Block masking works better than random for temporal data

### Collapse Prevention

- If embedding variance drops below 0.01 → collapse happening
- Add batch normalization on encoder output
- Use VICReg-style variance/covariance loss as regularizer if needed

---

## Training

- (Add lessons as you discover them)

---

## Evaluation

### Classification

- Embodiment classification baseline: ~85% with raw features
- If pretrained is much worse, encoder losing information

### Forecasting

- Copy-last baseline is surprisingly strong for slow-changing joints
- Linear baseline: ~0.03-0.06 MSE
- Need to beat linear to claim encoder helps

### Transfer

- Zero-pad 6-DOF robots to 7-DOF (pad joint 7 with zeros, mask in attention)
- Transfer ratio < 2.0 is good, < 1.5 is excellent

---

## Debugging

- (Add issues and fixes as you encounter them)
