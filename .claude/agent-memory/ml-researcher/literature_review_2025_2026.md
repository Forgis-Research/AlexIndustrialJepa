---
name: JEPA + PHM Literature Review 2025-2026
description: Concrete findings on JEPA for time series, collapse prevention, vibration foundation models, RUL SSL, and cross-domain transfer — searched April 2026
type: project
---

Key findings from focused literature search, April 2026.

## JEPA for Time Series

**"Joint Embeddings Go Temporal" (TS-JEPA)** — Ennadir, Golkar, Sarra. NeurIPS 2024 Workshop on Time Series in the Age of Large Models. arXiv:2509.25449. First systematic study of JEPA for time series. Uses patch tokens, >70% masking, lightweight Transformer. Outperforms MAE and contrastive baselines on classification; competitive on forecasting. No numbers in abstract-only page.

**"MTS-JEPA"** — He, Wen, Wang, Ma. arXiv:2602.04643 (Feb 2026). Multi-resolution JEPA + soft codebook bottleneck for multivariate TS anomaly detection. Datasets: MSL, SMAP, SWaT, PSM. Results: MSL F1=33.58/AUC=66.08 vs TS2Vec 23.48/64.86 (+44% F1); SWaT F1=72.89/AUC=84.95; PSM F1=61.61/AUC=77.85 vs iTransformer 54.12/63.09. Collapse prevented via soft codebook (bounded convex hull + dual entropy regularization). Beats TS-JEPA on SWaT.

**"Time-Series JEPA for Remote Control"** — Girgis, Valcarce, Bennis. arXiv:2406.04853. JEPA for bandwidth-limited control systems. Not fault diagnosis. Evaluated on cart-pole simulation only.

## Collapse Prevention

**LeJEPA** — Balestriero, LeCun. arXiv:2511.08544 (Nov 2025). Introduces SIGReg (Sketched Isotropic Gaussian Regularization). Forces embeddings toward isotropic Gaussian using Cramér-Wold theorem — projects latents onto M random directions, applies Epps-Pulley test. Achieves 79% ImageNet-1k linear eval with ViT-H/14. Validated on 10+ datasets, 60+ architectures. Replaces EMA, stop-gradient, and VICReg-style heuristics with one principled loss.

**LeWorldModel (LeWM)** — LeCun et al. arXiv:2603.19312 (Mar 2026). First JEPA to train end-to-end from raw pixels stably. Uses only two loss terms: next-embedding prediction + SIGReg. ~15M params, trains on single GPU in hours. Reduces hyperparameters from 6 to 1 vs prior end-to-end alternatives.

**SALT (Apple ML)** — Frozen teacher suffices instead of EMA. Train target encoder with pixel reconstruction, freeze it, train student to predict teacher's latents on masked regions.

## PHM Foundation Models

**RmGPT** — arXiv:2409.17604, IEEE IoT-J accepted 2025. GPT-style generative pretraining for rotating machinery. Tokens: Signal, Prompt, Time-Frequency Task, Fault. 82% accuracy in 16-class one-shot experiments. Evaluated on CWRU and Paderborn.

**PHM-GPT** — ScienceDirect 2025. LLM integrated with 15 public PHM datasets incl. CWRU and Paderborn. Total accuracy 96.92%, near 100% on CWRU.

**Masked SSL + Swin Transformer** — MDPI 2025. 99.53% on Paderborn, 100% on CWRU. MAE-style pretraining then fine-tune.

**OpenMAE** — ACM IMWUT 2025. MAE pretrained on 5M open-world vibration samples from Raspberry Shake datacenter. FreqCutMix augmentation. Up to 23% downstream accuracy improvement with enhanced generalizability.

## RUL with Self-Supervised Learning

**DCSSL** — Scientific Reports 2026. Dual-dimensional contrastive SSL for bearing RUL. Stage 1: random cropping + timestamp masking to build positive pairs, temporal-level + instance-level contrastive loss. Stage 2: fine-tune prediction head. Dataset: FEMTO/PRONOSTIA. Outperforms SOTA supervised methods.

**Self-Supervised RUL with Triplet Networks** — ScienceDirect 2024. Triplet network SSL leveraging unlabeled data for RUL. Handles limited labeled data.

**Contrastive Semi-Supervised RUL** — ScienceDirect 2025. CVAE for semi-supervised RUL with incomplete life histories.

JEPA + RUL: **NO PAPER EXISTS** combining JEPA with RUL prediction. This is a clear gap.

## Cross-Domain Transfer

Current SOTA approaches: domain adversarial networks, JMMD (joint maximum mean discrepancy), few-shot meta-learning + adversarial (MLAML). Best methods close but not eliminate domain gap.

JEPA + cross-machine transfer: **NO PAPER EXISTS**. Gap confirmed.

## MAE vs Contrastive vs JEPA for Time Series

- MAE reconstructs raw input — captures low-level detail, sensitive to noise
- Contrastive learning relies on invariance assumptions that don't always hold for time series
- JEPA predicts in latent space — naturally ignores unpredictable noise, learns high-level state transitions
- TS-JEPA (NeurIPS workshop 2024): JEPA outperforms both on classification

## Critical Gaps Confirmed (As of April 2026)

1. JEPA applied specifically to bearing fault diagnosis: NONE
2. JEPA + RUL prediction: NONE
3. JEPA + cross-machine/cross-component transfer: NONE
4. SIGReg applied to 1D vibration signals: NONE
5. JEPA + multi-component (bearing+gearbox+motor) unified pretraining: NONE

**Why:** and **How to apply:**
These gaps define the novelty space for IndustrialJEPA. Priority targets: (1) replace EMA with SIGReg for principled collapse prevention, (2) extend to RUL as downstream task, (3) frame cross-component transfer as JEPA's core advantage over MAE/contrastive.
