# Autoresearch Prompt

Paste this into Claude Code running with `--dangerously-skip-permissions` on SageMaker:

---

## Prompt

```
You are running autonomous research on KH-JEPA. Your goal: achieve MSE < 0.45 on ETTh1-96.

## Context

Read these files first:
- autoresearch/program.md (overall goals and tiers)
- autoresearch/ITERATION_STATUS.md (previous diagnosis - DELETE after reading)

## Previous Diagnosis (Summary)

Root cause identified: The encoder mean-pools all patches to a single vector, destroying temporal information needed for forecasting. The architecture outputs (B, latent_dim) but needs (B, pred_len, features) for forecasting.

Data pipeline verified:
- Predict zeros: ~1.0 MSE (data is normalized)
- Predict last value: ~0.4 MSE (persistence baseline)
- Predict mean: ~0.8 MSE

So any working model should beat 0.4 MSE. Current model gets 0.65+ which is worse than naive persistence.

## Your Mission

1. Fix the architecture so it actually forecasts (preserves temporal structure)
2. Iterate until MSE < 0.45
3. Then run transfer tests (Tier 2)

## Rules

1. ONE change at a time
2. After each change, run: python train.py
3. Log every result:
   echo "$(date) | $CHANGE | mse=$MSE | z_std=$Z_STD" >> experiment_log.txt
4. If MSE > 0.45: diagnose, hypothesize, fix
5. If MSE < 0.45: Tier 1 PASSED, check Tier 2 results
6. Maximum 20 experiments, then summarize findings
7. NO QUESTIONS - make decisions based on data

## Suggested Fix Order

1. First: Try baseline recipe (no JEPA) to verify architecture can learn
   - Edit: RECIPE = "baseline" in train.py
   - If baseline fails, the core architecture is broken

2. If baseline fails: The decoder needs to output (B, pred_len, C) not derive from (B, latent_dim)
   - Check how predictions are generated
   - May need to keep sequence dimension through encoder

3. If baseline works but JEPA fails: SIGReg or latent prediction is broken
   - Check z_std (should be ~1.0, not 0.15)
   - Try increasing SIGREG_WEIGHT

4. Remove double normalization: RevIN + pre-normalized data may conflict
   - Data is already normalized in prepare.py
   - RevIN normalizes again - may hurt

## Start Now

1. Read ITERATION_STATUS.md for full context
2. Delete ITERATION_STATUS.md (knowledge transferred to you)
3. Run baseline recipe first
4. Begin iteration loop

Go.
```
