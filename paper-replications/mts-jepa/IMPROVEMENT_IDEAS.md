# MTS-JEPA Improvement Ideas

All ideas from brainstorming (Phase 3J) with their current status.

## VALIDATED

*(None yet — pending full-scale experiments)*

## PROTOTYPED

### CC-JEPA: Causal Codebook JEPA
- **Status**: Prototype implemented (`cc_jepa.py`)
- **Results**: Pending comparison experiments
- **Core insight**: Causal masking in encoder ensures genuine future prediction, codebook prevents collapse
- **Next**: Run comparison on PSM and MSL

## DEFERRED (Promising, Needs More Time)

### Degradation Regime Codebook (DRC)
- **Idea**: Codebook entries naturally discover degradation regimes on run-to-failure data
- **Why deferred**: Requires C-MAPSS pre-training which is a separate infrastructure
- **Priority**: P0 for next session
- **Estimated effort**: 3 days

### Information Bottleneck Analysis
- **Idea**: Formally characterize codebook as information bottleneck; compute I(X;Z) and I(Z;Y)
- **Why deferred**: IB estimation requires careful implementation
- **Priority**: P2

### VICReg + Codebook
- **Idea**: Add variance-invariance-covariance loss to prevent collapse
- **Why deferred**: May be unnecessary if codebook already prevents collapse
- **Priority**: P1 (quick experiment)

### Multi-Resolution Codebook with Scale-Specific Codes
- **Idea**: Separate K_fine and K_coarse codebooks with cross-scale consistency
- **Why deferred**: Adds complexity, unclear benefit
- **Priority**: P2

### Anomaly Prediction as RUL Regression
- **Idea**: Predict time-to-next-anomaly (continuous) instead of binary
- **Why deferred**: Requires dataset-specific label engineering
- **Priority**: P1

### Optimal K Selection via MDL
- **Idea**: Use Minimum Description Length to automatically select codebook size
- **Why deferred**: Requires sweeping K which is expensive
- **Priority**: P2

### Codebook Utilization via VQBridge
- **Idea**: Replace entropy regularization with soft→hard quantization annealing
- **Why deferred**: Our utilization is reasonable (65-70%), may not be needed
- **Priority**: P3

## KILLED

### Continuous-Time Codebook JEPA (Neural ODE)
- **Reason**: Too complex, Neural ODEs are finicky, low chance of meaningful improvement
- **Effort would have been**: 5+ days

### Codebook as Language (Explanation Generation)
- **Reason**: Requires external knowledge or templates, tangential to NeurIPS contribution
- **Effort would have been**: 5+ days

### Adversarial Codebook Robustness
- **Reason**: Interesting but doesn't address the core gaps in the gap map
- **Effort would have been**: 3 days

### Frequency-Aware Codebook with Wavelet Views
- **Reason**: TimeVQVAE-AD already does frequency-domain codebook; incremental contribution
- **Effort would have been**: 3 days

### Self-Discovering Fault Taxonomy (Idea #9)
- **Reason**: High risk, depends entirely on codebook interpretability which hasn't been validated
- **Status**: Could be revived if DRC (#4) shows codebook entries are interpretable

### Predictive Codebook Transfer
- **Reason**: Cross-domain transfer is interesting but too risky for a first paper
- **Status**: Could become a follow-up paper

## LEAD-TIME-AWARE METRIC (LTAP)
- **Status**: Implemented (`lead_time_analysis.py`)
- **Results**: Will be computed on all datasets after replication
- **This is not a method improvement but an evaluation improvement — counts as P0**
