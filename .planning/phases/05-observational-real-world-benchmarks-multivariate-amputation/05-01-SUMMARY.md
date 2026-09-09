# Plan 05-01: Multivariate Amputation Engine (Schouten et al. 2018) - Summary

**Executed:** 2026-09-09
**Status:** Completed & Verified
**Requirements Satisfied:** `EVAL-01`

## Summary of Accomplishments

1. **Multivariate Amputation Engine (`ampute_multivariate`)**:
   - Implemented `ampute_multivariate` in `umbra/benchmark/amputation.py` following the methodology of Schouten, Lugtig, & Vink (2018).
   - Generates complex multivariate missingness patterns across numeric features with custom or automatic candidate pattern matrices.
   - Evaluates standardized weighted sum scores $S_{ik} = \sum_{j=1}^P w_{kj} \tilde{X}_{ij}$ per pattern group.
2. **Missingness Mechanism Enforcement**:
   - **MCAR**: Uniform random missingness without feature dependency.
   - **MAR Invariant**: Strictly verifies that weights for incomplete variables in any pattern are zero ($w_{kj} = 0$ if $P_{kj} = 0$). Violations immediately raise informative `ValueError`.
   - **MNAR**: Allows non-zero weights on incomplete variables ($w_{kj} \ne 0$), inducing direct self-masking missingness.
3. **Continuous Logistic Odds Types**:
   - Implemented four probability shift functions:
     - `RIGHT`: $P = \text{expit}(S - \bar{S} + b)$ (high scores amputed; $r(S, R) > 0$).
     - `LEFT`: $P = \text{expit}(-(S - \bar{S}) + b)$ (low scores amputed; $r(S, R) < 0$).
     - `MID`: $P = \text{expit}(-|S - \bar{S}| + 0.75 + b)$ (central scores amputed).
     - `TAIL`: $P = \text{expit}(|S - \bar{S}| - 0.75 + b)$ (bimodal tail scores amputed).
   - Offsets $b$ calibrated via Brent's method root-finding (`scipy.optimize.brentq`) ensuring average probability matches target `prop` within $\pm 0.05$ across $N \ge 1000$.
4. **Structured Container (`AmputationResult`)**:
   - Stores `data_amputed`, `data_complete`, boolean `mask`, `patterns`, `weights`, `probabilities`, and summary statistics.
