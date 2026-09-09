# Phase 4: Cost-Sensitive Risk Routing & Empirical Calibration - Validation Plan

**Domain:** Cost-Sensitive Decision Theory, Platt Scaling, Isotonic Regression, Automated Verification Suite

## Validation Criteria

### Mathematical Invariants
1. **Bayes-Optimal Threshold Invariant**:
   $$\tau^* = \frac{C_{\text{FA}}}{C_{\text{FA}} + C_{\text{FN}}}$$
   - When $C_{\text{FN}} > C_{\text{FA}}$, $\tau^* < 0.5$ (more sensitive to MNAR).
   - When $C_{\text{FN}} < C_{\text{FA}}$, $\tau^* > 0.5$ (more permissive of MAR).
   - Expected loss for action $a \in \{\text{MAR}, \text{MNAR}\}$ equals $\mathbb{E}[L \mid a]$.
2. **Probability Calibration Invariants**:
   - $P(\text{MNAR} \mid s) \in [0.0, 1.0]$ for all $s \in [0, 1]$.
   - Monotonicity: for $s_1 < s_2$, $P(\text{MNAR} \mid s_1) \le P(\text{MNAR} \mid s_2)$.
   - Brier score $\text{BS} \in [0.0, 1.0]$ and Expected Calibration Error $\text{ECE} \in [0.0, 1.0]$.
3. **Decision Profile Ordering**:
   For any dataset with intermediate missingness signals ($s \approx 0.35$):
   - `conservative_mnar` ($\tau^* = 0.20$) flags `HIGH` risk and routes to MNAR strategy.
   - `balanced` ($\tau^* = 0.50$) flags `MEDIUM` risk.
   - `permissive_mar` ($\tau^* = 0.80$) flags `LOW` risk and routes to standard MICE.

### Automated Test Suite (`tests/test_router_calibration.py`)
- `test_bayes_optimal_threshold_computation`: Verify exact formula, symmetric costs yielding 0.5, and extreme cost ratios.
- `test_platt_scaling_fit_and_predict`: Fit Platt scaling on synthetic diagnostic scores; verify monotonicity and output in $[0, 1]$.
- `test_isotonic_calibration_fit_and_predict`: Fit Isotonic regression; verify strict monotonicity and piecewise constant behavior.
- `test_calibrator_serialization`: Verify `to_dict`, `from_dict`, JSON serialization, and equivalence of `predict_proba` before and after round-trip.
- `test_router_decision_profiles`: Verify predefined profiles (`balanced`, `conservative_mnar`, `permissive_mar`) and custom profile creation.
- `test_assess_mnar_risk_with_profiles_and_calibration`: Verify `assess_mnar_risk` emits `calibrated_p_mnar`, `expected_loss`, and `decision_profile`.
- `test_umbra_imputer_decision_profile_routing`: End-to-end test verifying `UmbraImputer` routes differently under `conservative_mnar` vs `permissive_mar`.
- `test_umbra_imputer_custom_loss_matrix`: Verify `UmbraImputer(loss_matrix={"c_fn": 10.0, "c_fa": 1.0})` dynamically recalibrates thresholds.
- `test_sklearn_compatibility_with_profiles`: Verify `clone(UmbraImputer(decision_profile='conservative_mnar'))` and parameter introspection.

### Regression Checks
- All 139 existing unit tests in `tests/` must continue to pass without modification.
- `mypy` type checking passes with zero errors across all source files.
- `ruff` lint and format checks pass cleanly.
