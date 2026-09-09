# Plan 04-02: Empirical Probability Calibration & UmbraImputer Integration - Summary

**Executed:** 2026-09-09
**Status:** Completed & Verified
**Requirements Satisfied:** `ROUT-01`, `ROUT-02`, `ROUT-03`

## Summary of Accomplishments

1. **Probability Calibrator Engine**:
   - Implemented `MNARRiskCalibrator` supporting `"platt"`, `"isotonic"`, and `"linear"` calibration.
   - Guaranteed monotonicity: for Platt scaling, $w \ge 0$ is enforced via optimization bounds; for Isotonic regression, non-decreasing step functions are fitted via PAVA.
   - Built-in default parameters ($w_0 = 6.0, b_0 = -3.0$) providing reasonable out-of-the-box probability calibration without mandatory user training.
   - Implemented `brier_score()` and `expected_calibration_error()` (ECE).
   - Full JSON serialization and round-trip persistence via `save()` and `load()`.
2. **UmbraImputer Native Integration**:
   - Added parameters `decision_profile`, `loss_matrix`, and `calibrator` to `UmbraImputer.__init__`.
   - Exposed fitted attributes: `self.calibrated_p_mnar_` and `self.routing_expected_losses_`.
   - Added `transform_multiple()` and updated `fit_transform_multiple()` on `UmbraImputer` for multiple stochastic draw generation.
   - Verified scikit-learn compliance with `clone()`, `get_params()`, and `set_params()`.
3. **Public Re-Exports**:
   - Re-exported `MNARRiskCalibrator`, `RouterDecisionProfile`, `ROUTER_PROFILES`, `compute_bayes_optimal_threshold`, `compute_expected_losses`, and `get_decision_profile` across `umbra` and `umbra.diagnostics`.
4. **Verification**:
   - Authored `tests/test_router_calibration.py` with 12 comprehensive unit tests covering all components.
   - Full test suite: 151 passed, 0 failures.
   - 0 mypy type errors; 0 ruff lint errors.
