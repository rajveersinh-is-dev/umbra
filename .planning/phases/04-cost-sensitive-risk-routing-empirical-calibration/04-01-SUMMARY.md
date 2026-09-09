# Plan 04-01: Cost-Sensitive Decision Engine & Decision Profiles - Summary

**Executed:** 2026-09-09
**Status:** Completed & Verified
**Requirements Satisfied:** `ROUT-01`, `ROUT-03`

## Summary of Accomplishments

1. **Closed-Form Bayes-Optimal Thresholds**:
   - Implemented `compute_bayes_optimal_threshold(c_fn, c_fa)` in `umbra/diagnostics/risk_calibrator.py`:
     $$\tau^* = \frac{C_{\text{FA}}}{C_{\text{FA}} + C_{\text{FN}}}$$
   - Implemented `compute_expected_losses(p_mnar, c_fn, c_fa)` computing expected risk and regret for MAR and MNAR actions.
2. **Pluggable Decision Profiles**:
   - Created `RouterDecisionProfile` dataclass supporting `to_dict()`, `from_dict()`, `save()`, and `load()`.
   - Populated standard profiles in `ROUTER_PROFILES`:
     - `balanced`: $C_{\text{FN}} = 1, C_{\text{FA}} = 1 \implies \tau^* = 0.50, \tau_{\text{med}} = 0.25$.
     - `conservative_mnar`: $C_{\text{FN}} = 4, C_{\text{FA}} = 1 \implies \tau^* = 0.20, \tau_{\text{med}} = 0.10$.
     - `permissive_mar`: $C_{\text{FN}} = 1, C_{\text{FA}} = 4 \implies \tau^* = 0.80, \tau_{\text{med}} = 0.40$.
   - Implemented `get_decision_profile(profile_or_name, loss_matrix=None)` dynamically deriving $\tau^*$ for arbitrary user-specified cost matrices.
3. **Integration into Diagnostics**:
   - Extended `MNARRiskReport` with `calibrated_p_mnar`, `decision_profile`, `expected_loss`, and `loss_matrix`.
   - Updated `assess_mnar_risk` and `diagnose_dataframe` to accept `decision_profile` and `loss_matrix` to dynamically govern risk tiers.
