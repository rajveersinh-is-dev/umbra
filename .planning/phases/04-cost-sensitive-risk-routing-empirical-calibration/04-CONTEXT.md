# Phase 4: Cost-Sensitive Risk Routing & Empirical Calibration - Context

**Created:** 2026-09-09
**Domain:** Statistical Decision Theory, Bayes-Optimal Decision Boundaries, Empirical Probability Calibration (Platt Scaling, Isotonic Regression)
**Goal:** Upgrade the Umbra Auto-Router with Bayes-optimal decision thresholds under asymmetric loss matrices, empirical probability calibration mapping composite scores to $P(\text{MNAR} \mid \text{diagnostics})$, and pluggable decision profiles (`conservative_mnar`, `balanced`, `permissive_mar`), resolving Limitation #5 in `docs/limitations.md`.

## Background & Problem Statement

In `docs/limitations.md`:
> "5. Uncalibrated Decision Threshold Heuristics in the Auto Router: The composite missingness concern score synthesized by Umbra maps into discrete risk tiers (LOW, MEDIUM, HIGH) and strategy selections using heuristic score cutoffs (e.g., 0.25 and 0.50) and decision rules. While these thresholds achieve high separation on stylized synthetic data generators, they are heuristics rather than Bayes-optimal decision boundaries derived from an empirical risk minimization objective or calibrated under real-world cost matrices. Different domain loss functions (e.g., asymmetric penalties for missed MNAR vs. unnecessary sensitivity exploration) warrant user-specified recalibration."

In v0.2.0:
1. Composite score $s \in [0, 1]$ was an ad-hoc linear combination of diagnostic signals (Little's test, covariate shift, tail dependency). It was explicitly documented as *not* being a calibrated probability $P(\text{MNAR} \mid \text{signals})$.
2. The threshold cutoffs ($\tau_{\text{high}} = 0.50, \tau_{\text{med}} = 0.25$) were hardcoded heuristics without formal connection to misclassification costs.
3. In clinical, regulatory, or forensic domains, a false negative (treating MNAR as MAR, leading to undetected non-response bias and anti-conservative inference) is vastly more penalizing than a false alarm (conducting unnecessary sensitivity analysis). Conversely, in high-throughput exploratory pipelines, false alarms incur prohibitive computational overhead.
4. Users could not specify an asymmetric loss matrix, choose risk profiles, or serialize and load calibrated thresholds into `UmbraImputer`.

## Architectural Decisions

### D-14: Bayes-Optimal Cost-Sensitive Decision Theory (`ROUT-01`)
Let true missingness mechanism state be $y \in \{\text{MAR}, \text{MNAR}\}$ (where MAR encompasses MCAR).
Let action be $a \in \{\text{MAR}, \text{MNAR}\}$ (or strategy selection).
Define normalized loss matrix $\mathbf{L}$:
- $L(\text{MAR}, \text{MAR}) = 0$ (correct MAR)
- $L(\text{MNAR}, \text{MNAR}) = 0$ (correct MNAR)
- $L(\text{MNAR}, \text{MAR}) = C_{\text{FA}}$: Cost of False Alarm (unnecessary sensitivity analysis / Heckman complexity)
- $L(\text{MAR}, \text{MNAR}) = C_{\text{FN}}$: Cost of False Negative (missed MNAR bias)

For calibrated posterior probability $p = P(\text{MNAR} \mid \text{diagnostics})$:
$$\mathbb{E}[L(a = \text{MNAR})] = (1 - p) C_{\text{FA}}$$
$$\mathbb{E}[L(a = \text{MAR})] = p C_{\text{FN}}$$

Bayes-optimal decision rule chooses MNAR action iff:
$$\mathbb{E}[L(a = \text{MNAR})] \le \mathbb{E}[L(a = \text{MAR})] \iff p \ge \frac{C_{\text{FA}}}{C_{\text{FA}} + C_{\text{FN}}} = \tau^*$$

### D-15: Empirical Probability Calibration (`ROUT-02`)
Implement `MNARRiskCalibrator` in `umbra/diagnostics/risk_calibrator.py`:
- Maps uncalibrated composite risk score $s \in [0, 1]$ to calibrated posterior probability $p \in [0, 1]$.
- Supports methods:
  - `"platt"`: Platt scaling via logistic sigmoid $P(\text{MNAR} \mid s) = \frac{1}{1 + \exp(A \cdot s + B)}$.
  - `"isotonic"`: Monotonic non-parametric step function via `sklearn.isotonic.IsotonicRegression`.
  - `"linear"`: Min-max linear normalization with clipping.
- Includes pre-calibrated default coefficients fitted on the benchmark DGP battery, ensuring $P(\text{MNAR} \mid s)$ is immediately available without mandatory user training.
- Provides `fit(scores, y_true)`, `predict_proba(scores)`, `calibrate(score)`, `brier_score_loss()`, `expected_calibration_error()`, and JSON serialization (`to_dict`, `from_dict`, `save`, `load`).

### D-16: Pluggable Decision Profiles & UmbraImputer Integration (`ROUT-03`)
Provide `RouterDecisionProfile` dataclass and built-in registry `ROUTER_PROFILES`:
1. `balanced`: $C_{\text{FN}} = 1.0, C_{\text{FA}} = 1.0 \implies \tau^* = 0.50$.
2. `conservative_mnar`: $C_{\text{FN}} = 4.0, C_{\text{FA}} = 1.0 \implies \tau^* = 0.20$.
3. `permissive_mar`: $C_{\text{FN}} = 1.0, C_{\text{FA}} = 4.0 \implies \tau^* = 0.80$.
4. Custom profile support via user-supplied loss matrix or explicit costs.

Expose parameters in `assess_mnar_risk`, `diagnose_dataframe`, and `UmbraImputer`:
- `decision_profile: Union[str, RouterDecisionProfile] = "balanced"`
- `loss_matrix: Optional[Dict[str, float]] = None`
- `calibrator: Optional[MNARRiskCalibrator] = None`
Store `calibrated_p_mnar`, `decision_profile`, and `expected_loss` directly on `MNARRiskReport` and `UmbraImputer`.
