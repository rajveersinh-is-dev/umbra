# Phase 4: Cost-Sensitive Risk Routing & Empirical Calibration - Research

**Researched:** 2026-09-09
**Domain:** Cost-Sensitive Classification, Platt Scaling, Isotonic Regression, Probability Calibration Metrics

## 1. Statistical Decision Theory & Asymmetric Costs

### Foundations (Elkan 2001)
In binary classification under uncertainty, let the true class be $y \in \{0, 1\}$ where $0 \equiv \text{MAR}$ and $1 \equiv \text{MNAR}$.
Let the decision be $\hat{y} \in \{0, 1\}$.
The cost matrix $C(i, j)$ represents the penalty incurred by predicting $\hat{y} = i$ when the true state is $y = j$:

| Predicted $\hat{y}$ \ Actual $y$ | $y = 0$ (MAR) | $y = 1$ (MNAR) |
|---|---|---|
| $\hat{y} = 0$ (MAR) | $C_{00} = 0$ | $C_{01} = C_{\text{FN}}$ (Missed MNAR) |
| $\hat{y} = 1$ (MNAR) | $C_{10} = C_{\text{FA}}$ (False Alarm) | $C_{11} = 0$ |

Given conditional posterior probability $p = P(y = 1 \mid \mathbf{x})$:
- Expected loss of predicting $\hat{y} = 0$:
  $$\mathbb{E}[L \mid \hat{y} = 0] = (1 - p) C_{00} + p C_{01} = p C_{\text{FN}}$$
- Expected loss of predicting $\hat{y} = 1$:
  $$\mathbb{E}[L \mid \hat{y} = 1] = (1 - p) C_{10} + p C_{11} = (1 - p) C_{\text{FA}}$$

The Bayes-optimal decision minimizes conditional risk:
$$\hat{y}^* = 1 \iff \mathbb{E}[L \mid \hat{y} = 1] \le \mathbb{E}[L \mid \hat{y} = 0] \iff (1 - p) C_{\text{FA}} \le p C_{\text{FN}}$$
Solving for $p$:
$$p (C_{\text{FA}} + C_{\text{FN}}) \ge C_{\text{FA}} \iff p \ge \frac{C_{\text{FA}}}{C_{\text{FA}} + C_{\text{FN}}} \equiv \tau^*$$

### Operational Decision Profiles
From this closed-form threshold $\tau^*$:
1. **`balanced` Profile**:
   $C_{\text{FN}} = 1.0, C_{\text{FA}} = 1.0 \implies \tau^* = \frac{1}{1 + 1} = 0.50$.
   $\tau_{\text{med}} = 0.25$.
2. **`conservative_mnar` Profile** (Clinical, Regulatory, Causal Inference):
   Missed MNAR incurs catastrophic bias and anti-conservative coverage.
   $C_{\text{FN}} = 4.0, C_{\text{FA}} = 1.0 \implies \tau^* = \frac{1}{1 + 4} = 0.20$.
   $\tau_{\text{med}} = 0.10$.
3. **`permissive_mar` Profile** (Exploratory Pipelines, Low-Compute Budgets):
   Unnecessary sensitivity bounds and Heckman models incur excessive pipeline friction.
   $C_{\text{FN}} = 1.0, C_{\text{FA}} = 4.0 \implies \tau^* = \frac{4}{1 + 4} = 0.80$.
   $\tau_{\text{med}} = 0.40$.

---

## 2. Probability Calibration Algorithms

### Platt Scaling (Platt 1999, Niculescu-Mizil & Caruana 2005)
Fits a univariate logistic sigmoid mapping raw continuous score $s \in [0, 1]$ to calibrated posterior probability $p \in (0, 1)$:
$$P(\text{MNAR} \mid s) = \frac{1}{1 + \exp(- (w \cdot s + b))}$$
where $w > 0$ enforces monotonicity. Fitted via maximum likelihood (binary cross-entropy) with regularized parameters.

### Isotonic Regression (Zadrozny & Elkan 2002)
Fits a non-parametric piecewise constant isotonic (monotonic non-decreasing) step function minimizing squared residuals:
$$\min_{\hat{p}_1 \le \dots \le \hat{p}_n} \sum_{i=1}^n (y_i - \hat{p}_i)^2$$
Implemented via `sklearn.isotonic.IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")`.
Ideal when the relationship between composite score and empirical MNAR rate is non-linear but strictly monotonic.

### Calibration Quality Metrics
1. **Brier Score**: Mean squared error between calibrated probability and binary outcome:
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2 \in [0, 1]$$
2. **Expected Calibration Error (ECE)**: Bin predictions into $M$ bins $B_1, \dots, B_M$:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
   where $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} p_i$ and $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$.

---

## 3. Integration Plan & Backward Compatibility

1. **`umbra/diagnostics/risk_calibrator.py`**:
   - `MNARRiskCalibrator`: Standalone calibrator with `"platt"`, `"isotonic"`, and pre-fitted default weights.
   - `RouterDecisionProfile`: Dataclass encapsulating $(C_{\text{FN}}, C_{\text{FA}}, \tau^*_{\text{high}}, \tau^*_{\text{med}}, \tau_{\text{tail}})$.
   - `ROUTER_PROFILES`: Predefined dictionary (`"balanced"`, `"conservative_mnar"`, `"permissive_mar"`).
   - `compute_bayes_optimal_threshold(c_fn, c_fa) -> float`.
2. **`umbra/diagnostics/mnar_risk_score.py`**:
   - Extend `assess_mnar_risk` and `diagnose_dataframe` with optional `decision_profile` and `calibrator`.
   - Extend `MNARRiskReport` with `calibrated_p_mnar: Optional[float]`, `decision_profile: Optional[str]`, `expected_loss: Optional[float]`.
   - If profile is supplied, compute dynamic thresholds $\tau_{\text{high}}, \tau_{\text{med}}$ via Bayes-optimal rule.
3. **`umbra/api.py`**:
   - Add `decision_profile: str = "balanced"`, `loss_matrix: Optional[Dict[str, float]] = None`, `calibrator: Optional[MNARRiskCalibrator] = None` to `UmbraImputer.__init__`.
   - Expose `calibrated_p_mnar_` and `expected_losses_` on fitted `UmbraImputer`.
4. **`tests/test_router_calibration.py`**:
   - Comprehensive unit test suite covering:
     - Bayes-optimal threshold derivation and monotonicity under cost ratios.
     - Platt and Isotonic calibration convergence, bounding $p \in [0, 1]$, and ECE/Brier scores.
     - Pluggable decision profiles (`conservative_mnar` vs `balanced` vs `permissive_mar`).
     - JSON serialization and round-trip persistence of calibrators and profiles.
     - End-to-end `UmbraImputer` execution with custom loss matrices.
