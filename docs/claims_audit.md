# Umbra Mathematical & Scientific Claims Audit

This document audits every theoretical, statistical, and empirical claim across the Umbra repository, classifying them into formal epistemic categories to ensure rigorous scientific integrity.

---

## 1. Classification Taxonomy

- **Theorem**: Formally proven in peer-reviewed mathematical statistics literature.
- **Assumption-Dependent Claim**: True mathematically conditional on explicit, non-testable structural assumptions.
- **Empirical Finding**: Demonstrated through reproducible simulation or benchmark data.
- **Heuristic**: Computationally or pragmatically motivated rule of thumb without formal asymptotic optimality guarantees.
- **Unsupported (Purged)**: Any claim lacking mathematical or empirical justification (permanently purged from the codebase).

---

## 2. Comprehensive Mathematical & Algorithmic Audit Matrix

| Mathematical Component | Epistemic Classification | Theoretical Basis / Governing Reference | Operational Implementation & Audit Notes |
| :--- | :---: | :--- | :--- |
| **Little's MCAR Test** | **THEOREM** | Little, R. J. A. (1988), *JASA* | Implemented in `umbra/diagnostics/mcar_test.py`. Assumes multivariate normality of observed vectors under MCAR. |
| **Exact Degrees of Freedom** | **THEOREM** | Little, R. J. A. (1988), *JASA* | Exact formula $\text{df} = \sum_{j=1}^J p_j - p$. Verified against R `naniar` and Stata implementations. |
| **Covariance Regularization** | **EMPIRICAL / HEURISTIC** | Ledoit, O. & Wolf, M. (2004); Higham (2002) | Ledoit-Wolf shrinkage provides asymptotically optimal Frobenius loss reduction; eigenvalue floor ($\epsilon=10^{-6}$) ensures numerical inversion stability. |
| **Wasserstein Covariate Shift** | **EMPIRICAL** | Villani, C. (2008), *Optimal Transport* | Non-parametric 1D Wasserstein-1 distance measuring divergence between responder and non-responder marginal distributions. |
| **Missingness Concern Score** | **HEURISTIC** | Pragmatic composite scoring | Weighted sum (MCAR $0.25$, Shift $0.35$, Tail $0.40$). Domain regex receives $0.00$ weight. Fixed thresholds ($0.35, 0.65$) act as decision boundaries, not posterior probabilities. |
| **Auxiliary Variable F-Screening** | **HEURISTIC / EMPIRICAL** | Stock, Wright, & Yogo (2002), *JBES* | Evaluates candidate relevance via first-stage linear regression. $F > 10$ benchmark warns against weak instrument bias; labeled *candidate auxiliary variable*, never *proved instrument*. |
| **Heckman Selection Estimator** | **ASSUMPTION-DEPENDENT** | Heckman, J. J. (1979), *Econometrica* | Two-step estimator consistent conditional on joint bivariate Gaussian errors and valid exclusion ($Z \perp \epsilon \mid X$). Second-stage OLS standard errors are approximate; they do not include Murphy-Topel (1985) generated-regressor corrections. |
| **Pattern-Mixture Shift ($\delta$)** | **ASSUMPTION-DEPENDENT** | Little, R. J. A. (1993), *JASA* | Parameterizes unobserved counterfactual departure in residual standard deviation units: $Y_{\text{mis}} \sim \hat{\mu}_{\text{MAR}} + \delta \hat{\sigma}$. Bounded by user-specified prior belief. |
| **Tipping Point Frontier** | **EMPIRICAL / MATHEMATICAL** | Liublinska, V. & Rubin, D. B. (2014) | Deterministic grid search locating critical $\delta^*$ where substantive null hypothesis cannot be rejected or point estimate changes sign. |
| **Multiple Imputation Pooling** | **THEOREM** | Rubin, D. B. (1987); Barnard & Rubin (1999) | Exact pooling rules for point estimate $\bar{Q}$, total variance $T = \bar{U} + (1 + 1/m)B$, and adjusted degrees of freedom $\nu$. (Note: Benchmark leaderboard evaluates single-imputation plug-in coverage; full MI is exposed via `rubins_rules`). |
| **Non-Identifiability Theorem** | **THEOREM** | Molenberghs et al. (2008), *Stat. Sci.* | Proves any MNAR model has an empirically indistinguishable MAR counterpart on observed data. Governs all epistemic disclaimers in Umbra. |
| **Auto Router Policy** | **HEURISTIC (RISK-AWARE)** | Evidence-conditioned triage | Evaluates empirical signals to prevent catastrophic failures (Heckman on MCAR or MICE on MNAR). Not an intrinsically superior estimator; matches MICE on MCAR/MAR and Heckman on MNAR with instrument. |
| **Unrestricted Manski Bounds** | **THEOREM** | Manski, C. F. (1990), *AER* | For unbounded continuous distributions ($Y \in (-\infty, +\infty)$), worst-case bounds evaluate to $(-\infty, +\infty)$ and are vacuous. Valid on compact physical support $[y_{\min}, y_{\max}]$. |

---

## 3. Epistemic Guardrails Enforced in Code

1. **No Omniscience Claims**: Umbra never claims to identify true missingness mechanisms or uncover counterfactual distributions without unverifiable structural assumptions.
2. **Mandatory Wilson Uncertainty Intervals**: All routing benchmark accuracy and sensitivity figures must report binomial 95% Wilson score confidence intervals.
3. **Transparent Failure Modes**: Dedicated failure mode documentation (`docs/failure_modes.md`) and stress batteries (`benchmarks/misspecification_benchmark.py`) detail explicit breakdown boundaries (weak instruments, heavy-tailed errors, symmetric tail censoring).
