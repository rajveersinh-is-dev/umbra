# Phase 1: Heckman Selection Asymptotic SE Correction (Murphy-Topel & FIML) - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase addresses Limitation #2 in `docs/limitations.md`. It delivers exact analytical Murphy & Topel (1985) two-step asymptotic covariance correction and optional Full-Information Maximum Likelihood (FIML) estimation for `HeckmanSelectionImputer` in `umbra/imputers/heckman_selection.py`.

</domain>

<decisions>
## Implementation Decisions

### Murphy-Topel Asymptotic Standard Errors
- **D-01:** Implement the analytical Murphy & Topel (1985) two-step asymptotic covariance matrix formula in `HeckmanSelectionImputer`:
  $$\text{Var}(\hat{\beta}) = \sigma_2^2 (X_*^T X_*)^{-1} \left[ X_*^T (I - \rho^2 \Delta) X_* + X_*^T \Delta W \text{Var}(\hat{\gamma}) W^T \Delta X_* \right] (X_*^T X_*)^{-1}$$
  where $\Delta = \text{diag}(\delta_i)$, $\delta_i = \lambda_i (\lambda_i + w_i \hat{\gamma})$, and $W$ is the selection design matrix.
  — **Reversibility:** reversible — self-contained in `umbra/imputers/heckman_selection.py`.
- **D-02:** Murphy-Topel standard errors become the default parametric standard errors stored on `self.coef_stderr_` when `n_bootstrap_se == 0`, replacing naive uncorrected OLS standard errors (which ignore first-stage variance).

### Estimation Method Architecture (Two-Step vs FIML)
- **D-03:** Add `method: Literal["two-step", "fiml"] = "two-step"` to `HeckmanSelectionImputer.__init__`.
- **D-04:** For `method="fiml"`, initialize the numerical optimizer with the two-step estimates ($\hat{\beta}^{(0)}, \hat{\gamma}^{(0)}, \hat{\sigma}^{(0)}, \hat{\rho}^{(0)}$), maximize the joint bivariate normal log-likelihood via `scipy.optimize.minimize` (L-BFGS-B or trust-ncg with bounds $\sigma > 0$ and $-1 < \rho < 1$), and invert the observed Hessian for asymptotic standard errors.
- **D-05:** If FIML optimization fails to converge or produces a non-positive-definite Hessian, gracefully fall back to two-step with Murphy-Topel standard errors, emit a `HeckmanConvergenceWarning`, and record `self.method_used_ = "two-step"`.

### Imputation Draw Mechanism
- **D-06:** In `transform()`, default to deterministic conditional expectation $E[Y_i \mid S_i=0, X_i, W_i] = X_i \hat{\beta} + \hat{\rho} \hat{\sigma} \lambda(W_i \hat{\gamma})$ to adhere to scikit-learn standard deterministic transformer conventions. Provide `stochastic: bool = False` for single-draw perturbation with conditional variance $\sigma_i^2 = \sigma^2(1 - \rho^2 \delta_i)$.

### User Guidance & Exclusion Restrictions
- **D-07:** Support estimation without explicit exclusion restrictions via non-linear functional form identification, but emit a warning if the condition number of $[X_* \quad \lambda]$ indicates elevated multicollinearity.

### the agent's Discretion
- User gave full authority ("Whatever is best") to apply the most rigorous econometric formulations for matrix shapes, numerical safeguards on Mills ratio tails, and SciPy optimizer configuration.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Econometric Theory & Limitations
- `docs/limitations.md` §2 — Heckman Selection Two-Step Standard Error Adjustments
- Murphy, K.M. & Topel, R.H. (1985), *Estimation and Inference with Two-Step Econometric Estimators*, Journal of Business & Economic Statistics 3(4): 370–379.
- Cameron, A.C. & Trivedi, P.K. (2005), *Microeconometrics: Methods and Applications*, Cambridge University Press, Section 24.5 (Two-Step M-Estimators and Selection Models).
- Greene, W.H. (2003), *Econometric Analysis*, 5th edition, Prentice Hall, Chapter 22 (Sample Selection).

### Codebase Implementations
- `umbra/imputers/heckman_selection.py` — Current two-step implementation and Ridge fallback
- `umbra/diagnostics/shadow_variable_finder.py` — Auxiliary variable and instrument detection
- `tests/test_imputers.py` — Existing Heckman estimator unit tests

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `statsmodels.discrete.discrete_model.Probit` in `umbra/imputers/heckman_selection.py`: Already fits first-stage selection model and computes parameter covariance matrix `cov_params()`.
- `scipy.stats.norm`: Inverse Mills ratio tail approximations already implemented and numerical guards in place.

### Established Patterns
- Scikit-learn estimator interface: `fit`, `transform`, `check_is_fitted`, `n_features_in_`, `feature_names_in_`.
- Strict typing with mypy: all parameter annotations, return types, and Union narrowings explicitly typed.

### Integration Points
- `umbra/imputers/heckman_selection.py`: Core estimator logic.
- `umbra/__init__.py`: Public API export.
- `tests/test_imputers.py` & `tests/test_coverage.py`: Unit tests.

</code_context>

<specifics>
## Specific Ideas
- Provide both `self.coef_stderr_` (Murphy-Topel or FIML analytical SE) and `self.bootstrap_stderr_` (if `n_bootstrap_se > 0`).

</specifics>

<deferred>
## Deferred Ideas

### Reviewed Todos (not folded)
- None — discussion stayed within Phase 1 scope.

</deferred>

---

*Phase: 01-heckman-selection-asymptotic-se-correction-murphy-topel-fiml*
*Context gathered: 2026-09-09*
