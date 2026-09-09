# Plan 01-02 Summary: Full-Information Maximum Likelihood (FIML) Heckman Estimation

## Execution Details
- **Phase:** 01 (Heckman Selection Asymptotic SE Correction)
- **Plan:** 01-02 (Wave 2)
- **Status:** COMPLETED
- **Requirements Satisfied:** HECK-02

## What Changed
1. **`umbra/imputers/heckman_selection.py`**:
   - Added `HeckmanConvergenceWarning(UserWarning)` custom warning.
   - Added constructor parameters `method: Literal["two-step", "fiml"] = "two-step"` and `se_method: Optional[Literal["murphy-topel", "bootstrap", "naive"]] = None`.
   - Implemented `_fiml_neg_log_likelihood` evaluating joint bivariate normal log-likelihood across unobserved units ($\ln \Phi(-w_i \gamma)$) and observed units with conditional normal selection density.
   - Implemented unconstrained optimization over parameter vector $\theta = (\beta, \gamma, \ln\sigma, \text{atanh}(\rho))$ using L-BFGS-B / BFGS, initialized with two-step parameter estimates.
   - Inverted observed Hessian via `statsmodels.tools.numdiff.approx_hess` and applied the Delta method to compute standard errors for $\hat{\sigma}$, $\hat{\rho}$, and $\hat{\beta}_\lambda$.
   - Implemented graceful fallback on non-convergence or singular Hessian: emits `HeckmanConvergenceWarning`, retains two-step estimates with Murphy-Topel standard errors, and records `self.method_used_ = "two-step"`.
   - Exposed `self.method_used_` and `self.log_likelihood_`.
2. **`umbra/imputers/__init__.py`**:
   - Exported `HeckmanConvergenceWarning` and `HeckmanSEWarning`.
3. **`tests/test_heckman.py`**:
   - Added unit tests:
     - `test_fiml_estimation_accuracy`: parameter recovery on synthetic selection DGP with known ground truth.
     - `test_fiml_standard_errors_positive`: strictly positive standard errors and positive definite covariance via Hessian inversion.
     - `test_fiml_convergence_fallback`: fallback to two-step with `HeckmanConvergenceWarning` on collinear data.
     - `test_fiml_sklearn_pipeline_compatibility`: scikit-learn `Pipeline` compatibility with `fit_transform`.
     - `test_se_method_naive_emits_warning`: verified backward-compatible warning for naive OLS standard errors.
     - `test_heckman_stochastic_imputation_draws_vary`: verified stochastic imputation draws produce variation across missing rows.

## Verification
- `pytest tests/test_heckman.py -v`: 9/9 passed.
- `pytest tests/`: 124 passed, 0 failures.
- `mypy umbra/ tests/test_heckman.py`: 0 errors.
- `ruff check umbra/ tests/` & `ruff format --check`: clean.
- Test coverage: 93%.
