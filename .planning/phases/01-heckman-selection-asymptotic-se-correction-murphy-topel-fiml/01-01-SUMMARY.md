# Plan 01-01 Summary: Murphy-Topel Asymptotic Covariance Correction

## Execution Details
- **Phase:** 01 (Heckman Selection Asymptotic SE Correction)
- **Plan:** 01-01 (Wave 1)
- **Status:** COMPLETED
- **Requirements Satisfied:** HECK-01

## What Changed
1. **`umbra/imputers/heckman_selection.py`**:
   - Implemented private method `_compute_murphy_topel_covariance(self, X_star, W_obs, y_obs, beta_star, V1, delta) -> np.ndarray`.
   - Formula:
     $$V_2 = \hat{\sigma}^2 (X_*^T X_*)^{-1} \left[ X_*^T (I - \hat{\rho}^2 \Delta) X_* + X_*^T \Delta W_{\text{obs}} V_1 W_{\text{obs}}^T \Delta X_* \right] (X_*^T X_*)^{-1}$$
     with vectorized execution avoiding $N \times N$ matrix allocations.
   - Adjusted residual disturbance variance $\hat{\sigma}^2 = \frac{1}{n_1} \sum \hat{\nu}_i^2 + \hat{\beta}_\lambda^2 \bar{\delta}$ and correlation $\hat{\rho} = \text{clip}(\hat{\beta}_\lambda / \hat{\sigma}, -0.999, 0.999)$.
   - Exposed primary fitted attributes `self.coef_cov_`, `self.coef_stderr_`, `self.mills_stderr_`, `self.sigma_`, and `self.rho_`.
   - Made Murphy-Topel the default standard error estimator for two-step Heckman estimation when `n_bootstrap_se == 0`, retiring uncorrected naive OLS standard errors.
2. **`tests/test_heckman.py`**:
   - Created test suite with `test_murphy_topel_covariance_shape_and_positivity`, `test_murphy_topel_greater_than_naive_ols`, and `test_murphy_topel_matches_bootstrap_order_of_magnitude`.
   - Verified that Murphy-Topel standard errors account for generated regressor uncertainty and exceed naive OLS standard errors.

## Verification
- `pytest tests/test_heckman.py -v`: 9/9 passed.
- `mypy umbra/ tests/test_heckman.py`: 0 errors in 20 source files.
- `ruff check umbra/ tests/` and `ruff format --check`: 0 errors.
