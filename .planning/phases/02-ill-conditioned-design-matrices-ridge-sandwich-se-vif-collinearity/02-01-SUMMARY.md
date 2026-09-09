# Plan 02-01 Summary: Ridge Sandwich Covariance & VIF Collinearity Diagnostics

## Execution Details
- **Phase:** 02 (Ill-Conditioned Design Matrices: Ridge Sandwich SE & VIF Collinearity)
- **Plan:** 02-01 (Wave 1)
- **Status:** COMPLETED
- **Requirements Satisfied:** HECK-03, HECK-04

## What Changed
1. **`umbra/imputers/heckman_selection.py`**:
   - Implemented `HeckmanCollinearityWarning(UserWarning)` to warn users when second-stage design matrix $[X_* \quad \lambda_1]$ suffers from severe collinearity.
   - Added constructor arguments:
     - `ridge_alpha: float = 1.0`: Ridge regularization strength.
     - `ridge_se_method: Literal["sandwich", "nan"] = "sandwich"`: standard error method for regularized Ridge fallback.
   - Implemented `_compute_collinearity_diagnostics(self, X_star, col_names) -> Dict[str, Any]`:
     - Computes Belsley condition indices ($\kappa = s_{\max} / s_{\min}$) on unit-norm scaled $X_*$.
     - Computes Variance Inflation Factors (VIF) for all regressors including $\lambda_1$.
     - Emits `HeckmanCollinearityWarning` when $\kappa > 30$ or $\text{VIF}_{\lambda} > 10$.
     - Stores diagnostics in `self.models_[target]["collinearity_diagnostics"]` and exposes `self.collinearity_diagnostics_` on the estimator.
   - Implemented `_compute_ridge_sandwich_covariance(self, X_star, W_obs, y_obs, beta_star, V1, delta, alpha) -> np.ndarray`:
     - Regularized sandwich covariance formula:
       $$Q_\alpha = (X_*^T X_* + \alpha I)^{-1}$$
       $$M = X_*^T (I - \hat{\rho}^2 \Delta) X_* + X_*^T \Delta W_{\text{obs}} V_1 W_{\text{obs}}^T \Delta X_*$$
       $$V_{\text{ridge}} = \hat{\sigma}^2 Q_\alpha M Q_\alpha$$
     - Replaces NaNs with analytical sandwich standard errors under Ridge fallback, while maintaining backward compatibility via `ridge_se_method="nan"`.
2. **Exports in `umbra` and `umbra.imputers`**:
   - Exported `HeckmanCollinearityWarning` in `umbra/imputers/__init__.py` and `umbra/__init__.py`.
3. **Tests**:
   - Updated `tests/test_imputers.py::test_heckman_ridge_fallback_warning_and_nans` to pass `ridge_se_method="nan"` for legacy mode verification.
   - Created `tests/test_heckman_collinearity.py` with 7 unit tests covering:
     - VIF and condition index accuracy on collinear data.
     - `HeckmanCollinearityWarning` emission on severe collinearity.
     - Positive semi-definiteness, symmetry, and finiteness of $V_{\text{ridge}}$.
     - Replacement of NaNs with finite positive SEs under Ridge fallback.
     - Legacy NaN mode backward compatibility.
     - `ridge_alpha` parameter tuning and variance shrinking.
     - scikit-learn `clone()` compatibility and parameter validation.

## Verification
- `pytest tests/test_heckman_collinearity.py -v`: 7/7 passed.
- Full pytest test suite `pytest tests/ -v`: 131/131 passed.
- `mypy umbra/ tests/test_heckman_collinearity.py`: 0 issues found in 20 source files.
- `ruff check` and `ruff format --check`: 0 issues found across 40 files.
