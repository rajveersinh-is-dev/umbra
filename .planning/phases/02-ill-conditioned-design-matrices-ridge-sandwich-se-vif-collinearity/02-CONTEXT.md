# Phase 2: Ill-Conditioned Design Matrices (Ridge Sandwich SE & VIF Collinearity) - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning
**Mode:** Autonomous

<domain>
## Phase Boundary

This phase addresses Limitation #3 in `docs/limitations.md`. It resolves parametric uncertainty quantification when Heckman second-stage design matrices are near-singular or multicollinear with the inverse Mills ratio $\lambda_1$.
Delivers:
1. Exact analytical regularized Ridge sandwich covariance matrix ($V_{\text{ridge}} = \hat{\sigma}^2 Q_\alpha M Q_\alpha$) replacing NaN standard errors when Ridge fallback is triggered (`HECK-03`).
2. Variance Inflation Factors (VIF) and Condition Indices for the augmented design matrix $[X_* \quad \lambda_1]$ with automated `CollinearityWarning` when severe collinearity is detected (`HECK-04`).

</domain>

<decisions>
## Implementation Decisions

### Decision 1: Ridge Sandwich Covariance Matrix Formulation (D-08)
- When second-stage OLS design matrix $[X_* \quad \lambda_1]$ is near-singular or rank-deficient, an $L_2$-regularized Ridge estimator is fitted:
  $$\hat{\beta}_{\text{ridge}} = (X_*^T X_* + \alpha I)^{-1} X_*^T y$$
  where $\alpha > 0$ (default $1.0$, configurable via `ridge_alpha`).
- Compute the sandwich covariance matrix:
  $$Q_\alpha = (X_*^T X_* + \alpha I)^{-1}$$
  $$M = X_*^T (I - \hat{\rho}^2 \Delta) X_* + X_*^T \Delta W_{\text{obs}} V_1 W_{\text{obs}}^T \Delta X_*$$
  $$V_{\text{ridge}} = \hat{\sigma}^2 Q_\alpha M Q_\alpha$$
- Standard errors: $\text{se}_{\text{ridge}} = \sqrt{\max(10^{-16}, \text{diag}(V_{\text{ridge}}))}$.
- Supports `ridge_se: Literal["sandwich", "nan"] = "sandwich"` so users (or backward-compatible tests) can explicitly choose between analytical sandwich standard errors and legacy NaN markers.
- Emit `HeckmanSEWarning` informing the user that regularized Ridge estimation with sandwich standard errors was used.

### Decision 2: Collinearity Diagnostics (VIF & Condition Index) (D-09)
- For each column $j$ in $X_*$ (including $\lambda_1$):
  $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
  via auxiliary regression of column $j$ on remaining columns.
- Normalize columns of $X_*$ to unit Euclidean norm and compute singular values $s_1 \ge \dots \ge s_p$.
  Condition number: $\kappa = s_1 / s_p$.
  Condition indices: $\eta_j = s_1 / s_j$.
- If condition number $\kappa > 30$ or $\text{VIF}_{\lambda} > 10$, emit `HeckmanCollinearityWarning(UserWarning)`.
- Expose collinearity diagnostics dictionary in `self.models_[target]["collinearity_diagnostics"]` and `self.collinearity_diagnostics_` on the imputer instance.

### Decision 3: Custom Warnings & Exports (D-10)
- Add `HeckmanCollinearityWarning(UserWarning)` to `umbra.imputers.heckman_selection`.
- Export `HeckmanCollinearityWarning` in `umbra.imputers` and `umbra`.

</decisions>

<canonical_refs>
## Canonical References
- `docs/limitations.md` §3 — Near-Singular Design Matrices and Ridge Fallback Uncertainty
- White, H. (1982), Maximum Likelihood Estimation of Misspecified Models, Econometrica 50(1): 1–25.
- Belsley, D. A., Kuh, E., & Welsch, R. E. (1980), Regression Diagnostics: Identifying Influential Data and Sources of Collinearity, John Wiley & Sons.
- Murphy, K. M., & Topel, R. H. (1985), Estimation and Inference with Two-Step Econometric Estimators, JBES 3(4): 370–379.
</canonical_refs>
