# Phase 2: Ill-Conditioned Design Matrices (Ridge Sandwich SE & VIF Collinearity) - Research

**Researched:** 2026-09-09
**Domain:** Tikhonov Regularization, Sandwich Covariance Estimators, Collinearity Diagnostics (VIF, Condition Index)
**Confidence:** HIGH

<research_summary>
## Summary

Phase 2 resolves Limitation #3 in `docs/limitations.md`. When the Heckman second stage has collinear covariates or sample size is small, second-stage OLS design matrix $X_* = [1 \quad X \quad \lambda_1]$ can become rank-deficient or ill-conditioned. In v0.2.0, an $L_2$ Ridge regression fallback was used to prevent unhandled runtime errors, but parameter standard errors were set to NaN.

### 1. Ridge Sandwich Covariance Matrix
For Ridge objective $\min_\beta \frac{1}{2} \|y - X_* \beta\|^2 + \frac{\alpha}{2} \|\beta\|^2$:
The estimator is $\hat{\beta}_{\text{ridge}} = (X_*^T X_* + \alpha I)^{-1} X_*^T y = Q_\alpha X_*^T y$ where $Q_\alpha = (X_*^T X_* + \alpha I)^{-1}$.
Under the Heckman second stage, the asymptotic covariance taking into account the heteroskedasticity and first-stage generated regressor variance $M = X_*^T (I - \hat{\rho}^2 \Delta) X_* + X_*^T \Delta W_{\text{obs}} V_1 W_{\text{obs}}^T \Delta X_*$ is:
$$V_{\text{ridge}} = \hat{\sigma}^2 Q_\alpha M Q_\alpha$$
Properties:
- Strictly positive semi-definite (since $M$ is positive semi-definite and $Q_\alpha$ is strictly positive definite for any $\alpha > 0$).
- When $\alpha \to 0$, $V_{\text{ridge}} \to V_2$ (the Murphy-Topel covariance matrix).
- For ill-conditioned $X_*$, $Q_\alpha$ regularizes the inversion, yielding well-conditioned standard errors.

### 2. Collinearity Diagnostics (VIF and Condition Index)
- **Variance Inflation Factors (VIF)**:
  For each predictor $j = 1, \dots, p$ in $X_*$ (including $\lambda_1$):
  Regress $X_{*, j}$ on the remaining columns of $X_*$. Let $R_j^2$ be the coefficient of determination.
  $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
  A VIF $> 10$ indicates severe collinearity.
- **Condition Index**:
  Standardize columns of $X_*$ to have unit Euclidean norm ($x_j / \|x_j\|_2$).
  Compute singular value decomposition $U S V^T$.
  Condition number: $\kappa = s_{\max} / s_{\min}$.
  Condition indices: $\eta_j = s_{\max} / s_j$.
  Belsley et al. (1980) criterion: $\kappa > 30$ indicates moderate-to-severe collinearity; $\kappa > 100$ indicates extreme collinearity.
- Emits `HeckmanCollinearityWarning` when $\kappa > 30$ or $\text{VIF}_{\lambda} > 10$.
</research_summary>
