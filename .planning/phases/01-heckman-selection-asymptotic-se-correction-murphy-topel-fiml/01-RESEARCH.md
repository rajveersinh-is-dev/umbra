# Phase 1: Heckman Selection Asymptotic SE Correction (Murphy-Topel & FIML) - Research

**Researched:** 2026-09-09
**Domain:** Econometric Selection Models, Asymptotic M-Estimation, Maximum Likelihood Optimization
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01 / D-02:** Implement analytical Murphy & Topel (1985) asymptotic covariance matrix formula for two-step Heckman estimation:
  $$\text{Var}(\hat{\beta}) = \sigma_2^2 (X_*^T X_*)^{-1} \left[ X_*^T (I - \rho^2 \Delta) X_* + X_*^T \Delta W \text{Var}(\hat{\gamma}) W^T \Delta X_* \right] (X_*^T X_*)^{-1}$$
  Make it the default standard error stored on `self.coef_stderr_` when `n_bootstrap_se == 0`, retiring uncorrected naive OLS variance.
- **D-03 / D-04 / D-05:** Expose parameter `method: Literal["two-step", "fiml"] = "two-step"`. For FIML, initialize with two-step parameter estimates, optimize the joint bivariate normal log-likelihood via SciPy (`L-BFGS-B` or `BFGS` on unconstrained $(\beta, \gamma, \ln \sigma, \text{atanh}(\rho))$), invert the observed Hessian for asymptotic standard errors, and gracefully fall back to two-step with Murphy-Topel standard errors if optimization fails to converge.
- **D-06:** Deterministic conditional mean imputation $E[Y_i \mid S_i=0, X_i, W_i]$ by default; support stochastic perturbation with conditional selection variance.
- **D-07:** Non-linear identification without instruments supported with collinearity warning when condition number is high.

### the agent's Discretion
- Matrix formulation optimizations, vectorization via NumPy, gradient derivations, and optimizer tolerances.

### Deferred Ideas (OUT OF SCOPE)
- Non-parametric copula selection models (ECON-02) deferred to v2.
</user_constraints>

<architectural_responsibility_map>
## Architectural Responsibility Map

Single-tier library module — all capabilities reside in `umbra.imputers.heckman_selection`.
</architectural_responsibility_map>

<research_summary>
## Summary

This phase resolves Limitation #2 in `docs/limitations.md`. The Heckman (1979) two-step estimator includes a generated regressor—the inverse Mills ratio $\hat{\lambda}_i = \lambda(w_i \hat{\gamma})$—from the first-stage Probit selection equation into the second-stage outcome regression. Naive OLS standard errors from the second stage underestimate sampling variability by assuming homoskedastic errors and ignoring the estimation variance of $\hat{\gamma}$.

We implement the exact closed-form asymptotic covariance matrix derived by Murphy & Topel (1985), Greene (2003), and Cameron & Trivedi (2005). In addition, we implement Full-Information Maximum Likelihood (FIML) using unconstrained reparameterization $(\ln \sigma, \text{atanh}(\rho))$ to guarantee admissible variance and correlation parameters during optimization.

### Mathematical Foundations

#### 1. Murphy-Topel Two-Step Covariance Formula
Let the observed sample size be $n_1$.
Let $X_* = [X \quad \lambda]$ be the $n_1 \times (k + 1)$ design matrix for the second stage.
Let $W$ be the $n_1 \times m$ selection equation design matrix for the observed units.
Let $\hat{\gamma}$ be the Probit estimates with estimated covariance $V_1 = \text{Var}(\hat{\gamma}) = (W_{\text{all}}^T \hat{D} W_{\text{all}})^{-1}$.
For each observed unit $i$:
$$\lambda_i = \frac{\phi(w_i \hat{\gamma})}{\Phi(w_i \hat{\gamma})}, \quad \delta_i = \lambda_i (\lambda_i + w_i \hat{\gamma}) \in (0, 1)$$
Residuals: $\hat{\nu}_i = y_i - x_{*i} \hat{\beta}_*$, where $\hat{\beta}_* = (\hat{\beta}^T, \hat{\beta}_\lambda)^T$.
Second-stage variance adjustment:
$$\hat{\sigma}^2 = \frac{1}{n_1} \sum_{i=1}^{n_1} \hat{\nu}_i^2 + \hat{\beta}_\lambda^2 \bar{\delta}, \quad \text{where } \bar{\delta} = \frac{1}{n_1} \sum_{i=1}^{n_1} \delta_i$$
$$\hat{\rho} = \frac{\hat{\beta}_\lambda}{\hat{\sigma}}, \quad \text{clipped to } [-0.999, 0.999]$$
Let $\Delta = \text{diag}(\delta_1, \dots, \delta_{n_1})$. The asymptotic covariance matrix of $\hat{\beta}_*$ is:
$$V_2 = \hat{\sigma}^2 (X_*^T X_*)^{-1} \left[ X_*^T (I - \hat{\rho}^2 \Delta) X_* + X_*^T \Delta W V_1 W^T \Delta X_* \right] (X_*^T X_*)^{-1}$$
The standard errors are the square roots of the diagonal elements of $V_2$.

#### 2. Full-Information Maximum Likelihood (FIML)
The full log-likelihood for sample $(y_i, s_i)_{i=1}^N$ where $s_i = 1$ indicates observed $y_i$:
$$\ln L = \sum_{i: s_i=0} \ln(1 - \Phi(w_i \gamma)) + \sum_{i: s_i=1} \left[ -\frac{1}{2}\ln(2\pi) - \ln \sigma - \frac{(y_i - x_i \beta)^2}{2\sigma^2} + \ln \Phi\left(\frac{w_i \gamma + \frac{\rho}{\sigma}(y_i - x_i \beta)}{\sqrt{1 - \rho^2}}\right) \right]$$
To enforce $\sigma > 0$ and $\rho \in (-1, 1)$, we define parameter vector:
$$\theta = (\beta, \gamma, \tau, \zeta) \in \mathbb{R}^{k + m + 2}$$
where $\sigma = \exp(\tau)$ and $\rho = \tanh(\zeta)$.
The gradient $\nabla_\theta \ln L$ can be computed analytically or numerically, and the asymptotic parameter covariance is obtained via the inverted Hessian $(-H)^{-1}$. Standard errors of $\sigma$ and $\rho$ follow via the Delta method:
$$\text{SE}(\hat{\sigma}) = \hat{\sigma} \cdot \text{SE}(\hat{\tau}), \quad \text{SE}(\hat{\rho}) = (1 - \hat{\rho}^2) \cdot \text{SE}(\hat{\zeta})$$

</research_summary>

<validation_architecture>
## Validation Architecture

### Verification Plan
1. **Murphy-Topel Numerical Correctness**:
   - Compare analytical standard errors against 200-iteration paired bootstrap on simulated selection DGP.
   - Verify that Murphy-Topel standard errors are strictly positive and greater than naive OLS standard errors (reflecting the first-stage penalty).
2. **FIML vs Two-Step Consistency**:
   - Fit `HeckmanSelectionImputer(method="fiml")` on benchmark selection DGP.
   - Verify parameter estimates $\hat{\beta}_{\text{fiml}}$ are within 1.96 standard errors of two-step estimates and true DGP coefficients.
3. **Graceful Fallback & Warnings**:
   - On collinear or near-singular synthetic input, verify that FIML emits `HeckmanConvergenceWarning` and falls back to two-step with Murphy-Topel SE.
4. **Scikit-Learn Native API Verification**:
   - Test `fit`, `transform`, `get_feature_names_out`, `check_is_fitted`, DataFrame vs NumPy inputs.
   - Run `pytest tests/ -k heckman` and full test suite.
</validation_architecture>
