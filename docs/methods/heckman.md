# Method Specification: Heckman Selection Imputer

## 1. Mathematical Definition

Implements James Heckman's (1976, 1979) two-step selection estimator adapted for missing data imputation under Not-Missing-At-Random (MNAR) selection.

### Model Formulation
1. **Selection Equation**:
   $$z_i^* = W_i \gamma + u_i, \quad R_i = \mathbf{1}(z_i^* > 0)$$
   where $R_i = 1$ indicates that outcome $Y_i$ is observed, and $R_i = 0$ indicates that $Y_i$ is missing.
2. **Outcome Equation**:
   $$Y_i = X_i \beta + \epsilon_i, \quad \text{observed only when } R_i = 1$$
3. **Joint Error Distribution**:
   $$\begin{pmatrix} u_i \\ \epsilon_i \end{pmatrix} \sim \mathcal{N}\left( \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} 1 & \rho \sigma_\epsilon \\ \rho \sigma_\epsilon & \sigma_\epsilon^2 \end{pmatrix} \right)$$

### Conditional Expectations
For observed cases ($R_i = 1$):
$$E[Y_i \mid R_i = 1, X_i, W_i] = X_i \beta + \beta_\lambda \lambda_1(W_i \gamma)$$
where $\lambda_1(\eta) = \frac{\phi(\eta)}{\Phi(\eta)}$ is the Inverse Mills Ratio (hazard function) and $\beta_\lambda = \rho \sigma_\epsilon$.

For unobserved / missing cases ($R_i = 0$):
$$E[Y_i \mid R_i = 0, X_i, W_i] = X_i \beta + \beta_\lambda \lambda_0(W_i \gamma)$$
where $\lambda_0(\eta) = -\frac{\phi(\eta)}{1 - \Phi(\eta)} = -\frac{\phi(-\eta)}{\Phi(-\eta)} \le 0$.

### Imputation Rule
For missing entries ($R_i = 0$):
$$\hat{Y}_i = X_i \hat{\beta} + \hat{\beta}_\lambda \lambda_0(W_i \hat{\gamma}) + \tilde{\epsilon}_i$$
where in stochastic mode, $\tilde{\epsilon}_i \sim \mathcal{N}\left(0, \sigma_\epsilon^2 (1 - \rho^2 \delta_0)\right)$ with $\delta_0 = -\lambda_0 (\eta - \lambda_0)$.

---

## 2. Assumptions

1. **Bivariate Normality**: Errors $(u_i, \epsilon_i)$ are jointly Gaussian.
2. **Exclusion Restriction**: $W_i$ contains at least one variable $Z_i$ (shadow variable / instrument) that strongly predicts missingness propensity $R_i$ but does **not** directly determine outcome $Y_i$ conditional on $X_i$ ($Z_i \in W, Z_i \notin X$).
3. **Linearity**: Both selection propensity index and outcome equation are linear in parameters.

---

## 3. Inputs
- `target_cols`: Target column(s) to impute.
- `shadow_cols`: Dict `{target_col: shadow_var}` specifying instruments.
- `stochastic`: Whether to add conditional Gaussian residual variance.
- `n_imputations`: Number of stochastic draws ($M$).
- `random_state`: Reproducibility seed.

---

## 4. Outputs
- Imputed DataFrame (or List of DataFrames if $M > 1$).
- Fitted parameters: `gamma`, `beta`, `beta_lambda`, `rho`, `sigma_eps`, standard errors.

---

## 5. Failure Modes & Limitations
- **Absence of Exclusion Restriction ($W = X$)**:
  When no shadow variable is available, identification relies entirely on the non-linearity of the Inverse Mills Ratio. In linear ranges of the Probit function, $\lambda_1(W\gamma)$ is collinear with $W$, resulting in severe variance inflation, erratic coefficient estimates, and numerical instability.
- **Violation of Normality**: If true errors are heavy-tailed or bimodal, the two-step estimator is biased.
- **Weak Instruments**: If $Z$ has $F < 10$ in the first stage, the second-stage estimator suffers from finite-sample bias.

---

## 6. Numerical Considerations
- Inverse Mills Ratio is computed in log-space for extreme values ($|\eta| > 10$) using `stats.norm.logpdf - stats.norm.logcdf` to prevent divide-by-zero or numerical overflow.
- Tikhonov regularization is applied to OLS design matrices when condition number $> 10^5$.

---

## 7. Interpretation Guidance
- $\hat{\rho} > 0$: Units with unobservably higher outcomes are more likely to respond.
- $\hat{\rho} < 0$: Units with unobservably higher outcomes are more likely to be missing (self-censoring).
- $\hat{\rho} \approx 0$: Missingness is compatible with MAR; Heckman reduces to OLS.

---

## 8. References
- Heckman, J. J. (1976). "The common structure of statistical models of truncation, sample selection and limited dependent variables." *Annals of Economic and Social Measurement*, 5(4), 475-492.
- Heckman, J. J. (1979). "Sample selection bias as a specification error." *Econometrica*, 47(1), 153-161.
