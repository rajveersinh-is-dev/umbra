# Missing at Random (MAR)

## 1. Mathematical Definition

Data are **Missing at Random (MAR)** when conditional on all observed variables $X_{\text{obs}}$, missingness is statistically independent of the unobserved missing values $X_{\text{mis}}$:

$$P(M \mid X_{\text{obs}}, X_{\text{mis}}, \psi) = P(M \mid X_{\text{obs}}, \psi) \quad \forall X_{\text{mis}}$$

This means that while missingness is systematic and non-random across the population, all systematic factors governing non-response have been captured and measured in $X_{\text{obs}}$.

---

## 2. Statistical Consequences

- **Conditional Equivalence**: Units with identical observed covariate profiles share the same distribution of missing values, regardless of whether they responded or not:
  $$P(X_{\text{mis}} \mid X_{\text{obs}}, M = 1) = P(X_{\text{mis}} \mid X_{\text{obs}}, M = 0)$$
- **Ignorability**: If the substantive parameters $\theta$ and missingness parameters $\psi$ are functionally independent (parameter distinctness), valid likelihood inference does not require specifying a joint model for $M$.
- **Complete-Case Bias**: Complete-case analysis is generally biased under MAR unless the probability of missingness depends only on the independent variables in an outcome regression.
- **Validity of Standard MICE**: Multiple Imputation by Chained Equations (MICE) using observed covariates yields consistent estimators of parameters and correct asymptotic coverage when the imputation models are correctly specified.

---

## 3. Empirical Diagnostics in Umbra

While true MAR cannot be confirmed definitively against MNAR without untestable assumptions, Umbra tests for evidence consistent with MAR through:
1. **Covariate Distribution Shifts**: Two-sample Kolmogorov-Smirnov (KS) tests and standardized effect sizes (Cohen's $d$, Cliff's $\delta$) comparing $X_{\text{obs}}$ when target variable $Y$ is observed vs. missing.
2. **Predictive Sufficiency**: Assessing whether observed covariates $X_{\text{obs}}$ predict missingness propensity with low residual tail concentration.
