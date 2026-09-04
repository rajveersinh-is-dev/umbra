# Missing Completely at Random (MCAR)

## 1. Mathematical Definition

Data are **Missing Completely at Random (MCAR)** when the probability of missingness is completely independent of both observed values $X_{\text{obs}}$ and unobserved values $X_{\text{mis}}$:

$$P(M \mid X_{\text{obs}}, X_{\text{mis}}, \psi) = P(M \mid \psi)$$

In graphical causal models, this corresponds to a graph where the missingness indicator node $M$ has no incoming edges from any variable in the substantive model.

---

## 2. Statistical Consequences

- **Unbiased Subsampling**: Observed units constitute a pure random sub-sample of the population.
- **Complete-Case Consistency**: Sample means, variances, and regression coefficients computed from complete cases only are consistent estimators of population parameters.
- **Efficiency Loss**: Discarding incomplete records reduces effective sample size, leading to wider confidence intervals and reduced statistical power.
- **Covariate Balance**: The marginal and joint distributions of observed covariates in rows where $X_j$ is missing are statistically identical to rows where $X_j$ is observed.

---

## 3. Diagnostic Testing: Little's (1988) Test

Little's multivariate test is the canonical test for the MCAR hypothesis under the assumption of multivariate normality.

### Null and Alternative Hypotheses
$$H_0: \text{Data are MCAR} \quad \text{vs.} \quad H_1: \text{Data are Not MCAR (MAR or MNAR)}$$

### Test Statistic
Let $J$ be the number of distinct missingness patterns observed, and let $N_j$ be the count of observations in pattern $j$.
For pattern $j$, let $p_j$ be the number of observed variables, $\bar{y}_{\text{obs}, j}$ be the sample mean vector, and $\hat{\mu}_{\text{obs}, j}, \hat{\Sigma}_{\text{obs}, j}$ be the subvector and submatrix of the Expectation-Maximization (EM) maximum likelihood estimators.

The test statistic is:
$$d^2 = \sum_{j=1}^J N_j (\bar{y}_{\text{obs}, j} - \hat{\mu}_{\text{obs}, j})' \hat{\Sigma}_{\text{obs}, j}^{-1} (\bar{y}_{\text{obs}, j} - \hat{\mu}_{\text{obs}, j})$$

Under $H_0$:
$$d^2 \sim \chi^2(df), \quad df = \sum_{j=1}^J p_j - p$$
where $p$ is the total number of evaluated numeric features.

### Interpretation Guidance
- **$p \ge \alpha$ (Fail to Reject)**: Observed patterns are statistically compatible with MCAR. Standard MAR methods (like MICE) or complete-case methods do not suffer from selection bias.
- **$p < \alpha$ (Reject)**: There is statistically significant evidence of non-random missingness. The data are either MAR or MNAR. Crucially, Little's test cannot distinguish whether the departure is MAR or MNAR.
