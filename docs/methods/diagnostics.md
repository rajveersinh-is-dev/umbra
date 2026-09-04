# Method Specification: Umbra Diagnostic Battery

## 1. Overview
The Umbra diagnostic battery provides multi-signal empirical screening to quantify evidence consistent with departures from MCAR and MAR.

---

## 2. Little's (1988) MCAR Test

### 1. Mathematical Definition
Tests the global multivariate hypothesis $H_0: \text{Data are MCAR}$.
$$d^2 = \sum_{j=1}^J N_j (\bar{y}_{\text{obs}, j} - \hat{\mu}_{\text{obs}, j})' \hat{\Sigma}_{\text{obs}, j}^{-1} (\bar{y}_{\text{obs}, j} - \hat{\mu}_{\text{obs}, j})$$
Under $H_0$:
$$d^2 \sim \chi^2(df), \quad df = \sum_{j=1}^J p_j - p$$

### 2. Assumptions
- Data are generated from a multivariate normal distribution.
- Missingness pattern counts $N_j$ are sufficiently large for asymptotic chi-square approximation.

### 3. Inputs
- `data`: Numeric matrix or DataFrame $X \in \mathbb{R}^{n \times p}$ with missing values.
- `alpha`: Significance threshold (default 0.05).
- `ridge_reg`: Tikhonov regularization parameter $\lambda$ (default $10^{-4}$).

### 4. Outputs
- `LittleMCARResult` containing `statistic`, `p_value`, `degrees_of_freedom`, `is_rejected`, `pattern_details`.

### 5. Failure Modes
- Non-normality (skewed or heavy-tailed data): Can inflate test statistics, leading to over-rejection.
- Zero degrees of freedom ($df \le 0$): Occurs when there are no overidentifying restrictions.
- Highly sparse patterns: Singular covariance submatrices.

### 6. Numerical Considerations
- Regularized EM algorithm with eigenvalue clipping ensures positive semi-definiteness.
- Moore-Penrose pseudo-inverse fallback for ill-conditioned covariance sub-blocks.

### 7. Interpretation Guidance
- Rejection ($p < \alpha$) indicates non-random missingness (MAR or MNAR). It **cannot** distinguish MAR from MNAR.

### 8. References
- Little, R. J. A. (1988). "A test of missing completely at random for multivariate data with missing values." *JASA*, 83(404), 1198-1202.

---

## 3. Covariate Distribution Shift Analysis

### 1. Mathematical Definition
For each incomplete variable $Y$ and observed covariate $X_k$, compares $F(X_k \mid R_Y = 1)$ vs. $F(X_k \mid R_Y = 0)$:
- Two-sample Kolmogorov-Smirnov statistic:
  $$D = \sup_x |F_{\text{obs}}(x) - F_{\text{mis}}(x)|$$
- Standardized effect size (Cohen's $d$):
  $$d = \frac{\bar{X}_{\text{mis}} - \bar{X}_{\text{obs}}}{s_{\text{pooled}}}$$
- Non-parametric Cliff's delta:
  $$\delta = \frac{2U}{n_{\text{obs}} n_{\text{mis}}} - 1$$

### 2. Assumptions
- Independent observations across sample rows.

### 3. Inputs
- Dataset $X$, significance level $\alpha$.

### 4. Outputs
- `PatternAnalysisReport` containing per-covariate shift metrics and co-missingness correlation matrix.

### 5. Failure Modes
- Very small group sizes ($n_{\text{mis}} < 5$ or $n_{\text{obs}} < 5$): Low statistical power.
- Zero-variance covariates: Handled by returning 0 effect size and non-significance.

### 6. Numerical Considerations
- Pooled variance safeguarded with $\epsilon = 10^{-12}$.

### 7. Interpretation Guidance
- Significant shifts indicate missingness is systematically correlated with observed features (evidence against MCAR, compatible with MAR).

---

## 4. Residual Tail Dependency / Self-Censoring

### 1. Mathematical Definition
Regresses observed cases of target variable $Y$ on covariates $X$ via Ridge regression: $\hat{Y} = X \hat{\beta}$.
Discretizes predicted values into quantiles and evaluates the concentration ratio:
$$\text{Tail Ratio} = \frac{\max_q P(R_Y = 0 \mid \hat{Y} \in Q_q)}{P(R_Y = 0)}$$

### 2. Assumptions
- Linear relationship between observed covariates and target.

### 3. Interpretation Guidance
- High tail concentration (ratio $> 1.4$) indicates that missingness concentrates sharply at extreme predicted values (a characteristic signature of self-censoring).
