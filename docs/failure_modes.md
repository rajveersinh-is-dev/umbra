# Negative Results & Methodological Failure Modes

A scientific framework is defined as much by its boundaries and failure modes as by its strengths. This document details known edge cases, statistical regimes where diagnostics break down, and algorithmic failure modes in missing-data analysis.

---

## 1. Regimes Where Diagnostics Fail or Conflict

### 1.1 Symmetric U-Shaped Tail Dropout
- **Mechanism**: Missingness occurs with highest probability in both extreme tails of an unobserved distribution (e.g., $P(M=1 \mid |Y| > 2) = 0.8$, but $P(M=1 \mid |Y| \le 2) = 0.1$).
- **Diagnostic Behavior**:
  - The observed mean $\mathbb{E}[Y_{\text{obs}}]$ approximately equals the complete-data mean $\mathbb{E}[Y]$.
  - First-order covariate shift tests (Student's $t$, standard directional regression) show negligible linear shift ($\Delta \mu \approx 0$).
  - Two-sample Kolmogorov-Smirnov (KS) tests and variance ratio tests can detect dispersion differences, but directional MNAR risk scoring will under-score the risk if only mean shifts are evaluated.
- **Umbra Mitigation**: Umbra combines KS maximum empirical distribution divergence with tail concentration tests ($\kappa_4$, excess kurtosis) rather than relying purely on linear mean differences.

### 1.2 Underpowered MCAR Tests in High Dimensions ($p > n$)
- **Mechanism**: In high-dimensional settings where $p$ approaches or exceeds $n$, the number of distinct missingness patterns $J$ escalates exponentially, while pattern sample sizes $n_j$ become sparse ($n_j = 1$ or $2$).
- **Diagnostic Behavior**:
  - The sample covariance matrix $\hat{\Sigma}$ becomes singular or rank-deficient.
  - Little's (1988) test statistic $d^2 = \sum_j n_j (\bar{y}_j - \hat{\mu})^T \hat{\Sigma}_j^{-1} (\bar{y}_j - \hat{\mu})$ involves inverses of singular sub-matrices.
- **Umbra Mitigation**: Umbra applies Ledoit-Wolf optimal shrinkage and eigenvalue clipping ($\epsilon = 10^{-6} I$). However, when $p \gg n$, the asymptotic $\chi^2(\sum p_j - p)$ null distribution is poor, leading to potential Type I or Type II error distortion. Users are warned when $n / p < 5$.

### 1.3 Conflicting Diagnostic Signals (Rejection of MCAR without Instrument)
- **Scenario**: Little's test strongly rejects MCAR ($p < 10^{-6}$), and KS distance indicates substantial observable covariate shift ($d_{KS} > 0.35$), but the Shadow Variable Finder discovers **no** candidate instrument with $F > 10$.
- **Methodological Conflict**:
  - The data are definitively not MCAR.
  - Is the mechanism MAR (missingness fully explained by observed covariates) or MNAR (missingness additionally driven by the unobserved values)?
- **Scientific Resolution**:
  - By the Molenberghs et al. (2008) non-identifiability theorem, this question **cannot be resolved from observed data alone**.
  - **Auto Router Policy**: The router refuses to guess. It defaults to MAR (MICE) point estimation with Rubin's pooled uncertainty, and automatically executes a **Pattern-Mixture Tipping Point Analysis** across $\delta \in [-1.5, +1.5]$ standard deviations.

---

## 2. Estimator Breakdown Scenarios

### 2.1 Weak Instruments in Selection Models ($F < 10$)
- **Mechanism**: An auxiliary variable $Z$ is selected that has only weak empirical correlation with the missingness propensity $M$ (first-stage $F$-statistic $< 10$).
- **Consequences**:
  - In Heckman's two-step procedure, the predicted Inverse Mills Ratio $\hat{\lambda}_i = \phi(Z_i\hat{\gamma}) / \Phi(Z_i\hat{\gamma})$ is collinear with the intercept and linear regressors $X_i$.
  - The second-stage regression suffers from severe variance inflation:
    $$\text{Var}(\hat{\beta}) \propto \frac{1}{1 - R^2_{\lambda, X}}$$
  - Point estimates become wildly unstable, frequently flipping sign relative to naive estimators.
- **Umbra Safeguards**:
  - Umbra strictly checks the Stock-Yogo (2005) threshold $F > 10$ before allowing Heckman selection.
  - If $F \le 10$, Umbra emits a `WeakInstrumentWarning`, rejects the instrument, and falls back to MAR + sensitivity bounds.

### 2.2 Violation of Bivariate Normality in Heckman Selection
- **Mechanism**: The joint distribution of $(u_1, u_2)$ departs substantially from bivariate Gaussian (e.g., heavy-tailed Student-$t$, skewed Cauchy, or multi-modal errors).
- **Consequences**:
  - The functional form of the Inverse Mills Ratio $\lambda(\cdot)$ is derived strictly from the normal CDF and PDF. Under non-normal errors, $\mathbb{E}[u_2 \mid u_1 > -Z\gamma] \neq \rho \sigma_2 \lambda(Z\gamma)$.
  - As demonstrated by Manski (1989) and Little (1985), Heckman point estimates under non-normal errors can exhibit **higher asymptotic bias** than naive complete-case estimators.
- **Umbra Recommendation**: Always contrast Heckman estimates with non-parametric pattern-mixture tipping point intervals.

### 2.3 Boundary Solutions & Collapse in Deep Generative MNAR
- **Mechanism**: Variational autoencoders modeling joint $p_\theta(x, m \mid z)$ trained on small datasets ($n < 500$) or extreme missingness rates ($> 60\%$).
- **Consequences**:
  - Posterior collapse: The latent representation $q_\phi(z \mid x_{\text{obs}}, m)$ collapses to the uninformative prior $\mathcal{N}(0, I)$, ignoring $m$.
  - Generative hallucination: Missing values are imputed with high variance and unrealistic tail predictions.
- **Umbra Recommendation**: Deep generative MNAR is restricted to large-sample exploratory settings ($n \ge 1000$). For small tabular datasets, pattern-mixture models are strictly preferred.

---

## 3. Negative Results Summary Table

| Experimental Scenario | What Fails | Symptoms Observed | Recommended Scientific Practice |
| :--- | :--- | :--- | :--- |
| Symmetric tail dropout | Directional mean shift | Linear $\Delta \mu \approx 0$, false MAR confidence | Inspect KS distance and variance ratio; do not assume MCAR |
| Ultra-high dimensions ($p > n$) | Little's MCAR test | Inversion instability, inflated test statistics | Apply Ledoit-Wolf shrinkage; rely primarily on pairwise tests |
| Weak instrument ($F < 10$) | Heckman selection | Variance inflation, sign flipping of $\hat{\beta}$ | Reject instrument; transition to pattern-mixture sensitivity |
| Non-normal selection errors | Heckman selection | Asymptotic bias worse than OLS | Audit residual normality; prioritize sensitivity analysis |
| Extreme missingness ($> 70\%$) | All point estimators | Complete loss of statistical power | Report bounds only; acknowledge identification collapse |
| Small sample ($n < 50$) | Rubin's rules DF | Negative or fractional degrees of freedom | Use Barnard-Rubin finite-sample adjustment |
