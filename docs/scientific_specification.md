# Scientific Specification & Mathematical Foundations of Umbra

## 1. Mathematical Notation & Framework

Let the complete dataset be represented by the random matrix $Y \in \mathbb{R}^{n \times p}$, comprising $n$ independent and identically distributed realizations of a $p$-dimensional random vector $Y_i = (Y_{i1}, \dots, Y_{ip})^T$.

Partition each record $Y_i$ into its observed and unobserved components:
$$Y_i = (Y_{i, \text{obs}}, Y_{i, \text{mis}})$$
where $Y_{i, \text{obs}} = \{Y_{ij} : M_{ij} = 0\}$ and $Y_{i, \text{mis}} = \{Y_{ij} : M_{ij} = 1\}$.

The missingness indicator matrix $M \in \{0, 1\}^{n \times p}$ is defined elementwise by:
$$M_{ij} = \begin{cases} 1 & \text{if } Y_{ij} \text{ is missing (unobserved)} \\ 0 & \text{if } Y_{ij} \text{ is observed} \end{cases}$$

The joint distribution of the complete data $Y$ and missingness indicators $M$ is governed by parameters $(\theta, \psi) \in \Theta \times \Psi$:
$$P(Y, M \mid \theta, \psi)$$

This joint distribution can be factorized according to two complementary paradigms:

1. **Selection Model Formulation** (Heckman, 1979; Little & Rubin, 2019):
   $$P(Y, M \mid \theta, \psi) = P(Y \mid \theta) \, P(M \mid Y, \psi)$$
   where $P(Y \mid \theta)$ specifies the scientific data-generating process and $P(M \mid Y, \psi)$ specifies the missingness/selection mechanism.

2. **Pattern-Mixture Model Formulation** (Little, 1993):
   $$P(Y, M \mid \phi, \omega) = P(M \mid \phi) \, P(Y \mid M, \omega)$$
   where $P(M \mid \phi)$ models the marginal probability of missingness patterns and $P(Y \mid M, \omega)$ specifies distinct conditional distributions across pattern strata.

---

## 2. Formal Definitions of Missingness Mechanisms

Following the foundational taxonomy of Donald Rubin (1976):

### 2.1 Missing Completely at Random (MCAR)
Missingness is independent of both observed and unobserved values of the data matrix:
$$P(M \mid Y_{\text{obs}}, Y_{\text{mis}}, \psi) = P(M \mid \psi) \quad \forall Y_{\text{obs}}, Y_{\text{mis}}, \psi$$

**Empirical Consequences:**
- The observed sample $Y_{\text{obs}}$ constitutes a uniform, representative random subsample of the target population.
- Complete-case analysis is statistically unbiased for first-order moments $\mathbb{E}[Y]$, although it incurs severe efficiency loss.
- Pairwise covariances and correlation matrices remain asymptotically unbiased.

### 2.2 Missing at Random (MAR)
Missingness conditionally depends on observed data $Y_{\text{obs}}$, but is conditionally independent of unobserved data $Y_{\text{mis}}$:
$$P(M \mid Y_{\text{obs}}, Y_{\text{mis}}, \psi) = P(M \mid Y_{\text{obs}}, \psi) \quad \forall Y_{\text{mis}}, \psi$$

**Empirical Consequences:**
- Missingness is ignorable for likelihood-based and Bayesian inference, provided the parameter spaces $\Theta$ and $\Psi$ are distinct (Rubin, 1976).
- Complete-case estimators are biased ($\mathbb{E}[Y_{\text{obs}}] \neq \mathbb{E}[Y]$), but condition-specific weighting (IPW) or conditional mean imputation (e.g., MICE) completely removes bias.

### 2.3 Missing Not at Random (MNAR)
Missingness depends directly on the unobserved values themselves, even after conditioning on all available observed covariates:
$$\exists Y_{\text{mis}} \neq Y'_{\text{mis}} \quad \text{s.t.} \quad P(M \mid Y_{\text{obs}}, Y_{\text{mis}}, \psi) \neq P(M \mid Y_{\text{obs}}, Y'_{\text{mis}}, \psi)$$

**Empirical Consequences:**
- Missingness is non-ignorable.
- Standard estimators (complete-case, mean imputation, standard MICE, standard matrix completion) suffer from persistent, non-vanishing asymptotic bias.
- Point estimation requires untestable parametric assumptions (e.g., bivariate normality in Heckman selection) or explicit sensitivity analyses.

---

## 3. The Fundamental Non-Identifiability Theorem

A foundational principle underpinning Umbra's design is the mathematical impossibility of empirically proving MNAR from observed data alone without unverifiable structural assumptions.

### Theorem (Molenberghs et al., 2008; Robins & Ritov, 1997)
*Let $S = (Y_{\text{obs}}, M)$ denote the observed data. For any non-random missingness model $P_1(Y, M \mid \theta_1, \psi_1)$ (an MNAR model), there exists a missing-at-random model $P_0(Y, M \mid \theta_0, \psi_0)$ (an MAR model) that produces an identical marginal distribution over the observable variables:*
$$P_1(Y_{\text{obs}}, M) \equiv P_0(Y_{\text{obs}}, M) \quad \forall (Y_{\text{obs}}, M)$$

*Proof Sketch:*
Under pattern-mixture factorization:
$$P(Y_{\text{obs}}, Y_{\text{mis}}, M) = P(M) \, P(Y_{\text{obs}} \mid M) \, P(Y_{\text{mis}} \mid Y_{\text{obs}}, M)$$
The observed data identify only:
1. The pattern probabilities: $P(M)$
2. The conditional observed distributions: $P(Y_{\text{obs}} \mid M)$

The conditional distribution of the missing values given the observed data and missingness status:
$$P(Y_{\text{mis}} \mid Y_{\text{obs}}, M = 1)$$
is completely unconstrained by the observable data $S$. Any arbitrary choice of $P^*(Y_{\text{mis}} \mid Y_{\text{obs}}, M = 1)$ can be paired with the empirically identified components to yield a joint distribution whose observable marginal matches $P(Y_{\text{obs}}, M)$ exactly. Choosing $P^*(Y_{\text{mis}} \mid Y_{\text{obs}}, M = 1) = P(Y_{\text{mis}} \mid Y_{\text{obs}}, M = 0)$ yields an MAR model that fits the observed data with identical likelihood to the MNAR model. $\blacksquare$

### Implications for Umbra
1. **No "MNAR Detector"**: Umbra never claims to "detect" or "prove" MNAR.
2. **Diagnostic Framing**: Diagnostics report *observed-data evidence consistent with plausible MNAR mechanisms* (e.g., failure of MCAR, covariate shifts, instrument strength, residual tail concentrations).
3. **Mandatory Sensitivity Analysis**: Because point estimators under MNAR hinge on untestable assumptions, primary scientific inferences must be reported alongside tipping point and parametric sensitivity intervals.

---

## 4. The Epistemic Four-Tier Architecture

To enforce scientific integrity, Umbra partitions all outputs into four distinct epistemic tiers:

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Observed-Data Evidence                              │
│ • Little's MCAR d² statistic, p-value                       │
│ • Two-sample Kolmogorov-Smirnov covariate shift statistics  │
│ • Auxiliary instrument first-stage F-statistic              │
├─────────────────────────────────────────────────────────────┤
│ Tier 2: Model-Based Inferences                              │
│ • Point estimates θ̂ under specified modeling assumptions   │
│ • Rubin's pooled multiple imputation variance Ū + (1+1/M)B  │
├─────────────────────────────────────────────────────────────┤
│ Tier 3: Untestable Structural Assumptions                   │
│ • Bivariate error normality (u₁, u₂) ~ N₂(0, Σ)             │
│ • Exclusion restriction: Z ⊥ Y | (X, u₁)                    │
│ • Shift invariance: E[Y_mis|X] - E[Y_obs|X] = δ             │
├─────────────────────────────────────────────────────────────┤
│ Tier 4: Sensitivity Bounds & Tipping Points                 │
│ • Monotonic curve θ̂(δ) for δ ∈ [δ_min, δ_max]               │
│ • Tipping point δ* where null is crossed or sign flips      │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Algorithmic Specifications

### 5.1 Little's MCAR Test (Exact Formulation)

Let the sample be partitioned into $J$ distinct missingness patterns $S_1, \dots, S_J$, where pattern $j$ contains $n_j$ observations exhibiting an identical set of observed variables $O_j \subseteq \{1, \dots, p\}$ with cardinality $p_j = |O_j|$.

#### Expectation-Maximization (EM) Estimation
Under the null hypothesis $H_0: \text{MCAR}$, the complete data follow a multivariate normal distribution:
$$Y_i \sim \mathcal{N}_p(\mu, \Sigma)$$

1. **E-step**: For each observation $i$ in pattern $j$:
   $$\hat{y}_{ij} = \mathbb{E}[Y_i \mid Y_{i, O_j}, \hat{\mu}^{(t)}, \hat{\Sigma}^{(t)}]$$
   $$\hat{C}_{ij} = \text{Var}[Y_i \mid Y_{i, O_j}, \hat{\mu}^{(t)}, \hat{\Sigma}^{(t)}]$$

2. **M-step**:
   $$\hat{\mu}^{(t+1)} = \frac{1}{n} \sum_{i=1}^n \hat{y}_i$$
   $$\hat{\Sigma}^{(t+1)} = \frac{1}{n} \sum_{i=1}^n \left( (\hat{y}_i - \hat{\mu}^{(t+1)})(\hat{y}_i - \hat{\mu}^{(t+1)})^T + \hat{C}_i \right)$$

#### Regularization
To prevent singular covariance matrices in ill-conditioned or high-dimensional regimes, Umbra applies regularized shrinkage:
$$\hat{\Sigma}_{\text{reg}} = (1 - \lambda) \hat{\Sigma} + \lambda \, \text{diag}(\hat{\Sigma}) + \epsilon I_p$$
where $\lambda \in [0, 1]$ is determined via Ledoit-Wolf shrinkage and $\epsilon = 10^{-6}$.

#### Test Statistic and Degrees of Freedom
Little's test statistic $d^2$ measures the Mahalanobis distance between pattern-specific observed means and the corresponding sub-vectors of the overall ML estimate:
$$d^2 = \sum_{j=1}^J n_j \left( \bar{y}_{\text{obs}, j} - \hat{\mu}_{O_j} \right)^T \hat{\Sigma}_{O_j}^{-1} \left( \bar{y}_{\text{obs}, j} - \hat{\mu}_{O_j} \right)$$

Under $H_0$, Little (1988) proved:
$$d^2 \xrightarrow{d} \chi^2(\text{df})$$
$$\text{df} = \sum_{j=1}^J p_j - p$$
where $p_j = |O_j|$ is the number of observed variables in pattern $j$, and $p$ is the total number of variables.

---

### 5.2 Multiple Imputation via Chained Equations (MICE) & Rubin's Rules

MICE models the joint distribution $P(Y_1, \dots, Y_p \mid \theta)$ via a sequence of iteratively specified univariate conditional distributions:
$$Y_j \mid Y_{-j}, \theta_j \sim f_j(Y_j \mid Y_{-j}, \theta_j)$$

For iteration $t = 1, \dots, T$:
1. Draw parameters $\theta_j^{(t)} \sim P(\theta_j \mid Y_{j, \text{obs}}, Y_{-j}^{(t)})$.
2. Draw imputations $Y_{j, \text{mis}}^{(t)} \sim P(Y_{j, \text{mis}} \mid Y_{-j}^{(t)}, \theta_j^{(t)})$.

#### Rubin's Combination Rules (Rubin, 1987)
For $M$ multiply imputed datasets, let $\hat{Q}_m$ denote the scalar point estimator in dataset $m$, with estimated variance $\hat{U}_m$.

1. **Pooled Point Estimate**:
   $$\bar{Q} = \frac{1}{M} \sum_{m=1}^M \hat{Q}_m$$

2. **Within-Imputation Variance**:
   $$\bar{U} = \frac{1}{M} \sum_{m=1}^M \hat{U}_m$$

3. **Between-Imputation Variance**:
   $$B = \frac{1}{M - 1} \sum_{m=1}^M (\hat{Q}_m - \bar{Q})^2$$

4. **Total Variance**:
   $$T = \bar{U} + \left(1 + \frac{1}{M}\right) B$$

5. **Degrees of Freedom** (Barnard & Rubin, 1999 adjustment for small/moderate samples):
   $$\nu_{\text{old}} = (M - 1) \left( 1 + \frac{\bar{U}}{(1 + M^{-1}) B} \right)^2$$
   $$\nu_{\text{obs}} = \frac{n - k + 1}{n - k + 3} (n - k) (1 - \hat{\gamma})$$
   $$\nu = \frac{\nu_{\text{old}} \nu_{\text{obs}}}{\nu_{\text{old}} + \nu_{\text{obs}}}$$
   where $\hat{\gamma} = \frac{(1 + M^{-1}) B}{T}$ is the fraction of missing information, and $n - k$ is the complete-data degrees of freedom.

---

### 5.3 Heckman Two-Step Selection Estimator

To address sample selection bias under MNAR where an auxiliary instrument is available:

#### Structural Model
1. **Selection Equation** (Latent propensity):
   $$M_i^* = Z_i^T \gamma + u_{1i}$$
   $$M_i = \mathbf{1}(M_i^* > 0)$$
   where $Y_i$ is observed if and only if $M_i = 1$ (or $M_i = 0$ depending on coding; in Umbra, $M_i=1$ denotes missingness).

2. **Outcome Equation**:
   $$Y_i = X_i^T \beta + u_{2i}$$

3. **Joint Error Distribution**:
   $$\begin{pmatrix} u_{1i} \\ u_{2i} \end{pmatrix} \sim \mathcal{N}_2 \left( \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} 1 & \rho \sigma_2 \\ \rho \sigma_2 & \sigma_2^2 \end{pmatrix} \right)$$

#### Two-Step Estimation Procedure
1. **Step 1 (Probit Selection)**:
   Fit a probit model of $M$ on $Z$ via Maximum Likelihood to obtain $\hat{\gamma}$.
   Compute the Inverse Mills Ratio (IMR) $\hat{\lambda}_i$:
   $$\hat{\lambda}_i = \frac{\phi(Z_i^T \hat{\gamma})}{\Phi(Z_i^T \hat{\gamma})} \quad \text{(using log-space computation to prevent float underflow)}$$

2. **Step 2 (Augmented Outcome Regression)**:
   Fit OLS regression on observed cases:
   $$Y_i = X_i^T \beta + \beta_\lambda \hat{\lambda}_i + \varepsilon_i$$
   where $\beta_\lambda = \rho \sigma_2$.

3. **Exclusion Restriction**:
   $Z$ must contain at least one valid instrument $Z_{\text{aux}} \notin X$ such that:
   $$\text{Cov}(Z_{\text{aux}}, M^*) \neq 0 \quad \text{and} \quad \text{Cov}(Z_{\text{aux}}, u_2) = 0$$
   Umbra validates the relevance condition via the first-stage $F$-statistic (Stock-Yogo threshold $F > 10$).

---

### 5.4 Pattern-Mixture Sensitivity Analysis & Tipping Points

Under the pattern-mixture paradigm, missingness is modeled by shifting the unobserved counterfactual distribution relative to the observed distribution by an explicit sensitivity offset $\delta$:

$$Y_{i, \text{mis}} = \hat{Y}_{i, \text{MAR}} + \delta \cdot s$$

where:
- $\hat{Y}_{i, \text{MAR}}$ is the conditional mean prediction from an MAR model.
- $s$ is the scale factor (standard deviation $\sigma(Y_{\text{obs}})$, percentage offset, or raw units).
- $\delta \in [\delta_{\text{min}}, \delta_{\text{max}}]$ represents the departure from MAR.

#### Tipping Point Definition
Let $\hat{\theta}(\delta)$ be the downstream inferential estimand (e.g., regression coefficient, treatment effect, or population mean) as a function of $\delta$.

The **tipping point** $\delta^*$ is formally defined as:
$$\delta^* = \inf \left\{ |\delta| : \text{sign}(\hat{\theta}(\delta)) \neq \text{sign}(\hat{\theta}(0)) \quad \text{or} \quad \text{CI}_{95}(\hat{\theta}(\delta)) \ni 0 \right\}$$

If $\delta^*$ is outside the domain of scientifically plausible shifts $[\delta_{\text{plaus, min}}, \delta_{\text{plaus, max}}]$, the primary qualitative conclusion is robust to unmeasured MNAR departures.

---

## 6. Assumptions and Failure Modes Summary

| Method | Key Identifiability Assumption | Empirical Diagnostics Available | Catastrophic Failure Mode |
| :--- | :--- | :--- | :--- |
| **Complete-Case** | MCAR: $P(M \mid Y) = P(M)$ | Little's $d^2$ test ($p > 0.05$) | Asymptotically biased when missingness depends on covariates or outcomes |
| **MICE (MAR)** | MAR: $P(M \mid Y) = P(M \mid Y_{\text{obs}})$ | Covariate shift tests, KS distance | Severe undercoverage (nominal 95% CI achieves 0% actual coverage) under strong MNAR |
| **Heckman Selection** | Joint normality of $(u_1, u_2)$ & Instrument relevance ($F > 10$) | Probit goodness of fit, First-stage $F$ | Severe multicollinearity and inflated variance if instrument is weak ($F < 10$) or collinear |
| **Pattern-Mixture** | Shift invariance across covariate strata | Sensitivity grid exploration | Sensitivity parameter $\delta$ is fundamentally non-identifiable; misjudging plausible range yields invalid bounds |
| **Auto Router** | Diagnostic signals correlate with appropriate method family | Risk score concordance, Shadow variable $F$ | Symmetric U-shaped dropout where mean shifts cancel out |
