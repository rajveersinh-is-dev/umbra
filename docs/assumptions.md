# Scientific Assumptions & Identifiability Boundaries in Umbra

This document delineates the exact mathematical and structural assumptions underlying every diagnostic, estimator, and sensitivity tool in Umbra.

---

## 1. Rubin's Foundational Taxonomy

Let $Y = (Y_{\text{obs}}, Y_{\text{mis}})$ denote the complete data matrix and $M \in \{0, 1\}^{n \times p}$ denote the missingness indicator matrix ($M_{ij} = 1$ if missing, $0$ if observed).

### 1.1 Missing Completely at Random (MCAR)
- **Mathematical Definition**:
  $$P(M \mid Y_{\text{obs}}, Y_{\text{mis}}, \psi) = P(M \mid \psi) \quad \forall Y_{\text{obs}}, Y_{\text{mis}}, \psi$$
- **What is Assumed**: Missingness is an independent coin flip unrelated to any observed covariates or unobserved outcomes.
- **What is Identifiable**:
  - Sample means $\mathbb{E}[Y_{\text{obs}}] = \mathbb{E}[Y]$.
  - Complete-case estimates are asymptotically unbiased.
- **Empirical Testability**: **Testable** against observed covariates via Little's (1988) multivariate distance test.
- **Catastrophic Failure**: Complete-case analysis loses statistical efficiency ($\text{Var} \propto 1/n_{\text{obs}}$) and becomes severely biased if MCAR is violated.

### 1.2 Missing at Random (MAR)
- **Mathematical Definition**:
  $$P(M \mid Y_{\text{obs}}, Y_{\text{mis}}, \psi) = P(M \mid Y_{\text{obs}}, \psi) \quad \forall Y_{\text{mis}}, \psi$$
- **What is Assumed**: Conditional on observed covariates $Y_{\text{obs}}$, missingness is independent of the unobserved values $Y_{\text{mis}}$ (Conditional Ignorability).
- **What is Identifiable**:
  - The conditional distribution $P(Y_{\text{mis}} \mid Y_{\text{obs}})$ equals $P(Y_{\text{obs}} \mid Y_{\text{obs}})$.
  - Imputation via chained equations (MICE) or inverse probability weighting (IPW) yields consistent point estimates.
- **Empirical Testability**: **Non-testable**. As proven by Molenberghs et al. (2008), observed data can never prove that missingness is independent of $Y_{\text{mis}}$ without untestable assumptions.
- **Catastrophic Failure**: If missingness depends directly on $Y_{\text{mis}}$, standard MICE coverage collapses to **0%** and estimates suffer from uncorrectable asymptotic bias.

### 1.3 Missing Not at Random (MNAR)
- **Mathematical Definition**:
  $$\exists Y_{\text{mis}} \neq Y'_{\text{mis}} \quad \text{s.t.} \quad P(M \mid Y_{\text{obs}}, Y_{\text{mis}}, \psi) \neq P(M \mid Y_{\text{obs}}, Y'_{\text{mis}}, \psi)$$
- **What is Assumed**: Missingness depends on the unobserved quantities themselves, even after controlling for all observed features.
- **Identifiability Boundary**: Point estimates are **fundamentally non-identifiable** without external structural restrictions (e.g. valid instrumental variables or parametric distribution families).

---

## 2. Selection Models vs. Pattern-Mixture Models

### 2.1 Heckman Two-Step Selection Estimator
- **Structural Equations**:
  - Selection equation: $M_i^* = Z_i^T \gamma + u_{1i}, \quad M_i = \mathbf{1}(M_i^* > 0)$
  - Outcome equation: $Y_i = X_i^T \beta + u_{2i}$
- **Core Structural Assumptions**:
  1. **Bivariate Normality**: The error vector $(u_{1i}, u_{2i})$ follows a bivariate Gaussian distribution $\mathcal{N}_2(0, 0, 1, \sigma_2^2, \rho)$.
  2. **Exclusion Restriction**: $Z$ must contain at least one auxiliary variable $Z_{\text{aux}} \notin X$ such that:
     $$\text{Cov}(Z_{\text{aux}}, M^*) \neq 0 \quad \text{and} \quad \text{Cov}(Z_{\text{aux}}, u_2) = 0$$
- **What is Identifiable**:
  - Under valid exclusion restrictions and bivariate normality, $\beta$, $\sigma_2$, and the selection correlation $\rho$ are point-identified.
- **Failure Modes**:
  - If the instrument is weak ($F \le 10$), standard errors explode due to collinearity between the Inverse Mills Ratio $\lambda(Z\hat{\gamma})$ and regressors.
  - If errors are non-normal (e.g., Cauchy or Pareto), point estimates can be more biased than unadjusted OLS (Little, 1985).

### 2.2 Pattern-Mixture Models & Tipping Point Analysis
- **Structural Equation**:
  $$Y_{i, \text{mis}} = \hat{Y}_{i, \text{MAR}} + \delta \cdot s$$
  where $\delta$ is an explicit sensitivity parameter and $s$ is the outcome scale factor.
- **What is Assumed**:
  - The counterfactual distribution of non-responders is shifted by $\delta$ standard deviations relative to respondents with identical observed covariates.
- **What is Identifiable**:
  - For any fixed $\delta$, the point estimate $\hat{\theta}(\delta)$ and its confidence interval are identified.
  - The true value of $\delta$ is **unidentifiable from observed data alone**.
- **Scientific Purpose**:
  - To locate the **tipping point** $\delta^*$ where a qualitative conclusion flips sign or significance, allowing domain scientists to judge whether such a shift is scientifically plausible.

---

## 3. Candidate Auxiliary Variables vs. Proved Instruments

Umbra explicitly enforces the distinction:

| Concept | What It Means | How Umbra Evaluates It | Can Data Alone Prove It? |
| :--- | :--- | :--- | :---: |
| **Relevance** | $Z$ is associated with missingness propensity $M$. | First-stage $F$-statistic, Pearson correlation | **Yes** (empirically verifiable via $F > 10$) |
| **Exclusion** | $Z$ has no direct causal link to $Y$ except through $M$. | Conditional partial correlation $r(Z, Y \mid X) \approx 0$ on observed rows | **NO** (unobserved confounders $U$ can violate it) |

> [!WARNING]
> A variable exhibiting high empirical correlation with missingness is **NOT automatically a valid instrument**. Substantive domain expertise is strictly required to validate that the exclusion restriction holds causally.

---

## 4. Summary of Assumptions by Component

| Umbra Component | Identifiable Quantities | Required Untestable Assumptions | Diagnostic Safeguards |
| :--- | :--- | :--- | :--- |
| `littles_mcar_test` | Test statistic $d^2$, p-value | Multivariate normality of complete data under $H_0$ | Ledoit-Wolf shrinkage, eigenvalue clipping |
| `analyze_missingness_patterns` | Two-sample KS distance, Cohen's $d$, Cliff's $\delta$ | None (purely empirical observed distributions) | Non-parametric rank tests |
| `MARChainedEquationsImputer` | Imputed values under MAR | Ignorability of missingness given covariates | Rubin's rules pooled uncertainty |
| `HeckmanSelectionImputer` | Slope vector $\beta$, selection $\rho$ | Bivariate normality, valid exclusion restriction ($Z \notin X$) | Stock-Yogo $F > 10$ pre-check, log-space IMR |
| `PatternMixtureImputer` | Estimate conditional on shift $\delta$ | Shift invariance across strata | Continuous $\delta$ grid sweep |
| `UmbraImputer(strategy='auto')` | Evidence-based routing decision | Diagnostic signals correlate with best method family | Rejection of Heckman if $F \le 10$, fallback to sensitivity |
