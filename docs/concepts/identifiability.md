# Mathematical Foundations: Identifiability & The Missing Data Taxonomy

## 1. Formal Missing Data Taxonomy (Rubin, 1976)

Let $X = (X_{\text{obs}}, X_{\text{mis}})$ denote the complete rectangular data matrix for $n$ units, where:
- $X_{\text{obs}}$ represents the components observed in the sample.
- $X_{\text{mis}}$ represents the unobserved components that would have been measured in the absence of missingness.
- $M \in \{0, 1\}^{n \times p}$ denotes the missingness indicator matrix, where $M_{ij} = 1$ if $X_{ij}$ is missing and $M_{ij} = 0$ if $X_{ij}$ is observed.

The joint distribution of the complete data and the missingness mechanism factors as:
$$P(X, M \mid \theta, \psi) = P(X \mid \theta) P(M \mid X, \psi)$$
where $\theta$ indexes the substantive data-generating process and $\psi$ indexes the missingness mechanism parameters.

---

### A. Missing Completely at Random (MCAR)

**Definition**: Missingness is statistically independent of both observed and unobserved data values:
$$P(M \mid X_{\text{obs}}, X_{\text{mis}}, \psi) = P(M \mid \psi) \quad \forall X_{\text{obs}}, X_{\text{mis}}$$

**Implications**:
- The observed cases are a simple random sub-sample of the target population.
- Complete-case analysis (listwise deletion) yields unbiased point estimates, though it sacrifices statistical efficiency and power.
- **Testability**: **Testable**. Under multivariate normality, Little's (1988) multivariate test evaluates whether observed sample means across patterns are mutually consistent.

---

### B. Missing at Random (MAR)

**Definition**: Conditional on observed covariates $X_{\text{obs}}$, missingness is conditionally independent of the unobserved missing values $X_{\text{mis}}$:
$$P(M \mid X_{\text{obs}}, X_{\text{mis}}, \psi) = P(M \mid X_{\text{obs}}, \psi) \quad \forall X_{\text{mis}}$$

**Implications**:
- Responders and non-responders who share identical observed values $X_{\text{obs}}$ have the same conditional distribution for $X_{\text{mis}}$:
  $$P(X_{\text{mis}} \mid X_{\text{obs}}, M = 0) = P(X_{\text{mis}} \mid X_{\text{obs}}, M = 1)$$
- Under parameter distinctness ($\theta$ and $\psi$ are functionally independent in the Bayesian/likelihood sense), the missingness mechanism is **ignorable** for likelihood-based inference.
- Methods such as Multiple Imputation by Chained Equations (MICE), Full Information Maximum Likelihood (FIML), and Inverse Probability Weighting (IPW) produce consistent estimators.
- **Testability**: **Partially Testable**. Departures from MCAR into MAR can be assessed by observing whether $X_{\text{obs}}$ predicts $M$. However, whether $M$ depends on $X_{\text{mis}}$ *after* conditioning on $X_{\text{obs}}$ is untestable from observed data.

---

### C. Missing Not at Random (MNAR)

**Definition**: Missingness depends directly on the unobserved values themselves, even after conditioning on all observed data:
$$P(M \mid X_{\text{obs}}, X_{\text{mis}}, \psi) \neq P(M \mid X_{\text{obs}}, \psi)$$

**Implications**:
- The conditional distributions of responders and non-responders differ:
  $$P(X_{\text{mis}} \mid X_{\text{obs}}, M = 0) \neq P(X_{\text{mis}} \mid X_{\text{obs}}, M = 1)$$
- Standard MAR techniques (MICE, mean imputation, standard regression) are **inconsistent and asymptotically biased**.
- Imputing under a false MAR assumption yields artificially narrow confidence intervals centered on systematically biased point estimates.

---

## 2. The Fundamental Non-Identifiability Theorem

> [!IMPORTANT]
> **Theorem (Molenberghs et al., 2008)**:
> For any MNAR model specified for $(X, M)$, there exists an MAR model that produces an **identical joint distribution over the observed data** $(X_{\text{obs}}, M)$.
> Consequently, **it is mathematically impossible to distinguish MAR from MNAR using observed data alone**.

### Proof Intuition
The observed data likelihood is:
$$L(\theta, \psi \mid X_{\text{obs}}, M) = \int P(X_{\text{obs}}, X_{\text{mis}} \mid \theta) P(M \mid X_{\text{obs}}, X_{\text{mis}}, \psi) \, dX_{\text{mis}}$$
Because $X_{\text{mis}}$ is integrated out over its entire support, any departure in the conditional distribution $P(X_{\text{mis}} \mid X_{\text{obs}}, M = 1)$ can be mirrored by an alternative choice of prior distribution $P(X_{\text{mis}} \mid X_{\text{obs}})$ under an MAR assumption without altering the marginal observed likelihood by even $\epsilon > 0$.

---

## 3. Classification of Methodological Assumptions

To maintain scientific integrity, every assumption in Umbra is classified into one of five epistemological categories:

| Assumption Class | Definition | Examples in Umbra | Verification Status |
| :--- | :--- | :--- | :--- |
| **Testable Assumptions** | Can be formally falsified or confirmed using observed data | Little's MCAR null; Covariate distribution shifts; Complete vs missing balance | Empirical hypothesis test ($p$-values, effect sizes) |
| **Partially Testable** | Necessary conditions can be checked, but sufficient conditions cannot | Relevance of auxiliary shadow variables ($r(Z, M) \neq 0$, $F > 10$) | Empirical $F$-test confirms relevance; exclusion restriction remains unverified |
| **Untestable Assumptions** | Mathematical propositions regarding unobserved quantities that have no empirical test | Causal exclusion restriction ($Z \perp Y \mid X$); Absence of unmeasured confounders | Requires substantive domain theory |
| **Sensitivity Parameters** | Explicit quantities governing the magnitude of unobserved departure | Pattern-mixture shift parameter $\delta$; Selection error correlation $\rho$ | Swept systematically across plausible grids $[\delta_{\min}, \delta_{\max}]$ |
| **Modeling Assumptions** | Mathematical structure chosen for computational tractability | Bivariate normality of errors $(u, \epsilon)$; Linearity of regression equations | Assessed via standard residual diagnostics |

---

## 4. Umbra's Foundational Scientific Principle

> *«MNAR is generally not identifiable from observed data alone. Umbra therefore does not claim to prove that data are MNAR; it combines diagnostics and sensitivity analyses to quantify evidence and assess how conclusions change under plausible departures from MAR.»*

Umbra refuses to present single point estimates under arbitrary unverifiable assumptions as definitive truth. Instead, Umbra:
1. **Screens** for converging signals that suggest MAR is fragile (rejection of MCAR, covariate distribution shifts, extreme residual tail concentration).
2. **Surfaces** candidate auxiliary variables (instruments) while explicitly qualifying their limitations.
3. **Quantifies** conclusion stability via sensitivity bounds $\theta(\delta)$ and tipping-point analysis.
