# The Identifiability Map: Observable Evidence, Assumptions, and Epistemic Limits

This document establishes the formal theoretical foundation of [Umbra](https://github.com/Raj123-0/umbra), defining the exact mathematical boundaries between what can be empirically tested, what requires structural assumptions, and what is fundamentally unidentifiable from observed data alone.

---

## 1. The Central Epistemic Tri-Partition

Missing-data analysis cannot proceed responsibly without distinguishing three epistemic tiers:

```
+-------------------------------------------------------------------------+
| LEVEL 1: OBSERVABLE DATA EVIDENCE                                        |
| Testable directly from (X_obs, R) without unverifiable assumptions.     |
| Examples: Little's MCAR test, covariate shifts, residual tail shape,    |
|           candidate auxiliary variable relevance (first-stage F > 10).   |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
| LEVEL 2: ASSUMPTION-DEPENDENT ESTIMATION                                |
| Identifiable ONLY conditional on explicit, non-testable model assumptions|
| Examples: MAR conditional exchangeability (MICE),                       |
|           Bivariate Gaussian error structure & exclusion (Heckman),      |
|           Pattern-mixture shifts with specified delta.                  |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
| LEVEL 3: FUNDAMENTALLY UNIDENTIFIED QUANTITIES                           |
| Mathematically impossible to establish from observed data alone.         |
| Governed by Molenberghs et al. (2008) Non-Identifiability Theorem.      |
| Requires: Sensitivity bounds, tipping-point curves, scenario intervals.  |
+-------------------------------------------------------------------------+
```

---

## 2. Level 1: Observable Diagnostics (What Can Be Learned)

From the joint distribution of observed features and response indicators $(X_{\text{obs}}, R)$, the following quantities are empirically testable:

### A. Global Departures from MCAR (Little, 1988)
- **Null Hypothesis**: $H_0: P(R \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(R)$.
- **Empirical Test**: Regularized EM distance test comparing group-pattern means against the grand mean:
  $$d^2 = \sum_{j=1}^J n_j (\bar{y}_{\text{obs}, j} - \hat{\mu}_j)^T \hat{\Sigma}_j^{-1} (\bar{y}_{\text{obs}, j} - \hat{\mu}_j) \sim \chi^2(\text{df})$$
  where $\text{df} = \sum_{j=1}^J p_j - p$.
- **What Rejection Means**: Missingness is systematically correlated with observed covariates or outcomes.
- **What Rejection Does NOT Mean**: **Rejection of MCAR does NOT prove MNAR.** Data may be fully MAR.

### B. Observable Covariate Distribution Shifts
- Two-sample Kolmogorov-Smirnov and Mann-Whitney $U$ tests evaluating whether respondents ($R=1$) and non-respondents ($R=0$) have differing distributions across observed covariates $X$.
- Significant shifts indicate that $X$ predicts missingness, which is the defining requirement of MAR.

### C. Residual Tail Concentration
- Non-parametric tail concentration ratio evaluating whether missingness concentrates disproportionately in the extreme predicted quantiles of $X\beta$. While consistent with self-censoring, this remains an observable proxy.

### D. Candidate Auxiliary Variable Relevance
- Statistical correlation between an auxiliary variable $Z$ and the missingness indicator $R$, evaluated via first-stage partial $F$-statistic against the Stock-Yogo ($F > 10$) benchmark.

---

## 3. Level 2: Assumption-Dependent Estimation (What Can Be Modeled)

When data depart from MCAR, point identification requires explicit structural assumptions:

### A. Missing at Random (MAR) Framework
- **Structural Assumption**: $Y \perp\!\!\perp R \mid X$.
- **Estimator**: Chained equations (MICE) using Predictive Mean Matching (PMM) or Bayesian Ridge regression.
- **Limitation**: If missingness depends directly on unobserved $Y$ conditional on $X$, MAR point estimates are asymptotically biased and confidence interval coverage collapses.

### B. Heckman Selection Model Framework
- **Structural Assumptions**:
  1. Latent threshold selection: $R = \mathbf{1}(Z\gamma + u > 0)$.
  2. Joint bivariate normality:
     $$\begin{pmatrix} \epsilon \\ u \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} \sigma^2 & \rho\sigma \\ \rho\sigma & 1 \end{pmatrix}\right)$$
  3. **Exclusion Restriction**: Candidate instrument $Z$ influences $R$ directly but influences $Y$ ONLY through $R$ ($\text{Cov}(Z, \epsilon \mid X) = 0$).
- **Limitation**: As shown in Umbra's misspecification battery, if the exclusion restriction is violated ($Z \to Y$ direct path), Heckman selection produces massive omitted variable bias ($+1.087$) and 0% coverage.

### C. Pattern-Mixture Model Framework
- **Structural Assumption**: Non-responders have a mean shifted by $\delta \cdot \sigma_{\text{residual}}$ relative to responders with identical covariates $X$.
- **Limitation**: $\delta$ is unidentified from the data and must be varied across a sensitivity grid.

---

## 4. Level 3: Fundamentally Unidentified Quantities (What Cannot Be Known)

### The Molenberghs et al. (2008) Non-Identifiability Theorem
**Theorem**: *For any Not-Missing-at-Random (MNAR) model fitted to a set of data, there exists an observed-data Missing-at-Random (MAR) counterpart that produces an identical likelihood on observed data.*

**Proof Sketch**: The full data likelihood factorizes into:
$$P(Y, R \mid X) = P(Y_{\text{obs}}, R \mid X) \cdot P(Y_{\text{mis}} \mid Y_{\text{obs}}, R, X)$$
Because $Y_{\text{mis}}$ is never observed when $R=0$, the second term $P(Y_{\text{mis}} \mid Y_{\text{obs}}, R=0, X)$ receives zero empirical mass in the likelihood. Any arbitrary mathematical distribution can be assigned to $Y_{\text{mis}} \mid R=0$ without altering the goodness of fit to the observed data $(Y_{\text{obs}}, R=1)$.

**Direct Implication**: It is mathematically impossible for any algorithm to determine whether data are truly MNAR from observed data alone without unverifiable structural assumptions.

---

## 5. Partial Identification & The Bounds Boundary (Why Not Blind Manski Bounds?)

In econometrics, partial identification provides a continuum between non-parametric agnosticism and point-identified parametric models. Umbra distinguishes four distinct levels of partial identification:

### 1. Unrestricted Worst-Case Bounds (Manski, 1990; Horowitz & Manski, 2000)
For missing outcome data without any structural or distributional assumptions:
$$E[Y] \in \left[ P(R=1)E[Y \mid R=1] + P(R=0)y_{\text{min}},\, P(R=1)E[Y \mid R=1] + P(R=0)y_{\text{max}} \right]$$
- **Where They Break Down**: For continuous unbounded variables ($Y \in (-\infty, +\infty)$), $y_{\text{min}} = -\infty$ and $y_{\text{max}} = +\infty$. The resulting bounds are $(-\infty, +\infty)$ and completely vacuous.
- **Why Sample Extrema Fail**: Heuristically substituting sample minimums ($\min Y_{\text{obs}}$) and maximums ($\max Y_{\text{obs}}$) is **statistically invalid**, because under MNAR, non-respondents systematically reside in unobserved tails beyond the sample extremes.

### 2. Support-Restricted & Monotone Bounds (Informative Partial Identification)
- When the target variable $Y$ possesses **known, compact physical support** $[y_{\text{min}}, y_{\text{max}}]$ (e.g. survival proportions in $[0, 1]$, examination percentages in $[0, 100]$, 7-point Likert scales), worst-case Manski bounds are mathematically sharp and highly informative.
- Similarly, under monotone instrument assumptions (Manski & Pepper, 2000) or treatment selection monotonicity (Lee, 2009), bounds can be tightened substantially without parametric distributional assumptions.

### 3. Parametrically Identified Structural Models (Heckman, 1979)
- By introducing strong parametric restrictions (e.g., joint bivariate Gaussian errors between selection and outcome equations) and exclusion restrictions ($Z \to R$ but $Z \perp Y \mid X$), the parameter vector is point-identified.
- **Vulnerability**: As demonstrated in Umbra's misspecification battery, these models are fragile to heavy-tailed errors, non-linearities, or exclusion violations ($Z \to Y$).

### 4. Continuous Sensitivity Analysis & Tipping Points (Little, 1993)
- For general continuous data without compact physical support or valid instruments, Umbra avoids returning vacuous $(-\infty, +\infty)$ intervals or manufactured sample-extrema bounds.
- Instead, Umbra provides **Pattern-Mixture Sensitivity Analysis**, parameterizing the unobserved counterfactual mean in residual standard deviation units:
  $$Y_{\text{mis}} \sim \hat{\mu}_{\text{MAR}}(X) + \delta \cdot \hat{\sigma}_{\text{res}}$$
- By varying $\delta \in [-3, +3]$ and computing the **tipping point ($\delta^*$)**, researchers discover the exact severity of MNAR departure required to overturn scientific conclusions.
