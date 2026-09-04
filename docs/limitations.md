# Explicit Methodological Limitations & Failure Cases

A hallmark of mature research software is honesty about its boundaries. This document articulates where Umbra does **NOT** work, what assumptions are non-negotiable, and when users should seek alternative methods.

---

## 1. The Fundamental Non-Identifiability Limit

**True Missing-Not-At-Random (MNAR) is mathematically unidentifiable from observed data alone.**

No diagnostic battery, statistical test, machine learning model, or deep neural network can definitively "prove" that data are MNAR without imposing untestable assumptions.
- Umbra's diagnostic score is a **risk assessment of converging empirical indicators**, not a mathematical proof of the true mechanism.
- If an analyst assumes MAR, no observed data test can decisively refute the claim that unmeasured covariates (omitted from the dataset) would have satisfied MAR.

---

## 2. Shadow Variables & Auxiliary Instruments Are Not Proved Causal Instruments

The `shadow_variable_finder` ranks candidate features based purely on two empirical conditions:
1. Strong correlation with the missingness indicator $R_Y$ ($F > 10$).
2. Low conditional partial correlation with observed outcome values $Y \mid X$.

**CRITICAL LIMITATION**:
- Statistical association and partial correlation **CANNOT** prove the exclusion restriction.
- If a candidate variable $Z$ has an unmeasured direct causal path to unobserved $Y_{\text{mis}}$ that does not manifest on the observed sample, the Heckman selection model will be asymptotically biased.
- **Rule**: Empirical correlation is only a screening filter. Substantive domain theory is strictly required to validate that an exclusion restriction causally holds.

---

## 3. Heckman Selection Model Without an Exclusion Restriction

When Heckman selection is executed without an instrumental shadow variable ($W = X$):
- Identification rests entirely on the non-linearity of the Probit Inverse Mills Ratio $\lambda_1(X\gamma)$.
- In moderate probability ranges (e.g. $P(R=1) \in [0.20, 0.80]$), the Inverse Mills Ratio is nearly linear in $X\gamma$.
- This induces **severe multicollinearity** between $X$ and $\lambda_1$, leading to inflated standard errors, extreme variance, and numerical instability.
- Umbra will issue a warning and recommends falling back to pattern-mixture sensitivity sweeps rather than relying on functional form identification alone.

---

## 4. Sensitivity Parameter Selection ($\delta$)

Pattern-mixture models rely on an external sensitivity parameter $\delta$.
- $\delta$ is not estimated from data.
- The choice of grid (e.g. $[-1.5, +1.5]$ standard deviations) is a user-specified assumption. If the true unobserved departure exceeds $+1.5$ standard deviations, the sensitivity interval will not contain the true parameter.
- Users must justify the bounds of the sensitivity grid using domain benchmarks, historical surveys, or physical bounds.

---

## 5. Finite-Sample & Asymptotic Approximations

- **Little's MCAR Test**: Relies on asymptotic $\chi^2$ distributions. In small samples ($N < 100$) or when missingness patterns have very few observations ($N_j < 5$), $p$-values can be distorted.
- **Two-Step Heckman Estimator**: Is consistent asymptotically, but can exhibit notable finite-sample bias in small samples ($N < 250$).

---

## 6. High-Dimensional Tabular Data ($p > N$)

- Little's EM test and Heckman selection models require inverting covariance matrices of dimension $p \times p$.
- In high-dimensional regimes ($p > N$ or $p > 100$), classical selection models fail due to singularity. Regularization helps, but classical identification theorems break down.

---

## 7. Categorical & Text Features

- Umbra is optimized for numeric and mixed tabular data.
- High-cardinality nominal variables (e.g. free text, hundreds of categorical levels) must be preprocessed (e.g. via target encoding or dimensionality reduction) before fitting selection models.

---

## 8. Causal Interpretation Warning

- Umbra is an **imputation and missing-data sensitivity package**, not a causal discovery engine.
- Imputing missing data does **NOT** turn observational associations into causal effects. Controlling for confounding, collider bias, and exchangeability remains the responsibility of the investigator.
