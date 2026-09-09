# Codebase Concerns

**Analysis Date:** 2026-09-09

## Methodological & Theoretical Boundaries

**Fundamental Non-Identifiability of MNAR:**
- **Issue:** As proven in the missing data literature (Manski 2003, Robins 1997, Molenberghs et al. 2008), the true Not-Missing-At-Random mechanism is fundamentally non-identifiable from observational data without untestable structural assumptions.
- **Impact:** No algorithm can guarantee unbiased point imputation under arbitrary MNAR.
- **Current Mitigation:** Umbra explicitly communicates this limitation in documentation, emits `UserWarning` when MNAR risk is detected, and pairs point estimates with quantitative sensitivity bounds (`SensitivityReport`).

**Heckman Exclusion Restriction Dependence:**
- **Issue:** When a candidate shadow/instrumental variable $Z$ is not available, the Heckman two-stage selection estimator identifies the selection correlation $\rho$ solely via the non-linearity of the Probit inverse Mills ratio.
- **Impact:** In the absence of an exclusion restriction, the inverse Mills ratio $\lambda(W\gamma)$ is often approximately linear over typical ranges, inducing severe multicollinearity between the second-stage regressors and $\lambda$, causing variance inflation and parameter instability.
- **Current Mitigation:** Umbra automatically tests candidate instruments for exclusion validity, flags weak instruments with $F \le 10$ (`WeakInstrumentWarning`), emits an explicit warning when no exclusion restriction exists, and provides a regularized Ridge fallback when the design matrix is near-singular.

**First-Stage Estimation Uncertainty (Murphy-Topel Bias):**
- **Issue:** Standard two-step Heckman regression treats generated regressors $\hat{\lambda}_i$ as known data, underestimating second-stage standard errors (Murphy & Topel 1985).
- **Current Mitigation:** Umbra defaults to paired bootstrap standard errors (`n_bootstrap_se = 200`), resampling both stages to propagate estimation variance, and emits `HeckmanSEWarning` if `n_bootstrap_se=0` is requested.

## Numerical Stability Considerations

**Extreme Probit Quantile Inversion:**
- **Issue:** In the extreme tails of the normal distribution ($\eta < -10$), the direct ratio $\phi(\eta)/\Phi(\eta)$ suffers from floating-point underflow in the denominator.
- **Current Mitigation:** Implemented log-space tail evaluation: $\exp(\text{logpdf}(\eta) - \text{logcdf}(\eta))$ for $\eta \in [-30, -10]$, and hard clipping outside $[-30, 30]$.

**Vectorized Little's MCAR EM Convergence:**
- **Issue:** High-dimensional missingness patterns or ill-conditioned covariance matrices can cause slow EM convergence or singular covariance inversions.
- **Current Mitigation:** EM iterations group rows by distinct binary missingness patterns; regularized diagonal shrinkage is applied if condition numbers degrade.

## Optional Dependencies & Framework Footprint

**PyTorch Deep Generative Imputer:**
- **Issue:** `DeepGenerativeMNARImputer` requires `torch`. For lightweight environments (e.g. basic data pipelines), installing PyTorch adds significant download size.
- **Current Mitigation:** PyTorch is declared as an optional extra (`umbra-impute[deep]`). If PyTorch is absent, `DeepGenerativeMNARImputer` raises an informative `ImportError` directing the user to install the extra without impacting core features.

---

*Concerns analysis: 2026-09-09*
*Update when new methodological constraints or technical debt are identified*
