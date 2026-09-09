# Phase 3: Multiple Imputation & Rubin Pooling Engine - Context

**Created:** 2026-09-09
**Domain:** Multiple Imputation (Rubin 1987), Barnard & Rubin (1999) Small-Sample Degrees of Freedom, Fraction of Missing Information (FMI)
**Goal:** Implement multi-draw stochastic imputation ($M \ge 5$) across all stochastic imputers, and build a dedicated `RubinPooler` combining $M$ datasets with Barnard-Rubin small-sample corrections, resolving Limitation #4 in `docs/limitations.md`.

## Background & Problem Statement

In `docs/limitations.md`:
> "4. Single Plug-in Imputation vs. Multiple Imputation Pooling: Single deterministic or single stochastic draws fail to account for imputation uncertainty, underestimating standard errors and inflating Type I error rates."

In v0.2.0:
- `rubins_rules` in `umbra/imputers/mar_chained_equations.py` was a basic function with Rubin (1987) large-sample degrees of freedom: $\nu_m = (M - 1)(1 + r^{-1})^2$.
- When sample size $N$ is small, $\nu_m$ can exceed the complete-data degrees of freedom $\nu_0 = N - k$, which is mathematically invalid because missing data can never produce more information than complete data.
- Not all imputers provided a uniform `transform_multiple(X, m=5)` interface (`DeepGenerativeMNARImputer` lacked multiple stochastic sampling).
- Users had no scikit-learn compatible `RubinPooler` to fit downstream models across $M$ datasets and pool parameters automatically.

## Architectural Decisions

### D-11: Uniform `transform_multiple()` and `fit_transform_multiple()` Interface
- All stochastic imputers (`MARChainedEquationsImputer`, `HeckmanSelectionImputer`, `PatternMixtureImputer`, and `DeepGenerativeMNARImputer`) must support:
  - `transform_multiple(X, m=5, random_state=None) -> List[pd.DataFrame]`
  - `fit_transform_multiple(X, m=5, random_state=None) -> List[pd.DataFrame]`
  - `transform(X, return_all_imputations=True) -> List[pd.DataFrame]`
- In `DeepGenerativeMNARImputer`, add stochastic sampling `z ~ N(mu, sigma^2)` during evaluation to draw $M$ distinct reconstructions from the learned joint data-mask latent space.

### D-12: Barnard & Rubin (1999) Small-Sample Adjustment
- Complete-data degrees of freedom: $\nu_0$ (e.g. $N - k$, or user-supplied `df_complete`).
- Observed-data degrees of freedom:
  $$\nu_{\text{obs}} = \frac{\nu_0 + 1}{\nu_0 + 3} \nu_0 (1 - \hat{\gamma})$$
  where $\hat{\gamma} = \frac{(1 + M^{-1}) B}{T} = \frac{r}{1 + r}$.
- Barnard-Rubin adjusted degrees of freedom:
  $$\nu_{\text{adj}} = \left( \frac{1}{\nu_m} + \frac{1}{\nu_{\text{obs}}} \right)^{-1} = \frac{\nu_m \nu_{\text{obs}}}{\nu_m + \nu_{\text{obs}}}$$
- Guaranteed: $\nu_{\text{adj}} \le \nu_0$ and $\nu_{\text{adj}} \le \nu_m$.

### D-13: `RubinPooler` Class & Extended `RubinsRulesResult`
- Create `umbra.imputers.rubin_pooler` exposing:
  - `RubinPooler`: A high-level pooler that can pool lists of point/variance arrays, or run an estimator/function across $M$ imputed DataFrames and compute pooled coefficients, standard errors, $t$-statistics, $p$-values, CIs, FMI, and relative efficiency.
  - `rubins_rules()`: Extended functional API accepting optional `df_complete: Optional[float] = None`, returning `RubinsRulesResult` with backward-compatible attributes (`pooled_estimate`, `within_variance`, `between_variance`, `total_variance`, `standard_error`, `df`, `ci_lower`, `ci_upper`) plus new diagnostics (`fmi`, `relative_variance`, `relative_efficiency`, `df_adjusted`).
