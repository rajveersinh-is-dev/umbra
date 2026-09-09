# Phase 6: Identifiability & Sensitivity Auditing (Manski Bounds & Audit Certificates) - Context

**Phase:** 6 of 6
**Milestone:** Umbra Methodological & Statistical Hardening
**Requirements:** `IDENT-01`, `IDENT-02`

## 1. The Core Scientific Problem

Molenberghs et al. (2008) proved that for any Missing Not At Random (MNAR) model, there exists a Missing At Random (MAR) counterpart that fits the observed data equally well. Consequently:
- **True MNAR is fundamentally unidentifiable from observed data alone without untestable assumptions.**
- Standard point-imputation algorithms (MICE, Heckman selection, Pattern Mixture) always substitute an untestable structural assumption (e.g. conditional independence, bivariate normality, exclusion restriction, or sensitivity parameter $\delta$) to recover point estimates.
- When domain experts or auditors inspect missing data diagnostics, they require transparent quantification of what is mathematically guaranteed under *zero* untestable assumptions versus what relies on structural domain assumptions.

## 2. Solution Architecture

### A. Manski Partial Identifiability Intervals (`IDENT-01`)
Manski (1989, 1990, 2003) established sharp nonparametric bounds for population parameters under arbitrary missingness mechanisms:
1. **Mean Bounds**:
   For $Y \in [y_{\min}, y_{\max}]$ with observed rate $1 - p_{\text{miss}}$:
   $$\mathbb{E}[Y] \in \left[ \bar{Y}_{\text{obs}}(1 - p_{\text{miss}}) + y_{\min} p_{\text{miss}}, \; \bar{Y}_{\text{obs}}(1 - p_{\text{miss}}) + y_{\max} p_{\text{miss}} \right]$$
   The interval width $\Delta = p_{\text{miss}}(y_{\max} - y_{\min})$ is sharp and requires *zero* untestable assumptions.
2. **Quantile Bounds**:
   Manski bounds for population quantiles $q_\alpha$ under arbitrary missingness.
3. **Pluggable Support**:
   Supports user-provided domain support bounds $[y_{\min}, y_{\max}]$, empirical sample extremes, or trimmed quantiles.

### B. Machine-Verifiable Identifiability & Assumption Audit Certificate (`IDENT-02`)
1. An audit certificate module in `umbra/diagnostics/identifiability_audit.py` creating `IdentifiabilityCertificate` and `AssumptionAudit`.
2. Integrates directly into `UmbraDiagnosticReport` and `diagnose_dataframe`:
   - Categorizes every check into **Testable Implications** (Little's MCAR test, covariate shifts, tail correlation, instrument relevance) and **Untestable Domain Assumptions** (joint normality, exogeneity, conditional independence).
   - Generates an audit verdict (`ASSUMPTION_DEPENDENT`, `PARTIALLY_IDENTIFIABLE`, `COMPATIBLE_WITH_MAR`).
   - Exports machine-verifiable JSON (`to_dict()`) and publication-grade Markdown (`to_markdown()`).
