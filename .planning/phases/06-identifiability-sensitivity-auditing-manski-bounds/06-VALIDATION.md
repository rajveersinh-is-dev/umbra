# Phase 6: Identifiability & Sensitivity Auditing - Validation Plan

**Requirements:** `IDENT-01`, `IDENT-02`

## Validation Criteria

### Mathematical & Algorithmic Invariants
1. **Manski Mean Bounds (`IDENT-01`)**:
   - For any observed sample with missingness rate $p_{\text{miss}}$ and support $[y_L, y_U]$:
     - $\text{LB}_{\text{mean}} \le \bar{Y}_{\text{obs}} \le \text{UB}_{\text{mean}}$.
     - $\text{UB}_{\text{mean}} - \text{LB}_{\text{mean}} = p_{\text{miss}} (y_U - y_L)$ strictly holds.
     - When $p_{\text{miss}} \to 0$, $[\text{LB}, \text{UB}] \to [\bar{Y}_{\text{obs}}, \bar{Y}_{\text{obs}}]$.
     - The true population mean across ground-truth complete data MUST fall inside the Manski interval $[\text{LB}, \text{UB}]$.
2. **Manski Quantile Bounds (`IDENT-01`)**:
   - For quantile $\alpha \in (0, 1)$:
     - $\text{LB}_\alpha \le \text{UB}_\alpha$.
     - When $p_{\text{miss}} < 0.5$, median interval $[\text{LB}_{0.5}, \text{UB}_{0.5}]$ is strictly interior to $[y_L, y_U]$.
3. **Machine-Verifiable Audit Certificate (`IDENT-02`)**:
   - `UmbraDiagnosticReport` contains `identifiability_certificate: IdentifiabilityCertificate`.
   - Certificate explicitly lists testable evidence (Little's MCAR $p$-value, covariate shift max Cohen's $d$, instrument $F$-statistic) and untestable domain assumptions for each candidate imputer.
   - `to_dict()` and `to_markdown()` serialize with 100% schema integrity without runtime errors.
   - Molenberghs non-identifiability theorem and Manski partial identification citations are included.

### Automated Test Suite (`tests/test_identifiability_and_manski.py`)
- `test_manski_mean_bounds_math`: Verify interval width formula, sharpness, and bounds containment.
- `test_manski_quantile_bounds`: Verify quantile and median partial identification bounds.
- `test_manski_bounds_user_support_vs_empirical`: Verify user-supplied vs empirical support bounds.
- `test_identifiability_certificate_generation`: Verify certificate creation, testable implications, and untestable assumptions.
- `test_report_integration_with_certificate`: Verify `diagnose_report()` and `UmbraDiagnosticReport` embed the audit certificate.
- `test_certificate_markdown_and_json`: Verify serialization and formatting.

### Regression Checks
- 100% test pass rate across all test files (170+ tests).
- 0 mypy type errors and 0 ruff lint errors.
