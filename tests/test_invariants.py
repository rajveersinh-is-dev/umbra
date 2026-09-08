"""
Mathematical Invariants & Deterministic Reproducibility Tests.

Tests fundamental mathematical guarantees:
1. Little's test statistic d^2 >= 0 and p-value in [0, 1].
2. Rubin's rules total variance T >= U_bar (total variance never decreases under imputation uncertainty).
3. Barnard-Rubin degrees of freedom nu > 0.
4. Sensitivity parameter monotonicity: d(estimate)/d(delta) > 0 for positive scale shifts.
5. Determinism: Same random seed produces identical results to machine precision across runs.
"""

import numpy as np
import pandas as pd
import pytest

from umbra.api import UmbraImputer
from umbra.diagnostics.mcar_test import littles_mcar_test
from umbra.imputers.mar_chained_equations import rubins_rules
from umbra.sensitivity.grid_analysis import run_sensitivity_grid


def test_littles_test_mathematical_invariants():
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.randn(n),
            "x2": rng.randn(n),
            "x3": rng.randn(n),
        }
    )
    df.loc[0:30, "x1"] = np.nan
    df.loc[20:50, "x2"] = np.nan

    res = littles_mcar_test(df)
    # Non-negative Mahalanobis distance
    assert res.statistic >= 0.0
    # Probability bounds
    assert 0.0 <= res.p_value <= 1.0
    # Degrees of freedom must be positive integer
    assert res.degrees_of_freedom >= 1
    assert isinstance(res.degrees_of_freedom, int)


def test_rubins_rules_variance_invariants():
    # Invariant: Total variance T = U_bar + (1 + 1/M)*B >= U_bar
    estimates = [10.2, 10.5, 9.8, 10.1, 10.4]
    variances = [0.25, 0.28, 0.24, 0.26, 0.27]

    res = rubins_rules(estimates, variances)

    u_bar = float(np.mean(variances))
    assert res.total_variance >= u_bar
    assert res.within_variance == pytest.approx(u_bar, abs=1e-6)
    assert res.between_variance >= 0.0
    assert res.degrees_of_freedom > 0.0
    assert res.ci_lower < res.pooled_mean < res.ci_upper


def test_sensitivity_shift_monotonicity():
    rng = np.random.RandomState(42)
    n = 150
    x = rng.randn(n)
    y = 2.0 * x + rng.randn(n)
    y[0:30] = np.nan

    df = pd.DataFrame({"x": x, "y": y})

    grid_res = run_sensitivity_grid(
        df,
        target_column="y",
        delta_grid=list(np.linspace(-1.0, 1.0, 9)),
        shift_type="standardized",
        random_state=42,
    )

    df_grid = grid_res.grid_df
    # Invariant: As delta increases, the mean estimate of the incomplete variable must monotonically increase
    estimates = df_grid["target_mean"].to_numpy()
    deltas = df_grid["delta"].to_numpy()

    # Monotonicity check
    assert np.all(np.diff(deltas) > 0), "Delta grid must be monotonically increasing"
    diffs = np.diff(estimates)
    assert np.all(diffs >= -1e-10), "Sensitivity curve must be non-decreasing in delta"
    assert estimates[-1] > estimates[0] + 1e-6, "Sensitivity curve must have positive total range"


def test_transform_preserves_observed_values():
    """Critical invariant: imputation must not alter observed (non-missing) values."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({"x": rng.randn(n), "y": rng.randn(n)})
    df.loc[:30, "y"] = np.nan
    obs_mask = df["y"].notna()
    y_observed_original = df.loc[obs_mask, "y"].to_numpy().copy()

    imp = UmbraImputer(strategy="mar", random_state=42)
    result = imp.fit_transform(df)

    np.testing.assert_allclose(
        result.loc[obs_mask, "y"].to_numpy(),
        y_observed_original,
        rtol=1e-10,
        err_msg="Observed values must not be modified by imputation",
    )


def test_sensitivity_reports_only_medium_high_risk_columns():
    """Verify that sensitivity_reports_ is only populated for MEDIUM/HIGH risk columns."""
    rng = np.random.RandomState(42)
    n = 300
    df = pd.DataFrame(
        {
            "x": rng.randn(n),
            "y_mcar": rng.randn(n),
            "z": rng.randn(n),
            "y_mnar": rng.randn(n),
        }
    )
    # y_mcar is MCAR -> LOW risk
    df.loc[:30, "y_mcar"] = np.nan
    # y_mnar has tail selection -> MEDIUM/HIGH risk
    df.loc[df["y_mnar"] > 0.5, "y_mnar"] = np.nan

    imp = UmbraImputer(strategy="auto", random_state=42)
    imp.fit(df)
    for col in imp.sensitivity_reports_:
        risk = imp.diagnostics_[col].risk_level
        assert risk in ("MEDIUM", "HIGH"), f"Column '{col}' had {risk} risk but got sensitivity report"


def test_reproducibility_seed_determinism():
    """Verify that identical random seeds yield identical imputations across runs."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame(
        {
            "a": rng.randn(n),
            "b": rng.randn(n),
            "c": rng.randn(n),
        }
    )
    df.loc[0:40, "a"] = np.nan
    df.loc[30:70, "b"] = np.nan

    # Run 1
    imp1 = UmbraImputer(strategy="mar", random_state=123)
    res1 = imp1.fit_transform(df.copy())

    # Run 2
    imp2 = UmbraImputer(strategy="mar", random_state=123)
    res2 = imp2.fit_transform(df.copy())

    np.testing.assert_allclose(
        res1.to_numpy(),
        res2.to_numpy(),
        rtol=1e-10,
        atol=1e-10,
        err_msg="Imputations with identical seeds must be bitwise deterministic",
    )
