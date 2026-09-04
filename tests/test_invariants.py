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
    assert np.all(diffs > 0), "Sensitivity curve must be strictly monotonic in delta"


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
