"""
Unit tests for edge cases, numerical stability, and degenerate inputs.
"""

import numpy as np
import pandas as pd

from umbra.api import UmbraImputer, diagnose
from umbra.diagnostics.mcar_test import littles_mcar_test
from umbra.diagnostics.pattern_analysis import analyze_missingness_patterns
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer
from umbra.imputers.pattern_mixture import PatternMixtureImputer
from umbra.sensitivity.grid_analysis import run_sensitivity_grid


def test_complete_data_no_missing():
    """All methods should gracefully handle data with zero missing values."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    # Diagnostics
    mcar = littles_mcar_test(df)
    assert not mcar.is_rejected
    assert mcar.statistic == 0.0

    pat = analyze_missingness_patterns(df)
    assert len(pat.variable_reports) == 0

    diag = diagnose(df)
    assert diag.overall_missing_rate == 0.0

    # Imputers
    imp = UmbraImputer(strategy="auto")
    df_imp = imp.fit_transform(df)
    pd.testing.assert_frame_equal(df, df_imp)


def test_constant_column_zero_variance():
    """Handle columns with zero variance without division by zero."""
    df = pd.DataFrame(
        {
            "const": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "var": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "other": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )

    pat = analyze_missingness_patterns(df)
    assert "var" in pat.variable_reports

    imp = UmbraImputer(strategy="mar", random_state=42)
    df_imp = imp.fit_transform(df)
    assert not df_imp["var"].isna().any()


def test_high_missingness_rate():
    """Handle columns where 90% of data is missing."""
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.randn(n),
            "y": rng.randn(n),
        }
    )
    df.loc[10:, "y"] = np.nan  # 90% missing

    imp_mice = MARChainedEquationsImputer(random_state=42)
    df_imp = imp_mice.fit_transform(df)
    assert not df_imp["y"].isna().any()

    # Sensitivity grid on high missingness
    grid = run_sensitivity_grid(df, target_column="y", delta_grid=[-1.0, 0.0, 1.0])
    assert len(grid.grid_df) == 3


def test_extreme_delta_shifts():
    """Extreme delta shifts in Pattern Mixture should not cause overflow or NaN."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"x": rng.randn(50), "y": rng.randn(50)})
    df.loc[0:15, "y"] = np.nan

    pm_extreme = PatternMixtureImputer(delta=15.0, target_cols=["y"], random_state=42)
    df_pos = pm_extreme.fit_transform(df)
    assert not np.isinf(df_pos["y"]).any()
    assert not df_pos["y"].isna().any()

    pm_neg_extreme = PatternMixtureImputer(delta=-15.0, target_cols=["y"], random_state=42)
    df_neg = pm_neg_extreme.fit_transform(df)
    assert not np.isinf(df_neg["y"]).any()
    assert not df_neg["y"].isna().any()


def test_regularized_em_stability():
    """Near-singular or high-correlation data should be stabilized by eigenvalue clipping."""
    rng = np.random.RandomState(42)
    # Collinear data
    base = rng.randn(40)
    df = pd.DataFrame(
        {
            "x1": base,
            "x2": base + 1e-6 * rng.randn(40),
            "x3": base * 2.0 + 1e-6 * rng.randn(40),
        }
    )
    df.loc[0:10, "x1"] = np.nan

    # Little's test should not throw LinAlgError
    res = littles_mcar_test(df)
    assert res.statistic >= 0.0
    assert not np.isnan(res.p_value)
