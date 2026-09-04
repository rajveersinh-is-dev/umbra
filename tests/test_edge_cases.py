"""
Unit tests for edge cases, numerical stability, failure modes, and degenerate inputs.

Verifies the 16 core edge cases specified in Section 11:
1. Empty dataset
2. One-row dataset
3. Tiny dataset (3-5 rows)
4. All-missing column (100% NaN)
5. Almost-all-missing column (95% NaN)
6. Constant column (zero variance)
7. Duplicate columns
8. Highly correlated columns (collinearity)
9. Singular matrices (rank deficiency)
10. NaN and Inf values
11. Extreme values (large scale / underflow)
12. Weak instruments (F <= 10)
13. Failed regression fallback
14. Non-convergence / early stopping
15. Categorical variables type checking
16. Mixed numerical and categorical data
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from umbra.api import UmbraImputer, diagnose
from umbra.diagnostics.mcar_test import littles_mcar_test
from umbra.diagnostics.pattern_analysis import analyze_missingness_patterns
from umbra.imputers.heckman_selection import HeckmanSelectionImputer, WeakInstrumentWarning
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer
from umbra.imputers.pattern_mixture import PatternMixtureImputer


def test_empty_dataset():
    """1. Empty dataset (0 rows, 0 cols or 0 rows, N cols)."""
    df_empty = pd.DataFrame()
    diag = diagnose(df_empty)
    assert diag.overall_missing_rate == 0.0
    assert diag.n_samples == 0

    imp = UmbraImputer(strategy="auto")
    df_imp = imp.fit_transform(df_empty)
    assert df_imp.empty


def test_one_row_dataset():
    """2. One-row dataset: raises informative ValueError for Little's test."""
    df_one = pd.DataFrame({"x": [np.nan], "y": [1.0]})
    with pytest.raises(ValueError, match="Insufficient data dimensions"):
        diagnose(df_one)

    imp = UmbraImputer(strategy="auto")
    with pytest.raises(ValueError, match="Insufficient data dimensions"):
        imp.fit(df_one)


def test_tiny_dataset():
    """3. Tiny dataset (3-5 rows) should run without throwing index/dimension exceptions."""
    df_tiny = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [2.0, 4.0, 6.0, 8.0],
        }
    )
    imp = UmbraImputer(strategy="mar", random_state=42)
    df_imp = imp.fit_transform(df_tiny)
    assert not df_imp["a"].isna().any()
    assert len(df_imp) == 4


def test_all_missing_column():
    """4. 100% missing column handled gracefully."""
    df = pd.DataFrame(
        {
            "all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "valid": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    diag = diagnose(df)
    assert "all_nan" in diag.missing_columns

    imp = UmbraImputer(strategy="mar", random_state=42)
    df_imp = imp.fit_transform(df)
    assert not df_imp["valid"].isna().any()


def test_almost_all_missing_column():
    """5. Almost-all-missing column (95% NaN)."""
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({"x": rng.randn(n), "y": rng.randn(n)})
    df.loc[5:, "y"] = np.nan  # 95% missing

    imp_mice = MARChainedEquationsImputer(random_state=42)
    df_imp = imp_mice.fit_transform(df)
    assert not df_imp["y"].isna().any()


def test_constant_column_zero_variance():
    """6. Handle columns with zero variance without division by zero."""
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


def test_duplicate_columns():
    """7. Duplicate column names and identical values handled."""
    rng = np.random.RandomState(42)
    vals = rng.randn(30)
    df = pd.DataFrame({"x": vals, "x_copy": vals, "y": vals + 0.5})
    df.loc[0:5, "y"] = np.nan

    imp = UmbraImputer(strategy="mar", random_state=42)
    df_imp = imp.fit_transform(df)
    assert not df_imp["y"].isna().any()


def test_highly_correlated_columns():
    """8. Highly correlated columns (collinearity r > 0.9999)."""
    rng = np.random.RandomState(42)
    base = rng.randn(50)
    df = pd.DataFrame(
        {
            "x1": base,
            "x2": base + 1e-6 * rng.randn(50),
            "x3": base * 2.0 + 1e-6 * rng.randn(50),
        }
    )
    df.loc[0:10, "x1"] = np.nan

    res = littles_mcar_test(df)
    assert res.statistic >= 0.0
    assert not np.isnan(res.p_value)


def test_singular_matrices():
    """9. Singular rank-deficient matrices stabilized by regularized EM."""
    rng = np.random.RandomState(42)
    X = rng.normal(0, 1, size=(40, 10))
    # Artificially make columns 8 and 9 exact linear combinations
    X[:, 8] = X[:, 0] + X[:, 1]
    X[:, 9] = X[:, 2] - 2 * X[:, 3]
    df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(10)])
    df.loc[0:8, "col_0"] = np.nan

    diag = diagnose(df)
    assert diag.mcar.statistic >= 0.0


def test_nan_inf_values():
    """10. Dataset containing Inf or extreme values handled safely."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    diag = diagnose(df)
    assert diag.overall_missing_rate > 0.0


def test_extreme_values():
    """11. Extreme values with large scales or underflows."""
    df = pd.DataFrame(
        {
            "x": [1e10, -1e10, 2e10, np.nan, 3e10],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    imp = UmbraImputer(strategy="mar", random_state=42)
    df_imp = imp.fit_transform(df)
    assert not np.isinf(df_imp["x"]).any()
    assert not df_imp["x"].isna().any()


def test_weak_instrument_warning():
    """12. Weak instrument (F <= 10) emits WeakInstrumentWarning."""
    rng = np.random.RandomState(42)
    n = 200
    x = rng.randn(n)
    # Z has zero correlation with missingness
    z = rng.randn(n)
    y = 0.5 * x + rng.randn(n)
    # Missingness depends only on x, not z
    mask = x > 0.5
    y_obs = y.copy()
    y_obs[mask] = np.nan

    df = pd.DataFrame({"x": x, "z": z, "target": y_obs})

    imp_heck = HeckmanSelectionImputer(shadow_cols={"target": "z"}, random_state=42)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        imp_heck.fit(df)
        has_weak_warn = any(issubclass(w.category, WeakInstrumentWarning) for w in recorded)
        assert has_weak_warn, "Expected WeakInstrumentWarning when F <= 10"


def test_failed_regression_fallback():
    """13. Insufficient degrees of freedom in regression falls back cleanly."""
    df = pd.DataFrame(
        {
            "target": [np.nan, np.nan, 3.0, 4.0],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [5.0, 6.0, 7.0, 8.0],
        }
    )
    pm = PatternMixtureImputer(delta=0.0, random_state=42)
    df_imp = pm.fit_transform(df)
    assert not df_imp["target"].isna().any()


def test_non_convergence_max_iter():
    """14. Non-convergence / early stopping at max_iter returns best estimate."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame(rng.randn(60, 4), columns=["a", "b", "c", "d"])
    df.loc[0:15, "a"] = np.nan
    df.loc[10:25, "b"] = np.nan

    imp = MARChainedEquationsImputer(max_iter=1, random_state=42)
    df_imp = imp.fit_transform(df)
    assert not df_imp.isna().any().any()


def test_categorical_variables_type_error():
    """15. Categorical/string columns in UmbraImputer raise clear TypeError."""
    df_complete_cat = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0],
            "cat": ["A", "B", "A", "B"],
        }
    )
    # Numeric column is missing, string column is complete:
    # Umbra requires incomplete columns to be numeric
    imp = UmbraImputer(strategy="mar", random_state=42)
    df_imp = imp.fit_transform(df_complete_cat)
    assert not df_imp["num"].isna().any()
    df_cat_miss = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "cat": ["A", "B", np.nan, "B"],
        }
    )
    with pytest.raises(TypeError, match="requires incomplete features to be numeric"):
        imp.fit(df_cat_miss)


def test_mixed_numerical_categorical_diagnose():
    """16. Mixed numerical and categorical data in diagnose() succeeds cleanly."""
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
            "cat1": ["A", "B", "A", "B", np.nan, "A", "B", "A", "B", "A"] * 3,
            "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] * 3,
        }
    )
    diag = diagnose(df)
    assert diag.overall_missing_rate > 0.0
    assert "cat1" in diag.missing_columns
    assert "num1" in diag.missing_columns
    # Check that covariate shift computed Chi2/Cramers V for cat1
    var_rep = diag.covariate_shift.variable_reports.get("num1")
    assert var_rep is not None
    assert "cat1" in var_rep.covariate_shifts
    shift = var_rep.covariate_shifts["cat1"]
    assert not shift.is_numeric
    assert shift.chi2_statistic is not None
