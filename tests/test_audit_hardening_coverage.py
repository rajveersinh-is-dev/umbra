"""
Targeted tests for remaining edge cases to achieve >92% test coverage.
"""

import numpy as np
import pandas as pd
import pytest

from umbra.api import UmbraImputer
from umbra.diagnostics.report import diagnose_report
from umbra.imputers.heckman_selection import HeckmanSelectionImputer
from umbra.imputers.pattern_mixture import PatternMixtureImputer
from umbra.sensitivity.grid_analysis import run_sensitivity_grid


def test_api_fitted_explain_and_feature_names():
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({"x": rng.randn(n), "y": rng.randn(n)})
    df.loc[:20, "y"] = np.nan

    imp = UmbraImputer(strategy="mar", random_state=42)
    imp.fit(df)
    # Test explain on fitted imputer
    imp.explain()

    # Test get_feature_names_out
    names = imp.get_feature_names_out()
    assert list(names) == ["x", "y"]

    # Test transform with return_diagnostics and numpy array
    arr = df.to_numpy()
    res_arr, diags = imp.transform(arr, return_diagnostics=True)
    assert isinstance(res_arr, np.ndarray)
    assert "y" in diags


def test_api_pattern_mixture_in_auto_transform():
    # Force pattern_mixture in auto transform
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({"x": rng.randn(n), "income": rng.randn(n)})
    df.loc[:30, "income"] = np.nan

    imp = UmbraImputer(strategy="pattern_mixture", random_state=42)
    imp.fit(df)
    res = imp.transform(df)
    assert not res["income"].isna().any()


def test_pattern_mixture_edge_cases():
    rng = np.random.RandomState(42)
    n = 50
    df = pd.DataFrame({"x": rng.randn(n), "y": rng.randn(n)})
    df.loc[:10, "y"] = np.nan

    # Test percentage shift
    pm_pct = PatternMixtureImputer(delta=0.1, shift_type="percentage", random_state=42)
    res_pct = pm_pct.fit_transform(df)
    assert not res_pct["y"].isna().any()

    # Test raw shift
    pm_raw = PatternMixtureImputer(delta=2.0, shift_type="raw", random_state=42)
    res_raw = pm_raw.fit_transform(df)
    assert not res_raw["y"].isna().any()

    # Test invalid shift
    pm_inv = PatternMixtureImputer(delta=1.0, shift_type="invalid", random_state=42)
    pm_inv.fit(df)
    with pytest.raises(ValueError, match="Unknown shift_type"):
        pm_inv.transform(df)

    # Test unexpected kwargs
    with pytest.raises(TypeError, match="Unexpected keyword arguments"):
        PatternMixtureImputer(invalid_arg=123)

    # Test target with no other predictors (only target column in df)
    df_single = pd.DataFrame({"y": df["y"].copy()})
    pm_single = PatternMixtureImputer(delta=1.0, random_state=42)
    res_single = pm_single.fit_transform(df_single)
    assert not res_single["y"].isna().any()

    # Test get_feature_names_out and numpy input
    names = pm_pct.get_feature_names_out()
    assert list(names) == ["x", "y"]

    arr = df.to_numpy()
    res_arr = pm_pct.fit_transform(arr)
    assert not np.isnan(res_arr.to_numpy() if isinstance(res_arr, pd.DataFrame) else res_arr).any()

    # Test fit_transform_multiple
    mult = pm_pct.fit_transform_multiple(df)
    assert len(mult) >= 1


def test_heckman_edge_cases():
    rng = np.random.RandomState(42)
    n = 80
    z = rng.randn(n)
    x = rng.randn(n)
    y = 1.0 + 2.0 * x + rng.randn(n)
    mask = z > 0.5
    df = pd.DataFrame({"z": z, "x": x, "y": y})
    df.loc[mask, "y"] = np.nan

    # Unexpected kwargs
    with pytest.raises(TypeError, match="Unexpected keyword arguments"):
        HeckmanSelectionImputer(bogus_param=True)

    # Target not in df or target has no missing values
    h = HeckmanSelectionImputer(target_cols=["nonexistent", "x"], random_state=42)
    h.fit(df)
    assert "nonexistent" not in h.models_
    assert "x" not in h.models_

    # Feature names out and numpy array
    h_valid = HeckmanSelectionImputer(target_cols=["y"], shadow_cols={"y": "z"}, random_state=42)
    h_valid.fit(df)
    names = h_valid.get_feature_names_out()
    assert "y" in names

    # Fit and transform with numpy array
    arr = df.to_numpy()
    res_arr = h_valid.transform(arr)
    assert res_arr is not None


def test_sensitivity_edge_cases(tmp_path):
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({"x": rng.randn(n), "y": rng.randn(n), "outcome": rng.randn(n)})
    df.loc[:30, "y"] = np.nan

    # Target column not found
    with pytest.raises(ValueError, match="not found in data"):
        run_sensitivity_grid(df, target_column="nonexistent")

    # Downstream outcome and feature
    rep = run_sensitivity_grid(
        df,
        target_column="y",
        downstream_outcome="outcome",
        downstream_feature="y",
        delta_grid=[0.5, 1.0],  # No 0.0 in grid
        decision_threshold=0.0,
    )
    assert rep is not None
    assert rep.grid_df is not None

    # Decision threshold crossing
    rep2 = run_sensitivity_grid(
        df,
        target_column="y",
        delta_grid=[-2.0, 2.0],
        decision_threshold=df["y"].mean(),
    )
    assert rep2 is not None


def test_diagnostic_report_summary_and_export(tmp_path):
    rng = np.random.RandomState(42)
    n = 60
    df = pd.DataFrame({"x": rng.randn(n), "income": rng.randn(n)})
    df.loc[:15, "income"] = np.nan

    rep = diagnose_report(df)
    s = rep.summary()
    assert "income" in s

    # Export to paths
    p_json = tmp_path / "diag.json"
    p_md = tmp_path / "diag.md"
    p_html = tmp_path / "diag.html"

    rep.to_json(path=p_json)
    rep.to_markdown(path=p_md)
    rep.to_html(path=p_html)

    assert p_json.exists()
    assert p_md.exists()
    assert p_html.exists()
