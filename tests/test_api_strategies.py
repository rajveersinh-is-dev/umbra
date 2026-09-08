"""
Tests for explicit UmbraImputer strategies and fit_transform_multiple.
"""

import numpy as np
import pandas as pd

from umbra.api import UmbraImputer


def test_umbra_explicit_heckman_strategy():
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({"x": rng.randn(n), "z": rng.randn(n), "y": rng.randn(n)})
    df.loc[:50, "y"] = np.nan
    imp = UmbraImputer(strategy="heckman", shadow_cols={"y": "z"}, random_state=42)
    result = imp.fit_transform(df)
    assert not result["y"].isna().any()
    assert imp.routing_decisions_["y"] == "heckman_selection"
    assert imp.strategy_map_["y"] == "heckman"


def test_umbra_explicit_pattern_mixture_strategy():
    rng = np.random.RandomState(42)
    n = 150
    df = pd.DataFrame({"x": rng.randn(n), "y": rng.randn(n)})
    df.loc[:30, "y"] = np.nan
    imp = UmbraImputer(strategy="pattern_mixture", delta=0.5, random_state=42)
    result = imp.fit_transform(df)
    assert not result["y"].isna().any()
    assert imp.routing_decisions_["y"] == "pattern_mixture"
    assert imp.strategy_map_["y"] == "pattern_mixture"


def test_fit_transform_multiple_returns_list():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"x": rng.randn(100), "y": rng.randn(100)})
    df.loc[:20, "y"] = np.nan
    imp = UmbraImputer(strategy="mar", n_imputations=3, random_state=42)
    results = imp.fit_transform_multiple(df)
    assert isinstance(results, list)
    assert len(results) == 3
    for res in results:
        assert not res["y"].isna().any()


def test_fit_transform_multiple_n1_returns_singleton_list():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"x": rng.randn(50), "y": rng.randn(50)})
    df.loc[:10, "y"] = np.nan
    imp = UmbraImputer(strategy="mar", n_imputations=1, random_state=42)
    results = imp.fit_transform_multiple(df)
    assert isinstance(results, list)
    assert len(results) == 1
    assert not results[0]["y"].isna().any()


def test_fit_transform_multiple_auto_strategy():
    rng = np.random.RandomState(42)
    n = 300
    z = rng.randn(n)
    x = rng.randn(n)
    R = (0.5 * z + 0.8 * x + rng.randn(n) > 0).astype(int)
    y = 1.0 + 2.0 * x + rng.randn(n)
    y[R == 0] = np.nan
    df = pd.DataFrame({"x": x, "z": z, "y": y})

    imp = UmbraImputer(strategy="auto", shadow_cols={"y": "z"}, n_imputations=2, random_state=42)
    results = imp.fit_transform_multiple(df)
    assert len(results) == 2
    for r in results:
        assert not r["y"].isna().any()
