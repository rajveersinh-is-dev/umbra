"""
Unit tests for scikit-learn API compatibility.
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from scripts.build_synthetic_benchmarks import generate_benchmark_battery
from umbra.api import UmbraImputer


@pytest.fixture(scope="module")
def benchmarks():
    return generate_benchmark_battery(n_samples=1000, random_state=42)


def test_umbra_imputer_fit_transform(benchmarks):
    data = benchmarks["MAR"].data_observed.copy()
    imputer = UmbraImputer(strategy="mar", random_state=42)
    imputer.fit(data)
    res = imputer.transform(data)

    assert not res.isna().any().any()
    assert len(res) == len(data)


def test_umbra_imputer_auto_strategy(benchmarks):
    data = benchmarks["MNAR_MEDIUM"].data_observed.copy()
    imputer = UmbraImputer(
        strategy="auto",
        shadow_cols={"income": "shadow_z"},
        run_sensitivity=True,
        random_state=42,
    )
    res, diags = imputer.fit_transform(data, return_diagnostics=True)

    assert not res.isna().any().any()
    assert "income" in diags
    assert diags["income"].risk_level == "HIGH"
    assert imputer.get_sensitivity("income") is not None


def test_pipeline_integration(benchmarks):
    data = benchmarks["MNAR_LOW"].data_observed.copy()
    y = np.random.randn(len(data))

    pipe = Pipeline(
        [
            ("imputer", UmbraImputer(strategy="auto", run_sensitivity=False, random_state=42)),
            ("reg", Ridge()),
        ]
    )

    pipe.fit(data, y)
    preds = pipe.predict(data)
    assert len(preds) == len(data)
    assert not np.isnan(preds).any()


def test_numpy_array_input():
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [np.nan, 3.0, 4.0],
            [2.0, np.nan, 5.0],
            [3.0, 4.0, 6.0],
            [4.0, 5.0, 7.0],
        ]
    )
    imputer = UmbraImputer(strategy="mar", run_sensitivity=False, random_state=42)
    X_imp = imputer.fit_transform(X)
    assert isinstance(X_imp, np.ndarray)
    assert not np.isnan(X_imp).any()
