"""
Unit tests for scikit-learn API compatibility.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from scripts.build_synthetic_benchmarks import generate_benchmark_battery
from umbra.api import UmbraImputer
from umbra.imputers.deep_generative_mnar import DeepGenerativeMNARImputer
from umbra.imputers.heckman_selection import HeckmanSelectionImputer
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer
from umbra.imputers.pattern_mixture import PatternMixtureImputer


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


def test_explain_and_get_sensitivity_not_fitted():
    imputer = UmbraImputer()
    with pytest.raises(NotFittedError):
        imputer.explain()
    with pytest.raises(NotFittedError):
        imputer.get_sensitivity("col")
    with pytest.raises(NotFittedError):
        imputer.get_feature_names_out()
    with pytest.raises(NotFittedError):
        imputer.transform(np.array([[1.0, 2.0]]))


def test_sub_imputers_not_fitted_error():
    X_dummy = np.array([[1.0, 2.0], [np.nan, 3.0]])
    imputers = [
        HeckmanSelectionImputer(),
        PatternMixtureImputer(),
        MARChainedEquationsImputer(),
        DeepGenerativeMNARImputer(),
    ]
    for imp in imputers:
        with pytest.raises(NotFittedError):
            imp.transform(X_dummy)
        with pytest.raises(NotFittedError):
            imp.get_feature_names_out()


def test_feature_dimension_mismatch_error():
    df_fit = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [3.0, 4.0, 5.0], "c": [6.0, 7.0, 8.0]})
    df_wrong_dim = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    estimators = [
        UmbraImputer(strategy="mar", run_sensitivity=False, random_state=42),
        MARChainedEquationsImputer(random_state=42),
        PatternMixtureImputer(random_state=42),
        HeckmanSelectionImputer(random_state=42),
        DeepGenerativeMNARImputer(epochs=1, random_state=42),
    ]

    for est in estimators:
        est.fit(df_fit)
        with pytest.raises(ValueError, match="is expecting 3 features"):
            est.transform(df_wrong_dim)


def test_feature_names_mismatch_error():
    df_fit = pd.DataFrame({"col_x": [1.0, 2.0, np.nan], "col_y": [3.0, 4.0, 5.0]})
    df_wrong_names = pd.DataFrame({"wrong_1": [1.0, 2.0, np.nan], "wrong_2": [3.0, 4.0, 5.0]})

    imputer = UmbraImputer(strategy="mar", run_sensitivity=False, random_state=42)
    imputer.fit(df_fit)
    with pytest.raises(ValueError, match="The feature names should match"):
        imputer.transform(df_wrong_names)


def test_get_feature_names_out_protocol():
    # 1. Fit with DataFrame
    df = pd.DataFrame({"feature_a": [1.0, np.nan, 3.0], "feature_b": [4.0, 5.0, 6.0]})
    imputer = UmbraImputer(strategy="mar", run_sensitivity=False, random_state=42)
    imputer.fit(df)

    names = imputer.get_feature_names_out()
    assert list(names) == ["feature_a", "feature_b"]

    # Custom input_features
    custom_names = imputer.get_feature_names_out(["f1", "f2"])
    assert list(custom_names) == ["f1", "f2"]

    with pytest.raises(ValueError, match="input_features should have length equal"):
        imputer.get_feature_names_out(["too_short"])

    # 2. Fit with NumPy array
    arr = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
    imp_arr = UmbraImputer(strategy="mar", run_sensitivity=False, random_state=42)
    imp_arr.fit(arr)
    assert not hasattr(imp_arr, "feature_names_in_")
    names_arr = imp_arr.get_feature_names_out()
    assert list(names_arr) == ["x0", "x1"]

    # 3. Sub-imputers with DataFrame and NumPy array
    sub_classes = [
        lambda: MARChainedEquationsImputer(random_state=42),
        lambda: PatternMixtureImputer(random_state=42),
        lambda: HeckmanSelectionImputer(random_state=42),
        lambda: DeepGenerativeMNARImputer(epochs=1, random_state=42),
    ]
    for factory in sub_classes:
        # Fit on DataFrame
        sub_imp = factory()
        sub_imp.fit(df)
        assert list(sub_imp.get_feature_names_out()) == ["feature_a", "feature_b"]
        assert list(sub_imp.get_feature_names_out(["c1", "c2"])) == ["c1", "c2"]
        with pytest.raises(ValueError, match="input_features should have length equal"):
            sub_imp.get_feature_names_out(["too_short"])

        # Fit on array
        sub_arr = factory()
        sub_arr.fit(arr)
        assert not hasattr(sub_arr, "feature_names_in_")
        assert list(sub_arr.get_feature_names_out()) == ["x0", "x1"]
        # Transform array
        trans_arr = sub_arr.transform(arr)
        assert trans_arr is not None
