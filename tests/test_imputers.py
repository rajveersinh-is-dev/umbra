"""
Unit tests for Umbra imputers module.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.build_synthetic_benchmarks import generate_benchmark_battery
from umbra.imputers.deep_generative_mnar import HAS_TORCH, DeepGenerativeMNARImputer
from umbra.imputers.heckman_selection import HeckmanSelectionImputer
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer
from umbra.imputers.pattern_mixture import PatternMixtureImputer


@pytest.fixture(scope="module")
def benchmarks():
    return generate_benchmark_battery(n_samples=1000, random_state=42)


def test_mar_chained_equations(benchmarks):
    data = benchmarks["MAR"].data_observed.copy()
    assert data["income"].isna().any()

    # Test PMM
    pmm_imp = MARChainedEquationsImputer(imputation_method="pmm", max_iter=5, random_state=42)
    pmm_res = pmm_imp.fit_transform(data)
    assert not pmm_res.isna().any().any()
    assert len(pmm_res) == len(data)

    # Test Bayesian Ridge
    br_imp = MARChainedEquationsImputer(
        imputation_method="bayesian_ridge", max_iter=3, random_state=42
    )
    br_res = br_imp.fit_transform(data)
    assert not br_res.isna().any().any()


def test_heckman_selection_imputer(benchmarks):
    bench = benchmarks["MNAR_MEDIUM"]
    data = bench.data_observed.copy()
    obs_mean = data["income"].mean()
    true_mean = bench.true_params["true_mean"]

    heckman = HeckmanSelectionImputer(
        target_cols=["income"],
        shadow_cols={"income": "shadow_z"},
        random_state=42,
    )
    res = heckman.fit_transform(data)
    assert not res["income"].isna().any()

    heck_mean = res["income"].mean()
    # Bias reduction: Heckman mean should be closer to true mean than observed mean
    heck_bias = abs(heck_mean - true_mean)
    naive_bias = abs(obs_mean - true_mean)
    assert heck_bias < naive_bias, (
        f"Expected Heckman bias {heck_bias:.3f} < naive bias {naive_bias:.3f}"
    )


def test_pattern_mixture_delta_shift(benchmarks):
    data = benchmarks["MNAR_LOW"].data_observed.copy()

    pm_base = PatternMixtureImputer(delta=0.0, target_cols=["income"], random_state=42)
    df_base = pm_base.fit_transform(data)

    pm_pos = PatternMixtureImputer(delta=1.0, target_cols=["income"], random_state=42)
    df_pos = pm_pos.fit_transform(data)

    pm_neg = PatternMixtureImputer(delta=-1.0, target_cols=["income"], random_state=42)
    df_neg = pm_neg.fit_transform(data)

    # Missing rows should be shifted upwards with positive delta
    mis_mask = data["income"].isna()
    assert df_pos.loc[mis_mask, "income"].mean() > df_base.loc[mis_mask, "income"].mean()
    assert df_neg.loc[mis_mask, "income"].mean() < df_base.loc[mis_mask, "income"].mean()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_deep_generative_imputer():
    X = pd.DataFrame(
        {
            "a": np.random.randn(200),
            "b": np.random.randn(200),
            "c": np.random.randn(200),
        }
    )
    X.loc[0:40, "c"] = np.nan

    imp = DeepGenerativeMNARImputer(epochs=10, batch_size=32, random_state=42)
    X_imp = imp.fit_transform(X)
    assert not X_imp.isna().any().any()
