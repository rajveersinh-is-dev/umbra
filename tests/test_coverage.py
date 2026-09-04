"""
Unit tests for Rubin's rules and empirical coverage calculations.
"""

import numpy as np
import pytest

from umbra.imputers.heckman_selection import HeckmanSelectionImputer
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer, rubins_rules
from umbra.imputers.pattern_mixture import PatternMixtureImputer


def test_rubins_rules_basic():
    # 5 imputations
    point_estimates = [10.0, 10.2, 9.8, 10.1, 9.9]
    variances = [0.25, 0.24, 0.26, 0.25, 0.25]

    pooled = rubins_rules(point_estimates, variances, alpha=0.05)

    assert pytest.approx(pooled.pooled_mean, abs=1e-5) == 10.0
    assert pytest.approx(pooled.within_variance, abs=1e-3) == 0.25
    assert pooled.between_variance > 0.0
    assert pooled.total_variance > pooled.within_variance
    assert pooled.ci_lower < 10.0 < pooled.ci_upper
    assert pooled.degrees_of_freedom > 1.0


def test_rubins_rules_single_imputation():
    pooled = rubins_rules([5.0], [0.5], alpha=0.05)
    assert pytest.approx(pooled.pooled_mean) == 5.0
    assert pytest.approx(pooled.total_variance) == 0.5
    assert pooled.between_variance == 0.0


def test_rubins_rules_confidence_levels():
    point_estimates = [5.0, 5.1, 4.9, 5.2, 4.8]
    variances = [0.2, 0.2, 0.2, 0.2, 0.2]

    ci_95 = rubins_rules(point_estimates, variances, alpha=0.05)
    ci_90 = rubins_rules(point_estimates, variances, alpha=0.10)
    ci_80 = rubins_rules(point_estimates, variances, alpha=0.20)

    # Higher confidence level -> wider interval
    width_95 = ci_95.ci_upper - ci_95.ci_lower
    width_90 = ci_90.ci_upper - ci_90.ci_lower
    width_80 = ci_80.ci_upper - ci_80.ci_lower

    assert width_95 > width_90 > width_80


def test_multiple_draws_imputers():
    rng = np.random.RandomState(42)
    n = 100
    df = {"x": rng.randn(n), "y": rng.randn(n)}
    import pandas as pd

    df = pd.DataFrame(df)
    df.loc[0:20, "y"] = np.nan

    # Test MICE multiple draws
    mice = MARChainedEquationsImputer(n_draws=5, random_state=42)
    draws_mice = mice.fit_transform_multiple(df)
    assert len(draws_mice) == 5
    assert all(not d.isna().any().any() for d in draws_mice)
    # Check variation across draws for imputed values
    mis_mask = df["y"].isna()
    val1 = draws_mice[0].loc[mis_mask, "y"].values
    val2 = draws_mice[1].loc[mis_mask, "y"].values
    assert not np.allclose(val1, val2)

    # Test Pattern Mixture multiple draws
    pm = PatternMixtureImputer(n_draws=5, delta=1.0, random_state=42)
    draws_pm = pm.fit_transform_multiple(df)
    assert len(draws_pm) == 5
    assert all(not d.isna().any().any() for d in draws_pm)

    # Test Heckman multiple draws
    df["z"] = rng.randn(n)  # instrument
    heck = HeckmanSelectionImputer(
        target_cols=["y"], shadow_cols={"y": "z"}, n_draws=5, random_state=42
    )
    draws_heck = heck.fit_transform_multiple(df)
    assert len(draws_heck) == 5
    assert all(not d.isna().any().any() for d in draws_heck)


def test_heckman_rubin_pooled_ci_wider_than_single_plug_in():
    """Verify that Rubin-pooled 95% CI is strictly wider than single-imputation plug-in CI."""
    from benchmarks.dgps import generate_simulation_dataset
    from benchmarks.simulation_runner import evaluate_imputer_replication

    sim_data = generate_simulation_dataset(
        "MNAR_SELECTION", n_samples=500, missing_rate=0.30, random_state=42
    )

    res_single = evaluate_imputer_replication(
        lambda: HeckmanSelectionImputer(
            shadow_cols={"income": "shadow_z"},
            n_imputations=1,
            n_bootstrap_se=0,
            random_state=42,
        ),
        sim_data,
        rep_idx=0,
        n_imputations=1,
    )

    res_multiple = evaluate_imputer_replication(
        lambda: HeckmanSelectionImputer(
            shadow_cols={"income": "shadow_z"},
            n_imputations=5,
            stochastic=True,
            n_bootstrap_se=0,
            random_state=42,
        ),
        sim_data,
        rep_idx=0,
        n_imputations=5,
    )

    assert res_multiple.ci_width_95 > res_single.ci_width_95, (
        f"Expected Rubin pooled CI width ({res_multiple.ci_width_95:.4f}) > "
        f"single plug-in width ({res_single.ci_width_95:.4f})"
    )
