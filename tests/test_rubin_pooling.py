"""
Unit tests for Multiple Imputation Rubin Pooling Engine,
Barnard-Rubin (1999) small-sample degrees of freedom, and multi-draw interfaces.
"""

from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pytest

from umbra.imputers.deep_generative_mnar import HAS_TORCH, DeepGenerativeMNARImputer
from umbra.imputers.heckman_selection import HeckmanSelectionImputer
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer
from umbra.imputers.pattern_mixture import PatternMixtureImputer
from umbra.imputers.rubin_pooler import RubinPooler, RubinsRulesResult, rubins_rules


def _make_test_data(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    z = rng.randn(n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + rng.randn(n)

    # Induce missingness
    p_mis = 1.0 / (1.0 + np.exp(-(0.5 * x1 + 0.8 * z)))
    mis_mask = rng.rand(n) < p_mis
    y_obs = y.copy()
    y_obs[mis_mask] = np.nan

    return pd.DataFrame({"x1": x1, "x2": x2, "z": z, "y": y_obs})


def test_transform_multiple_all_imputers() -> None:
    """Verify transform_multiple(X, m=5) returns 5 complete DataFrames across all imputers."""
    df = _make_test_data(n=100, seed=42)

    imputers: List[Tuple[str, Any]] = [
        ("MICE", MARChainedEquationsImputer(imputation_method="pmm", random_state=42)),
        (
            "Heckman",
            HeckmanSelectionImputer(
                target_cols=["y"], shadow_cols={"y": "z"}, stochastic=True, random_state=42
            ),
        ),
        ("PM", PatternMixtureImputer(delta=0.5, stochastic=True, random_state=42)),
    ]
    if HAS_TORCH:
        imputers.append(
            ("DeepGen", DeepGenerativeMNARImputer(epochs=10, stochastic=True, random_state=42))
        )

    for name, imp in imputers:
        imp.fit(df)
        dfs = imp.transform_multiple(df, m=5, random_state=123)
        assert len(dfs) == 5, f"{name}: expected 5 imputed DataFrames, got {len(dfs)}"
        for i, d in enumerate(dfs):
            assert isinstance(d, pd.DataFrame), f"{name}: item {i} is not a DataFrame"
            assert not d["y"].isna().any(), f"{name}: draw {i} still has NaNs"

        # Verify draws are stochastic and vary across imputations
        y_draws = np.array([d["y"].to_numpy() for d in dfs])
        var_across_draws = np.var(y_draws, axis=0)
        assert np.any(var_across_draws > 0), f"{name}: draws are identical, not stochastic"


def test_fit_transform_multiple_shortcut() -> None:
    """Verify fit_transform_multiple is a convenient one-step equivalent."""
    df = _make_test_data(n=80, seed=42)
    imp = MARChainedEquationsImputer(imputation_method="pmm", random_state=42)
    dfs = imp.fit_transform_multiple(df, m=4)
    assert len(dfs) == 4
    for d in dfs:
        assert not d["y"].isna().any()


def test_rubin_pooler_scalar_and_vector() -> None:
    """Verify Rubin's rules variance pooling matches analytical formulas."""
    # Q_m: point estimates, U_m: within variances
    q = [10.0, 10.5, 9.5, 10.2, 9.8]
    u = [1.0, 1.1, 0.9, 1.05, 0.95]
    m = len(q)

    res = rubins_rules(point_estimates=q, variance_estimates=u, alpha=0.05)
    assert isinstance(res, RubinsRulesResult)

    expected_q_bar = float(np.mean(q))
    expected_u_bar = float(np.mean(u))
    expected_b_var = float(np.var(q, ddof=1))
    expected_t_var = float(expected_u_bar + (1.0 + 1.0 / m) * expected_b_var)
    expected_se = float(np.sqrt(expected_t_var))

    assert np.isclose(res.pooled_estimate, expected_q_bar)
    assert np.isclose(res.within_variance, expected_u_bar)
    assert np.isclose(res.between_variance, expected_b_var)
    assert np.isclose(res.total_variance, expected_t_var)
    assert np.isclose(res.standard_error, expected_se)
    assert res.ci_lower < res.pooled_estimate < res.ci_upper

    # Test vector pooling via RubinPooler.pool_estimates
    q_mat = np.array([[10.0, 2.0], [10.5, 2.2], [9.5, 1.8], [10.2, 2.1], [9.8, 1.9]])
    u_mat = np.array([[1.0, 0.1], [1.1, 0.11], [0.9, 0.09], [1.05, 0.1], [0.95, 0.09]])

    pooler = RubinPooler()
    vec_res = pooler.pool_estimates(q_mat, u_mat)
    assert isinstance(vec_res, dict)
    assert "param_0" in vec_res and "param_1" in vec_res
    assert np.isclose(vec_res["param_0"].pooled_estimate, expected_q_bar)
    assert np.isclose(vec_res["param_1"].pooled_estimate, 2.0)


def test_barnard_rubin_adjustment_bounds() -> None:
    """Verify Barnard & Rubin (1999) small-sample adjusted degrees of freedom."""
    q = [1.2, 1.5, 0.9, 1.3, 1.1]
    u = [0.04, 0.05, 0.03, 0.045, 0.035]

    # 1. Without df_complete (standard Rubin 1987 large-sample dof)
    res_large = rubins_rules(q, u, df_complete=None)
    df_rubin = res_large.df_rubin
    assert np.isfinite(df_rubin)
    assert res_large.df == df_rubin
    assert res_large.df_adjusted is None

    # 2. With small complete-sample dof: nu_0 = 15
    nu_0 = 15.0
    res_small = rubins_rules(q, u, df_complete=nu_0)
    assert res_small.df_adjusted is not None
    # Key invariant: nu_adj <= nu_0 strictly
    assert res_small.df <= nu_0, f"Expected nu_adj <= nu_0 ({nu_0}), got {res_small.df}"
    # Key invariant: nu_adj <= nu_rubin strictly
    assert res_small.df <= df_rubin, f"Expected nu_adj <= nu_rubin ({df_rubin}), got {res_small.df}"

    # Critical t-value is larger for smaller dof -> wider CI
    width_large = res_large.ci_upper - res_large.ci_lower
    width_small = res_small.ci_upper - res_small.ci_lower
    assert width_small > width_large, (
        f"Small-sample adjusted CI ({width_small:.4f}) should be wider than large-sample ({width_large:.4f})"
    )

    # 3. As nu_0 -> infinity, nu_adj -> nu_rubin
    res_inf = rubins_rules(q, u, df_complete=1e8)
    assert np.isclose(res_inf.df, df_rubin, rtol=1e-3)


def test_rubin_pooler_model_fit() -> None:
    """Verify RubinPooler fits downstream OLS across M DataFrames and pools parameters."""
    rng = np.random.RandomState(42)
    n = 150
    x1 = rng.randn(n)
    x2 = rng.randn(n)

    # Create M=5 imputed datasets with slight imputation variance
    imputed_dfs: List[pd.DataFrame] = []
    for m_idx in range(5):
        y_m = 2.0 + 3.0 * x1 - 1.5 * x2 + rng.randn(n) * 0.8
        imputed_dfs.append(pd.DataFrame({"x1": x1, "x2": x2, "y": y_m}))

    pooler = RubinPooler()
    pooler.fit(imputed_dfs, target="y", feature_cols=["x1", "x2"])

    assert pooler.is_fitted_ is True
    assert len(pooler.coef_) == 2
    assert np.isclose(pooler.intercept_, 2.0, atol=0.3)
    assert np.isclose(pooler.coef_[0], 3.0, atol=0.3)
    assert np.isclose(pooler.coef_[1], -1.5, atol=0.3)

    # Standard errors and p-values
    assert np.all(pooler.stderr_ > 0.0)
    assert pooler.pvalues_[0] < 0.001  # x1 is highly significant
    assert pooler.pvalues_[1] < 0.001  # x2 is highly significant

    # Diagnostics
    assert len(pooler.fmi_) == 3  # const, x1, x2
    assert np.all((pooler.fmi_ >= 0.0) & (pooler.fmi_ <= 1.0))

    # Summary table
    summary = pooler.summary()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 3
    assert "parameter" in summary.columns
    assert "std_error" in summary.columns
    assert "fmi" in summary.columns


def test_rubin_pooled_ci_wider_than_single_plugin() -> None:
    """Verify that Rubin pooling produces strictly wider CIs than naive single plug-in imputation."""
    rng = np.random.RandomState(42)
    # Simulated estimates across M=10 imputations with genuine imputation variation
    m = 10
    q_draws = 5.0 + rng.randn(m) * 0.4
    u_draws = np.full(m, 0.15)  # within-imputation variance

    # 1. Naive single plug-in: ignores between-imputation variance B
    naive_se = float(np.sqrt(u_draws[0]))
    naive_ci_width = 2.0 * 1.96 * naive_se

    # 2. Rubin's rules: total variance T = U_bar + (1 + 1/M) * B
    rubin_res = rubins_rules(point_estimates=q_draws.tolist(), variance_estimates=u_draws.tolist())
    rubin_ci_width = rubin_res.ci_upper - rubin_res.ci_lower

    assert rubin_res.between_variance > 0.0
    assert rubin_res.total_variance > rubin_res.within_variance
    assert rubin_ci_width > naive_ci_width, (
        f"Rubin CI width ({rubin_ci_width:.4f}) must strictly exceed single plug-in ({naive_ci_width:.4f})"
    )


def test_rubin_pooler_fmi_and_efficiency() -> None:
    """Verify Fraction of Missing Information and Relative Efficiency."""
    q_low_b = [10.0, 10.01, 9.99, 10.0, 10.01]  # low missingness variation
    u_base = [1.0, 1.0, 1.0, 1.0, 1.0]
    res_low = rubins_rules(q_low_b, u_base)

    q_high_b = [10.0, 12.0, 8.0, 11.5, 8.5]  # high missingness variation
    res_high = rubins_rules(q_high_b, u_base)

    assert res_high.fmi > res_low.fmi, (
        f"High B FMI ({res_high.fmi}) should exceed Low B ({res_low.fmi})"
    )
    assert res_high.relative_variance > res_low.relative_variance
    assert res_high.relative_efficiency <= 1.0
    assert res_low.relative_efficiency <= 1.0


def test_rubins_rules_input_validation_and_edge_cases() -> None:
    """Verify input validation and edge cases in rubins_rules."""
    with pytest.raises(ValueError, match="requires at least M=1"):
        rubins_rules([], [])

    with pytest.raises(ValueError, match="Length mismatch"):
        rubins_rules([1.0, 2.0], [0.1])

    # M=1 emits warning and produces zero between-variance
    with pytest.warns(UserWarning, match="rubins_rules called with M=1"):
        m1_res = rubins_rules([5.0], [0.5], df_complete=20.0)
    assert m1_res.between_variance == 0.0
    assert m1_res.total_variance == 0.5
    assert m1_res.df == 20.0

    # Test dictionary conversion
    d = m1_res.to_dict()
    assert "pooled_estimate" in d
    assert "total_variance" in d
