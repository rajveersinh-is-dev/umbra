"""
Unit tests for Umbra diagnostics module.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.build_synthetic_benchmarks import generate_benchmark_battery
from umbra.diagnostics.mcar_test import littles_mcar_test
from umbra.diagnostics.mnar_risk_score import (
    _evaluate_residual_tail_dependency,
    assess_mnar_risk,
    diagnose_dataframe,
)
from umbra.diagnostics.pattern_analysis import (
    analyze_missingness_patterns,
    compute_cliffs_delta,
    compute_cohens_d,
)
from umbra.diagnostics.shadow_variable_finder import (
    _compute_first_stage_f_stat,
    find_shadow_variables,
)
from umbra.sensitivity.grid_analysis import run_sensitivity_grid


@pytest.fixture(scope="module")
def benchmarks():
    return generate_benchmark_battery(n_samples=1000, random_state=42)


def test_littles_mcar_on_complete_data():
    X = np.random.randn(100, 3)
    res = littles_mcar_test(X)
    assert not res.is_rejected
    assert res.degrees_of_freedom == 0
    assert res.statistic == 0.0


def test_littles_mcar_synthetic_battery(benchmarks):
    # MCAR should fail to reject (p > 0.01)
    mcar_res = littles_mcar_test(benchmarks["MCAR"].data_observed)
    assert mcar_res.p_value > 0.01, f"MCAR failed: p={mcar_res.p_value}"

    # MAR and MNAR should reject MCAR (p < 0.001)
    mar_res = littles_mcar_test(benchmarks["MAR"].data_observed)
    assert mar_res.is_rejected
    assert mar_res.p_value < 1e-4

    mnar_res = littles_mcar_test(benchmarks["MNAR_MEDIUM"].data_observed)
    assert mnar_res.is_rejected
    assert mnar_res.p_value < 1e-4


def test_pattern_analysis_effect_sizes():
    # Known shift
    x1 = np.zeros(100)
    x2 = np.ones(100)
    d = compute_cohens_d(x1, x2)
    # Variance of each is 0 so handled gracefully
    assert isinstance(d, float)

    x1 = np.random.normal(0, 1, 200)
    x2 = np.random.normal(1, 1, 200)
    d = compute_cohens_d(x1, x2)
    assert 0.7 < d < 1.3

    delta = compute_cliffs_delta(x1, x2)
    assert delta > 0.3


def test_pattern_analysis_covariate_shift(benchmarks):
    rep = analyze_missingness_patterns(benchmarks["MAR"].data_observed)
    var_rep = rep.variable_reports["income"]
    assert var_rep.n_missing > 0
    assert var_rep.max_ks_statistic > 0.10
    assert var_rep.n_significant_shifts > 0


def test_shadow_variable_finder(benchmarks):
    # In MNAR_MEDIUM, shadow_z is designed as the instrument
    rep = find_shadow_variables(benchmarks["MNAR_MEDIUM"].data_observed, target_column="income")
    assert len(rep.candidates) > 0
    best = rep.best_candidate
    assert best is not None
    assert best.variable_name == "shadow_z"
    assert best.is_promising_candidate


def test_mnar_risk_scoring(benchmarks):
    # MCAR with generic name
    df_mcar = benchmarks["MCAR"].data_observed.rename(columns={"income": "target_feature"})
    rep_mcar = assess_mnar_risk(df_mcar, target_col="target_feature")
    assert rep_mcar.risk_level in ("LOW", "MEDIUM")
    assert rep_mcar.composite_score < 0.40

    # MNAR High should be flagged as HIGH risk
    rep_mnar = assess_mnar_risk(benchmarks["MNAR_HIGH"].data_observed, target_col="income")
    assert rep_mnar.risk_level == "HIGH"
    assert rep_mnar.composite_score >= 0.50
    assert rep_mnar.domain_heuristic_matched

    # Diagnose dataframe
    all_reports = diagnose_dataframe(benchmarks["MNAR_MEDIUM"].data_observed)
    assert "income" in all_reports
    assert all_reports["income"].risk_level == "HIGH"


def test_littles_mcar_performance_vectorized():
    """Verify Little's MCAR test on N=5000, p=8 completes within 15 seconds."""
    import time

    rng = np.random.RandomState(42)
    n = 5000
    p = 8
    X = rng.randn(n, p)
    # Introduce missingness in multiple columns
    mask1 = rng.rand(n) < 0.20
    mask2 = rng.rand(n) < 0.15
    X[mask1, 0] = np.nan
    X[mask2, 1] = np.nan

    t0 = time.perf_counter()
    res = littles_mcar_test(X)
    elapsed = time.perf_counter() - t0

    assert res.statistic >= 0.0
    assert 0.0 <= res.p_value <= 1.0
    assert elapsed < 15.0, f"Little's test on N=5000 took {elapsed:.2f}s, expected < 15s"


def test_littles_mcar_to_json(tmp_path):
    X = np.random.randn(50, 3)
    X[0:5, 0] = np.nan
    res = littles_mcar_test(X)

    # String export
    json_str = res.to_json()
    assert '"statistic"' in json_str
    assert '"p_value"' in json_str

    # File export
    out_file = tmp_path / "mcar_result.json"
    res.to_json(path=out_file)
    assert out_file.exists()
    assert '"statistic"' in out_file.read_text(encoding="utf-8")


def test_first_stage_f_stat_zero_residual():
    # If Z perfectly predicts R, ssr_unres == 0
    R = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    Z = R.copy()  # perfect predictor
    X_covars = np.ones((len(R), 1))
    f_stat = _compute_first_stage_f_stat(R, Z, X_covars)
    assert f_stat >= 0.0
    assert np.isfinite(f_stat)


def test_tail_dependency_all_nan_covariate():
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame(
        {
            "target": np.where(rng.rand(n) < 0.3, np.nan, rng.randn(n)),
            "covar_good": rng.randn(n),
            "covar_all_nan": [np.nan] * n,
        }
    )
    # Should not crash on all-NaN covariate
    r_pred, tail_ratio, meta = _evaluate_residual_tail_dependency(
        df, "target", ["covar_good", "covar_all_nan"]
    )
    assert np.isfinite(r_pred)
    assert np.isfinite(tail_ratio)


def test_sensitivity_grid_non_overlapping_plausible_range():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    # Custom delta grid entirely outside [-1.0, 1.0]
    report = run_sensitivity_grid(df, "y", delta_grid=[2.0, 3.0], random_state=42)
    assert report.estimate_min <= report.estimate_max
    assert np.isfinite(report.uncertainty_spread)


def test_littles_mcar_summary():
    X = np.random.randn(50, 3)
    X[0:5, 0] = np.nan
    res = littles_mcar_test(X)
    summary_text = res.summary()
    assert "Little's MCAR Test" in summary_text
    assert "Chi-squared Statistic" in summary_text


def test_covariate_shift_categorical_summary():
    from umbra.diagnostics.pattern_analysis import CovariateShift

    shift = CovariateShift(
        covariate_name="cat_var",
        is_numeric=False,
        chi2_statistic=12.4,
        chi2_p_value=0.002,
        cramers_v=0.35,
        is_significant=True,
    )
    s = shift.summary()
    assert "Chi2=" in s
    assert "Cramer's V=" in s
    assert "SHIFT DETECTED" in s
