"""
Unit tests for Umbra diagnostics module.
"""

import numpy as np
import pytest

from scripts.build_synthetic_benchmarks import generate_benchmark_battery
from umbra.diagnostics.mcar_test import littles_mcar_test
from umbra.diagnostics.mnar_risk_score import assess_mnar_risk, diagnose_dataframe
from umbra.diagnostics.pattern_analysis import (
    analyze_missingness_patterns,
    compute_cliffs_delta,
    compute_cohens_d,
)
from umbra.diagnostics.shadow_variable_finder import find_shadow_variables


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
