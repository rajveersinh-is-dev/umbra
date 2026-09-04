"""
Unit tests for the Umbra Auto Router and MNAR Risk Assessor.
"""

import pytest

from scripts.build_synthetic_benchmarks import generate_benchmark_battery
from umbra.api import UmbraImputer
from umbra.diagnostics.mnar_risk_score import assess_mnar_risk, diagnose_dataframe


@pytest.fixture(scope="module")
def benchmarks():
    return generate_benchmark_battery(n_samples=800, random_state=42)


def test_router_mcar_no_false_high_risk(benchmarks):
    """
    Critical requirement: Column naming heuristics must NOT override
    statistically clear MCAR data.
    """
    df_mcar = benchmarks["MCAR"].data_observed.copy()
    # Even if column is called 'income', data is strictly MCAR (p-value > 0.05)
    report = assess_mnar_risk(df_mcar, target_col="income")
    assert report.risk_level in ("LOW", "MEDIUM"), f"MCAR misclassified as {report.risk_level}"
    assert report.composite_score < 0.45


def test_router_mar_classification(benchmarks):
    df_mar = benchmarks["MAR"].data_observed.copy()
    report = assess_mnar_risk(df_mar, target_col="income")
    # MAR exhibits covariate shifts but Little's test rejects MCAR
    assert report.covariate_shift_score > 0.0
    assert report.risk_level in ("LOW", "MEDIUM")


def test_router_mnar_high_classification(benchmarks):
    df_mnar = benchmarks["MNAR_HIGH"].data_observed.copy()
    report = assess_mnar_risk(df_mnar, target_col="income")
    assert report.risk_level == "HIGH"
    assert report.composite_score >= 0.50


def test_auto_strategy_selection(benchmarks):
    # Case 1: MNAR with shadow column -> Heckman
    df_mnar = benchmarks["MNAR_MEDIUM"].data_observed.copy()
    imp_heck = UmbraImputer(
        strategy="auto",
        shadow_cols={"income": "shadow_z"},
        random_state=42,
    )
    imp_heck.fit(df_mnar)
    assert imp_heck.strategy_map_["income"] == "heckman"

    # Case 2: MNAR without shadow column -> Pattern mixture
    df_no_shadow = df_mnar.drop(columns=["shadow_z"])
    imp_pm = UmbraImputer(
        strategy="auto",
        shadow_cols=None,
        random_state=42,
    )
    imp_pm.fit(df_no_shadow)
    assert imp_pm.strategy_map_["income"] == "pattern_mixture"

    # Case 3: MCAR data -> MAR / MICE
    df_mcar = benchmarks["MCAR"].data_observed.copy()
    imp_mcar = UmbraImputer(
        strategy="auto",
        random_state=42,
    )
    imp_mcar.fit(df_mcar)
    assert imp_mcar.strategy_map_["income"] == "mar"


def test_diagnose_dataframe_all_columns(benchmarks):
    df = benchmarks["MNAR_LOW"].data_observed.copy()
    reports = diagnose_dataframe(df)
    assert isinstance(reports, dict)
    assert "income" in reports
    for col, rep in reports.items():
        assert 0.0 <= rep.composite_score <= 1.0
        assert rep.risk_level in ("LOW", "MEDIUM", "HIGH")


def test_router_strict_dispatch_and_expected_field():
    """Verify that evaluate_single_routing returns expected_dispatch and adheres to strict correctness."""
    from benchmarks.router_benchmark import evaluate_single_routing

    res_mcar = evaluate_single_routing("MCAR", n_samples=300, missing_rate=0.20, random_state=42)
    assert "expected_dispatch" in res_mcar
    assert res_mcar["expected_dispatch"] == "mar_chained_equations"
    assert res_mcar["is_correct"] is True

    res_mnar = evaluate_single_routing("MNAR_SELECTION", n_samples=300, missing_rate=0.30, random_state=42)
    assert "expected_dispatch" in res_mnar
    assert isinstance(res_mnar["is_correct"], bool)
