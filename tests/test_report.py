"""
Unit tests for Umbra diagnostic report generation (umbra.diagnose).
"""

import json

import numpy as np
import pytest

import umbra
from scripts.build_synthetic_benchmarks import generate_benchmark_battery
from umbra.diagnostics.report import UmbraDiagnosticReport


@pytest.fixture(scope="module")
def benchmarks():
    return generate_benchmark_battery(n_samples=600, random_state=42)


def test_diagnose_mcar(benchmarks):
    df_mcar = benchmarks["MCAR"].data_observed.copy()
    report = umbra.diagnose(df_mcar, target_cols=["income"])

    assert isinstance(report, UmbraDiagnosticReport)
    assert report.mcar is not None
    assert report.mcar.statistic >= 0.0
    assert "income" in report.covariate_shift.variable_reports
    assert "income" in report.mnar_evidence

    # Tiered presentation
    assert len(report.recommendations) > 0

    # Serialization
    d = report.to_dict()
    assert isinstance(d, dict)
    assert "tier1_observed_data_evidence" in d
    assert "tier2_model_based_inference" in d
    assert "tier3_untestable_assumptions" in d
    assert "tier4_sensitivity_results" in d
    assert "recommendations" in d
    assert "mcar_test" in d["tier1_observed_data_evidence"]

    js = report.to_json()
    assert isinstance(js, str)
    parsed = json.loads(js)
    assert parsed["dataset_info"]["n_samples"] == len(df_mcar)

    md = report.to_markdown()
    assert "1. Observed-Data Evidence" in md
    assert "2. Model-Based Inference" in md
    assert "3. Untestable Assumptions" in md

    html = report.to_html()
    assert "<!DOCTYPE html>" in html
    assert "Umbra Diagnostic Audit" in html


def test_diagnose_mnar_with_sensitivity(benchmarks):
    df_mnar = benchmarks["MNAR_HIGH"].data_observed.copy()
    report = umbra.diagnose(
        df_mnar,
        target_cols=["income"],
        shadow_cols={"income": "shadow_z"},
        run_sensitivity=True,
    )

    assert "income" in report.mnar_evidence
    assert report.mnar_evidence["income"].risk_level == "HIGH"
    assert "income" in report.sensitivity
    sens = report.sensitivity["income"]
    assert sens.uncertainty_spread > 0.0
    assert len(sens.grid_df) > 0
    assert len(report.warnings) > 0


def test_diagnose_numpy_array():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    X[0:20, 0] = np.nan
    X[10:30, 1] = np.nan

    report = umbra.diagnose(X)
    assert isinstance(report, UmbraDiagnosticReport)
    assert report.n_samples == 100
    assert report.n_features == 4
    assert len(report.missing_counts) == 2
