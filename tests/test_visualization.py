"""
Unit tests for Umbra scientific visualizations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts.build_synthetic_benchmarks import generate_benchmark_battery
from umbra.diagnostics.mnar_risk_score import assess_mnar_risk
from umbra.diagnostics.pattern_analysis import analyze_missingness_patterns
from umbra.sensitivity.grid_analysis import run_sensitivity_grid
from umbra.visualization.figures import (
    plot_covariate_shifts,
    plot_diagnostic_evidence,
    plot_missingness_matrix,
    plot_router_confusion_matrix,
    plot_sensitivity_curve,
)


@pytest.fixture(scope="module")
def benchmarks():
    return generate_benchmark_battery(n_samples=500, random_state=42)


def test_plot_missingness_matrix(benchmarks, tmp_path: Path):
    df = benchmarks["MAR"].data_observed
    out_file = tmp_path / "matrix.png"
    fig = plot_missingness_matrix(df, max_rows=50, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    plt.close(fig)


def test_plot_covariate_shifts(benchmarks, tmp_path: Path):
    df = benchmarks["MAR"].data_observed
    pat = analyze_missingness_patterns(df)
    out_file = tmp_path / "shifts.png"
    fig = plot_covariate_shifts(pat, target_column="income", save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    plt.close(fig)


def test_plot_sensitivity_curve(benchmarks, tmp_path: Path):
    df = benchmarks["MNAR_HIGH"].data_observed
    sens = run_sensitivity_grid(df, target_column="income", random_state=42)
    out_file = tmp_path / "sens.png"
    fig = plot_sensitivity_curve(sens, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    plt.close(fig)


def test_plot_diagnostic_evidence(benchmarks, tmp_path: Path):
    df = benchmarks["MNAR_HIGH"].data_observed
    report = assess_mnar_risk(df, target_col="income")
    out_file = tmp_path / "evidence.png"
    fig = plot_diagnostic_evidence(report, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    plt.close(fig)


def test_plot_router_confusion_matrix(tmp_path: Path):
    data = np.array([[0.95, 0.05, 0.00], [0.10, 0.85, 0.05], [0.00, 0.15, 0.85]])
    df_cm = pd.DataFrame(
        data,
        index=["MCAR", "MAR", "MNAR"],
        columns=["mar", "pattern_mixture", "heckman"],
    )
    out_file = tmp_path / "confusion.png"
    fig = plot_router_confusion_matrix(df_cm, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    plt.close(fig)
