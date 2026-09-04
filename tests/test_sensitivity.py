"""
Unit tests for Umbra sensitivity grid analysis.
"""

import pandas as pd
import pytest

from scripts.build_synthetic_benchmarks import generate_benchmark_battery
from umbra.sensitivity.grid_analysis import run_sensitivity_grid


@pytest.fixture(scope="module")
def benchmarks():
    return generate_benchmark_battery(n_samples=1000, random_state=42)


def test_sensitivity_grid_monotonicity(benchmarks):
    data = benchmarks["MNAR_MEDIUM"].data_observed.copy()
    report = run_sensitivity_grid(
        data, target_column="income", delta_grid=[-1.0, -0.5, 0.0, 0.5, 1.0]
    )

    means = report.grid_df["target_mean"].values
    # Monotonically increasing with delta
    assert all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    assert report.uncertainty_spread > 0.0
    assert report.mar_baseline_estimate > 0.0


def test_sensitivity_custom_downstream_evaluator(benchmarks):
    data = benchmarks["MNAR_LOW"].data_observed.copy()

    # Custom evaluator: difference in mean between high age and low age
    def custom_eval(df: pd.DataFrame) -> float:
        high_age = df[df["age"] > 0]["income"].mean()
        low_age = df[df["age"] <= 0]["income"].mean()
        return float(high_age - low_age)

    report = run_sensitivity_grid(
        data,
        target_column="income",
        delta_grid=[-1.0, 0.0, 1.0],
        downstream_evaluator=custom_eval,
    )
    assert len(report.grid_df) == 3
    assert "downstream_metric" in report.grid_df.columns
