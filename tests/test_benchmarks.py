"""
Smoke tests for benchmark runner and scaling modules.
"""

from benchmarks.performance_scaling import benchmark_runtime_vs_dimension, benchmark_runtime_vs_n


def test_performance_scaling_smoke():
    df_n = benchmark_runtime_vs_n(sample_sizes=[50, 100], n_features=3, random_state=42)
    assert len(df_n) == 2
    assert "N" in df_n.columns
    assert "Umbra (Auto)" in df_n.columns

    df_p = benchmark_runtime_vs_dimension(feature_counts=[3, 4], n_samples=100, random_state=42)
    assert len(df_p) == 2
    assert "p" in df_p.columns
    assert "Umbra (Auto)" in df_p.columns
