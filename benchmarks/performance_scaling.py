"""
Runtime & Memory Performance Scaling Benchmarks for Umbra.

Benchmarks computation time as a function of:
1. Sample size N in {250, 500, 1000, 2500, 5000, 10000}
2. Dimensionality p in {4, 10, 20, 40}
3. Missingness rate in {10%, 20%, 30%, 40%, 50%}
4. Imputation method (MICE PMM, MICE Ridge, Heckman, Pattern Mixture, Umbra Auto)
"""

import time
from typing import List, Optional

import numpy as np
import pandas as pd

from umbra.api import UmbraImputer
from umbra.imputers.heckman_selection import HeckmanSelectionImputer
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer
from umbra.imputers.pattern_mixture import PatternMixtureImputer


def benchmark_runtime_vs_n(
    sample_sizes: Optional[List[int]] = None,
    n_features: int = 5,
    missing_rate: float = 0.30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure runtime (seconds) across sample sizes N."""
    if sample_sizes is None:
        sample_sizes = [250, 500, 1000, 2500, 5000, 10000]

    rng = np.random.RandomState(random_state)
    results = []

    for n in sample_sizes:
        # Generate synthetic data matrix
        X = rng.normal(0, 1, size=(n, n_features))
        # Mask 30% of column 0
        mask = rng.uniform(0, 1, size=n) < missing_rate
        X[mask, 0] = np.nan
        df = pd.DataFrame(X, columns=[f"x_{i}" for i in range(n_features)])

        methods = {
            "MAR MICE (Ridge)": MARChainedEquationsImputer(
                imputation_method="ridge", max_iter=5, random_state=random_state
            ),
            "MAR MICE (PMM)": MARChainedEquationsImputer(
                imputation_method="pmm", max_iter=5, random_state=random_state
            ),
            "Heckman Selection": HeckmanSelectionImputer(
                shadow_cols={"x_0": "x_1"}, random_state=random_state
            ),
            "Pattern Mixture": PatternMixtureImputer(delta=0.0, random_state=random_state),
            "Umbra (Auto)": UmbraImputer(
                strategy="auto",
                shadow_cols={"x_0": "x_1"},
                run_sensitivity=False,
                random_state=random_state,
            ),
        }

        row = {"N": n, "p": n_features, "missing_rate": missing_rate}
        for name, imp in methods.items():
            t0 = time.perf_counter()
            imp.fit_transform(df)
            elapsed = time.perf_counter() - t0
            row[name] = round(elapsed, 4)

        results.append(row)

    return pd.DataFrame(results)


def benchmark_runtime_vs_dimension(
    feature_counts: Optional[List[int]] = None,
    n_samples: int = 2000,
    missing_rate: float = 0.30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure runtime (seconds) across feature count p."""
    if feature_counts is None:
        feature_counts = [4, 8, 16, 32, 64]

    rng = np.random.RandomState(random_state)
    results = []

    for p in feature_counts:
        X = rng.normal(0, 1, size=(n_samples, p))
        mask = rng.uniform(0, 1, size=n_samples) < missing_rate
        X[mask, 0] = np.nan
        df = pd.DataFrame(X, columns=[f"x_{i}" for i in range(p)])

        methods = {
            "MAR MICE (Ridge)": MARChainedEquationsImputer(
                imputation_method="ridge", max_iter=5, random_state=random_state
            ),
            "Heckman Selection": HeckmanSelectionImputer(
                shadow_cols={"x_0": "x_1"}, random_state=random_state
            ),
            "Pattern Mixture": PatternMixtureImputer(delta=0.0, random_state=random_state),
            "Umbra (Auto)": UmbraImputer(
                strategy="auto",
                shadow_cols={"x_0": "x_1"},
                run_sensitivity=False,
                random_state=random_state,
            ),
        }

        row = {"N": n_samples, "p": p, "missing_rate": missing_rate}
        for name, imp in methods.items():
            t0 = time.perf_counter()
            imp.fit_transform(df)
            elapsed = time.perf_counter() - t0
            row[name] = round(elapsed, 4)

        results.append(row)

    return pd.DataFrame(results)
