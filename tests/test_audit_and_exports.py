"""
Tests for report formatting, export methods, edge summaries, and diagnostics.
"""

import numpy as np
import pandas as pd
import pytest

import umbra
from umbra.api import UmbraImputer
from umbra.diagnostics.pattern_analysis import analyze_missingness_patterns
from umbra.diagnostics.shadow_variable_finder import find_shadow_variables
from umbra.sensitivity.grid_analysis import TippingPoint


def test_report_summary_and_properties():
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.randn(n),
            "x2": rng.randn(n),
            "x3": rng.randn(n),
        }
    )
    df.loc[0:30, "x1"] = np.nan
    df.loc[20:50, "x2"] = np.nan

    report = umbra.diagnose(df, run_sensitivity=True)

    # Test summary string
    summary = report.summary()
    assert "UMBRA SCIENTIFIC MISSING DATA DIAGNOSTIC REPORT" in summary
    assert "TIER 1: OBSERVED-DATA EVIDENCE" in summary
    assert "TIER 2: MODEL-BASED INFERENCE" in summary
    assert "TIER 3: UNTESTABLE ASSUMPTIONS" in summary

    # Test properties
    assert 0.0 < report.overall_missing_rate < 1.0
    assert len(report.missing_counts) == 2
    assert len(report.missing_rates) == 2


def test_shadow_variable_candidate_methods():
    rng = np.random.RandomState(42)
    n = 200
    z = rng.randn(n)
    y = 0.5 * z + rng.randn(n)
    x = rng.randn(n)

    df = pd.DataFrame({"z": z, "x": x, "y": y})
    # Induce missingness correlated with z
    mis = z > 0.5
    df.loc[mis, "y"] = np.nan

    rep = find_shadow_variables(df, target_column="y")
    summary = rep.summary()
    assert "Candidate Auxiliary Variable Report" in summary

    if rep.best_candidate:
        cand = rep.best_candidate
        d = cand.to_dict()
        assert "variable_name" in d
        assert "first_stage_f_stat" in d
        cand_sum = cand.summary()
        assert "Auxiliary Candidate" in cand_sum
        # Test legacy aliases
        assert cand.missingness_correlation == cand.relevance_correlation
        assert cand.outcome_correlation == cand.direct_outcome_correlation
        assert cand.shadow_score == cand.candidate_score
        assert isinstance(cand.is_promising_candidate, bool)
        assert isinstance(cand.rationale, str)


def test_pattern_reporting():
    rng = np.random.RandomState(42)
    n = 150
    df = pd.DataFrame(
        {
            "a": rng.randn(n),
            "b": rng.randn(n),
            "c": rng.randn(n),
        }
    )
    df.loc[0:20, "a"] = np.nan
    df.loc[0:20, "b"] = np.nan  # identical pattern

    patterns = analyze_missingness_patterns(df)
    d = patterns.to_dict()
    assert "variable_reports" in d
    assert "patterns" in d
    assert patterns.pattern_table is not None
    assert len(patterns.pattern_table) > 0

    var_rep = patterns.variable_reports["a"]
    assert "Variable 'a'" in var_rep.summary()
    assert len(var_rep.covariate_shifts) > 0

    shift = list(var_rep.covariate_shifts.values())[0]
    assert isinstance(shift.summary(), str)
    assert isinstance(shift.to_dict(), dict)


def test_tipping_point_dataclass():
    tp = TippingPoint(
        metric_name="sign_flip",
        tipping_delta=0.85,
        original_value=2.4,
        tipping_value=0.0,
        description="Estimate flips sign from positive to negative",
    )
    d = tp.to_dict()
    assert d["metric_name"] == "sign_flip"
    assert d["tipping_delta"] == 0.85
    assert d["original_value"] == 2.4


def test_umbra_imputer_invalid_strategy():
    imputer = UmbraImputer(strategy="non_existent_strategy")
    with pytest.raises(ValueError, match="Invalid strategy"):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [2.0, 3.0, 4.0]})
        imputer.fit(df)


def test_deep_generative_imputer_direct():
    from umbra.imputers.deep_generative_mnar import DeepGenerativeMNARImputer

    rng = np.random.RandomState(42)
    df = pd.DataFrame(rng.randn(100, 3), columns=["x1", "x2", "x3"])
    df.loc[0:15, "x1"] = np.nan

    imputer = DeepGenerativeMNARImputer(
        latent_dim=2,
        epochs=3,
        random_state=42,
    )
    imputed = imputer.fit_transform(df)
    assert isinstance(imputed, pd.DataFrame)
    assert not imputed.isna().any().any()
