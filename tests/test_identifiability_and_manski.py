"""
Unit and integration tests for Manski Partial Identifiability Bounds and Audit Certificates (Phase 6).

Verifies:
1. Sharp Manski population mean bounds and interval width invariant Delta = p_miss * (y_U - y_L).
2. Sharp Manski quantile and median partial identification bounds.
3. Domain support handling (custom support, empirical bounds, quantile trimming).
4. Machine-verifiable IdentifiabilityCertificate generation and strategy contracts.
5. Integration into UmbraDiagnosticReport, diagnose_report, to_dict(), and to_markdown().
"""

import numpy as np
import pandas as pd
import pytest

from umbra.diagnostics.identifiability_audit import (
    IdentifiabilityCertificate,
    audit_identifiability,
)
from umbra.diagnostics.manski_bounds import (
    ManskiBoundsResult,
    compute_dataframe_manski_bounds,
    compute_manski_bounds,
)
from umbra.diagnostics.report import diagnose_report


def test_manski_mean_bounds_math() -> None:
    """Verify Manski mean bounds formula, sharpness, and interval width invariant."""
    # Complete population of 1000 items from Uniform(10, 50)
    rng = np.random.default_rng(42)
    y_true = rng.uniform(10.0, 50.0, size=1000)
    true_mean = float(np.mean(y_true))

    # Introduce 30% missingness
    y_obs = y_true.copy()
    miss_idx = rng.choice(1000, size=300, replace=False)
    y_obs[miss_idx] = np.nan

    support = (10.0, 50.0)
    res = compute_manski_bounds(y_obs, feature_name="score", support=support)

    assert isinstance(res, ManskiBoundsResult)
    assert res.feature == "score"
    assert res.n_total == 1000
    assert res.n_observed == 700
    assert res.n_missing == 300
    assert abs(res.missing_rate - 0.30) < 1e-6

    # Mathematical Invariant: interval width Delta = p_miss * (y_U - y_L)
    expected_width = 0.30 * (50.0 - 10.0)  # 12.0
    assert abs(res.mean_interval_width - expected_width) < 1e-6
    assert abs((res.mean_upper_bound - res.mean_lower_bound) - expected_width) < 1e-6

    # Invariant: observed mean is inside the Manski interval
    assert res.mean_lower_bound <= res.observed_mean <= res.mean_upper_bound

    # Invariant: true population mean MUST be contained in the sharp Manski interval
    assert res.contains(true_mean, parameter="mean")
    assert res.mean_lower_bound <= true_mean <= res.mean_upper_bound


def test_manski_mean_bounds_zero_missing() -> None:
    """Verify Manski bounds collapse to point estimate when missingness rate is zero."""
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    res = compute_manski_bounds(y, support=(0.0, 12.0))

    assert res.n_missing == 0
    assert res.missing_rate == 0.0
    assert res.mean_interval_width == 0.0
    assert res.mean_lower_bound == 6.0
    assert res.mean_upper_bound == 6.0
    assert res.observed_mean == 6.0


def test_manski_quantile_bounds() -> None:
    """Verify Manski sharp quantile and median bounds."""
    rng = np.random.default_rng(123)
    y_comp = rng.normal(100.0, 15.0, size=1000)
    true_median = float(np.median(y_comp))

    # 20% missingness
    y_obs = y_comp.copy()
    y_obs[:200] = np.nan

    res = compute_manski_bounds(
        y_obs,
        feature_name="iq_score",
        support=(40.0, 160.0),
        quantiles=(0.25, 0.50, 0.75),
    )

    # Check quantiles dictionary
    assert 0.25 in res.quantile_bounds
    assert 0.50 in res.quantile_bounds
    assert 0.75 in res.quantile_bounds

    for q, (lb, ub) in res.quantile_bounds.items():
        assert lb <= ub, f"Quantile {q}: lower bound {lb} > upper bound {ub}"

    # Median interval must contain true population median
    assert res.contains(true_median, parameter="median")
    assert res.median_lower_bound <= true_median <= res.median_upper_bound
    assert res.median_interval_width > 0.0


def test_manski_bounds_support_and_trimming() -> None:
    """Verify custom support bounds and empirical quantile trimming."""
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan])

    # 1. Custom support
    res_custom = compute_manski_bounds(y, support=(0.0, 10.0))
    assert res_custom.support_lower == 0.0
    assert res_custom.support_upper == 10.0

    # 2. Empirical support (defaults to min and max of observed)
    res_emp = compute_manski_bounds(y)
    assert res_emp.support_lower == 1.0
    assert res_emp.support_upper == 5.0

    # 3. Trimmed quantile support
    y_outliers = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 100.0, np.nan])
    res_trim = compute_manski_bounds(y_outliers, trim_quantile=0.1)
    assert res_trim.support_upper < 100.0


def test_compute_dataframe_manski_bounds() -> None:
    """Verify compute_dataframe_manski_bounds iterates across incomplete columns."""
    df = pd.DataFrame(
        {
            "full": [1.0, 2.0, 3.0, 4.0],
            "inc1": [10.0, np.nan, 30.0, 40.0],
            "inc2": [np.nan, 2.5, np.nan, 5.5],
            "category": ["A", "B", "C", "D"],
        }
    )

    bounds_map = compute_dataframe_manski_bounds(df)
    assert "inc1" in bounds_map
    assert "inc2" in bounds_map
    assert "full" not in bounds_map
    assert "category" not in bounds_map
    assert bounds_map["inc1"].missing_rate == 0.25
    assert bounds_map["inc2"].missing_rate == 0.50


def test_identifiability_certificate_generation() -> None:
    """Verify audit_identifiability generates a complete, valid IdentifiabilityCertificate."""
    rng = np.random.default_rng(42)
    n = 300
    x = rng.normal(0, 1, size=n)
    z = rng.normal(0, 1, size=n)
    y = 5.0 + 1.5 * x + rng.normal(0, 1, size=n)
    # Selection dependent on z
    r = (z > 0).astype(int)
    y_obs = np.where(r == 1, y, np.nan)

    df = pd.DataFrame({"x": x, "z": z, "y": y_obs})

    cert = audit_identifiability(feature="y", df=df, shadow_var="z")

    assert isinstance(cert, IdentifiabilityCertificate)
    assert cert.feature == "y"
    assert cert.missing_rate > 0.3
    assert cert.manski_bounds is not None
    assert "Manski Nonparametric Bounds" in cert.strategy_assumptions
    assert "MICE / MAR Chained Equations" in cert.strategy_assumptions
    assert "Heckman Selection Model" in cert.strategy_assumptions
    assert "Pattern Mixture Sensitivity" in cert.strategy_assumptions
    assert len(cert.citations) >= 3

    # Check to_dict() serialization
    d = cert.to_dict()
    assert d["feature"] == "y"
    assert "manski_bounds" in d
    assert "strategy_assumptions" in d

    # Check to_markdown()
    md = cert.to_markdown()
    assert "### Identifiability & Assumption Audit Certificate" in md
    assert "Molenberghs Non-Identifiability Theorem" in md
    assert "Manski" in md


def test_report_integration_with_certificate() -> None:
    """Verify diagnose_report generates and embeds identifiability certificates in UmbraDiagnosticReport."""
    rng = np.random.default_rng(999)
    df = pd.DataFrame(
        {
            "age": rng.normal(40, 10, size=200),
            "income": rng.normal(50000, 15000, size=200),
            "contact": rng.poisson(2, size=200),
        }
    )
    # Add missingness in income
    df.loc[df["contact"] < 2, "income"] = np.nan

    rep = diagnose_report(df, shadow_cols={"income": "contact"})

    assert hasattr(rep, "identifiability_certificates")
    assert "income" in rep.identifiability_certificates
    cert = rep.identifiability_certificates["income"]
    assert isinstance(cert, IdentifiabilityCertificate)
    assert cert.feature == "income"
    assert cert.manski_bounds.mean_interval_width > 0.0

    # Verify summary() contains Tier 5
    summary_text = rep.summary()
    assert "TIER 5: MANSKI PARTIAL IDENTIFICATION" in summary_text
    assert "income" in summary_text

    # Verify to_dict() contains tier5
    rep_dict = rep.to_dict()
    assert "tier5_identifiability_certificates" in rep_dict
    assert "income" in rep_dict["tier5_identifiability_certificates"]

    # Verify to_markdown() contains Section 6
    rep_md = rep.to_markdown()
    assert "6. Identifiability & Assumption Audit Certificates" in rep_md
    assert "Manski Partial Identification" in rep_md


def test_manski_bounds_invalid_inputs() -> None:
    """Verify error handling on empty arrays or inverted support bounds."""
    with pytest.raises(ValueError, match="empty array"):
        compute_manski_bounds([])

    with pytest.raises(ValueError, match="lower bound .* must be <= upper bound"):
        compute_manski_bounds([1.0, 2.0, np.nan], support=(10.0, 5.0))
