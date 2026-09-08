"""
Unit tests for human-readable explain and reporting module.
"""

import pandas as pd

from umbra.diagnostics.mnar_risk_score import DiagnosticSignal, MNARRiskReport
from umbra.explain import (
    diagnostics_to_markdown,
    explain_diagnostics,
    explain_sensitivity,
    format_risk_badge,
)
from umbra.sensitivity.grid_analysis import SensitivityReport, TippingPoint


def test_format_risk_badge():
    assert "HIGH" in format_risk_badge("HIGH")
    assert "MEDIUM" in format_risk_badge("MEDIUM")
    assert "LOW" in format_risk_badge("LOW")


def test_explain_diagnostics_all_levels():
    reports = {
        "col_high": MNARRiskReport(
            target_column="col_high",
            risk_level="HIGH",
            composite_score=0.75,
            missing_rate=0.30,
            signals=[
                DiagnosticSignal(
                    name="Test Signal High",
                    weight=0.5,
                    score=0.8,
                    is_triggered=True,
                    description="Triggered high signal",
                )
            ],
            explanation="High MNAR risk",
            recommended_strategy="mnar_heckman",
            shadow_candidate="shadow_z",
            domain_heuristic_matched=True,
            citation="Test Citation 2024",
        ),
        "col_med": MNARRiskReport(
            target_column="col_med",
            risk_level="MEDIUM",
            composite_score=0.40,
            missing_rate=0.20,
            signals=[
                DiagnosticSignal(
                    name="Test Signal Med",
                    weight=0.5,
                    score=0.4,
                    is_triggered=False,
                    description="Untriggered signal",
                )
            ],
            explanation="Moderate risk",
            recommended_strategy="mnar_pattern_mixture_sensitivity",
        ),
        "col_low": MNARRiskReport(
            target_column="col_low",
            risk_level="LOW",
            composite_score=0.10,
            missing_rate=0.05,
            signals=[],
            explanation="Low risk",
            recommended_strategy="mar_chained_equations",
        ),
    }

    text = explain_diagnostics(reports)
    assert isinstance(text, str)

    # Empty reports
    empty_text = explain_diagnostics({})
    assert isinstance(empty_text, str)

    # Markdown export
    md = diagnostics_to_markdown(reports)
    assert "col_high" in md
    assert "col_med" in md
    assert "col_low" in md
    assert "Test Citation 2024" in md

    # Empty markdown
    empty_md = diagnostics_to_markdown({})
    assert "# Umbra Missingness Diagnostics Audit" in empty_md


def test_explain_sensitivity_fragile_and_robust():
    df_grid = pd.DataFrame(
        {
            "delta": [-1.0, 0.0, 1.0],
            "target_mean": [5.0, 10.0, 15.0],
            "downstream_metric": [5.0, 10.0, 15.0],
        }
    )

    fragile_rep = SensitivityReport(
        target_column="income",
        grid_df=df_grid,
        mar_baseline_estimate=10.0,
        estimate_min=-2.0,
        estimate_max=15.0,
        uncertainty_spread=17.0,
        tipping_points=[
            TippingPoint(
                metric_name="sign_flip",
                tipping_delta=-0.5,
                original_value=10.0,
                tipping_value=0.0,
                description="Sign flip at delta=-0.5",
            )
        ],
        is_fragile=True,
        interpretation="Fragile to plausible shifts",
    )
    s_fragile = explain_sensitivity(fragile_rep)
    assert "FRAGILE" in s_fragile
    assert "income" in s_fragile

    robust_rep = SensitivityReport(
        target_column="score",
        grid_df=df_grid,
        mar_baseline_estimate=10.0,
        estimate_min=8.0,
        estimate_max=12.0,
        uncertainty_spread=4.0,
        is_fragile=False,
        interpretation="Robust finding",
    )
    s_robust = explain_sensitivity(robust_rep)
    assert "ROBUST" in s_robust
    assert "score" in s_robust
