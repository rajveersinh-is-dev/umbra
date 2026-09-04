"""
Publication-Quality Scientific Visualizations for Umbra.

Generates plots for:
1. Missingness Matrix & Pattern Overview
2. Covariate Distribution Shifts (Forest / Tornado plot)
3. Diagnostic Signal Evidence Radar / Bar Chart
4. Sensitivity Curves with Confidence Bands & Tipping Points
5. Router Accuracy & Confusion Matrix

All plots adhere to scientific publishing standards (proper labels, units,
typography, accessible palettes, and reproducible metadata).
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from umbra.diagnostics.mnar_risk_score import MNARRiskReport
from umbra.diagnostics.pattern_analysis import PatternAnalysisReport
from umbra.sensitivity.grid_analysis import SensitivityReport


def plot_missingness_matrix(
    data: pd.DataFrame,
    max_rows: int = 150,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot binary missingness matrix indicating pattern signatures across rows.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to visualize.
    max_rows : int, default=150
        Maximum rows to display in matrix view.
    save_path : Optional[str or Path]
        If provided, saves figure to disk.
    dpi : int, default=300
        Resolution for saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df_sample = data.head(max_rows)
    mask = df_sample.isna().to_numpy()

    fig, (ax_mat, ax_bar) = plt.subplots(
        1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [4, 1]}, sharey=False
    )

    # Missingness matrix (0 = Observed, 1 = Missing)
    ax_mat.imshow(mask, cmap="Blues", aspect="auto", interpolation="none")
    ax_mat.set_title(
        f"Missingness Matrix (First {len(df_sample)} Observations)", fontsize=12, fontweight="bold"
    )
    ax_mat.set_xlabel("Variables", fontsize=11)
    ax_mat.set_ylabel("Row Index", fontsize=11)
    ax_mat.set_xticks(range(len(data.columns)))
    ax_mat.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=9)

    # Right side bar: Missingness percentage per column
    miss_pct = data.isna().mean() * 100.0
    ax_bar.barh(
        range(len(data.columns)), miss_pct.values, color="#1f77b4", edgecolor="black", alpha=0.8
    )
    ax_bar.set_yticks(range(len(data.columns)))
    ax_bar.set_yticklabels([])
    ax_bar.set_xlabel("Missing %", fontsize=11)
    ax_bar.set_xlim(0, max(10, min(100, miss_pct.max() * 1.25)))
    ax_bar.grid(axis="x", linestyle="--", alpha=0.5)
    ax_bar.set_title("Column Missing %", fontsize=11)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_covariate_shifts(
    pattern_report: PatternAnalysisReport,
    target_column: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot distribution shifts (KS statistic and Cohen's d) for incomplete variables.

    Parameters
    ----------
    pattern_report : PatternAnalysisReport
        Output from analyze_missingness_patterns.
    target_column : Optional[str]
        If provided, plots only for this incomplete variable.
    save_path : Optional[str or Path]
        Path to save figure.
    dpi : int, default=300

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not pattern_report.variable_reports:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No incomplete variables found.", ha="center", va="center")
        return fig

    target = target_column or list(pattern_report.variable_reports.keys())[0]
    var_rep = pattern_report.variable_reports.get(target)
    if not var_rep or not var_rep.covariate_shifts:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"No covariate shifts recorded for {target}.", ha="center", va="center")
        return fig

    covars = list(var_rep.covariate_shifts.keys())
    ks_vals = [var_rep.covariate_shifts[c].ks_statistic or 0.0 for c in covars]
    d_vals = [var_rep.covariate_shifts[c].cohens_d or 0.0 for c in covars]
    sig_mask = [var_rep.covariate_shifts[c].is_significant for c in covars]

    colors = ["#d62728" if s else "#1f77b4" for s in sig_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, max(4, len(covars) * 0.45)), sharey=True)

    y_pos = range(len(covars))
    ax1.barh(y_pos, ks_vals, color=colors, edgecolor="black", alpha=0.85)
    ax1.axvline(0.15, color="grey", linestyle="--", label="Relevance threshold (0.15)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(covars, fontsize=10)
    ax1.set_xlabel("Two-Sample KS Statistic", fontsize=11)
    ax1.set_title("Kolmogorov-Smirnov Shift", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    ax2.barh(y_pos, d_vals, color=colors, edgecolor="black", alpha=0.85)
    ax2.axvline(0.0, color="black", linestyle="-", linewidth=0.8)
    ax2.axvline(0.2, color="grey", linestyle="--", label="Small effect (|d|=0.2)")
    ax2.axvline(-0.2, color="grey", linestyle="--")
    ax2.set_xlabel("Cohen's d (Standardized Diff)", fontsize=11)
    ax2.set_title("Mean Standardized Shift", fontsize=11, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    fig.suptitle(
        f"Covariate Distribution Shifts when '{target}' is Missing",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_sensitivity_curve(
    report: SensitivityReport,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot MNAR sensitivity curve theta(delta) with 95% confidence bands and tipping points.

    Parameters
    ----------
    report : SensitivityReport
        Output from run_sensitivity_grid.
    save_path : Optional[str or Path]
        Path to save figure.
    dpi : int, default=300

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = report.grid_df
    deltas = df["delta"].values
    metrics = df["downstream_metric"].values
    ci_lower = df["ci_lower"].values
    ci_upper = df["ci_upper"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot 95% confidence band
    ax.fill_between(
        deltas, ci_lower, ci_upper, color="#1f77b4", alpha=0.2, label="95% Confidence Band"
    )

    # Plot estimate curve theta(delta)
    ax.plot(
        deltas,
        metrics,
        color="#1f77b4",
        marker="o",
        linewidth=2.0,
        label=r"Downstream Parameter $\theta(\delta)$",
    )

    # Reference lines: delta = 0 (MAR assumption) and theta = 0 (Null hypothesis)
    ax.axvline(0.0, color="grey", linestyle=":", label="MAR Baseline (delta=0)")
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.7, label="Null Threshold (0.0)")

    # Plausible assumption bounds [-1.0, +1.0]
    ax.axvspan(
        -1.0, 1.0, color="lightyellow", alpha=0.3, label="Plausible MNAR departures [-1, +1]"
    )

    # Tipping points
    for tp in report.tipping_points:
        ax.plot(
            tp.tipping_delta,
            tp.tipping_value,
            marker="*",
            markersize=14,
            color="#d62728",
            linestyle="None",
            label=f"Tipping Point ({tp.metric_name} at delta={tp.tipping_delta:+.2f})",
        )

    ax.set_xlabel(
        r"MNAR Sensitivity Departure Parameter $\delta$ (Standard Deviations)", fontsize=11
    )
    ax.set_ylabel(r"Estimated Downstream Parameter $\theta(\delta)$", fontsize=11)
    ax.set_title(
        f"MNAR Sensitivity & Tipping Point Analysis for '{report.target_column}'\n"
        f"Conclusion Fragility: {'FRAGILE' if report.is_fragile else 'ROBUST'}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_diagnostic_evidence(
    report: MNARRiskReport,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot diagnostic evidence signals breakdown for an incomplete variable."""
    signals = report.signals
    names = [s.name for s in signals]
    scores = [s.score for s in signals]
    triggered = [s.is_triggered for s in signals]

    colors = ["#d62728" if t else "#2ca02c" for t in triggered]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = range(len(names))

    bars = ax.barh(y_pos, scores, color=colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Signal Score [0.0 = MCAR/MAR, 1.0 = Severe Departure]", fontsize=11)
    ax.set_title(
        f"Diagnostic Evidence Battery: '{report.target_column}'\n"
        f"Synthesized Risk: {report.risk_level} (Score: {report.composite_score:.2f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.axvline(0.5, color="grey", linestyle="--", label="High-Risk Threshold (0.50)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_router_confusion_matrix(
    conf_matrix: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot heatmap of Auto router confusion matrix across missingness regimes."""
    # Exclude margins if present
    plot_df = conf_matrix.copy()
    if "All" in plot_df.index:
        plot_df = plot_df.drop(index="All")
    if "All" in plot_df.columns:
        plot_df = plot_df.drop(columns="All")

    fig, ax = plt.subplots(figsize=(7, 5))
    cax = ax.imshow(plot_df.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index, fontsize=10)

    ax.set_xlabel("Chosen Imputation Strategy", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Missingness Regime", fontsize=11, fontweight="bold")
    ax.set_title(
        "Auto Strategy Router Confusion Matrix\n(Normalized by True Regime)",
        fontsize=12,
        fontweight="bold",
    )

    # Overlay numeric percentages
    for i in range(len(plot_df.index)):
        for j in range(len(plot_df.columns)):
            val = plot_df.values[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.1%}",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
                fontsize=10,
            )

    fig.colorbar(cax, ax=ax, label="Proportion")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
