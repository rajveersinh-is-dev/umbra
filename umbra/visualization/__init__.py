"""
Scientific visualization tools for missingness diagnostics and sensitivity analysis.
"""

from umbra.visualization.figures import (
    plot_covariate_shifts,
    plot_diagnostic_evidence,
    plot_missingness_matrix,
    plot_router_confusion_matrix,
    plot_sensitivity_curve,
)

__all__ = [
    "plot_missingness_matrix",
    "plot_covariate_shifts",
    "plot_diagnostic_evidence",
    "plot_sensitivity_curve",
    "plot_router_confusion_matrix",
]
