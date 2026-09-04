"""
Umbra Sensitivity Module.

Provides sensitivity analysis under untestable MNAR assumption shifts:
- run_sensitivity_grid: Evaluates metric stability across delta-shift departure grids
- SensitivityReport: Structured report of parameter bounds and fragile conclusions
- TippingPoint: Critical departure magnitude where conclusions or decisions flip
"""

from umbra.sensitivity.grid_analysis import (
    SensitivityReport,
    TippingPoint,
    run_sensitivity_grid,
)

__all__ = [
    "run_sensitivity_grid",
    "SensitivityReport",
    "TippingPoint",
]
