"""
Real-world observational and semi-synthetic benchmark dataset loaders for Umbra.
"""

from umbra.data.loaders import (
    load_california_housing,
    load_clinical_trial_attrition,
    load_cps_wage,
    load_nhanes_biomarkers,
)

__all__ = [
    "load_cps_wage",
    "load_nhanes_biomarkers",
    "load_california_housing",
    "load_clinical_trial_attrition",
]
