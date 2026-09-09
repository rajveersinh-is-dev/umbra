"""
Umbra Imputers Module.

Provides statistically principled imputation models across missingness mechanisms:
- MARChainedEquationsImputer: Multivariable Imputation by Chained Equations (PMM, Ridge, Bayesian)
- HeckmanSelectionImputer: Parametric bivariate normal selection model with Mills ratio correction
- PatternMixtureImputer: Sensitivity shift imputation across non-response strata
- DeepGenerativeMNARImputer: Neural generative imputation for complex non-linear manifolds
- rubins_rules: Exact multiple imputation pooling with Barnard-Rubin degrees of freedom
"""

from umbra.imputers.deep_generative_mnar import DeepGenerativeMNARImputer
from umbra.imputers.heckman_selection import (
    HeckmanConvergenceWarning,
    HeckmanSelectionImputer,
    HeckmanSEWarning,
    WeakInstrumentWarning,
)
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer, rubins_rules
from umbra.imputers.pattern_mixture import PatternMixtureImputer

__all__ = [
    "MARChainedEquationsImputer",
    "HeckmanSelectionImputer",
    "WeakInstrumentWarning",
    "HeckmanSEWarning",
    "HeckmanConvergenceWarning",
    "PatternMixtureImputer",
    "DeepGenerativeMNARImputer",
    "rubins_rules",
]
