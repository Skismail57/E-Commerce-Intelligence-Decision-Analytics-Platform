"""
Customer Intelligence Module
Provides advanced customer analytics including behavioral features, survival analysis, and customer fingerprinting.
"""

from .behavioral_features import (
    CustomerBehavioralFeatures,
    compute_behavioral_features_pipeline,
)

from .survival_analysis import (
    CustomerSurvivalAnalysis,
    run_survival_analysis_pipeline,
)

__all__ = [
    'CustomerBehavioralFeatures',
    'compute_behavioral_features_pipeline',
    'CustomerSurvivalAnalysis',
    'run_survival_analysis_pipeline',
]
