"""
Decision Support Module
Provides what-if analysis engine and executive decision center for business decision support.
"""

from .what_if_analysis import WhatIfAnalyzer, run_what_if_pipeline
from .executive_center import ExecutiveDecisionCenter, run_executive_center_pipeline

__all__ = [
    'WhatIfAnalyzer',
    'run_what_if_pipeline',
    'ExecutiveDecisionCenter',
    'run_executive_center_pipeline',
]
