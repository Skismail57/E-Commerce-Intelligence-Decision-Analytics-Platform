"""
Governance Module
Provides data lineage tracking and quality scoring for data governance.
"""

from .data_lineage import DataLineageTracker, run_lineage_pipeline

__all__ = [
    'DataLineageTracker',
    'run_lineage_pipeline',
]
