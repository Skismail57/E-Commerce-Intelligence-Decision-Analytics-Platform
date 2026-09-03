"""
Graph Analytics Module
Provides customer-product graph analytics for relationship mining and community detection.
"""

from .graph_builder import GraphAnalytics, run_graph_analytics_pipeline

__all__ = [
    'GraphAnalytics',
    'run_graph_analytics_pipeline',
]
