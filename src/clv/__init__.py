"""
CLV Module
Provides Customer Lifetime Value prediction using both probabilistic (BG/NBD + Gamma-Gamma) and ML-based approaches.
"""

from .clv_predictor import CLVPredictor
from .ml_clv_predictor import MLCLVPredictor, compare_clv_approaches, run_ml_clv_pipeline

__all__ = [
    'CLVPredictor',
    'MLCLVPredictor',
    'compare_clv_approaches',
    'run_ml_clv_pipeline',
]
