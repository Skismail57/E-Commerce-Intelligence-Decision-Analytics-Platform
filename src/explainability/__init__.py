"""
Explainability Module
Provides SHAP-based model explainability for interpretability.
"""

from .shap_explainer import SHAPExplainer, run_shap_pipeline

__all__ = [
    'SHAPExplainer',
    'run_shap_pipeline',
]
