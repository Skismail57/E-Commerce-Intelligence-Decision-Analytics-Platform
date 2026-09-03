"""
Causal Inference Module
Provides uplift modeling and causal inference methods for marketing campaign effectiveness and treatment effect estimation.
"""

from .uplift_modeling import (
    UpliftModeler,
    run_uplift_pipeline,
)

from .causal_inference import (
    CausalInferenceEngine,
    run_causal_inference_pipeline,
)

__all__ = [
    'UpliftModeler',
    'run_uplift_pipeline',
    'CausalInferenceEngine',
    'run_causal_inference_pipeline',
]
