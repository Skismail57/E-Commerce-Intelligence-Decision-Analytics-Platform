"""
Synthetic Data Module
Provides behavioral synthetic data generation for testing and development.
"""

from .behavioral_generator import BehavioralDataGenerator, run_behavioral_simulation_pipeline

__all__ = [
    'BehavioralDataGenerator',
    'run_behavioral_simulation_pipeline',
]
