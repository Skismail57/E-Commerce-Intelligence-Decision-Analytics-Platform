"""
Experimentation Module
Provides A/B testing framework for marketing campaigns, UI changes, and feature rollouts.
"""

from .ab_platform import (
    ABExperimentPlatform,
    run_ab_experiment,
)

__all__ = [
    'ABExperimentPlatform',
    'run_ab_experiment',
]
