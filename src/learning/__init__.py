"""
Learning Module
Provides closed-loop learning system for continuous model improvement.
"""

from .closed_loop import ClosedLoopLearningSystem, run_closed_loop_pipeline

__all__ = [
    'ClosedLoopLearningSystem',
    'run_closed_loop_pipeline',
]
