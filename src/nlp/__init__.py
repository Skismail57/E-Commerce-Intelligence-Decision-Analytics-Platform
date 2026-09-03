"""
NLP Module
Provides natural language processing capabilities for review intelligence and sentiment analysis.
"""

from .review_intelligence import ReviewIntelligenceEngine, run_review_intelligence_pipeline

__all__ = [
    'ReviewIntelligenceEngine',
    'run_review_intelligence_pipeline',
]
