"""
Adaptation module for self-learning and continuous improvement.

This module provides adaptive learning capabilities including performance tracking,
automated optimization, and system parameter tuning based on real-world usage patterns.
"""

from .learning_engine import (
    AdaptationExperiment,
    AdaptationParameter,
    AdaptationStatus,
    AdaptationStrategy,
    AdaptationType,
    BayesianOptimizationStrategy,
    GradientDescentStrategy,
    LearningEngine,
    LearningInsight,
    RandomSearchStrategy,
)

__all__ = [
    "LearningEngine",
    "AdaptationType",
    "AdaptationStatus",
    "AdaptationParameter",
    "AdaptationExperiment",
    "LearningInsight",
    "AdaptationStrategy",
    "GradientDescentStrategy",
    "BayesianOptimizationStrategy",
    "RandomSearchStrategy",
]
