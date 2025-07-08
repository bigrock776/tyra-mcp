"""
Adaptive learning engine for continuous system improvement.

Implements self-learning capabilities including parameter optimization,
model adaptation, and automated system tuning based on performance feedback.
"""

import asyncio
import json
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from ..analytics.performance_tracker import (
    MetricType,
    OptimizationRecommendation,
    PerformanceTracker,
)
from ..utils.config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AdaptationType(Enum):
    """Types of adaptations the system can perform."""

    PARAMETER_TUNING = "parameter_tuning"
    MODEL_SELECTION = "model_selection"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    RESOURCE_SCALING = "resource_scaling"
    PROMPT_OPTIMIZATION = "prompt_optimization"
    CACHE_STRATEGY = "cache_strategy"
    SEARCH_TUNING = "search_tuning"


class AdaptationStatus(Enum):
    """Status of adaptation experiments."""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class AdaptationParameter:
    """A parameter that can be adapted."""

    name: str
    current_value: Any
    value_type: str  # "float", "int", "str", "bool", "list"
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    step_size: Optional[float] = None
    description: str = ""
    impact_metrics: List[MetricType] = field(default_factory=list)


@dataclass
class AdaptationExperiment:
    """An adaptation experiment with its configuration and results."""

    id: str
    adaptation_type: AdaptationType
    status: AdaptationStatus
    parameters: Dict[str, Any]
    original_parameters: Dict[str, Any]
    target_metrics: List[MetricType]
    start_time: datetime
    end_time: Optional[datetime] = None
    baseline_metrics: Dict[MetricType, float] = field(default_factory=dict)
    experiment_metrics: Dict[MetricType, float] = field(default_factory=dict)
    improvement: Dict[MetricType, float] = field(default_factory=dict)
    confidence: float = 0.0
    success: bool = False
    notes: str = ""


@dataclass
class LearningInsight:
    """Insights learned from experiments and observations."""

    category: str
    insight: str
    confidence: float
    supporting_experiments: List[str]
    timestamp: datetime
    actionable: bool
    impact_estimate: str


class AdaptationStrategy(ABC):
    """Abstract base class for adaptation strategies."""

    @abstractmethod
    async def suggest_adaptation(
        self,
        performance_data: Dict[str, Any],
        available_parameters: List[AdaptationParameter],
    ) -> Optional[AdaptationExperiment]:
        """Suggest an adaptation experiment."""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        pass


class GradientDescentStrategy(AdaptationStrategy):
    """Gradient descent-like parameter optimization."""

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.parameter_history: Dict[str, List[Tuple[Any, float]]] = defaultdict(list)

    async def suggest_adaptation(
        self,
        performance_data: Dict[str, Any],
        available_parameters: List[AdaptationParameter],
    ) -> Optional[AdaptationExperiment]:
        """Suggest parameter changes based on gradient estimation."""
        try:
            # Find parameters with degrading performance
            for param in available_parameters:
                if (
                    param.value_type in ["float", "int"]
                    and param.min_value is not None
                    and param.max_value is not None
                ):
                    # Estimate gradient from history
                    history = self.parameter_history[param.name]
                    if len(history) >= 2:
                        gradient = self._estimate_gradient(
                            history, param.impact_metrics, performance_data
                        )
                        if abs(gradient) > 0.01:  # Significant gradient
                            new_value = self._calculate_new_value(param, gradient)
                            if new_value != param.current_value:
                                return AdaptationExperiment(
                                    id=f"gradient_{param.name}_{int(time.time())}",
                                    adaptation_type=AdaptationType.PARAMETER_TUNING,
                                    status=AdaptationStatus.PLANNED,
                                    parameters={param.name: new_value},
                                    original_parameters={
                                        param.name: param.current_value
                                    },
                                    target_metrics=param.impact_metrics,
                                    start_time=datetime.utcnow(),
                                    notes=f"Gradient descent optimization for {param.name}",
                                )

            return None

        except Exception as e:
            logger.error("Gradient descent suggestion failed", error=str(e))
            return None

    def _estimate_gradient(
        self,
        history: List[Tuple[Any, float]],
        impact_metrics: List[MetricType],
        performance_data: Dict[str, Any],
    ) -> float:
        """Estimate gradient from parameter history."""
        if len(history) < 2:
            return 0.0

        # Simple finite difference
        (val1, score1), (val2, score2) = history[-2], history[-1]

        if val1 == val2:
            return 0.0

        return (score2 - score1) / (val2 - val1)

    def _calculate_new_value(self, param: AdaptationParameter, gradient: float) -> Any:
        """Calculate new parameter value based on gradient."""
        if param.value_type == "float":
            step = (
                self.learning_rate
                * gradient
                * (param.max_value - param.min_value)
                * 0.1
            )
            new_value = float(param.current_value) + step
            return max(param.min_value, min(param.max_value, new_value))
        elif param.value_type == "int":
            step = max(
                1,
                int(
                    self.learning_rate
                    * gradient
                    * (param.max_value - param.min_value)
                    * 0.1
                ),
            )
            new_value = int(param.current_value) + step
            return max(param.min_value, min(param.max_value, new_value))

        return param.current_value

    def get_strategy_name(self) -> str:
        return "gradient_descent"


class BayesianOptimizationStrategy(AdaptationStrategy):
    """Bayesian optimization for parameter tuning."""

    def __init__(self):
        self.observations: List[Tuple[Dict[str, Any], float]] = []
        self.acquisition_function = "expected_improvement"

    async def suggest_adaptation(
        self,
        performance_data: Dict[str, Any],
        available_parameters: List[AdaptationParameter],
    ) -> Optional[AdaptationExperiment]:
        """Suggest parameter changes using Bayesian optimization."""
        try:
            # Simple Bayesian optimization implementation
            # In production, use libraries like scikit-optimize or GPyOpt

            # Select parameters to optimize
            numeric_params = [
                p
                for p in available_parameters
                if p.value_type in ["float", "int"] and p.min_value is not None
            ]

            if not numeric_params:
                return None

            # Select parameter to optimize (random for simplicity)
            param = random.choice(numeric_params)

            # Use expected improvement to suggest new value
            new_value = self._suggest_value_bayesian(param)

            if new_value != param.current_value:
                return AdaptationExperiment(
                    id=f"bayesian_{param.name}_{int(time.time())}",
                    adaptation_type=AdaptationType.PARAMETER_TUNING,
                    status=AdaptationStatus.PLANNED,
                    parameters={param.name: new_value},
                    original_parameters={param.name: param.current_value},
                    target_metrics=param.impact_metrics,
                    start_time=datetime.utcnow(),
                    notes=f"Bayesian optimization for {param.name}",
                )

            return None

        except Exception as e:
            logger.error("Bayesian optimization suggestion failed", error=str(e))
            return None

    def _suggest_value_bayesian(self, param: AdaptationParameter) -> Any:
        """Suggest new value using simplified Bayesian approach."""
        # Simplified: explore around current value with some randomness
        if param.value_type == "float":
            range_size = param.max_value - param.min_value
            exploration_range = range_size * 0.1  # Explore within 10% of range
            new_value = float(param.current_value) + random.uniform(
                -exploration_range, exploration_range
            )
            return max(param.min_value, min(param.max_value, new_value))
        elif param.value_type == "int":
            range_size = param.max_value - param.min_value
            exploration_range = max(1, int(range_size * 0.1))
            new_value = int(param.current_value) + random.randint(
                -exploration_range, exploration_range
            )
            return max(param.min_value, min(param.max_value, new_value))

        return param.current_value

    def get_strategy_name(self) -> str:
        return "bayesian_optimization"


class RandomSearchStrategy(AdaptationStrategy):
    """Random search for exploration."""

    async def suggest_adaptation(
        self,
        performance_data: Dict[str, Any],
        available_parameters: List[AdaptationParameter],
    ) -> Optional[AdaptationExperiment]:
        """Suggest random parameter changes for exploration."""
        try:
            # Randomly select a parameter to modify
            param = random.choice(available_parameters)
            new_value = self._generate_random_value(param)

            if new_value != param.current_value:
                return AdaptationExperiment(
                    id=f"random_{param.name}_{int(time.time())}",
                    adaptation_type=AdaptationType.PARAMETER_TUNING,
                    status=AdaptationStatus.PLANNED,
                    parameters={param.name: new_value},
                    original_parameters={param.name: param.current_value},
                    target_metrics=param.impact_metrics,
                    start_time=datetime.utcnow(),
                    notes=f"Random exploration for {param.name}",
                )

            return None

        except Exception as e:
            logger.error("Random search suggestion failed", error=str(e))
            return None

    def _generate_random_value(self, param: AdaptationParameter) -> Any:
        """Generate random value for parameter."""
        if param.allowed_values:
            return random.choice(param.allowed_values)
        elif (
            param.value_type == "float"
            and param.min_value is not None
            and param.max_value is not None
        ):
            return random.uniform(param.min_value, param.max_value)
        elif (
            param.value_type == "int"
            and param.min_value is not None
            and param.max_value is not None
        ):
            return random.randint(param.min_value, param.max_value)
        elif param.value_type == "bool":
            return not param.current_value

        return param.current_value

    def get_strategy_name(self) -> str:
        return "random_search"


class LearningEngine:
    """
    Adaptive learning engine for continuous system improvement.

    Features:
    - Multiple optimization strategies (gradient descent, Bayesian, random)
    - Automated A/B testing and experimentation
    - Performance-based parameter adaptation
    - Learning insights extraction
    - Rollback mechanisms for failed experiments
    - Multi-objective optimization
    """

    def __init__(self, performance_tracker: PerformanceTracker):
        self.performance_tracker = performance_tracker
        self.settings = get_settings()

        # Experiment management
        self._experiments: Dict[str, AdaptationExperiment] = {}
        self._experiment_queue: deque = deque()
        self._active_experiment: Optional[AdaptationExperiment] = None

        # Parameters and strategies
        self._available_parameters: List[AdaptationParameter] = []
        self._strategies: List[AdaptationStrategy] = []
        self._learning_insights: List[LearningInsight] = []

        # Configuration
        self.config = self.settings.adaptation.learning
        self._max_concurrent_experiments = self.config.get(
            "max_concurrent_experiments", 1
        )
        self._experiment_duration_minutes = self.config.get(
            "experiment_duration_minutes", 60
        )
        self._min_improvement_threshold = self.config.get(
            "min_improvement_threshold", 0.05
        )
        self._rollback_threshold = self.config.get("rollback_threshold", -0.1)

        # Performance tracking
        self._total_experiments = 0
        self._successful_experiments = 0
        self._total_adaptation_time = 0.0

        # Initialize strategies
        self._initialize_strategies()
        self._initialize_parameters()

    def _initialize_strategies(self) -> None:
        """Initialize adaptation strategies."""
        try:
            # Load strategy configurations
            strategies_config = self.config.get("strategies", {})

            if strategies_config.get("gradient_descent", {}).get("enabled", True):
                learning_rate = strategies_config.get("gradient_descent", {}).get(
                    "learning_rate", 0.1
                )
                self._strategies.append(GradientDescentStrategy(learning_rate))

            if strategies_config.get("bayesian_optimization", {}).get("enabled", True):
                self._strategies.append(BayesianOptimizationStrategy())

            if strategies_config.get("random_search", {}).get("enabled", True):
                self._strategies.append(RandomSearchStrategy())

            logger.info(
                "Learning strategies initialized",
                strategies=[s.get_strategy_name() for s in self._strategies],
            )

        except Exception as e:
            logger.error("Failed to initialize strategies", error=str(e))

    def _initialize_parameters(self) -> None:
        """Initialize available parameters for adaptation."""
        try:
            params_config = self.config.get("parameters", {})

            # Response time optimization parameters
            if "response_time" in params_config:
                rt_config = params_config["response_time"]
                self._available_parameters.extend(
                    [
                        AdaptationParameter(
                            name="embedding_batch_size",
                            current_value=rt_config.get("embedding_batch_size", 32),
                            value_type="int",
                            min_value=1,
                            max_value=128,
                            step_size=1,
                            description="Batch size for embedding generation",
                            impact_metrics=[
                                MetricType.RESPONSE_TIME,
                                MetricType.MEMORY_USAGE,
                            ],
                        ),
                        AdaptationParameter(
                            name="search_top_k",
                            current_value=rt_config.get("search_top_k", 20),
                            value_type="int",
                            min_value=5,
                            max_value=100,
                            step_size=5,
                            description="Number of initial search results",
                            impact_metrics=[
                                MetricType.RESPONSE_TIME,
                                MetricType.SEARCH_RELEVANCE,
                            ],
                        ),
                        AdaptationParameter(
                            name="rerank_top_k",
                            current_value=rt_config.get("rerank_top_k", 5),
                            value_type="int",
                            min_value=1,
                            max_value=20,
                            step_size=1,
                            description="Number of results to rerank",
                            impact_metrics=[
                                MetricType.RESPONSE_TIME,
                                MetricType.ACCURACY,
                            ],
                        ),
                    ]
                )

            # Quality optimization parameters
            if "quality" in params_config:
                quality_config = params_config["quality"]
                self._available_parameters.extend(
                    [
                        AdaptationParameter(
                            name="confidence_threshold",
                            current_value=quality_config.get(
                                "confidence_threshold", 0.7
                            ),
                            value_type="float",
                            min_value=0.1,
                            max_value=0.95,
                            step_size=0.05,
                            description="Minimum confidence threshold for responses",
                            impact_metrics=[
                                MetricType.ACCURACY,
                                MetricType.CONFIDENCE,
                                MetricType.HALLUCINATION_RATE,
                            ],
                        ),
                        AdaptationParameter(
                            name="similarity_threshold",
                            current_value=quality_config.get(
                                "similarity_threshold", 0.6
                            ),
                            value_type="float",
                            min_value=0.1,
                            max_value=0.9,
                            step_size=0.05,
                            description="Minimum similarity threshold for search results",
                            impact_metrics=[
                                MetricType.SEARCH_RELEVANCE,
                                MetricType.ACCURACY,
                            ],
                        ),
                        AdaptationParameter(
                            name="hybrid_search_weight",
                            current_value=quality_config.get(
                                "hybrid_search_weight", 0.7
                            ),
                            value_type="float",
                            min_value=0.1,
                            max_value=0.9,
                            step_size=0.1,
                            description="Weight for vector vs text search in hybrid search",
                            impact_metrics=[
                                MetricType.SEARCH_RELEVANCE,
                                MetricType.RESPONSE_TIME,
                            ],
                        ),
                    ]
                )

            logger.info(
                "Adaptation parameters initialized",
                parameter_count=len(self._available_parameters),
                parameters=[p.name for p in self._available_parameters],
            )

        except Exception as e:
            logger.error("Failed to initialize parameters", error=str(e))

    async def start_learning_cycle(self) -> None:
        """Start the continuous learning cycle."""
        try:
            logger.info("Starting adaptive learning cycle")

            while True:
                # Check if we should run an experiment
                if await self._should_run_experiment():
                    experiment = await self._plan_next_experiment()
                    if experiment:
                        await self._run_experiment(experiment)

                # Process completed experiments
                await self._process_completed_experiments()

                # Extract learning insights
                await self._extract_insights()

                # Wait before next cycle
                await asyncio.sleep(self.config.get("cycle_interval", 300))  # 5 minutes

        except Exception as e:
            logger.error("Learning cycle failed", error=str(e))

    async def _should_run_experiment(self) -> bool:
        """Determine if we should run a new experiment."""
        try:
            # Check if we have capacity
            if self._active_experiment is not None:
                return False

            # Check if there are queued experiments
            if self._experiment_queue:
                return True

            # Check if performance suggests we need optimization
            performance_summary = (
                await self.performance_tracker.get_performance_summary(days=1)
            )

            # Look for degrading trends
            trends = performance_summary.get("trends", {})
            for metric_name, trend_data in trends.items():
                if (
                    trend_data.get("direction") == "degrading"
                    and trend_data.get("magnitude", 0) > 10
                    and trend_data.get("confidence", 0) > 0.6
                ):
                    return True

            # Check alert levels
            alerts = performance_summary.get("alerts", {})
            if alerts.get("active", 0) > 0:
                return True

            return False

        except Exception as e:
            logger.error("Experiment readiness check failed", error=str(e))
            return False

    async def _plan_next_experiment(self) -> Optional[AdaptationExperiment]:
        """Plan the next experiment using available strategies."""
        try:
            # Use queued experiment if available
            if self._experiment_queue:
                return self._experiment_queue.popleft()

            # Get current performance data
            performance_data = await self.performance_tracker.get_performance_summary(
                days=7
            )

            # Try each strategy to suggest an experiment
            for strategy in self._strategies:
                experiment = await strategy.suggest_adaptation(
                    performance_data, self._available_parameters
                )
                if experiment:
                    logger.info(
                        "Experiment planned",
                        experiment_id=experiment.id,
                        strategy=strategy.get_strategy_name(),
                        parameters=experiment.parameters,
                    )
                    return experiment

            return None

        except Exception as e:
            logger.error("Experiment planning failed", error=str(e))
            return None

    async def _run_experiment(self, experiment: AdaptationExperiment) -> None:
        """Run an adaptation experiment."""
        start_time = time.time()

        try:
            experiment.status = AdaptationStatus.RUNNING
            experiment.start_time = datetime.utcnow()
            self._active_experiment = experiment
            self._experiments[experiment.id] = experiment

            # Record baseline metrics
            baseline_summary = await self.performance_tracker.get_performance_summary(
                days=1
            )
            for metric_name, metric_data in baseline_summary.get("metrics", {}).items():
                try:
                    metric_type = MetricType(metric_name)
                    experiment.baseline_metrics[metric_type] = metric_data.get(
                        "mean", 0.0
                    )
                except ValueError:
                    continue

            # Apply parameter changes
            await self._apply_experiment_parameters(experiment)

            logger.info(
                "Experiment started",
                experiment_id=experiment.id,
                parameters=experiment.parameters,
                duration_minutes=self._experiment_duration_minutes,
            )

            # Wait for experiment duration
            await asyncio.sleep(self._experiment_duration_minutes * 60)

            # Collect experiment results
            await self._collect_experiment_results(experiment)

            # Evaluate experiment success
            await self._evaluate_experiment(experiment)

            # Update experiment status
            experiment.status = AdaptationStatus.COMPLETED
            experiment.end_time = datetime.utcnow()

            # Update performance tracking
            experiment_time = time.time() - start_time
            self._total_experiments += 1
            self._total_adaptation_time += experiment_time
            if experiment.success:
                self._successful_experiments += 1

            logger.info(
                "Experiment completed",
                experiment_id=experiment.id,
                success=experiment.success,
                improvements=experiment.improvement,
                confidence=experiment.confidence,
            )

        except Exception as e:
            experiment.status = AdaptationStatus.FAILED
            experiment.notes = f"Experiment failed: {str(e)}"
            logger.error(
                "Experiment execution failed", experiment_id=experiment.id, error=str(e)
            )

        finally:
            self._active_experiment = None

    async def _apply_experiment_parameters(
        self, experiment: AdaptationExperiment
    ) -> None:
        """Apply experiment parameters to the system."""
        try:
            # This would integrate with the actual system configuration
            # For now, we simulate parameter application
            logger.info(
                "Applied experiment parameters",
                experiment_id=experiment.id,
                parameters=experiment.parameters,
            )

            # Update parameter current values
            for param in self._available_parameters:
                if param.name in experiment.parameters:
                    param.current_value = experiment.parameters[param.name]

        except Exception as e:
            logger.error(
                "Parameter application failed",
                experiment_id=experiment.id,
                error=str(e),
            )
            raise

    async def _collect_experiment_results(
        self, experiment: AdaptationExperiment
    ) -> None:
        """Collect metrics during experiment period."""
        try:
            # Get performance data for experiment period
            experiment_summary = await self.performance_tracker.get_performance_summary(
                days=1
            )

            for metric_name, metric_data in experiment_summary.get(
                "metrics", {}
            ).items():
                try:
                    metric_type = MetricType(metric_name)
                    if metric_type in experiment.target_metrics:
                        experiment.experiment_metrics[metric_type] = metric_data.get(
                            "mean", 0.0
                        )
                except ValueError:
                    continue

        except Exception as e:
            logger.error(
                "Result collection failed", experiment_id=experiment.id, error=str(e)
            )

    async def _evaluate_experiment(self, experiment: AdaptationExperiment) -> None:
        """Evaluate experiment success and calculate improvements."""
        try:
            total_improvement = 0.0
            improvement_count = 0

            for metric_type in experiment.target_metrics:
                if (
                    metric_type in experiment.baseline_metrics
                    and metric_type in experiment.experiment_metrics
                ):
                    baseline = experiment.baseline_metrics[metric_type]
                    experimental = experiment.experiment_metrics[metric_type]

                    if baseline > 0:  # Avoid division by zero
                        # Calculate improvement percentage
                        if metric_type in [
                            MetricType.RESPONSE_TIME,
                            MetricType.MEMORY_USAGE,
                            MetricType.HALLUCINATION_RATE,
                        ]:
                            # Lower is better
                            improvement = (baseline - experimental) / baseline
                        else:
                            # Higher is better
                            improvement = (experimental - baseline) / baseline

                        experiment.improvement[metric_type] = improvement
                        total_improvement += improvement
                        improvement_count += 1

            # Calculate overall success
            if improvement_count > 0:
                avg_improvement = total_improvement / improvement_count
                experiment.confidence = min(
                    1.0, max(0.0, (improvement_count / len(experiment.target_metrics)))
                )

                # Consider experiment successful if average improvement exceeds threshold
                experiment.success = (
                    avg_improvement >= self._min_improvement_threshold
                    and all(
                        imp >= self._rollback_threshold
                        for imp in experiment.improvement.values()
                    )
                )
            else:
                experiment.success = False
                experiment.confidence = 0.0

            # Handle rollback if experiment failed badly
            if any(
                imp < self._rollback_threshold
                for imp in experiment.improvement.values()
            ):
                await self._rollback_experiment(experiment)

        except Exception as e:
            logger.error(
                "Experiment evaluation failed",
                experiment_id=experiment.id,
                error=str(e),
            )

    async def _rollback_experiment(self, experiment: AdaptationExperiment) -> None:
        """Rollback failed experiment parameters."""
        try:
            # Restore original parameters
            for param in self._available_parameters:
                if param.name in experiment.original_parameters:
                    param.current_value = experiment.original_parameters[param.name]

            experiment.status = AdaptationStatus.ROLLED_BACK
            experiment.notes += " | Parameters rolled back due to poor performance"

            logger.warning(
                "Experiment rolled back",
                experiment_id=experiment.id,
                improvements=experiment.improvement,
            )

        except Exception as e:
            logger.error(
                "Experiment rollback failed", experiment_id=experiment.id, error=str(e)
            )

    async def _process_completed_experiments(self) -> None:
        """Process completed experiments and update learning state."""
        try:
            completed_experiments = [
                exp
                for exp in self._experiments.values()
                if exp.status
                in [
                    AdaptationStatus.COMPLETED,
                    AdaptationStatus.FAILED,
                    AdaptationStatus.ROLLED_BACK,
                ]
            ]

            for experiment in completed_experiments:
                # Update strategy state (for gradient descent, etc.)
                for strategy in self._strategies:
                    if hasattr(strategy, "parameter_history"):
                        for param_name, param_value in experiment.parameters.items():
                            # Calculate score based on improvements
                            score = sum(experiment.improvement.values()) / max(
                                len(experiment.improvement), 1
                            )
                            strategy.parameter_history[param_name].append(
                                (param_value, score)
                            )

        except Exception as e:
            logger.error("Experiment processing failed", error=str(e))

    async def _extract_insights(self) -> None:
        """Extract learning insights from experiment history."""
        try:
            # Analyze successful experiments for patterns
            successful_experiments = [
                exp
                for exp in self._experiments.values()
                if exp.success and exp.status == AdaptationStatus.COMPLETED
            ]

            if len(successful_experiments) < 3:  # Need sufficient data
                return

            # Find parameter patterns that lead to improvements
            parameter_impacts = defaultdict(list)

            for experiment in successful_experiments:
                for param_name, param_value in experiment.parameters.items():
                    total_improvement = sum(experiment.improvement.values())
                    parameter_impacts[param_name].append(
                        (param_value, total_improvement)
                    )

            # Generate insights
            for param_name, impacts in parameter_impacts.items():
                if len(impacts) >= 3:
                    insight = self._analyze_parameter_impact(param_name, impacts)
                    if insight:
                        self._learning_insights.append(insight)

            # Keep only recent insights
            cutoff_time = datetime.utcnow() - timedelta(days=30)
            self._learning_insights = [
                insight
                for insight in self._learning_insights
                if insight.timestamp >= cutoff_time
            ]

        except Exception as e:
            logger.error("Insight extraction failed", error=str(e))

    def _analyze_parameter_impact(
        self, param_name: str, impacts: List[Tuple[Any, float]]
    ) -> Optional[LearningInsight]:
        """Analyze the impact of a parameter on performance."""
        try:
            if len(impacts) < 3:
                return None

            # Sort by parameter value
            impacts.sort(key=lambda x: x[0])
            values, improvements = zip(*impacts)

            # Calculate correlation
            correlation = (
                np.corrcoef(values, improvements)[0, 1] if len(values) > 1 else 0
            )

            if abs(correlation) > 0.5:  # Significant correlation
                direction = "increases" if correlation > 0 else "decreases"
                confidence = min(0.95, abs(correlation))

                insight_text = f"Parameter '{param_name}' shows strong correlation with performance. "
                insight_text += f"Performance {direction} as {param_name} increases. "
                insight_text += f"Optimal range appears to be {min(values):.3f} - {max(values):.3f}."

                return LearningInsight(
                    category="parameter_optimization",
                    insight=insight_text,
                    confidence=confidence,
                    supporting_experiments=[],  # Could add experiment IDs
                    timestamp=datetime.utcnow(),
                    actionable=True,
                    impact_estimate=f"Potential {abs(correlation)*20:.1f}% performance improvement",
                )

            return None

        except Exception as e:
            logger.error(
                "Parameter impact analysis failed", param_name=param_name, error=str(e)
            )
            return None

    async def queue_experiment(self, experiment: AdaptationExperiment) -> None:
        """Queue an experiment for execution."""
        self._experiment_queue.append(experiment)
        logger.info("Experiment queued", experiment_id=experiment.id)

    async def get_experiment_history(
        self, days: int = 30
    ) -> List[AdaptationExperiment]:
        """Get experiment history."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        return [
            exp for exp in self._experiments.values() if exp.start_time >= cutoff_time
        ]

    async def get_learning_insights(
        self, category: Optional[str] = None
    ) -> List[LearningInsight]:
        """Get learning insights with optional filtering."""
        insights = self._learning_insights

        if category:
            insights = [insight for insight in insights if insight.category == category]

        # Sort by confidence and recency
        insights.sort(key=lambda i: (i.confidence, i.timestamp), reverse=True)
        return insights

    async def get_current_parameters(self) -> List[AdaptationParameter]:
        """Get current parameter configuration."""
        return self._available_parameters.copy()

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning engine statistics."""
        success_rate = self._successful_experiments / max(self._total_experiments, 1)
        avg_adaptation_time = self._total_adaptation_time / max(
            self._total_experiments, 1
        )

        return {
            "total_experiments": self._total_experiments,
            "successful_experiments": self._successful_experiments,
            "success_rate": success_rate,
            "avg_adaptation_time": avg_adaptation_time,
            "active_experiment": (
                self._active_experiment.id if self._active_experiment else None
            ),
            "queued_experiments": len(self._experiment_queue),
            "available_parameters": len(self._available_parameters),
            "strategies": [s.get_strategy_name() for s in self._strategies],
            "learning_insights": len(self._learning_insights),
            "config": {
                "experiment_duration_minutes": self._experiment_duration_minutes,
                "min_improvement_threshold": self._min_improvement_threshold,
                "rollback_threshold": self._rollback_threshold,
            },
        }
