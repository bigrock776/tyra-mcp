"""
Adaptive configuration optimization system for Tyra MCP Memory Server.

This module provides intelligent configuration optimization based on performance
patterns, usage analytics, and system conditions.
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from .memory_health import HealthStatus, get_health_manager
from .performance_tracker import MetricType, get_performance_tracker

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Configuration optimization strategies."""

    PERFORMANCE = "performance"  # Optimize for speed
    EFFICIENCY = "efficiency"  # Optimize for resource usage
    RELIABILITY = "reliability"  # Optimize for stability
    BALANCED = "balanced"  # Balance all factors


class ConfigParameter(Enum):
    """Configurable parameters that can be optimized."""

    # Cache settings
    CACHE_TTL_EMBEDDINGS = "cache.ttl.embeddings"
    CACHE_TTL_SEARCH = "cache.ttl.search"
    CACHE_TTL_RERANK = "cache.ttl.rerank"
    CACHE_MAX_SIZE = "cache.max_size"

    # Database settings
    DB_POOL_SIZE = "database.pool_size"
    DB_TIMEOUT = "database.timeout"
    DB_MAX_CONNECTIONS = "database.max_connections"

    # Performance settings
    BATCH_SIZE_EMBEDDINGS = "embeddings.batch_size"
    WORKER_COUNT = "api.worker_count"
    REQUEST_TIMEOUT = "api.request_timeout"

    # RAG settings
    RETRIEVAL_LIMIT = "rag.retrieval_limit"
    RERANK_LIMIT = "rag.rerank_limit"
    HYBRID_WEIGHT = "rag.hybrid_weight"
    CONFIDENCE_THRESHOLD = "rag.confidence_threshold"

    # Model settings
    MODEL_DEVICE = "embeddings.device"
    MODEL_PRECISION = "embeddings.precision"


@dataclass
class ConfigValue:
    """Configuration parameter value with metadata."""

    parameter: ConfigParameter
    value: Union[int, float, str, bool]
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step_size: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Union[int, float, str, bool]]] = None
    description: str = ""
    impact_level: str = "medium"  # low, medium, high
    restart_required: bool = False


@dataclass
class OptimizationExperiment:
    """Single optimization experiment."""

    experiment_id: str
    parameter: ConfigParameter
    original_value: Union[int, float, str, bool]
    test_value: Union[int, float, str, bool]
    strategy: OptimizationStrategy
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: int = 15
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    improvement_score: Optional[float] = None
    success: Optional[bool] = None
    notes: str = ""


@dataclass
class OptimizationResult:
    """Result of configuration optimization."""

    parameter: ConfigParameter
    original_value: Union[int, float, str, bool]
    optimized_value: Union[int, float, str, bool]
    improvement_percentage: float
    confidence_score: float
    experiments_count: int
    strategy_used: OptimizationStrategy
    timestamp: datetime
    metrics_improved: List[str]
    trade_offs: List[str] = field(default_factory=list)


class ConfigurationOptimizer:
    """
    Intelligent configuration optimization system.

    Features:
    - Automatic parameter tuning based on performance metrics
    - A/B testing for configuration changes
    - Multi-objective optimization (performance vs efficiency vs reliability)
    - Safe experimentation with rollback capabilities
    - Learning from historical optimization results
    - Integration with performance tracking and health monitoring
    """

    def __init__(
        self,
        config_file_path: str = "config/config.yaml",
        optimization_interval_hours: int = 6,
        experiment_duration_minutes: int = 15,
        min_confidence_threshold: float = 0.7,
    ):
        self.config_file_path = Path(config_file_path)
        self.optimization_interval_hours = optimization_interval_hours
        self.experiment_duration_minutes = experiment_duration_minutes
        self.min_confidence_threshold = min_confidence_threshold

        # Current configuration
        self.current_config: Dict[str, Any] = {}
        self.parameter_definitions: Dict[ConfigParameter, ConfigValue] = {}

        # Optimization state
        self.active_experiments: Dict[str, OptimizationExperiment] = {}
        self.completed_experiments: List[OptimizationExperiment] = []
        self.optimization_results: List[OptimizationResult] = []

        # Performance baselines
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        self.last_optimization: Optional[datetime] = None

        # Background tasks
        self._running = False
        self._optimization_task: Optional[asyncio.Task] = None

        # Initialize parameter definitions
        self._initialize_parameter_definitions()

        # Load current configuration
        self._load_configuration()

        logger.info("ConfigurationOptimizer initialized")

    def _initialize_parameter_definitions(self):
        """Initialize configurable parameter definitions."""

        # Cache parameters
        self.parameter_definitions[ConfigParameter.CACHE_TTL_EMBEDDINGS] = ConfigValue(
            parameter=ConfigParameter.CACHE_TTL_EMBEDDINGS,
            value=3600,  # 1 hour default
            min_value=300,  # 5 minutes
            max_value=86400,  # 24 hours
            step_size=300,
            description="TTL for embedding cache in seconds",
            impact_level="medium",
        )

        self.parameter_definitions[ConfigParameter.CACHE_TTL_SEARCH] = ConfigValue(
            parameter=ConfigParameter.CACHE_TTL_SEARCH,
            value=1800,  # 30 minutes default
            min_value=60,  # 1 minute
            max_value=7200,  # 2 hours
            step_size=300,
            description="TTL for search cache in seconds",
            impact_level="medium",
        )

        self.parameter_definitions[ConfigParameter.DB_POOL_SIZE] = ConfigValue(
            parameter=ConfigParameter.DB_POOL_SIZE,
            value=20,
            min_value=5,
            max_value=100,
            step_size=5,
            description="Database connection pool size",
            impact_level="high",
            restart_required=True,
        )

        self.parameter_definitions[ConfigParameter.BATCH_SIZE_EMBEDDINGS] = ConfigValue(
            parameter=ConfigParameter.BATCH_SIZE_EMBEDDINGS,
            value=32,
            min_value=8,
            max_value=128,
            step_size=8,
            description="Batch size for embedding generation",
            impact_level="medium",
        )

        self.parameter_definitions[ConfigParameter.WORKER_COUNT] = ConfigValue(
            parameter=ConfigParameter.WORKER_COUNT,
            value=4,
            min_value=1,
            max_value=16,
            step_size=1,
            description="Number of API worker processes",
            impact_level="high",
            restart_required=True,
        )

        self.parameter_definitions[ConfigParameter.RETRIEVAL_LIMIT] = ConfigValue(
            parameter=ConfigParameter.RETRIEVAL_LIMIT,
            value=20,
            min_value=5,
            max_value=100,
            step_size=5,
            description="Maximum number of documents to retrieve",
            impact_level="medium",
        )

        self.parameter_definitions[ConfigParameter.RERANK_LIMIT] = ConfigValue(
            parameter=ConfigParameter.RERANK_LIMIT,
            value=10,
            min_value=3,
            max_value=50,
            step_size=1,
            description="Maximum number of documents to rerank",
            impact_level="medium",
        )

        self.parameter_definitions[ConfigParameter.HYBRID_WEIGHT] = ConfigValue(
            parameter=ConfigParameter.HYBRID_WEIGHT,
            value=0.7,
            min_value=0.1,
            max_value=0.9,
            step_size=0.1,
            description="Weight for vector search in hybrid search",
            impact_level="medium",
        )

        self.parameter_definitions[ConfigParameter.CONFIDENCE_THRESHOLD] = ConfigValue(
            parameter=ConfigParameter.CONFIDENCE_THRESHOLD,
            value=75.0,
            min_value=50.0,
            max_value=95.0,
            step_size=5.0,
            description="Minimum confidence threshold for responses",
            impact_level="high",
        )

        logger.info(
            f"Initialized {len(self.parameter_definitions)} parameter definitions"
        )

    def _load_configuration(self):
        """Load current configuration from file."""
        try:
            if self.config_file_path.exists():
                with open(self.config_file_path, "r") as f:
                    self.current_config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_file_path}")
            else:
                logger.warning(f"Configuration file not found: {self.config_file_path}")
                self.current_config = {}
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.current_config = {}

    def _save_configuration(self):
        """Save current configuration to file."""
        try:
            # Ensure directory exists
            self.config_file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file_path, "w") as f:
                yaml.dump(self.current_config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_file_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    async def start_optimization(self):
        """Start automatic configuration optimization."""
        if self._running:
            return

        self._running = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("Configuration optimization started")

    async def stop_optimization(self):
        """Stop configuration optimization."""
        self._running = False
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        logger.info("Configuration optimization stopped")

    async def _optimization_loop(self):
        """Main optimization loop."""
        while self._running:
            try:
                # Check if it's time for optimization
                if self._should_run_optimization():
                    await self._run_optimization_cycle()

                # Check active experiments
                await self._check_active_experiments()

                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(300)

    def _should_run_optimization(self) -> bool:
        """Check if optimization should run."""
        if not self.last_optimization:
            return True

        time_since_last = datetime.utcnow() - self.last_optimization
        return time_since_last >= timedelta(hours=self.optimization_interval_hours)

    async def _run_optimization_cycle(self):
        """Run a full optimization cycle."""
        logger.info("Starting optimization cycle")

        # Update performance baselines
        await self._update_performance_baselines()

        # Determine optimization strategy
        strategy = await self._determine_optimization_strategy()

        # Select parameters to optimize
        parameters_to_optimize = await self._select_optimization_targets(strategy)

        # Run experiments
        for parameter in parameters_to_optimize:
            if not self._running:
                break

            await self._run_parameter_experiment(parameter, strategy)

            # Wait between experiments
            await asyncio.sleep(60)

        self.last_optimization = datetime.utcnow()
        logger.info("Optimization cycle completed")

    async def _update_performance_baselines(self):
        """Update performance baselines from recent data."""
        performance_tracker = get_performance_tracker()
        health_manager = get_health_manager()

        # Get recent performance summary
        perf_summary = performance_tracker.get_performance_summary(hours=1)

        if "metrics" in perf_summary:
            for metric_name, metric_data in perf_summary["metrics"].items():
                if metric_name not in self.performance_baselines:
                    self.performance_baselines[metric_name] = {}

                self.performance_baselines[metric_name].update(
                    {
                        "mean": metric_data["mean"],
                        "p95": metric_data.get("p95", metric_data["mean"] * 1.2),
                        "count": metric_data["count"],
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                )

        # Get health metrics
        current_health = health_manager.get_current_health()
        if current_health and current_health.overall_status == HealthStatus.HEALTHY:
            for metric_name, metric in current_health.metrics.items():
                baseline_key = f"health_{metric_name}"
                if baseline_key not in self.performance_baselines:
                    self.performance_baselines[baseline_key] = {}

                self.performance_baselines[baseline_key].update(
                    {
                        "value": metric.value,
                        "status": metric.status.value,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                )

        logger.info(f"Updated {len(self.performance_baselines)} performance baselines")

    async def _determine_optimization_strategy(self) -> OptimizationStrategy:
        """Determine the best optimization strategy based on current conditions."""
        health_manager = get_health_manager()
        current_health = health_manager.get_current_health()

        if not current_health:
            return OptimizationStrategy.BALANCED

        # Check system health
        unhealthy_metrics = current_health.get_unhealthy_metrics()

        if unhealthy_metrics:
            # If there are resource issues, optimize for efficiency
            resource_issues = [
                m
                for m in unhealthy_metrics
                if "cpu" in m.name.lower() or "memory" in m.name.lower()
            ]
            if resource_issues:
                return OptimizationStrategy.EFFICIENCY

            # If there are reliability issues, optimize for reliability
            return OptimizationStrategy.RELIABILITY

        # If system is healthy, check performance trends
        performance_tracker = get_performance_tracker()
        patterns = performance_tracker.get_patterns(min_confidence=0.7)

        performance_declining = any(
            "increasing" in pattern["description"]
            and "latency" in pattern["description"].lower()
            for pattern in patterns
        )

        if performance_declining:
            return OptimizationStrategy.PERFORMANCE

        # Default to balanced optimization
        return OptimizationStrategy.BALANCED

    async def _select_optimization_targets(
        self, strategy: OptimizationStrategy
    ) -> List[ConfigParameter]:
        """Select parameters to optimize based on strategy."""
        targets = []

        if strategy == OptimizationStrategy.PERFORMANCE:
            targets.extend(
                [
                    ConfigParameter.CACHE_TTL_SEARCH,
                    ConfigParameter.CACHE_TTL_EMBEDDINGS,
                    ConfigParameter.BATCH_SIZE_EMBEDDINGS,
                    ConfigParameter.RETRIEVAL_LIMIT,
                ]
            )

        elif strategy == OptimizationStrategy.EFFICIENCY:
            targets.extend(
                [
                    ConfigParameter.DB_POOL_SIZE,
                    ConfigParameter.WORKER_COUNT,
                    ConfigParameter.CACHE_MAX_SIZE,
                    ConfigParameter.RERANK_LIMIT,
                ]
            )

        elif strategy == OptimizationStrategy.RELIABILITY:
            targets.extend(
                [
                    ConfigParameter.DB_TIMEOUT,
                    ConfigParameter.REQUEST_TIMEOUT,
                    ConfigParameter.CONFIDENCE_THRESHOLD,
                ]
            )

        else:  # BALANCED
            targets.extend(
                [
                    ConfigParameter.HYBRID_WEIGHT,
                    ConfigParameter.CACHE_TTL_SEARCH,
                    ConfigParameter.RETRIEVAL_LIMIT,
                    ConfigParameter.CONFIDENCE_THRESHOLD,
                ]
            )

        # Filter to only include parameters we have definitions for
        available_targets = [t for t in targets if t in self.parameter_definitions]

        # Limit to 3 parameters per cycle to avoid too many simultaneous changes
        return available_targets[:3]

    async def _run_parameter_experiment(
        self, parameter: ConfigParameter, strategy: OptimizationStrategy
    ):
        """Run an optimization experiment for a specific parameter."""
        param_def = self.parameter_definitions[parameter]
        current_value = self._get_current_parameter_value(parameter)

        # Generate test value
        test_value = self._generate_test_value(parameter, current_value, strategy)

        if test_value == current_value:
            logger.info(f"No optimization needed for {parameter.value}")
            return

        experiment_id = f"exp_{parameter.value}_{int(datetime.utcnow().timestamp())}"

        experiment = OptimizationExperiment(
            experiment_id=experiment_id,
            parameter=parameter,
            original_value=current_value,
            test_value=test_value,
            strategy=strategy,
            start_time=datetime.utcnow(),
            duration_minutes=self.experiment_duration_minutes,
        )

        logger.info(
            f"Starting experiment: {parameter.value} {current_value} -> {test_value}"
        )

        # Collect baseline metrics
        experiment.baseline_metrics = await self._collect_metrics()

        # Apply test configuration
        await self._apply_test_configuration(parameter, test_value)

        # Store active experiment
        self.active_experiments[experiment_id] = experiment

        logger.info(
            f"Experiment {experiment_id} started, will run for {self.experiment_duration_minutes} minutes"
        )

    def _get_current_parameter_value(
        self, parameter: ConfigParameter
    ) -> Union[int, float, str, bool]:
        """Get current value of a configuration parameter."""
        param_path = parameter.value.split(".")
        value = self.current_config

        try:
            for key in param_path:
                value = value[key]
            return value
        except (KeyError, TypeError):
            # Return default value if not found
            return self.parameter_definitions[parameter].value

    def _generate_test_value(
        self,
        parameter: ConfigParameter,
        current_value: Union[int, float, str, bool],
        strategy: OptimizationStrategy,
    ) -> Union[int, float, str, bool]:
        """Generate a test value for the parameter based on strategy."""
        param_def = self.parameter_definitions[parameter]

        if param_def.allowed_values:
            # Categorical parameter
            available_values = [
                v for v in param_def.allowed_values if v != current_value
            ]
            if available_values:
                return np.random.choice(available_values)
            return current_value

        if not isinstance(current_value, (int, float)):
            return current_value

        # Numeric parameter
        min_val = param_def.min_value or current_value * 0.5
        max_val = param_def.max_value or current_value * 2.0
        step = param_def.step_size or (max_val - min_val) / 20

        # Generate test value based on strategy
        if strategy == OptimizationStrategy.PERFORMANCE:
            # Try higher values for performance (more resources)
            if current_value < max_val:
                test_value = min(max_val, current_value + step)
            else:
                test_value = max(min_val, current_value - step)

        elif strategy == OptimizationStrategy.EFFICIENCY:
            # Try lower values for efficiency (fewer resources)
            if current_value > min_val:
                test_value = max(min_val, current_value - step)
            else:
                test_value = min(max_val, current_value + step)

        else:
            # Random search for balanced/reliability strategies
            direction = np.random.choice([-1, 1])
            test_value = current_value + (direction * step)
            test_value = max(min_val, min(max_val, test_value))

        # Ensure step size compliance
        if step and isinstance(step, (int, float)):
            test_value = round(test_value / step) * step

        return type(current_value)(test_value)

    async def _collect_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics."""
        metrics = {}

        # Get performance metrics
        performance_tracker = get_performance_tracker()
        perf_summary = performance_tracker.get_performance_summary(hours=0.5)

        if "metrics" in perf_summary:
            for metric_name, metric_data in perf_summary["metrics"].items():
                metrics[f"perf_{metric_name}_mean"] = metric_data["mean"]
                metrics[f"perf_{metric_name}_p95"] = metric_data.get(
                    "p95", metric_data["mean"] * 1.2
                )

        # Get health metrics
        health_manager = get_health_manager()
        current_health = health_manager.get_current_health()

        if current_health:
            for metric_name, metric in current_health.metrics.items():
                metrics[f"health_{metric_name}"] = metric.value

        # Add system utilization
        metrics["timestamp"] = datetime.utcnow().timestamp()

        return metrics

    async def _apply_test_configuration(
        self, parameter: ConfigParameter, test_value: Union[int, float, str, bool]
    ):
        """Apply test configuration value."""
        param_path = parameter.value.split(".")
        config = self.current_config

        # Navigate to the correct nested location
        for key in param_path[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the value
        config[param_path[-1]] = test_value

        # Save configuration
        self._save_configuration()

        logger.info(f"Applied test configuration: {parameter.value} = {test_value}")

        # Note: In a production system, you might need to restart services
        # or reload configuration dynamically depending on the parameter

    async def _check_active_experiments(self):
        """Check and complete active experiments."""
        completed_experiment_ids = []

        for experiment_id, experiment in self.active_experiments.items():
            elapsed_time = datetime.utcnow() - experiment.start_time

            if elapsed_time.total_seconds() >= experiment.duration_minutes * 60:
                # Experiment completed
                await self._complete_experiment(experiment)
                completed_experiment_ids.append(experiment_id)

        # Remove completed experiments
        for experiment_id in completed_experiment_ids:
            del self.active_experiments[experiment_id]

    async def _complete_experiment(self, experiment: OptimizationExperiment):
        """Complete an experiment and analyze results."""
        logger.info(f"Completing experiment: {experiment.experiment_id}")

        # Collect final metrics
        experiment.test_metrics = await self._collect_metrics()
        experiment.end_time = datetime.utcnow()

        # Calculate improvement score
        improvement_score = self._calculate_improvement_score(experiment)
        experiment.improvement_score = improvement_score

        # Determine if experiment was successful
        experiment.success = (
            improvement_score > 0 and improvement_score >= self.min_confidence_threshold
        )

        # Decide whether to keep the change
        if experiment.success:
            logger.info(
                f"Experiment {experiment.experiment_id} successful (score: {improvement_score:.3f})"
            )

            # Create optimization result
            result = OptimizationResult(
                parameter=experiment.parameter,
                original_value=experiment.original_value,
                optimized_value=experiment.test_value,
                improvement_percentage=improvement_score * 100,
                confidence_score=improvement_score,
                experiments_count=1,
                strategy_used=experiment.strategy,
                timestamp=datetime.utcnow(),
                metrics_improved=self._identify_improved_metrics(experiment),
            )

            self.optimization_results.append(result)

            # Keep the configuration change
            experiment.notes = "Configuration change applied"

        else:
            logger.info(
                f"Experiment {experiment.experiment_id} failed (score: {improvement_score:.3f})"
            )

            # Revert to original configuration
            await self._apply_test_configuration(
                experiment.parameter, experiment.original_value
            )
            experiment.notes = "Configuration reverted to original value"

        # Store completed experiment
        self.completed_experiments.append(experiment)

        # Limit history size
        if len(self.completed_experiments) > 100:
            self.completed_experiments = self.completed_experiments[-100:]

    def _calculate_improvement_score(self, experiment: OptimizationExperiment) -> float:
        """Calculate improvement score for an experiment."""
        if not experiment.baseline_metrics or not experiment.test_metrics:
            return 0.0

        improvements = []

        # Compare key performance metrics
        key_metrics = [
            "perf_latency_mean",
            "perf_latency_p95",
            "perf_throughput_mean",
            "perf_error_rate_mean",
            "health_cpu_usage",
            "health_memory_usage",
        ]

        for metric in key_metrics:
            if (
                metric in experiment.baseline_metrics
                and metric in experiment.test_metrics
            ):
                baseline = experiment.baseline_metrics[metric]
                test = experiment.test_metrics[metric]

                if baseline > 0:
                    # For latency, error rate, and resource usage: lower is better
                    if (
                        "latency" in metric
                        or "error_rate" in metric
                        or "usage" in metric
                    ):
                        improvement = (baseline - test) / baseline
                    else:
                        # For throughput: higher is better
                        improvement = (test - baseline) / baseline

                    improvements.append(improvement)

        if not improvements:
            return 0.0

        # Calculate weighted average improvement
        return np.mean(improvements)

    def _identify_improved_metrics(
        self, experiment: OptimizationExperiment
    ) -> List[str]:
        """Identify which metrics were improved by the experiment."""
        improved = []

        if not experiment.baseline_metrics or not experiment.test_metrics:
            return improved

        for metric in experiment.baseline_metrics:
            if metric in experiment.test_metrics:
                baseline = experiment.baseline_metrics[metric]
                test = experiment.test_metrics[metric]

                if baseline > 0:
                    # Check for significant improvement (>5%)
                    if (
                        "latency" in metric
                        or "error_rate" in metric
                        or "usage" in metric
                    ):
                        if (baseline - test) / baseline > 0.05:
                            improved.append(metric)
                    else:
                        if (test - baseline) / baseline > 0.05:
                            improved.append(metric)

        return improved

    # Public API methods
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            "running": self._running,
            "last_optimization": (
                self.last_optimization.isoformat() if self.last_optimization else None
            ),
            "active_experiments": len(self.active_experiments),
            "completed_experiments": len(self.completed_experiments),
            "optimization_results": len(self.optimization_results),
            "next_optimization": (
                (
                    self.last_optimization
                    + timedelta(hours=self.optimization_interval_hours)
                ).isoformat()
                if self.last_optimization
                else "Pending"
            ),
        }

    def get_active_experiments(self) -> List[Dict[str, Any]]:
        """Get list of active experiments."""
        experiments = []

        for experiment in self.active_experiments.values():
            elapsed_minutes = (
                datetime.utcnow() - experiment.start_time
            ).total_seconds() / 60
            remaining_minutes = max(0, experiment.duration_minutes - elapsed_minutes)

            experiments.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "parameter": experiment.parameter.value,
                    "original_value": experiment.original_value,
                    "test_value": experiment.test_value,
                    "strategy": experiment.strategy.value,
                    "elapsed_minutes": round(elapsed_minutes, 1),
                    "remaining_minutes": round(remaining_minutes, 1),
                    "progress_percentage": min(
                        100, (elapsed_minutes / experiment.duration_minutes) * 100
                    ),
                }
            )

        return experiments

    def get_optimization_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get optimization history."""
        history = []

        recent_results = sorted(
            self.optimization_results, key=lambda r: r.timestamp, reverse=True
        )[:limit]

        for result in recent_results:
            history.append(
                {
                    "parameter": result.parameter.value,
                    "original_value": result.original_value,
                    "optimized_value": result.optimized_value,
                    "improvement_percentage": round(result.improvement_percentage, 2),
                    "confidence_score": round(result.confidence_score, 3),
                    "strategy": result.strategy_used.value,
                    "timestamp": result.timestamp.isoformat(),
                    "metrics_improved": result.metrics_improved,
                }
            )

        return history

    def get_parameter_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for manual parameter tuning."""
        recommendations = []

        # Analyze recent patterns and health
        health_manager = get_health_manager()
        current_health = health_manager.get_current_health()

        if current_health:
            unhealthy_metrics = current_health.get_unhealthy_metrics()

            for metric in unhealthy_metrics:
                if "cpu" in metric.name.lower():
                    recommendations.append(
                        {
                            "parameter": ConfigParameter.WORKER_COUNT.value,
                            "suggestion": "Consider reducing worker count",
                            "reason": f"High CPU usage detected: {metric.value}%",
                            "priority": (
                                "high" if metric.status.name == "CRITICAL" else "medium"
                            ),
                        }
                    )

                elif "memory" in metric.name.lower():
                    recommendations.append(
                        {
                            "parameter": ConfigParameter.CACHE_MAX_SIZE.value,
                            "suggestion": "Consider reducing cache size",
                            "reason": f"High memory usage detected: {metric.value}%",
                            "priority": (
                                "high" if metric.status.name == "CRITICAL" else "medium"
                            ),
                        }
                    )

        # Analyze performance patterns
        performance_tracker = get_performance_tracker()
        patterns = performance_tracker.get_patterns(min_confidence=0.6)

        for pattern in patterns:
            if (
                "latency" in pattern["description"].lower()
                and "increasing" in pattern["description"]
            ):
                recommendations.append(
                    {
                        "parameter": ConfigParameter.CACHE_TTL_SEARCH.value,
                        "suggestion": "Consider increasing cache TTL",
                        "reason": "Increasing latency trend detected",
                        "priority": "medium",
                    }
                )

        return recommendations

    def export_optimization_data(self, filepath: Path):
        """Export optimization data for analysis."""
        export_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "configuration": self.current_config,
            "parameter_definitions": {},
            "completed_experiments": [],
            "optimization_results": [],
            "performance_baselines": self.performance_baselines,
        }

        # Export parameter definitions
        for param, param_def in self.parameter_definitions.items():
            export_data["parameter_definitions"][param.value] = asdict(param_def)

        # Export completed experiments
        for experiment in self.completed_experiments[-50:]:  # Last 50 experiments
            exp_data = asdict(experiment)
            exp_data["parameter"] = experiment.parameter.value
            exp_data["strategy"] = experiment.strategy.value
            exp_data["start_time"] = experiment.start_time.isoformat()
            if experiment.end_time:
                exp_data["end_time"] = experiment.end_time.isoformat()
            export_data["completed_experiments"].append(exp_data)

        # Export optimization results
        for result in self.optimization_results[-30:]:  # Last 30 results
            result_data = asdict(result)
            result_data["parameter"] = result.parameter.value
            result_data["strategy_used"] = result.strategy_used.value
            result_data["timestamp"] = result.timestamp.isoformat()
            export_data["optimization_results"].append(result_data)

        # Write to file
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Optimization data exported to {filepath}")


# Global optimizer instance
_config_optimizer: Optional[ConfigurationOptimizer] = None


def get_config_optimizer() -> ConfigurationOptimizer:
    """Get global configuration optimizer instance."""
    global _config_optimizer
    if _config_optimizer is None:
        _config_optimizer = ConfigurationOptimizer()
    return _config_optimizer
