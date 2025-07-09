"""Telemetry optimizer for reducing observability overhead."""

import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from ...utils.logger import get_logger
from ...utils.config import settings
from .performance_optimized_telemetry import get_telemetry, TelemetryLevel

logger = get_logger(__name__)


class OptimizationStrategy(Enum):
    """Telemetry optimization strategies."""
    AGGRESSIVE = "aggressive"  # Minimal telemetry, maximum performance
    BALANCED = "balanced"      # Balance between telemetry and performance
    CONSERVATIVE = "conservative"  # Prefer telemetry over performance


@dataclass
class PerformanceThresholds:
    """Performance thresholds for telemetry optimization."""
    cpu_threshold: float = 80.0
    memory_threshold: float = 80.0
    latency_threshold: float = 100.0  # ms
    error_rate_threshold: float = 0.05  # 5%
    telemetry_overhead_threshold: float = 0.001  # 1ms per operation


class TelemetryOptimizer:
    """Optimizer for reducing telemetry overhead while maintaining observability."""
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        thresholds: Optional[PerformanceThresholds] = None,
        optimization_interval: float = 60.0
    ):
        """Initialize telemetry optimizer.
        
        Args:
            strategy: Optimization strategy
            thresholds: Performance thresholds
            optimization_interval: How often to run optimization (seconds)
        """
        self.strategy = strategy
        self.thresholds = thresholds or PerformanceThresholds()
        self.optimization_interval = optimization_interval
        
        self.telemetry = get_telemetry()
        self._last_optimization = time.time()
        self._optimization_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self._performance_samples: List[Dict[str, float]] = []
        self._max_samples = 100
        
        # Optimization state
        self._current_level = TelemetryLevel.STANDARD
        self._current_sampling_rate = 1.0
        self._optimization_active = False
        
    async def optimize(self) -> Dict[str, Any]:
        """Run telemetry optimization."""
        if time.time() - self._last_optimization < self.optimization_interval:
            return {"status": "skipped", "reason": "too_soon"}
            
        self._last_optimization = time.time()
        optimization_start = time.time()
        
        try:
            # Collect current performance metrics
            performance_data = await self._collect_performance_data()
            
            # Analyze performance
            analysis = self._analyze_performance(performance_data)
            
            # Determine optimization actions
            actions = self._determine_optimization_actions(analysis)
            
            # Apply optimizations
            changes = await self._apply_optimizations(actions)
            
            # Record optimization
            optimization_record = {
                "timestamp": time.time(),
                "strategy": self.strategy.value,
                "performance_data": performance_data,
                "analysis": analysis,
                "actions": actions,
                "changes": changes,
                "optimization_time": time.time() - optimization_start
            }
            
            self._optimization_history.append(optimization_record)
            
            # Keep only recent history
            if len(self._optimization_history) > 50:
                self._optimization_history = self._optimization_history[-50:]
                
            return {
                "status": "completed",
                "changes": changes,
                "performance_impact": analysis.get("performance_impact", "unknown"),
                "optimization_time": optimization_record["optimization_time"]
            }
            
        except Exception as e:
            logger.error(f"Telemetry optimization failed: {e}")
            return {"status": "failed", "error": str(e)}
            
    async def _collect_performance_data(self) -> Dict[str, Any]:
        """Collect current performance data."""
        data = {
            "telemetry_stats": self.telemetry.get_performance_stats(),
            "system_metrics": {}
        }
        
        try:
            import psutil
            data["system_metrics"] = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
            }
        except ImportError:
            logger.warning("psutil not available, using telemetry-only metrics")
            
        # Add application-specific metrics
        metrics_summary = self.telemetry.get_metrics_summary()
        data["application_metrics"] = {
            "request_count": metrics_summary.get("counters", {}).get("requests_total", 0),
            "error_count": metrics_summary.get("counters", {}).get("errors_total", 0),
            "avg_latency": metrics_summary.get("histograms", {}).get("request_duration", {}).get("avg", 0)
        }
        
        return data
        
    def _analyze_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance data and identify issues."""
        analysis = {
            "performance_impact": "low",
            "issues": [],
            "recommendations": []
        }
        
        # Check system metrics
        system_metrics = data.get("system_metrics", {})
        if system_metrics:
            cpu_percent = system_metrics.get("cpu_percent", 0)
            memory_percent = system_metrics.get("memory_percent", 0)
            
            if cpu_percent > self.thresholds.cpu_threshold:
                analysis["issues"].append(f"High CPU usage: {cpu_percent}%")
                analysis["performance_impact"] = "high"
                
            if memory_percent > self.thresholds.memory_threshold:
                analysis["issues"].append(f"High memory usage: {memory_percent}%")
                analysis["performance_impact"] = "high"
                
        # Check telemetry overhead
        telemetry_stats = data.get("telemetry_stats", {})
        overhead_per_op = telemetry_stats.get("avg_overhead_per_op", 0)
        
        if overhead_per_op > self.thresholds.telemetry_overhead_threshold:
            analysis["issues"].append(f"High telemetry overhead: {overhead_per_op*1000:.3f}ms/op")
            analysis["recommendations"].append("reduce_telemetry_sampling")
            
        # Check application metrics
        app_metrics = data.get("application_metrics", {})
        avg_latency = app_metrics.get("avg_latency", 0)
        
        if avg_latency > self.thresholds.latency_threshold / 1000:  # Convert to seconds
            analysis["issues"].append(f"High latency: {avg_latency*1000:.1f}ms")
            analysis["recommendations"].append("reduce_telemetry_level")
            
        # Check error rate
        request_count = app_metrics.get("request_count", 0)
        error_count = app_metrics.get("error_count", 0)
        
        if request_count > 0:
            error_rate = error_count / request_count
            if error_rate > self.thresholds.error_rate_threshold:
                analysis["issues"].append(f"High error rate: {error_rate*100:.1f}%")
                analysis["recommendations"].append("increase_telemetry_level")
                
        return analysis
        
    def _determine_optimization_actions(self, analysis: Dict[str, Any]) -> List[str]:
        """Determine optimization actions based on analysis."""
        actions = []
        recommendations = analysis.get("recommendations", [])
        performance_impact = analysis.get("performance_impact", "low")
        
        if self.strategy == OptimizationStrategy.AGGRESSIVE:
            # Always prioritize performance
            if performance_impact in ["high", "medium"]:
                actions.extend(["reduce_sampling", "disable_spans", "minimal_level"])
            elif "reduce_telemetry_sampling" in recommendations:
                actions.append("reduce_sampling")
                
        elif self.strategy == OptimizationStrategy.BALANCED:
            # Balance performance and observability
            if performance_impact == "high":
                actions.extend(["reduce_sampling", "standard_level"])
            elif "reduce_telemetry_sampling" in recommendations:
                actions.append("reduce_sampling")
            elif "reduce_telemetry_level" in recommendations:
                actions.append("standard_level")
                
        elif self.strategy == OptimizationStrategy.CONSERVATIVE:
            # Prefer observability over performance
            if performance_impact == "high" and "reduce_telemetry_sampling" in recommendations:
                actions.append("moderate_reduce_sampling")
                
        # Handle error rate increase recommendations
        if "increase_telemetry_level" in recommendations:
            actions.append("detailed_level")
            
        return actions
        
    async def _apply_optimizations(self, actions: List[str]) -> Dict[str, Any]:
        """Apply optimization actions."""
        changes = {}
        
        for action in actions:
            if action == "reduce_sampling":
                old_rate = self.telemetry.sampling_rate
                new_rate = max(0.1, old_rate * 0.7)
                self.telemetry.sampling_rate = new_rate
                changes["sampling_rate"] = {"old": old_rate, "new": new_rate}
                
            elif action == "moderate_reduce_sampling":
                old_rate = self.telemetry.sampling_rate
                new_rate = max(0.3, old_rate * 0.9)
                self.telemetry.sampling_rate = new_rate
                changes["sampling_rate"] = {"old": old_rate, "new": new_rate}
                
            elif action == "increase_sampling":
                old_rate = self.telemetry.sampling_rate
                new_rate = min(1.0, old_rate * 1.2)
                self.telemetry.sampling_rate = new_rate
                changes["sampling_rate"] = {"old": old_rate, "new": new_rate}
                
            elif action == "minimal_level":
                old_level = self.telemetry.telemetry_level
                self.telemetry.set_telemetry_level(TelemetryLevel.MINIMAL)
                changes["telemetry_level"] = {"old": old_level.name, "new": "MINIMAL"}
                
            elif action == "standard_level":
                old_level = self.telemetry.telemetry_level
                self.telemetry.set_telemetry_level(TelemetryLevel.STANDARD)
                changes["telemetry_level"] = {"old": old_level.name, "new": "STANDARD"}
                
            elif action == "detailed_level":
                old_level = self.telemetry.telemetry_level
                self.telemetry.set_telemetry_level(TelemetryLevel.DETAILED)
                changes["telemetry_level"] = {"old": old_level.name, "new": "DETAILED"}
                
            elif action == "disable_spans":
                self.telemetry._spans_enabled = False
                changes["spans_enabled"] = {"old": True, "new": False}
                
            elif action == "enable_spans":
                self.telemetry._spans_enabled = True
                changes["spans_enabled"] = {"old": False, "new": True}
                
        return changes
        
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self._optimization_history.copy()
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get current optimization state."""
        return {
            "strategy": self.strategy.value,
            "thresholds": {
                "cpu_threshold": self.thresholds.cpu_threshold,
                "memory_threshold": self.thresholds.memory_threshold,
                "latency_threshold": self.thresholds.latency_threshold,
                "error_rate_threshold": self.thresholds.error_rate_threshold,
                "telemetry_overhead_threshold": self.thresholds.telemetry_overhead_threshold
            },
            "telemetry_state": self.telemetry.get_performance_stats(),
            "last_optimization": self._last_optimization,
            "optimization_history_count": len(self._optimization_history)
        }
        
    async def emergency_optimization(self) -> Dict[str, Any]:
        """Emergency optimization for critical performance issues."""
        logger.warning("Emergency telemetry optimization triggered")
        
        # Immediate actions for critical performance
        changes = await self._apply_optimizations([
            "reduce_sampling",
            "minimal_level",
            "disable_spans"
        ])
        
        # Force aggressive sampling reduction
        self.telemetry.sampling_rate = 0.05  # 5% sampling
        changes["emergency_sampling"] = {"old": "unknown", "new": 0.05}
        
        return {
            "status": "emergency_optimization_applied",
            "changes": changes,
            "message": "Telemetry reduced to minimal level for performance recovery"
        }
        
    def reset_optimizations(self) -> Dict[str, Any]:
        """Reset telemetry to default settings."""
        old_state = self.telemetry.get_performance_stats()
        
        # Reset to defaults
        self.telemetry.sampling_rate = 1.0
        self.telemetry.set_telemetry_level(TelemetryLevel.STANDARD)
        self.telemetry._spans_enabled = True
        
        new_state = self.telemetry.get_performance_stats()
        
        return {
            "status": "reset_completed",
            "old_state": old_state,
            "new_state": new_state
        }


# Global optimizer instance
_optimizer_instance: Optional[TelemetryOptimizer] = None


def get_telemetry_optimizer() -> TelemetryOptimizer:
    """Get the global telemetry optimizer instance."""
    global _optimizer_instance
    
    if _optimizer_instance is None:
        # Configure based on settings
        config = getattr(settings, 'observability', {})
        
        strategy = OptimizationStrategy.BALANCED
        if config.get('environment') == 'production':
            strategy = OptimizationStrategy.AGGRESSIVE
        elif config.get('environment') == 'development':
            strategy = OptimizationStrategy.CONSERVATIVE
            
        _optimizer_instance = TelemetryOptimizer(
            strategy=strategy,
            optimization_interval=config.get('optimization_interval', 60.0)
        )
        
    return _optimizer_instance


async def optimization_loop():
    """Background optimization loop."""
    optimizer = get_telemetry_optimizer()
    
    while True:
        try:
            result = await optimizer.optimize()
            if result["status"] == "completed":
                logger.debug(f"Telemetry optimization completed: {result}")
                
            await asyncio.sleep(optimizer.optimization_interval)
            
        except Exception as e:
            logger.error(f"Optimization loop error: {e}")
            await asyncio.sleep(120)  # Wait longer on error