"""
Self-Training Scheduler and Coordination System.

Coordinates scheduled improvement jobs, pattern detection, and automated system
optimization across all self-learning components.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .memory_health import MemoryHealthManager
from .config_optimizer import ConfigOptimizer
from .ab_testing import ABTestingFramework
from .prompt_evolution import PromptEvolutionEngine
from ..analytics.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Self-training job status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels."""
    
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SelfTrainingJob:
    """Self-training job specification."""
    
    job_id: str
    job_type: str
    priority: JobPriority
    schedule_pattern: str  # cron-like pattern
    target_component: str
    configuration: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout_minutes: int = 60
    retry_count: int = 3
    status: JobStatus = JobStatus.PENDING
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    error_history: List[str] = field(default_factory=list)


@dataclass
class ImprovementAction:
    """Represents a system improvement action."""
    
    action_id: str
    action_type: str
    component: str
    description: str
    confidence: float
    impact_score: float
    risk_score: float
    auto_apply: bool
    configuration_changes: Dict[str, Any]
    rollback_plan: Dict[str, Any]
    validation_checks: List[str]
    applied: bool = False
    applied_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None


class SelfTrainingScheduler:
    """Coordinates all self-learning and improvement activities."""
    
    def __init__(self):
        self.memory_health = None
        self.config_optimizer = None
        self.ab_testing = None
        self.prompt_evolution = None
        self.performance_tracker = None
        
        self.scheduled_jobs: Dict[str, SelfTrainingJob] = {}
        self.improvement_actions: Dict[str, ImprovementAction] = {}
        self.active_experiments: Set[str] = set()
        
        self._scheduler_running = False
        self._job_executor_pool = None
        self._initialized = False
        
    async def initialize(self, config: Dict[str, Any]):
        """Initialize self-training scheduler."""
        try:
            # Initialize component managers
            self.memory_health = MemoryHealthManager()
            await self.memory_health.initialize(config.get("memory_health", {}))
            
            self.config_optimizer = ConfigOptimizer()
            await self.config_optimizer.initialize(config.get("config_optimizer", {}))
            
            self.ab_testing = ABTestingFramework()
            await self.ab_testing.initialize(config.get("ab_testing", {}))
            
            self.prompt_evolution = PromptEvolutionEngine()
            await self.prompt_evolution.initialize(config.get("prompt_evolution", {}))
            
            self.performance_tracker = PerformanceTracker()
            
            # Create default scheduled jobs
            await self._create_default_jobs(config.get("default_jobs", {}))
            
            # Start scheduler
            await self._start_scheduler()
            
            self._initialized = True
            logger.info("Self-training scheduler initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize self-training scheduler: {e}")
            raise
    
    async def add_scheduled_job(self, job: SelfTrainingJob):
        """Add a new scheduled job."""
        try:
            # Validate job configuration
            self._validate_job(job)
            
            # Calculate next run time
            job.next_run = self._calculate_next_run(job.schedule_pattern)
            
            # Store job
            self.scheduled_jobs[job.job_id] = job
            
            # Persist to database
            await self._store_job(job)
            
            logger.info(f"Added scheduled job: {job.job_id}")
            
        except Exception as e:
            logger.error(f"Failed to add scheduled job: {e}")
            raise
    
    async def run_comprehensive_analysis(self, force: bool = False) -> Dict[str, Any]:
        """Run comprehensive system analysis and generate improvement recommendations."""
        try:
            analysis_results = {
                "timestamp": datetime.utcnow(),
                "memory_health": {},
                "performance_analysis": {},
                "prompt_analysis": {},
                "configuration_analysis": {},
                "improvement_recommendations": [],
                "risk_assessment": {},
                "overall_score": 0.0
            }
            
            # Memory health analysis
            logger.info("Running memory health analysis...")
            memory_analysis = await self.memory_health.analyze_memory_health()
            analysis_results["memory_health"] = memory_analysis
            
            # Performance analysis
            logger.info("Running performance analysis...")
            performance_analysis = await self.performance_tracker.analyze_latency_trends("all", hours=24)
            analysis_results["performance_analysis"] = performance_analysis
            
            # Prompt analysis
            logger.info("Running prompt analysis...")
            prompt_analysis = await self.prompt_evolution.analyze_prompt_performance()
            analysis_results["prompt_analysis"] = prompt_analysis
            
            # Configuration analysis
            logger.info("Running configuration analysis...")
            config_analysis = await self.config_optimizer.analyze_current_configuration()
            analysis_results["configuration_analysis"] = config_analysis
            
            # Generate improvement recommendations
            recommendations = await self._generate_improvement_recommendations(analysis_results)
            analysis_results["improvement_recommendations"] = recommendations
            
            # Calculate overall health score
            overall_score = self._calculate_overall_health_score(analysis_results)
            analysis_results["overall_score"] = overall_score
            
            # Risk assessment
            risk_assessment = await self._assess_system_risks(analysis_results)
            analysis_results["risk_assessment"] = risk_assessment
            
            # Store analysis results
            await self._store_analysis_results(analysis_results)
            
            # Trigger immediate actions if needed
            if overall_score < 0.7:
                await self._trigger_emergency_improvements(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise
    
    async def execute_improvement_action(self, action_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """Execute a specific improvement action."""
        try:
            action = self.improvement_actions.get(action_id)
            if not action:
                raise ValueError(f"Improvement action {action_id} not found")
            
            if action.applied:
                raise ValueError(f"Action {action_id} has already been applied")
            
            execution_result = {
                "action_id": action_id,
                "dry_run": dry_run,
                "started_at": datetime.utcnow(),
                "status": "success",
                "changes_applied": [],
                "validation_results": {},
                "rollback_info": None
            }
            
            if dry_run:
                # Simulate the action
                execution_result["simulated_changes"] = action.configuration_changes
                execution_result["estimated_impact"] = action.impact_score
                execution_result["risk_factors"] = self._assess_action_risks(action)
                return execution_result
            
            try:
                # Pre-execution validation
                validation_results = await self._validate_action_preconditions(action)
                execution_result["validation_results"] = validation_results
                
                if not validation_results["all_checks_passed"]:
                    execution_result["status"] = "failed"
                    execution_result["error"] = "Pre-execution validation failed"
                    return execution_result
                
                # Create rollback checkpoint
                rollback_checkpoint = await self._create_rollback_checkpoint(action)
                execution_result["rollback_info"] = rollback_checkpoint
                
                # Execute the action based on component
                if action.component == "memory_health":
                    changes = await self._execute_memory_health_action(action)
                elif action.component == "configuration":
                    changes = await self._execute_configuration_action(action)
                elif action.component == "prompts":
                    changes = await self._execute_prompt_action(action)
                elif action.component == "performance":
                    changes = await self._execute_performance_action(action)
                else:
                    raise ValueError(f"Unknown component: {action.component}")
                
                execution_result["changes_applied"] = changes
                
                # Post-execution validation
                post_validation = await self._validate_action_results(action, changes)
                execution_result["post_validation"] = post_validation
                
                if post_validation["success"]:
                    # Mark action as applied
                    action.applied = True
                    action.applied_at = datetime.utcnow()
                    action.results = execution_result
                    
                    # Store updated action
                    await self._store_improvement_action(action)
                    
                    logger.info(f"Successfully executed improvement action: {action_id}")
                else:
                    # Rollback changes
                    await self._rollback_action(action, rollback_checkpoint)
                    execution_result["status"] = "rolled_back"
                    execution_result["rollback_reason"] = "Post-execution validation failed"
                
            except Exception as e:
                execution_result["status"] = "failed"
                execution_result["error"] = str(e)
                
                # Attempt rollback if rollback info exists
                if execution_result.get("rollback_info"):
                    try:
                        await self._rollback_action(action, execution_result["rollback_info"])
                        execution_result["rollback_completed"] = True
                    except Exception as rollback_error:
                        execution_result["rollback_error"] = str(rollback_error)
                
                logger.error(f"Failed to execute improvement action {action_id}: {e}")
            
            execution_result["completed_at"] = datetime.utcnow()
            return execution_result
            
        except Exception as e:
            logger.error(f"Failed to execute improvement action: {e}")
            raise
    
    async def start_automated_optimization_experiment(self, experiment_name: str, components: List[str]) -> str:
        """Start an automated optimization experiment."""
        try:
            # Generate experiment configurations
            experiment_configs = await self._generate_optimization_experiments(components)
            
            # Create A/B test
            experiment_id = await self.ab_testing.create_configuration_experiment(
                experiment_name=experiment_name,
                base_config=experiment_configs["baseline"],
                parameter_variations=experiment_configs["variations"]
            )
            
            # Start experiment
            await self.ab_testing.start_experiment(experiment_id)
            
            # Track experiment
            self.active_experiments.add(experiment_id)
            
            # Schedule experiment monitoring
            await self._schedule_experiment_monitoring(experiment_id)
            
            logger.info(f"Started automated optimization experiment: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to start optimization experiment: {e}")
            raise
    
    async def get_system_improvement_status(self) -> Dict[str, Any]:
        """Get current status of all system improvements."""
        try:
            status = {
                "overall_health_score": await self._calculate_current_health_score(),
                "active_jobs": len([j for j in self.scheduled_jobs.values() if j.status == JobStatus.RUNNING]),
                "pending_actions": len([a for a in self.improvement_actions.values() if not a.applied]),
                "active_experiments": len(self.active_experiments),
                "recent_improvements": [],
                "upcoming_jobs": [],
                "system_trends": {}
            }
            
            # Get recent improvements
            recent_improvements = [
                {
                    "action_id": a.action_id,
                    "description": a.description,
                    "applied_at": a.applied_at,
                    "impact_score": a.impact_score
                }
                for a in self.improvement_actions.values()
                if a.applied and a.applied_at and a.applied_at > datetime.utcnow() - timedelta(days=7)
            ]
            status["recent_improvements"] = sorted(recent_improvements, key=lambda x: x["applied_at"], reverse=True)
            
            # Get upcoming jobs
            upcoming_jobs = [
                {
                    "job_id": j.job_id,
                    "job_type": j.job_type,
                    "next_run": j.next_run,
                    "priority": j.priority.value
                }
                for j in self.scheduled_jobs.values()
                if j.next_run and j.next_run > datetime.utcnow()
            ]
            status["upcoming_jobs"] = sorted(upcoming_jobs, key=lambda x: x["next_run"])[:10]
            
            # Get system trends
            status["system_trends"] = await self._analyze_improvement_trends()
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get improvement status: {e}")
            raise
    
    # Private methods
    
    async def _create_default_jobs(self, config: Dict[str, Any]):
        """Create default scheduled jobs."""
        default_jobs = [
            SelfTrainingJob(
                job_id="memory_health_check",
                job_type="memory_health_analysis",
                priority=JobPriority.HIGH,
                schedule_pattern="0 2 * * *",  # Daily at 2 AM
                target_component="memory_health",
                configuration={"comprehensive": True},
                timeout_minutes=30
            ),
            SelfTrainingJob(
                job_id="prompt_optimization",
                job_type="prompt_analysis",
                priority=JobPriority.MEDIUM,
                schedule_pattern="0 4 * * 0",  # Weekly on Sunday at 4 AM
                target_component="prompt_evolution",
                configuration={"generate_improvements": True},
                timeout_minutes=60
            ),
            SelfTrainingJob(
                job_id="configuration_optimization",
                job_type="config_analysis",
                priority=JobPriority.MEDIUM,
                schedule_pattern="0 3 1 * *",  # Monthly on 1st at 3 AM
                target_component="config_optimizer",
                configuration={"deep_analysis": True},
                timeout_minutes=90
            ),
            SelfTrainingJob(
                job_id="performance_trending",
                job_type="performance_analysis",
                priority=JobPriority.HIGH,
                schedule_pattern="0 */6 * * *",  # Every 6 hours
                target_component="performance_tracker",
                configuration={"trend_analysis": True},
                timeout_minutes=15
            ),
            SelfTrainingJob(
                job_id="comprehensive_system_review",
                job_type="comprehensive_analysis",
                priority=JobPriority.CRITICAL,
                schedule_pattern="0 1 * * 0",  # Weekly on Sunday at 1 AM
                target_component="scheduler",
                configuration={"full_analysis": True},
                timeout_minutes=120
            )
        ]
        
        for job in default_jobs:
            await self.add_scheduled_job(job)
    
    async def _start_scheduler(self):
        """Start the job scheduler."""
        if self._scheduler_running:
            return
        
        self._scheduler_running = True
        asyncio.create_task(self._scheduler_loop())
        logger.info("Self-training scheduler started")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._scheduler_running:
            try:
                current_time = datetime.utcnow()
                
                # Check for jobs ready to run
                for job in self.scheduled_jobs.values():
                    if (job.status == JobStatus.PENDING and 
                        job.next_run and 
                        job.next_run <= current_time):
                        
                        # Execute job
                        asyncio.create_task(self._execute_job(job))
                
                # Check experiment status
                await self._check_experiment_status()
                
                # Apply ready improvements
                await self._apply_ready_improvements()
                
                # Sleep for 1 minute before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(60)
    
    async def _execute_job(self, job: SelfTrainingJob):
        """Execute a scheduled job."""
        try:
            job.status = JobStatus.RUNNING
            job.last_run = datetime.utcnow()
            
            logger.info(f"Executing job: {job.job_id}")
            
            # Execute based on job type
            if job.job_type == "memory_health_analysis":
                results = await self.memory_health.analyze_memory_health()
            elif job.job_type == "prompt_analysis":
                results = await self.prompt_evolution.analyze_prompt_performance()
            elif job.job_type == "config_analysis":
                results = await self.config_optimizer.analyze_current_configuration()
            elif job.job_type == "performance_analysis":
                results = await self.performance_tracker.analyze_latency_trends("all")
            elif job.job_type == "comprehensive_analysis":
                results = await self.run_comprehensive_analysis()
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")
            
            job.results = results
            job.status = JobStatus.COMPLETED
            
            # Calculate next run time
            job.next_run = self._calculate_next_run(job.schedule_pattern)
            
            logger.info(f"Job completed successfully: {job.job_id}")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_history.append(f"{datetime.utcnow()}: {str(e)}")
            
            # Retry logic
            if job.retry_count > 0:
                job.retry_count -= 1
                job.status = JobStatus.PENDING
                job.next_run = datetime.utcnow() + timedelta(minutes=5)  # Retry in 5 minutes
            
            logger.error(f"Job failed: {job.job_id}: {e}")
        
        finally:
            await self._store_job(job)
    
    async def _generate_improvement_recommendations(self, analysis_results: Dict[str, Any]) -> List[ImprovementAction]:
        """Generate improvement recommendations based on analysis results."""
        recommendations = []
        
        # Memory health improvements
        memory_health = analysis_results.get("memory_health", {})
        if memory_health.get("overall_score", 1.0) < 0.8:
            action = ImprovementAction(
                action_id=f"memory_cleanup_{int(datetime.utcnow().timestamp())}",
                action_type="memory_cleanup",
                component="memory_health",
                description="Clean up stale and redundant memories",
                confidence=0.9,
                impact_score=0.7,
                risk_score=0.2,
                auto_apply=True,
                configuration_changes={"execute_cleanup": True, "aggressive": False},
                rollback_plan={"restore_from_backup": True},
                validation_checks=["verify_important_memories_preserved"]
            )
            recommendations.append(action)
        
        # Performance improvements
        performance_analysis = analysis_results.get("performance_analysis", {})
        if performance_analysis.get("trend_direction") == "declining":
            action = ImprovementAction(
                action_id=f"perf_optimization_{int(datetime.utcnow().timestamp())}",
                action_type="performance_optimization",
                component="configuration",
                description="Optimize configuration for better performance",
                confidence=0.8,
                impact_score=0.8,
                risk_score=0.3,
                auto_apply=False,  # Requires approval
                configuration_changes={"cache_ttl": 3600, "batch_size": 64},
                rollback_plan={"restore_previous_config": True},
                validation_checks=["verify_performance_improvement", "check_accuracy_maintained"]
            )
            recommendations.append(action)
        
        # Prompt improvements
        prompt_analysis = analysis_results.get("prompt_analysis", {})
        low_performing_prompts = [
            template_id for template_id, analysis in prompt_analysis.items()
            if analysis.get("metrics", {}).get("success_rate", 1.0) < 0.8
        ]
        
        if low_performing_prompts:
            action = ImprovementAction(
                action_id=f"prompt_improvement_{int(datetime.utcnow().timestamp())}",
                action_type="prompt_optimization",
                component="prompts",
                description=f"Improve {len(low_performing_prompts)} low-performing prompt templates",
                confidence=0.7,
                impact_score=0.6,
                risk_score=0.4,
                auto_apply=False,
                configuration_changes={"templates": low_performing_prompts},
                rollback_plan={"restore_original_templates": True},
                validation_checks=["verify_prompt_performance_improvement"]
            )
            recommendations.append(action)
        
        return recommendations
    
    def _calculate_overall_health_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        scores = []
        
        # Memory health score
        memory_score = analysis_results.get("memory_health", {}).get("overall_score", 0.5)
        scores.append(memory_score * 0.3)
        
        # Performance score (derived from trends)
        performance_analysis = analysis_results.get("performance_analysis", {})
        perf_score = 0.8 if performance_analysis.get("trend_direction") == "stable" else 0.6
        scores.append(perf_score * 0.4)
        
        # Prompt health score
        prompt_analysis = analysis_results.get("prompt_analysis", {})
        if prompt_analysis:
            prompt_scores = [
                analysis.get("metrics", {}).get("success_rate", 0.5)
                for analysis in prompt_analysis.values()
            ]
            avg_prompt_score = sum(prompt_scores) / len(prompt_scores) if prompt_scores else 0.5
            scores.append(avg_prompt_score * 0.3)
        else:
            scores.append(0.5 * 0.3)
        
        return sum(scores)
    
    async def _assess_system_risks(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess system risks based on analysis."""
        risks = {
            "high_risk_areas": [],
            "medium_risk_areas": [],
            "low_risk_areas": [],
            "overall_risk_level": "low"
        }
        
        # Check memory health risks
        memory_health = analysis_results.get("memory_health", {})
        if memory_health.get("overall_score", 1.0) < 0.6:
            risks["high_risk_areas"].append("Memory system health critically low")
        
        # Check performance risks
        performance_analysis = analysis_results.get("performance_analysis", {})
        if performance_analysis.get("trend_direction") == "declining":
            risks["medium_risk_areas"].append("Performance trending downward")
        
        # Determine overall risk level
        if risks["high_risk_areas"]:
            risks["overall_risk_level"] = "high"
        elif risks["medium_risk_areas"]:
            risks["overall_risk_level"] = "medium"
        
        return risks
    
    def _calculate_next_run(self, schedule_pattern: str) -> datetime:
        """Calculate next run time from cron-like pattern."""
        # Simplified implementation - would use proper cron parsing
        # For now, just add 24 hours for daily jobs
        if "* * *" in schedule_pattern:  # Daily pattern
            return datetime.utcnow() + timedelta(days=1)
        elif "* * 0" in schedule_pattern:  # Weekly pattern
            return datetime.utcnow() + timedelta(days=7)
        elif "1 * *" in schedule_pattern:  # Monthly pattern
            return datetime.utcnow() + timedelta(days=30)
        else:  # Default to 6 hours
            return datetime.utcnow() + timedelta(hours=6)
    
    def _validate_job(self, job: SelfTrainingJob):
        """Validate job configuration."""
        if not job.job_id:
            raise ValueError("Job ID is required")
        if not job.job_type:
            raise ValueError("Job type is required")
        if not job.target_component:
            raise ValueError("Target component is required")
    
    # Placeholder methods for database operations
    
    async def _store_job(self, job: SelfTrainingJob):
        """Store job in database."""
        pass
    
    async def _store_analysis_results(self, results: Dict[str, Any]):
        """Store analysis results in database."""
        pass
    
    async def _store_improvement_action(self, action: ImprovementAction):
        """Store improvement action in database."""
        pass
    
    async def _trigger_emergency_improvements(self, analysis_results: Dict[str, Any]):
        """Trigger emergency improvements for critical issues."""
        pass
    
    async def _validate_action_preconditions(self, action: ImprovementAction) -> Dict[str, Any]:
        """Validate preconditions for an action."""
        return {"all_checks_passed": True}
    
    async def _create_rollback_checkpoint(self, action: ImprovementAction) -> Dict[str, Any]:
        """Create rollback checkpoint before applying action."""
        return {"checkpoint_id": f"checkpoint_{int(datetime.utcnow().timestamp())}"}
    
    async def _execute_memory_health_action(self, action: ImprovementAction) -> List[str]:
        """Execute memory health improvement action."""
        return ["memory_cleanup_executed"]
    
    async def _execute_configuration_action(self, action: ImprovementAction) -> List[str]:
        """Execute configuration improvement action."""
        return ["configuration_updated"]
    
    async def _execute_prompt_action(self, action: ImprovementAction) -> List[str]:
        """Execute prompt improvement action."""
        return ["prompts_updated"]
    
    async def _execute_performance_action(self, action: ImprovementAction) -> List[str]:
        """Execute performance improvement action."""
        return ["performance_settings_updated"]
    
    async def _validate_action_results(self, action: ImprovementAction, changes: List[str]) -> Dict[str, Any]:
        """Validate results of an applied action."""
        return {"success": True}
    
    async def _rollback_action(self, action: ImprovementAction, rollback_info: Dict[str, Any]):
        """Rollback an applied action."""
        pass
    
    def _assess_action_risks(self, action: ImprovementAction) -> List[str]:
        """Assess risks for an action."""
        return ["low_risk"]
    
    async def _generate_optimization_experiments(self, components: List[str]) -> Dict[str, Any]:
        """Generate optimization experiment configurations."""
        return {"baseline": {}, "variations": {}}
    
    async def _schedule_experiment_monitoring(self, experiment_id: str):
        """Schedule monitoring for an experiment."""
        pass
    
    async def _calculate_current_health_score(self) -> float:
        """Calculate current system health score."""
        return 0.85  # Placeholder
    
    async def _analyze_improvement_trends(self) -> Dict[str, Any]:
        """Analyze trends in system improvements."""
        return {"trend": "improving"}
    
    async def _check_experiment_status(self):
        """Check status of active experiments."""
        pass
    
    async def _apply_ready_improvements(self):
        """Apply improvements that are ready for auto-application."""
        for action in self.improvement_actions.values():
            if (not action.applied and action.auto_apply and 
                action.risk_score < 0.3 and action.confidence > 0.8):
                try:
                    await self.execute_improvement_action(action.action_id)
                except Exception as e:
                    logger.error(f"Failed to auto-apply improvement {action.action_id}: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the self-training scheduler."""
        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "scheduler_running": self._scheduler_running,
            "active_jobs": len([j for j in self.scheduled_jobs.values() if j.status == JobStatus.RUNNING]),
            "pending_actions": len([a for a in self.improvement_actions.values() if not a.applied]),
            "active_experiments": len(self.active_experiments)
        }