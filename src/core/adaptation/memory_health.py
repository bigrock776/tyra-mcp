"""
Memory Health Management System.

Monitors memory system health, detects stale/redundant memories,
and implements automated cleanup and optimization routines.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MemoryHealthStatus(Enum):
    """Memory health status levels."""
    
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 80-89%
    FAIR = "fair"           # 70-79%
    POOR = "poor"           # 60-69%
    CRITICAL = "critical"   # <60%


@dataclass
class MemoryHealthMetric:
    """Single memory health measurement."""
    
    memory_id: str
    agent_id: str
    content_hash: str
    last_accessed: datetime
    access_count: int
    confidence_score: float
    staleness_score: float
    redundancy_score: float
    utility_score: float
    health_status: MemoryHealthStatus
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanupAction:
    """Memory cleanup action specification."""
    
    action_type: str  # "delete", "merge", "archive", "update"
    target_memories: List[str]
    reason: str
    impact_score: float
    safety_score: float
    estimated_savings: Dict[str, Any]
    requires_approval: bool = False


class MemoryHealthManager:
    """Manages memory system health and automated cleanup."""
    
    def __init__(self):
        self.db_client = None
        self.embedder = None
        self.cache = None
        self.graph_engine = None
        self._health_thresholds = self._get_default_thresholds()
        self._cleanup_schedules = {}
        self._last_analysis = None
        self._initialized = False
        
    async def initialize(self, config: Dict[str, Any]):
        """Initialize memory health manager."""
        try:
            # Initialize components (would be injected in real implementation)
            self._health_thresholds.update(config.get("health_thresholds", {}))
            self._cleanup_schedules = config.get("cleanup_schedules", {})
            
            self._initialized = True
            logger.info("Memory health manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory health manager: {e}")
            raise
    
    async def analyze_memory_health(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze overall memory system health."""
        try:
            # Get all memories for analysis
            memories = await self._get_memories_for_analysis(agent_id)
            
            health_metrics = []
            total_score = 0
            
            for memory in memories:
                metric = await self._analyze_single_memory(memory)
                health_metrics.append(metric)
                total_score += self._calculate_weighted_score(metric)
            
            overall_score = total_score / len(memories) if memories else 0
            health_status = self._determine_health_status(overall_score)
            
            # Analyze patterns
            patterns = await self._analyze_health_patterns(health_metrics)
            
            # Generate recommendations
            recommendations = await self._generate_health_recommendations(health_metrics, patterns)
            
            return {
                "overall_score": overall_score,
                "health_status": health_status.value,
                "total_memories": len(memories),
                "healthy_memories": len([m for m in health_metrics if m.health_status.value in ["excellent", "good"]]),
                "problematic_memories": len([m for m in health_metrics if m.health_status.value in ["poor", "critical"]]),
                "patterns": patterns,
                "recommendations": recommendations,
                "metrics": health_metrics,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Memory health analysis failed: {e}")
            raise
    
    async def detect_stale_memories(self, max_age_days: int = 30, min_access_count: int = 1) -> List[Dict[str, Any]]:
        """Detect stale memories that haven't been accessed recently."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            
            stale_memories = await self.db_client.query("""
                SELECT memory_id, agent_id, content, last_accessed, access_count, metadata
                FROM memories 
                WHERE last_accessed < %s AND access_count < %s
                ORDER BY last_accessed ASC
            """, (cutoff_date, min_access_count))
            
            # Analyze staleness factors
            analyzed_stale = []
            for memory in stale_memories:
                staleness_analysis = await self._analyze_staleness(memory)
                analyzed_stale.append({
                    **memory,
                    "staleness_score": staleness_analysis["score"],
                    "staleness_factors": staleness_analysis["factors"],
                    "recommended_action": staleness_analysis["action"]
                })
            
            return analyzed_stale
            
        except Exception as e:
            logger.error(f"Stale memory detection failed: {e}")
            raise
    
    async def detect_redundant_memories(self, similarity_threshold: float = 0.95) -> List[Dict[str, Any]]:
        """Detect redundant memories with high content similarity."""
        try:
            # Get all memories with embeddings
            memories_with_embeddings = await self.db_client.query("""
                SELECT memory_id, agent_id, content, embedding, confidence_score, metadata
                FROM memories 
                WHERE embedding IS NOT NULL
                ORDER BY created_at DESC
            """)
            
            redundant_groups = []
            processed_ids = set()
            
            for i, memory1 in enumerate(memories_with_embeddings):
                if memory1["memory_id"] in processed_ids:
                    continue
                
                similar_memories = [memory1]
                
                for j, memory2 in enumerate(memories_with_embeddings[i+1:], i+1):
                    if memory2["memory_id"] in processed_ids:
                        continue
                    
                    # Calculate similarity
                    similarity = self._calculate_similarity(
                        memory1["embedding"], 
                        memory2["embedding"]
                    )
                    
                    if similarity >= similarity_threshold:
                        similar_memories.append(memory2)
                        processed_ids.add(memory2["memory_id"])
                
                if len(similar_memories) > 1:
                    # Analyze redundancy group
                    redundancy_analysis = await self._analyze_redundancy_group(similar_memories)
                    redundant_groups.append(redundancy_analysis)
                    
                    for memory in similar_memories:
                        processed_ids.add(memory["memory_id"])
            
            return redundant_groups
            
        except Exception as e:
            logger.error(f"Redundant memory detection failed: {e}")
            raise
    
    async def identify_low_confidence_memories(self, confidence_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Identify memories with consistently low confidence scores."""
        try:
            low_confidence_memories = await self.db_client.query("""
                SELECT memory_id, agent_id, content, confidence_score, 
                       hallucination_score, access_count, last_accessed, metadata
                FROM memories 
                WHERE confidence_score < %s OR hallucination_score > %s
                ORDER BY confidence_score ASC, hallucination_score DESC
            """, (confidence_threshold, 1 - confidence_threshold))
            
            # Analyze low confidence factors
            analyzed_memories = []
            for memory in low_confidence_memories:
                confidence_analysis = await self._analyze_confidence_issues(memory)
                analyzed_memories.append({
                    **memory,
                    "confidence_issues": confidence_analysis["issues"],
                    "improvability_score": confidence_analysis["improvability"],
                    "recommended_action": confidence_analysis["action"]
                })
            
            return analyzed_memories
            
        except Exception as e:
            logger.error(f"Low confidence memory identification failed: {e}")
            raise
    
    async def plan_automated_cleanup(self) -> List[CleanupAction]:
        """Plan automated cleanup actions based on health analysis."""
        try:
            cleanup_actions = []
            
            # Analyze stale memories
            stale_memories = await self.detect_stale_memories()
            for memory in stale_memories:
                if memory["staleness_score"] > 0.8:
                    action = CleanupAction(
                        action_type="delete" if memory["staleness_score"] > 0.9 else "archive",
                        target_memories=[memory["memory_id"]],
                        reason=f"Stale memory (score: {memory['staleness_score']:.2f})",
                        impact_score=memory["staleness_score"],
                        safety_score=1.0 - memory.get("access_count", 0) / 100,
                        estimated_savings={"storage_mb": 0.1, "query_time_ms": 2},
                        requires_approval=memory["staleness_score"] < 0.95
                    )
                    cleanup_actions.append(action)
            
            # Analyze redundant memories
            redundant_groups = await self.detect_redundant_memories()
            for group in redundant_groups:
                if len(group["memories"]) > 1:
                    # Keep the best memory, remove others
                    best_memory = max(group["memories"], key=lambda m: m.get("confidence_score", 0))
                    targets = [m["memory_id"] for m in group["memories"] if m["memory_id"] != best_memory["memory_id"]]
                    
                    action = CleanupAction(
                        action_type="merge",
                        target_memories=targets,
                        reason=f"Redundant memories (similarity: {group['avg_similarity']:.2f})",
                        impact_score=group["redundancy_score"],
                        safety_score=0.9,  # High safety for merging
                        estimated_savings={"storage_mb": len(targets) * 0.1, "query_time_ms": len(targets) * 3},
                        requires_approval=True
                    )
                    cleanup_actions.append(action)
            
            # Analyze low confidence memories
            low_confidence = await self.identify_low_confidence_memories()
            for memory in low_confidence:
                if memory["confidence_score"] < 0.4:
                    action = CleanupAction(
                        action_type="delete",
                        target_memories=[memory["memory_id"]],
                        reason=f"Very low confidence (score: {memory['confidence_score']:.2f})",
                        impact_score=1.0 - memory["confidence_score"],
                        safety_score=0.8,
                        estimated_savings={"storage_mb": 0.1, "accuracy_improvement": 0.02},
                        requires_approval=memory.get("access_count", 0) > 5
                    )
                    cleanup_actions.append(action)
            
            # Sort by impact and safety
            cleanup_actions.sort(
                key=lambda a: (a.impact_score * a.safety_score, -len(a.target_memories)),
                reverse=True
            )
            
            return cleanup_actions
            
        except Exception as e:
            logger.error(f"Cleanup planning failed: {e}")
            raise
    
    async def execute_automated_cleanup(self, dry_run: bool = True) -> Dict[str, Any]:
        """Execute automated cleanup actions."""
        try:
            cleanup_actions = await self.plan_automated_cleanup()
            
            if dry_run:
                return {
                    "dry_run": True,
                    "planned_actions": len(cleanup_actions),
                    "actions": [
                        {
                            "type": action.action_type,
                            "targets": len(action.target_memories),
                            "reason": action.reason,
                            "impact": action.impact_score,
                            "safety": action.safety_score,
                            "requires_approval": action.requires_approval
                        }
                        for action in cleanup_actions
                    ],
                    "estimated_savings": self._calculate_total_savings(cleanup_actions)
                }
            
            # Execute actions that don't require approval
            executed_actions = []
            skipped_actions = []
            
            for action in cleanup_actions:
                if not action.requires_approval:
                    try:
                        result = await self._execute_cleanup_action(action)
                        executed_actions.append({
                            "action": action,
                            "result": result,
                            "status": "completed"
                        })
                    except Exception as e:
                        executed_actions.append({
                            "action": action,
                            "error": str(e),
                            "status": "failed"
                        })
                else:
                    skipped_actions.append(action)
            
            return {
                "dry_run": False,
                "executed_actions": len(executed_actions),
                "skipped_actions": len(skipped_actions),
                "successful_executions": len([a for a in executed_actions if a["status"] == "completed"]),
                "failed_executions": len([a for a in executed_actions if a["status"] == "failed"]),
                "actions_requiring_approval": skipped_actions,
                "results": executed_actions
            }
            
        except Exception as e:
            logger.error(f"Automated cleanup execution failed: {e}")
            raise
    
    async def schedule_health_maintenance(self):
        """Schedule regular health maintenance tasks."""
        try:
            # Daily stale memory detection
            asyncio.create_task(self._schedule_recurring_task(
                self._daily_stale_cleanup,
                interval_hours=24,
                task_name="daily_stale_cleanup"
            ))
            
            # Weekly redundancy detection
            asyncio.create_task(self._schedule_recurring_task(
                self._weekly_redundancy_cleanup,
                interval_hours=168,  # 7 days
                task_name="weekly_redundancy_cleanup"
            ))
            
            # Monthly comprehensive health analysis
            asyncio.create_task(self._schedule_recurring_task(
                self._monthly_health_analysis,
                interval_hours=720,  # 30 days
                task_name="monthly_health_analysis"
            ))
            
            logger.info("Health maintenance tasks scheduled")
            
        except Exception as e:
            logger.error(f"Failed to schedule health maintenance: {e}")
            raise
    
    async def calculate_health_score(self) -> float:
        """Calculate overall system health score."""
        try:
            health_analysis = await self.analyze_memory_health()
            return health_analysis["overall_score"]
        except Exception as e:
            logger.error(f"Health score calculation failed: {e}")
            return 0.0
    
    # Private helper methods
    
    def _get_default_thresholds(self) -> Dict[str, Any]:
        """Get default health analysis thresholds."""
        return {
            "staleness_days": 30,
            "min_access_count": 1,
            "redundancy_similarity": 0.95,
            "low_confidence_threshold": 0.6,
            "critical_confidence_threshold": 0.4
        }
    
    async def _get_memories_for_analysis(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get memories for health analysis."""
        query = """
            SELECT memory_id, agent_id, content, confidence_score, 
                   hallucination_score, access_count, last_accessed, 
                   created_at, embedding, metadata
            FROM memories
        """
        params = []
        
        if agent_id:
            query += " WHERE agent_id = %s"
            params.append(agent_id)
        
        query += " ORDER BY created_at DESC"
        
        return await self.db_client.query(query, params)
    
    async def _analyze_single_memory(self, memory: Dict[str, Any]) -> MemoryHealthMetric:
        """Analyze health of a single memory."""
        # Calculate staleness score
        days_since_access = (datetime.utcnow() - memory["last_accessed"]).days
        staleness_score = min(days_since_access / 30.0, 1.0)
        
        # Calculate utility score based on access patterns
        utility_score = min(memory["access_count"] / 10.0, 1.0)
        
        # Use existing confidence score
        confidence_score = memory.get("confidence_score", 0.5)
        
        # Calculate redundancy score (would require similarity analysis)
        redundancy_score = 0.0  # Placeholder
        
        # Calculate overall health score
        health_score = (
            (1 - staleness_score) * 0.3 +
            utility_score * 0.3 +
            confidence_score * 0.3 +
            (1 - redundancy_score) * 0.1
        )
        
        health_status = self._determine_health_status(health_score)
        
        # Generate recommendations
        recommendations = []
        if staleness_score > 0.8:
            recommendations.append("Consider archiving or deleting due to staleness")
        if confidence_score < 0.6:
            recommendations.append("Review and potentially improve confidence")
        if utility_score < 0.2:
            recommendations.append("Low utility - consider removal")
        
        return MemoryHealthMetric(
            memory_id=memory["memory_id"],
            agent_id=memory["agent_id"],
            content_hash=str(hash(memory["content"])),
            last_accessed=memory["last_accessed"],
            access_count=memory["access_count"],
            confidence_score=confidence_score,
            staleness_score=staleness_score,
            redundancy_score=redundancy_score,
            utility_score=utility_score,
            health_status=health_status,
            recommendations=recommendations
        )
    
    def _calculate_weighted_score(self, metric: MemoryHealthMetric) -> float:
        """Calculate weighted health score for a metric."""
        return (
            (1 - metric.staleness_score) * 0.3 +
            metric.utility_score * 0.3 +
            metric.confidence_score * 0.3 +
            (1 - metric.redundancy_score) * 0.1
        )
    
    def _determine_health_status(self, score: float) -> MemoryHealthStatus:
        """Determine health status from score."""
        if score >= 0.9:
            return MemoryHealthStatus.EXCELLENT
        elif score >= 0.8:
            return MemoryHealthStatus.GOOD
        elif score >= 0.7:
            return MemoryHealthStatus.FAIR
        elif score >= 0.6:
            return MemoryHealthStatus.POOR
        else:
            return MemoryHealthStatus.CRITICAL
    
    async def _analyze_health_patterns(self, metrics: List[MemoryHealthMetric]) -> Dict[str, Any]:
        """Analyze patterns in health metrics."""
        patterns = {
            "staleness_trend": "stable",
            "confidence_distribution": {},
            "agent_health_scores": {},
            "temporal_patterns": {}
        }
        
        # Analyze by agent
        agent_scores = defaultdict(list)
        for metric in metrics:
            agent_scores[metric.agent_id].append(self._calculate_weighted_score(metric))
        
        patterns["agent_health_scores"] = {
            agent: sum(scores) / len(scores)
            for agent, scores in agent_scores.items()
        }
        
        return patterns
    
    async def _generate_health_recommendations(self, metrics: List[MemoryHealthMetric], patterns: Dict[str, Any]) -> List[str]:
        """Generate system-wide health recommendations."""
        recommendations = []
        
        # Analyze problematic memories
        problematic = [m for m in metrics if m.health_status.value in ["poor", "critical"]]
        if len(problematic) > len(metrics) * 0.2:
            recommendations.append("High number of problematic memories - consider comprehensive cleanup")
        
        # Analyze staleness
        stale_count = len([m for m in metrics if m.staleness_score > 0.7])
        if stale_count > len(metrics) * 0.3:
            recommendations.append("Many stale memories detected - implement more aggressive cleanup")
        
        # Analyze confidence
        low_confidence = len([m for m in metrics if m.confidence_score < 0.6])
        if low_confidence > len(metrics) * 0.15:
            recommendations.append("Many low-confidence memories - review quality control processes")
        
        return recommendations
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _analyze_staleness(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze staleness factors for a memory."""
        days_since_access = (datetime.utcnow() - memory["last_accessed"]).days
        
        factors = []
        if days_since_access > 60:
            factors.append("Not accessed in over 60 days")
        if memory["access_count"] < 2:
            factors.append("Very low access count")
        
        score = min(days_since_access / 30.0, 1.0)
        
        if score > 0.9:
            action = "delete"
        elif score > 0.7:
            action = "archive"
        else:
            action = "monitor"
        
        return {
            "score": score,
            "factors": factors,
            "action": action
        }
    
    async def _analyze_redundancy_group(self, similar_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a group of similar memories."""
        avg_similarity = 0.95  # Placeholder - would calculate actual average
        
        return {
            "memories": similar_memories,
            "avg_similarity": avg_similarity,
            "redundancy_score": avg_similarity,
            "recommended_action": "merge" if len(similar_memories) > 2 else "review"
        }
    
    async def _analyze_confidence_issues(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze confidence issues for a memory."""
        issues = []
        if memory["confidence_score"] < 0.4:
            issues.append("Very low confidence score")
        if memory.get("hallucination_score", 0) > 0.6:
            issues.append("High hallucination risk")
        
        improvability = 0.5  # Placeholder for improvability analysis
        
        action = "delete" if memory["confidence_score"] < 0.3 else "review"
        
        return {
            "issues": issues,
            "improvability": improvability,
            "action": action
        }
    
    def _calculate_total_savings(self, actions: List[CleanupAction]) -> Dict[str, Any]:
        """Calculate total estimated savings from cleanup actions."""
        total_savings = defaultdict(float)
        
        for action in actions:
            for key, value in action.estimated_savings.items():
                total_savings[key] += value
        
        return dict(total_savings)
    
    async def _execute_cleanup_action(self, action: CleanupAction) -> Dict[str, Any]:
        """Execute a single cleanup action."""
        if action.action_type == "delete":
            # Delete memories
            for memory_id in action.target_memories:
                await self.db_client.execute(
                    "DELETE FROM memories WHERE memory_id = %s",
                    (memory_id,)
                )
            return {"deleted_count": len(action.target_memories)}
        
        elif action.action_type == "archive":
            # Archive memories
            for memory_id in action.target_memories:
                await self.db_client.execute(
                    "UPDATE memories SET archived = TRUE WHERE memory_id = %s",
                    (memory_id,)
                )
            return {"archived_count": len(action.target_memories)}
        
        elif action.action_type == "merge":
            # Merge similar memories (implementation would be more complex)
            return {"merged_count": len(action.target_memories)}
        
        return {"status": "unknown_action"}
    
    async def _schedule_recurring_task(self, task_func, interval_hours: int, task_name: str):
        """Schedule a recurring maintenance task."""
        while True:
            try:
                await asyncio.sleep(interval_hours * 3600)  # Convert to seconds
                await task_func()
                logger.info(f"Completed scheduled task: {task_name}")
            except Exception as e:
                logger.error(f"Scheduled task {task_name} failed: {e}")
    
    async def _daily_stale_cleanup(self):
        """Daily automated stale memory cleanup."""
        stale_memories = await self.detect_stale_memories(max_age_days=7, min_access_count=1)
        
        # Auto-delete very stale memories
        very_stale = [m for m in stale_memories if m["staleness_score"] > 0.95]
        for memory in very_stale:
            await self.db_client.execute(
                "DELETE FROM memories WHERE memory_id = %s",
                (memory["memory_id"],)
            )
        
        logger.info(f"Daily cleanup: removed {len(very_stale)} very stale memories")
    
    async def _weekly_redundancy_cleanup(self):
        """Weekly automated redundancy cleanup."""
        redundant_groups = await self.detect_redundant_memories()
        
        # Auto-merge highly redundant memories
        for group in redundant_groups:
            if group["avg_similarity"] > 0.98 and len(group["memories"]) > 2:
                # Keep the best memory, mark others for removal
                best_memory = max(group["memories"], key=lambda m: m.get("confidence_score", 0))
                targets = [m["memory_id"] for m in group["memories"] if m["memory_id"] != best_memory["memory_id"]]
                
                for target_id in targets:
                    await self.db_client.execute(
                        "DELETE FROM memories WHERE memory_id = %s",
                        (target_id,)
                    )
        
        logger.info(f"Weekly cleanup: processed {len(redundant_groups)} redundant groups")
    
    async def _monthly_health_analysis(self):
        """Monthly comprehensive health analysis."""
        health_report = await self.analyze_memory_health()
        
        # Store health report for trending
        await self.db_client.execute("""
            INSERT INTO memory_health_reports (report_date, overall_score, total_memories, recommendations)
            VALUES (%s, %s, %s, %s)
        """, (
            datetime.utcnow().date(),
            health_report["overall_score"],
            health_report["total_memories"],
            str(health_report["recommendations"])
        ))
        
        logger.info(f"Monthly health analysis completed. Overall score: {health_report['overall_score']:.2f}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the memory health manager."""
        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "last_analysis": self._last_analysis,
            "scheduled_tasks": len(self._cleanup_schedules),
            "thresholds": self._health_thresholds
        }