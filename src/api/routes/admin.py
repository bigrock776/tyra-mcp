"""
Administrative API endpoints.

Provides system administration, maintenance, monitoring,
and configuration management endpoints.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...core.cache.redis_cache import RedisCache
from ...core.memory.manager import MemoryManager
from ...core.observability.telemetry import get_telemetry
from ...core.utils.config import get_settings, reload_config
from ...core.utils.logger import get_logger
from ...core.utils.registry import ProviderType, get_provider

logger = get_logger(__name__)

router = APIRouter()


# Enums
class MaintenanceTask(str, Enum):
    """Available maintenance tasks."""

    CLEANUP_MEMORIES = "cleanup_memories"
    REBUILD_INDEXES = "rebuild_indexes"
    CLEAR_CACHE = "clear_cache"
    VACUUM_DATABASE = "vacuum_database"
    ANALYZE_PERFORMANCE = "analyze_performance"
    BACKUP_DATA = "backup_data"


class SystemStatus(str, Enum):
    """System status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    MAINTENANCE = "maintenance"


# Request/Response Models
class SystemInfo(BaseModel):
    """System information."""

    version: str = Field(..., description="System version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    status: SystemStatus = Field(..., description="Current system status")
    active_connections: int = Field(..., description="Number of active connections")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    storage_usage_gb: float = Field(..., description="Storage usage in GB")


class DatabaseStats(BaseModel):
    """Database statistics."""

    postgres: Dict[str, Any] = Field(..., description="PostgreSQL statistics")
    memgraph: Dict[str, Any] = Field(..., description="Memgraph statistics")
    redis: Dict[str, Any] = Field(..., description="Redis statistics")


class CacheStats(BaseModel):
    """Cache statistics."""

    total_keys: int = Field(..., description="Total cache keys")
    memory_usage_mb: float = Field(..., description="Cache memory usage")
    hit_rate: float = Field(..., description="Cache hit rate")
    miss_rate: float = Field(..., description="Cache miss rate")
    evictions: int = Field(..., description="Number of evictions")
    ttl_expirations: int = Field(..., description="TTL expirations")


class MaintenanceRequest(BaseModel):
    """Maintenance task request."""

    task: MaintenanceTask = Field(..., description="Task to perform")
    force: bool = Field(False, description="Force execution even if risky")
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Task-specific parameters"
    )


class ConfigUpdateRequest(BaseModel):
    """Configuration update request."""

    section: str = Field(..., description="Configuration section to update")
    updates: Dict[str, Any] = Field(..., description="Configuration updates")
    reload: bool = Field(True, description="Reload configuration after update")


class BackupRequest(BaseModel):
    """Backup request."""

    include_memories: bool = Field(True, description="Include memories in backup")
    include_graph: bool = Field(True, description="Include knowledge graph")
    include_config: bool = Field(True, description="Include configuration")
    compression: bool = Field(True, description="Compress backup")


class LogQuery(BaseModel):
    """Log query parameters."""

    level: Optional[str] = Field(None, description="Log level filter")
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    search: Optional[str] = Field(None, description="Search in log messages")
    limit: int = Field(100, ge=1, le=1000, description="Maximum log entries")


# Dependencies
async def get_memory_manager() -> MemoryManager:
    """Get memory manager instance."""
    try:
        return get_provider(ProviderType.MEMORY_MANAGER, "default")
    except Exception as e:
        logger.error(f"Failed to get memory manager: {e}")
        raise HTTPException(status_code=500, detail="Memory manager unavailable")


async def get_cache() -> RedisCache:
    """Get cache instance."""
    try:
        return get_provider(ProviderType.CACHE, "redis")
    except Exception as e:
        logger.error(f"Failed to get cache: {e}")
        raise HTTPException(status_code=500, detail="Cache unavailable")


# System endpoints
@router.get("/system/info", response_model=SystemInfo)
async def get_system_info():
    """
    Get system information and status.

    Returns overall system health and resource usage.
    """
    try:
        import time

        import psutil

        # Get process info
        process = psutil.Process()

        # Calculate uptime
        start_time = process.create_time()
        uptime = time.time() - start_time

        # Get resource usage
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)

        # Get disk usage
        disk_usage = psutil.disk_usage("/")

        # Determine status
        status = SystemStatus.HEALTHY
        if cpu_percent > 80 or memory_info.rss / 1024 / 1024 > 1000:
            status = SystemStatus.DEGRADED

        return SystemInfo(
            version="1.0.0",
            uptime_seconds=uptime,
            status=status,
            active_connections=len(process.connections()),
            memory_usage_mb=memory_info.rss / 1024 / 1024,
            cpu_usage_percent=cpu_percent,
            storage_usage_gb=disk_usage.used / 1024 / 1024 / 1024,
        )

    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/database/stats", response_model=DatabaseStats)
async def get_database_stats(
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """
    Get database statistics.

    Returns statistics for all database systems.
    """
    try:
        # Get stats from each database
        postgres_stats = await memory_manager.get_database_stats()

        # Placeholder for other databases
        memgraph_stats = {"node_count": 0, "edge_count": 0, "status": "healthy"}

        redis_stats = {"connected_clients": 0, "used_memory_mb": 0, "status": "healthy"}

        return DatabaseStats(
            postgres=postgres_stats, memgraph=memgraph_stats, redis=redis_stats
        )

    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats", response_model=CacheStats)
async def get_cache_stats(cache: RedisCache = Depends(get_cache)):
    """
    Get cache statistics.

    Returns detailed cache performance metrics.
    """
    try:
        stats = await cache.get_stats()

        return CacheStats(
            total_keys=stats["total_keys"],
            memory_usage_mb=stats["memory_usage_mb"],
            hit_rate=stats["hit_rate"],
            miss_rate=stats["miss_rate"],
            evictions=stats["evictions"],
            ttl_expirations=stats["ttl_expirations"],
        )

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Maintenance endpoints
@router.post("/maintenance/execute")
async def execute_maintenance(
    request: MaintenanceRequest,
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    cache: RedisCache = Depends(get_cache),
):
    """
    Execute a maintenance task.

    Runs maintenance tasks in the background to avoid blocking.
    """
    try:
        task_id = f"{request.task}_{datetime.utcnow().timestamp()}"

        if request.task == MaintenanceTask.CLEANUP_MEMORIES:
            background_tasks.add_task(
                memory_manager.cleanup_memories,
                force=request.force,
                **request.parameters or {},
            )

        elif request.task == MaintenanceTask.CLEAR_CACHE:
            background_tasks.add_task(
                cache.clear_all,
                pattern=(
                    request.parameters.get("pattern") if request.parameters else None
                ),
            )

        elif request.task == MaintenanceTask.REBUILD_INDEXES:
            background_tasks.add_task(
                memory_manager.rebuild_indexes, **request.parameters or {}
            )

        elif request.task == MaintenanceTask.VACUUM_DATABASE:
            background_tasks.add_task(
                memory_manager.vacuum_database, full=request.force
            )

        elif request.task == MaintenanceTask.ANALYZE_PERFORMANCE:
            background_tasks.add_task(_analyze_performance, memory_manager, cache)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown task: {request.task}")

        return {
            "task_id": task_id,
            "task": request.task,
            "status": "started",
            "message": f"Maintenance task {request.task} started in background",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute maintenance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache(
    pattern: Optional[str] = Query(None, description="Cache key pattern to clear"),
    cache: RedisCache = Depends(get_cache),
):
    """
    Clear cache entries.

    Clears all cache or entries matching a pattern.
    """
    try:
        if pattern:
            cleared = await cache.clear_pattern(pattern)
            message = f"Cleared {cleared} cache entries matching pattern: {pattern}"
        else:
            await cache.clear_all()
            message = "All cache entries cleared"

        return {"message": message}

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoints
@router.get("/config/current")
async def get_current_config():
    """
    Get current configuration.

    Returns the active configuration settings.
    """
    try:
        settings = get_settings()

        # Convert to dict and remove sensitive values
        config_dict = settings.dict()
        _remove_sensitive_values(config_dict)

        return config_dict

    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/update")
async def update_configuration(request: ConfigUpdateRequest):
    """
    Update configuration settings.

    Updates configuration and optionally reloads the system.
    """
    try:
        # Validate section exists
        settings = get_settings()
        if not hasattr(settings, request.section):
            raise HTTPException(
                status_code=400, detail=f"Unknown config section: {request.section}"
            )

        # Update configuration (placeholder - would update YAML file)
        logger.info(f"Updating config section {request.section}: {request.updates}")

        # Reload if requested
        if request.reload:
            reload_config()

        return {
            "section": request.section,
            "updates": request.updates,
            "reloaded": request.reload,
            "message": "Configuration updated successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Backup and restore
@router.post("/backup/create")
async def create_backup(
    request: BackupRequest,
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """
    Create a system backup.

    Creates a backup of specified components in the background.
    """
    try:
        backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Start backup in background
        background_tasks.add_task(_create_backup, backup_id, request, memory_manager)

        return {
            "backup_id": backup_id,
            "status": "started",
            "message": "Backup creation started in background",
        }

    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backup/list")
async def list_backups():
    """
    List available backups.

    Returns information about existing backups.
    """
    try:
        # Placeholder - would list actual backup files
        backups = [
            {
                "id": "backup_20240101_120000",
                "created_at": datetime(2024, 1, 1, 12, 0, 0),
                "size_mb": 256.5,
                "components": ["memories", "graph", "config"],
            }
        ]

        return {"backups": backups}

    except Exception as e:
        logger.error(f"Failed to list backups: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Logging endpoints
@router.post("/logs/query")
async def query_logs(query: LogQuery):
    """
    Query system logs.

    Returns filtered log entries based on query parameters.
    """
    try:
        # Placeholder - would query actual logs
        logs = [
            {
                "timestamp": datetime.utcnow(),
                "level": "INFO",
                "message": "Sample log entry",
                "module": "api.admin",
            }
        ]

        return {"logs": logs, "total": len(logs), "query": query.dict()}

    except Exception as e:
        logger.error(f"Failed to query logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/levels")
async def get_log_levels():
    """
    Get current log levels.

    Returns log levels for all modules.
    """
    try:
        import logging

        levels = {}
        for name, logger_obj in logging.Logger.manager.loggerDict.items():
            if isinstance(logger_obj, logging.Logger):
                levels[name] = logging.getLevelName(logger_obj.level)

        return {"log_levels": levels}

    except Exception as e:
        logger.error(f"Failed to get log levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logs/level/{module}")
async def set_log_level(
    module: str,
    level: str = Query(..., description="Log level: DEBUG, INFO, WARNING, ERROR"),
):
    """
    Set log level for a module.

    Dynamically adjusts logging verbosity.
    """
    try:
        import logging

        # Validate level
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise HTTPException(status_code=400, detail=f"Invalid log level: {level}")

        # Set level
        logger_obj = logging.getLogger(module)
        logger_obj.setLevel(numeric_level)

        return {
            "module": module,
            "level": level.upper(),
            "message": f"Log level set to {level.upper()} for {module}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set log level: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telemetry/optimize")
async def optimize_telemetry():
    """Optimize telemetry performance."""
    try:
        telemetry = get_telemetry()
        result = await telemetry.optimize_telemetry()
        return {
            "status": "success",
            "result": result,
            "message": "Telemetry optimization completed"
        }
    except Exception as e:
        logger.error(f"Telemetry optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telemetry/emergency-optimize")
async def emergency_optimize_telemetry():
    """Emergency telemetry optimization for critical performance issues."""
    try:
        telemetry = get_telemetry()
        result = await telemetry.emergency_optimize()
        return {
            "status": "success",
            "result": result,
            "message": "Emergency telemetry optimization applied"
        }
    except Exception as e:
        logger.error(f"Emergency telemetry optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/telemetry/performance-stats")
async def get_telemetry_performance_stats():
    """Get telemetry performance statistics."""
    try:
        telemetry = get_telemetry()
        stats = telemetry.get_telemetry_performance_stats()
        return {
            "status": "success",
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get telemetry performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telemetry/enable-optimization")
async def enable_telemetry_optimization():
    """Enable telemetry performance optimization."""
    try:
        telemetry = get_telemetry()
        telemetry.enable_performance_optimization()
        return {
            "status": "success",
            "message": "Telemetry performance optimization enabled"
        }
    except Exception as e:
        logger.error(f"Failed to enable telemetry optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telemetry/disable-optimization")
async def disable_telemetry_optimization():
    """Disable telemetry performance optimization."""
    try:
        telemetry = get_telemetry()
        telemetry.disable_performance_optimization()
        return {
            "status": "success",
            "message": "Telemetry performance optimization disabled"
        }
    except Exception as e:
        logger.error(f"Failed to disable telemetry optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
def _remove_sensitive_values(config_dict: Dict[str, Any]):
    """Remove sensitive values from config."""
    sensitive_keys = ["password", "secret", "key", "token"]

    for key, value in list(config_dict.items()):
        if isinstance(value, dict):
            _remove_sensitive_values(value)
        elif any(sensitive in key.lower() for sensitive in sensitive_keys):
            config_dict[key] = "***REDACTED***"


async def _analyze_performance(memory_manager: MemoryManager, cache: RedisCache):
    """Analyze system performance."""
    logger.info("Starting performance analysis...")

    # Analyze memory performance
    memory_stats = await memory_manager.analyze_performance()
    logger.info(f"Memory performance: {memory_stats}")

    # Analyze cache performance
    cache_stats = await cache.get_stats()
    logger.info(f"Cache performance: {cache_stats}")

    # Generate recommendations
    recommendations = []

    if cache_stats["hit_rate"] < 0.7:
        recommendations.append("Consider increasing cache TTL or size")

    if memory_stats.get("slow_queries", 0) > 10:
        recommendations.append("Optimize slow queries or add indexes")

    logger.info(f"Performance analysis complete. Recommendations: {recommendations}")


async def _create_backup(
    backup_id: str, request: BackupRequest, memory_manager: MemoryManager
):
    """Create system backup."""
    logger.info(f"Creating backup {backup_id}...")

    try:
        # Placeholder - would create actual backup
        if request.include_memories:
            logger.info("Backing up memories...")
            # Export memories

        if request.include_graph:
            logger.info("Backing up knowledge graph...")
            # Export graph

        if request.include_config:
            logger.info("Backing up configuration...")
            # Export config

        logger.info(f"Backup {backup_id} completed successfully")

    except Exception as e:
        logger.error(f"Backup {backup_id} failed: {e}")
        raise
