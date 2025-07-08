"""
Health check API endpoints.

Provides system health monitoring, readiness checks,
and component status verification.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...core.utils.logger import get_logger
from ..app import get_memory_manager

logger = get_logger(__name__)

router = APIRouter()


# Enums
class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentStatus(str, Enum):
    """Component status."""

    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"


# Response Models
class ComponentHealth(BaseModel):
    """Individual component health status."""

    name: str = Field(..., description="Component name")
    status: ComponentStatus = Field(..., description="Component status")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    details: Dict[str, Any] = Field(default={}, description="Additional details")
    last_check: datetime = Field(..., description="Last health check timestamp")


class HealthResponse(BaseModel):
    """Overall health check response."""

    status: HealthStatus = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="System version")
    uptime_seconds: float = Field(..., description="System uptime")
    components: List[ComponentHealth] = Field(
        ..., description="Component health statuses"
    )


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool = Field(..., description="Whether system is ready")
    timestamp: datetime = Field(..., description="Check timestamp")
    checks: Dict[str, bool] = Field(..., description="Individual readiness checks")


class LivenessResponse(BaseModel):
    """Liveness check response."""

    alive: bool = Field(..., description="Whether system is alive")
    timestamp: datetime = Field(..., description="Check timestamp")


@router.get("/health", response_model=HealthResponse)
async def health_check(memory_manager=Depends(get_memory_manager)):
    """
    Comprehensive health check.

    Checks all system components and returns detailed health status.
    """
    import time

    import psutil

    start_time = time.time()
    components = []
    unhealthy_count = 0
    degraded_count = 0

    try:
        # Get health from memory manager
        health_result = await memory_manager.health_check()

        # Convert memory manager health to component health format
        for component_name, component_data in health_result.get(
            "components", {}
        ).items():
            status = ComponentStatus.UP
            if not component_data.get("healthy", True):
                status = ComponentStatus.DOWN
                unhealthy_count += 1
            elif component_data.get("degraded", False):
                status = ComponentStatus.DEGRADED
                degraded_count += 1

            components.append(
                ComponentHealth(
                    name=component_name,
                    status=status,
                    latency_ms=component_data.get("response_time", 0) * 1000,
                    details=component_data.get("details", {}),
                    last_check=datetime.utcnow(),
                )
            )

        # Check system resources
        system_health = await _check_system_resources()
        components.append(system_health)
        if system_health.status == ComponentStatus.DOWN:
            unhealthy_count += 1
        elif system_health.status == ComponentStatus.DEGRADED:
            degraded_count += 1

        # Determine overall status
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        # Get process uptime
        process = psutil.Process()
        uptime = time.time() - process.create_time()

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            uptime_seconds=uptime,
            components=components,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status=HealthStatus.UNHEALTHY,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            uptime_seconds=0,
            components=[
                ComponentHealth(
                    name="memory_system",
                    status=ComponentStatus.DOWN,
                    latency_ms=0,
                    details={"error": str(e)},
                    last_check=datetime.utcnow(),
                )
            ],
        )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(memory_manager=Depends(get_memory_manager)):
    """
    Readiness check for load balancers.

    Returns whether the system is ready to handle requests.
    """
    try:
        # Use memory manager health check
        health_result = await memory_manager.health_check()

        checks = {}
        for component_name, component_data in health_result.get(
            "components", {}
        ).items():
            checks[component_name] = component_data.get("healthy", False)

        # System is ready if memory manager reports ready
        ready = health_result.get("status") == "healthy"

        return ReadinessResponse(
            ready=ready, timestamp=datetime.utcnow(), checks=checks
        )

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return ReadinessResponse(
            ready=False,
            timestamp=datetime.utcnow(),
            checks={"memory_system": False, "error": str(e)},
        )


@router.get("/live", response_model=LivenessResponse)
async def liveness_check():
    """
    Simple liveness check.

    Returns whether the application process is running.
    """
    return LivenessResponse(alive=True, timestamp=datetime.utcnow())


@router.get("/startup")
async def startup_check(memory_manager=Depends(get_memory_manager)):
    """
    Startup probe for Kubernetes.

    Used to know when application has started.
    """
    try:
        # Check if memory manager is initialized
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory system not initialized")

        # Basic health check
        health_result = await memory_manager.health_check()
        if health_result.get("status") != "healthy":
            raise HTTPException(status_code=503, detail="System not healthy")

        return {"status": "started", "timestamp": datetime.utcnow()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Startup check failed: {e}")
        raise HTTPException(status_code=503, detail="System not ready")


@router.get("/components/{component_name}")
async def check_component(
    component_name: str, memory_manager=Depends(get_memory_manager)
):
    """
    Check specific component health.

    Returns detailed health information for a single component.
    """
    try:
        component_name = component_name.lower()

        if component_name == "system":
            return await _check_system_resources()

        # Get component health from memory manager
        health_result = await memory_manager.health_check()
        components = health_result.get("components", {})

        if component_name not in components:
            raise HTTPException(
                status_code=404, detail=f"Unknown component: {component_name}"
            )

        component_data = components[component_name]
        status = ComponentStatus.UP
        if not component_data.get("healthy", True):
            status = ComponentStatus.DOWN
        elif component_data.get("degraded", False):
            status = ComponentStatus.DEGRADED

        return ComponentHealth(
            name=component_name,
            status=status,
            latency_ms=component_data.get("response_time", 0) * 1000,
            details=component_data.get("details", {}),
            last_check=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Component check failed for {component_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions


async def _check_system_resources() -> ComponentHealth:
    """Check system resource health."""
    import psutil

    details = {}

    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        details["cpu_percent"] = cpu_percent

        # Memory usage
        memory = psutil.virtual_memory()
        details["memory_percent"] = memory.percent
        details["memory_available_gb"] = memory.available / 1024 / 1024 / 1024

        # Disk usage
        disk = psutil.disk_usage("/")
        details["disk_percent"] = disk.percent
        details["disk_free_gb"] = disk.free / 1024 / 1024 / 1024

        # Determine status
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            status = ComponentStatus.DOWN
        elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 80:
            status = ComponentStatus.DEGRADED
        else:
            status = ComponentStatus.UP

        return ComponentHealth(
            name="system",
            status=status,
            latency_ms=0,
            details=details,
            last_check=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"System resource check failed: {e}")
        return ComponentHealth(
            name="system",
            status=ComponentStatus.DOWN,
            latency_ms=0,
            details={"error": str(e)},
            last_check=datetime.utcnow(),
        )
