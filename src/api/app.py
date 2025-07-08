"""
FastAPI application for Tyra MCP Memory Server.

Provides REST API endpoints for memory operations, search, RAG features,
and administrative functions beyond the MCP protocol.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from ..core.memory.manager import MemoryManager
from ..core.observability import get_memory_metrics, get_telemetry, get_tracer
from ..core.utils.config import get_settings
from ..core.utils.logger import get_logger
from .middleware.auth import AuthMiddleware
from .middleware.error_handler import ErrorHandlerMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .routes.admin import router as admin_router
from .routes.graph import router as graph_router
from .routes.health import router as health_router
from .routes.memory import router as memory_router
from .routes.rag import router as rag_router
from .routes.search import router as search_router
from .routes.webhooks import router as webhooks_router

logger = get_logger(__name__)


class MemorySystemState:
    """Global application state for memory system components."""

    def __init__(self):
        self.memory_manager: Optional[MemoryManager] = None
        self.telemetry = None
        self.tracer = None
        self.metrics = None
        self.settings = None


# Global state instance
app_state = MemorySystemState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting Tyra MCP Memory Server...")

    try:
        # Initialize configuration
        app_state.settings = get_settings()

        # Initialize observability
        app_state.telemetry = get_telemetry()
        app_state.tracer = get_tracer()
        app_state.metrics = get_memory_metrics()

        # Initialize memory manager
        app_state.memory_manager = MemoryManager()
        await app_state.memory_manager.initialize(app_state.settings.to_dict())

        logger.info(
            f"Tyra MCP Memory Server started on port {app_state.settings.api.port}"
        )
        yield

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Shutdown
        if app_state.memory_manager:
            await app_state.memory_manager.close()

        logger.info("Tyra MCP Memory Server shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    settings = get_settings()

    app = FastAPI(
        title="Tyra MCP Memory Server",
        description="Advanced memory system with RAG capabilities and MCP protocol support",
        version="1.0.0",
        docs_url="/docs" if settings.api.enable_docs else None,
        redoc_url="/redoc" if settings.api.enable_docs else None,
        lifespan=lifespan,
    )

    # Add middleware
    _setup_middleware(app, settings)

    # Add routes
    _setup_routes(app)

    # Add exception handlers
    _setup_exception_handlers(app)

    return app


def _setup_middleware(app: FastAPI, settings):
    """Setup application middleware."""

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Custom middleware
    if settings.api.enable_auth:
        app.add_middleware(AuthMiddleware)

    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)

    # Request timing and tracing middleware
    @app.middleware("http")
    async def request_tracking_middleware(request: Request, call_next):
        """Track requests with tracing and metrics."""
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Add request ID to state
        request.state.request_id = request_id

        try:
            # Trace the request
            async with app_state.tracer.telemetry.trace(
                f"api.{request.method}.{request.url.path}",
                attributes={
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.request_id": request_id,
                    "tyra.component": "api",
                },
            ):
                response = await call_next(request)

                # Record metrics
                duration = time.time() - start_time
                await app_state.metrics.record_api_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code,
                    duration_ms=duration * 1000,
                    request_id=request_id,
                )

                # Add response headers
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = f"{duration:.3f}s"

                return response

        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            await app_state.metrics.record_api_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=500,
                duration_ms=duration * 1000,
                request_id=request_id,
                error=str(e),
            )
            raise


def _setup_routes(app: FastAPI):
    """Setup application routes."""

    # Health check (no prefix)
    app.include_router(health_router, tags=["health"])

    # API routes with v1 prefix
    app.include_router(memory_router, prefix="/v1/memory", tags=["memory"])

    app.include_router(search_router, prefix="/v1/search", tags=["search"])

    app.include_router(rag_router, prefix="/v1/rag", tags=["rag"])

    app.include_router(graph_router, prefix="/v1/graph", tags=["graph"])

    app.include_router(admin_router, prefix="/v1/admin", tags=["admin"])

    app.include_router(webhooks_router, tags=["webhooks"])


def _setup_exception_handlers(app: FastAPI):
    """Setup global exception handlers."""

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler with logging and tracing."""
        request_id = getattr(request.state, "request_id", "unknown")

        logger.error(
            "API request failed",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            error=str(exc),
            exc_info=True,
        )

        # Record exception in trace
        if app_state.tracer:
            app_state.tracer.record_exception(
                exc, {"request_id": request_id, "endpoint": request.url.path}
            )

        # Return appropriate error response
        if isinstance(exc, HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.detail, "request_id": request_id},
            )

        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "request_id": request_id},
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not Found",
                "message": "The requested resource was not found",
                "path": str(request.url),
                "request_id": request_id,
            },
        )


# Dependency injection functions
async def get_memory_manager() -> MemoryManager:
    """Dependency for getting the memory manager."""
    if not app_state.memory_manager:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    return app_state.memory_manager


async def get_request_context(request: Request) -> Dict[str, Any]:
    """Get request context for tracing and logging."""
    return {
        "request_id": getattr(request.state, "request_id", "unknown"),
        "user_agent": request.headers.get("user-agent"),
        "client_ip": request.client.host if request.client else None,
    }


def get_app_instance() -> FastAPI:
    """Get the FastAPI application instance."""
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "src.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_level=settings.logging.level.lower(),
    )
