"""
Structured logging system.

This module provides a centralized logging system with structured output,
contextual information, and integration with monitoring systems.
"""

import logging
import logging.handlers
import sys
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .config import get_settings

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
agent_id_var: ContextVar[Optional[str]] = ContextVar("agent_id", default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


def add_context_info(logger, method_name, event_dict):
    """Add contextual information to log records."""
    # Add request context
    request_id = request_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id

    agent_id = agent_id_var.get()
    if agent_id:
        event_dict["agent_id"] = agent_id

    session_id = session_id_var.get()
    if session_id:
        event_dict["session_id"] = session_id

    # Add timestamp
    event_dict["timestamp"] = datetime.utcnow().isoformat()

    return event_dict


def add_performance_info(logger, method_name, event_dict):
    """Add performance-related information to log records."""
    # This processor can be extended to add performance metrics
    return event_dict


class LoggerManager:
    """Manages logging configuration and provides logger instances."""

    def __init__(self):
        self.configured = False
        self.console = Console()
        self.settings = None

    def configure_logging(self):
        """Configure the logging system based on settings."""
        if self.configured:
            return

        try:
            self.settings = get_settings()
            log_config = self.settings.logging
        except Exception:
            # Fallback to basic configuration if settings not available
            log_config = type(
                "LogConfig",
                (),
                {
                    "level": "INFO",
                    "format": "structured",
                    "file": None,
                    "max_size": "100MB",
                    "backup_count": 5,
                },
            )()

        # Configure structlog
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            add_context_info,
            add_performance_info,
        ]

        if log_config.format == "structured":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Configure standard logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, log_config.level.upper()),
        )

        # Set up file logging if configured
        if log_config.file:
            self._setup_file_logging(log_config)

        # Set up rich console logging for development
        if hasattr(self.settings, "development") and self.settings.development.debug:
            self._setup_rich_logging()

        self.configured = True

    def _setup_file_logging(self, log_config):
        """Set up file-based logging with rotation."""
        log_file = Path(log_config.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Parse max_size
        max_bytes = self._parse_size(log_config.max_size)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=log_config.backup_count,
            encoding="utf-8",
        )

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

    def _setup_rich_logging(self):
        """Set up rich console logging for development."""
        rich_handler = RichHandler(
            console=self.console, show_time=True, show_path=True, rich_tracebacks=True
        )

        root_logger = logging.getLogger()
        root_logger.addHandler(rich_handler)

    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '100MB' to bytes."""
        size_str = size_str.upper()
        if size_str.endswith("KB"):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)

    def get_logger(self, name: str) -> structlog.BoundLogger:
        """Get a logger instance with the given name."""
        if not self.configured:
            self.configure_logging()

        return structlog.get_logger(name)

    def set_request_context(
        self,
        request_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Set context variables for the current request."""
        if request_id:
            request_id_var.set(request_id)
        if agent_id:
            agent_id_var.set(agent_id)
        if session_id:
            session_id_var.set(session_id)

    def clear_request_context(self):
        """Clear all context variables."""
        request_id_var.set(None)
        agent_id_var.set(None)
        session_id_var.set(None)


# Global logger manager
logger_manager = LoggerManager()


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance with the given name."""
    return logger_manager.get_logger(name)


def set_request_context(
    request_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Set context variables for the current request."""
    logger_manager.set_request_context(request_id, agent_id, session_id)


def clear_request_context():
    """Clear all context variables."""
    logger_manager.clear_request_context()


class LoggingContextManager:
    """Context manager for temporary logging context."""

    def __init__(
        self,
        request_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.request_id = request_id
        self.agent_id = agent_id
        self.session_id = session_id
        self.previous_context = {}

    def __enter__(self):
        # Save previous context
        self.previous_context = {
            "request_id": request_id_var.get(),
            "agent_id": agent_id_var.get(),
            "session_id": session_id_var.get(),
        }

        # Set new context
        set_request_context(self.request_id, self.agent_id, self.session_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        set_request_context(
            self.previous_context["request_id"],
            self.previous_context["agent_id"],
            self.previous_context["session_id"],
        )


def with_logging_context(
    request_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Decorator to add logging context to a function."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            with LoggingContextManager(request_id, agent_id, session_id):
                return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            with LoggingContextManager(request_id, agent_id, session_id):
                return func(*args, **kwargs)

        if hasattr(func, "__code__") and "async" in str(func.__code__.co_flags):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class PerformanceLogger:
    """Logger for performance metrics."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.logger = get_logger(f"performance.{operation_name}")
        self.start_time = None

    def __enter__(self):
        import time

        self.start_time = time.time()
        self.logger.info("Operation started", operation=self.operation_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        duration = time.time() - self.start_time

        if exc_type is None:
            self.logger.info(
                "Operation completed",
                operation=self.operation_name,
                duration=duration,
                status="success",
            )
        else:
            self.logger.error(
                "Operation failed",
                operation=self.operation_name,
                duration=duration,
                status="error",
                error_type=exc_type.__name__ if exc_type else None,
                error_message=str(exc_val) if exc_val else None,
            )


def log_performance(operation_name: str):
    """Context manager for logging performance metrics."""
    return PerformanceLogger(operation_name)


# Initialize logging on module import
logger_manager.configure_logging()
