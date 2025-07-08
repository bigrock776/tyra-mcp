"""
Simple logging system for Tyra Memory Server.

Provides basic logging functionality without external dependencies.
Can be upgraded to structured logging when dependencies are available.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class SimpleFormatter(logging.Formatter):
    """Simple JSON formatter for structured logging."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if they exist
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, default=str)


class ContextLogger:
    """Simple logger with context support."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        """Setup basic logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(SimpleFormatter())
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log with additional context."""
        extra_fields = {}
        for key, value in kwargs.items():
            if key not in ["exc_info", "stack_info", "extra"]:
                extra_fields[key] = value

        # Create a custom log record
        record = self.logger.makeRecord(
            self.logger.name, level, "", 0, message, (), None
        )
        record.extra_fields = extra_fields
        self.logger.handle(record)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)


def get_logger(name: str) -> ContextLogger:
    """Get a logger instance."""
    return ContextLogger(name)


def setup_logging(level: str = "INFO"):
    """Setup basic logging configuration."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
