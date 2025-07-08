"""
Agent-aware logging system.

Provides contextual logging with agent, session, and user information.
"""

import contextvars
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from src.core.utils.logger import get_logger

# Context variables for agent information
agent_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "agent_id", default=None
)
session_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "session_id", default=None
)
user_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "user_id", default=None
)
operation_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "operation", default=None
)


class AgentContextFilter(logging.Filter):
    """
    Logging filter that adds agent context to log records.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Add agent context to log record
        record.agent_id = agent_id_context.get()
        record.session_id = session_id_context.get()
        record.user_id = user_id_context.get()
        record.operation = operation_context.get()
        record.timestamp_iso = datetime.utcnow().isoformat()

        return True


class AgentLogger:
    """
    Logger wrapper that provides agent-aware logging capabilities.
    """

    def __init__(self, name: str):
        self.logger = get_logger(name)

        # Add agent context filter if not already present
        if not any(isinstance(f, AgentContextFilter) for f in self.logger.filters):
            self.logger.addFilter(AgentContextFilter())

    def debug(self, message: str, **kwargs):
        """Log debug message with agent context."""
        self.logger.debug(self._format_message(message), extra=kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with agent context."""
        self.logger.info(self._format_message(message), extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with agent context."""
        self.logger.warning(self._format_message(message), extra=kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with agent context."""
        self.logger.error(self._format_message(message), extra=kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message with agent context."""
        self.logger.critical(self._format_message(message), extra=kwargs)

    def log_agent_action(
        self,
        action: str,
        result: str = "success",
        details: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Log agent-specific actions with structured data.

        Args:
            action: The action being performed
            result: Result of the action (success, error, etc.)
            details: Additional details about the action
            performance_metrics: Performance metrics for the action
        """
        log_data = {
            "action": action,
            "result": result,
            "details": details or {},
            "performance_metrics": performance_metrics or {},
            "agent_id": agent_id_context.get(),
            "session_id": session_id_context.get(),
            "user_id": user_id_context.get(),
            "operation": operation_context.get(),
        }

        if result == "success":
            self.info(f"Agent action completed: {action}", extra=log_data)
        elif result == "error":
            self.error(f"Agent action failed: {action}", extra=log_data)
        else:
            self.info(f"Agent action {result}: {action}", extra=log_data)

    def log_memory_operation(
        self,
        operation: str,
        memory_id: Optional[str] = None,
        query: Optional[str] = None,
        result_count: Optional[int] = None,
        processing_time: Optional[float] = None,
        confidence_score: Optional[float] = None,
    ):
        """
        Log memory-specific operations.

        Args:
            operation: Type of memory operation (store, search, delete, etc.)
            memory_id: ID of memory being operated on
            query: Search query if applicable
            result_count: Number of results returned
            processing_time: Time taken for operation (ms)
            confidence_score: Confidence score if applicable
        """
        log_data = {
            "operation_type": "memory",
            "memory_operation": operation,
            "memory_id": memory_id,
            "query": query,
            "result_count": result_count,
            "processing_time_ms": processing_time,
            "confidence_score": confidence_score,
        }

        self.info(f"Memory operation: {operation}", extra=log_data)

    def log_rag_operation(
        self,
        stage: str,
        query: Optional[str] = None,
        retrieved_count: Optional[int] = None,
        reranked_count: Optional[int] = None,
        hallucination_detected: Optional[bool] = None,
        confidence_score: Optional[float] = None,
        processing_time: Optional[float] = None,
    ):
        """
        Log RAG pipeline operations.

        Args:
            stage: RAG stage (retrieval, reranking, hallucination_detection)
            query: The query being processed
            retrieved_count: Number of documents retrieved
            reranked_count: Number of documents after reranking
            hallucination_detected: Whether hallucination was detected
            confidence_score: Overall confidence score
            processing_time: Time taken for stage (ms)
        """
        log_data = {
            "operation_type": "rag",
            "rag_stage": stage,
            "query": query,
            "retrieved_count": retrieved_count,
            "reranked_count": reranked_count,
            "hallucination_detected": hallucination_detected,
            "confidence_score": confidence_score,
            "processing_time_ms": processing_time,
        }

        self.info(f"RAG {stage}: {query}", extra=log_data)

    def log_agent_interaction(
        self,
        interaction_type: str,
        target_agent: Optional[str] = None,
        message: Optional[str] = None,
        success: bool = True,
    ):
        """
        Log agent-to-agent interactions.

        Args:
            interaction_type: Type of interaction (message, coordination, etc.)
            target_agent: Target agent ID
            message: Interaction message
            success: Whether interaction was successful
        """
        log_data = {
            "operation_type": "agent_interaction",
            "interaction_type": interaction_type,
            "target_agent": target_agent,
            "message": message,
            "success": success,
        }

        level = self.info if success else self.error
        level(
            f"Agent interaction {interaction_type} with {target_agent}", extra=log_data
        )

    def _format_message(self, message: str) -> str:
        """Format message with agent context."""
        agent_id = agent_id_context.get()
        session_id = session_id_context.get()

        if agent_id and session_id:
            return f"[{agent_id}:{session_id[:8]}] {message}"
        elif agent_id:
            return f"[{agent_id}] {message}"
        else:
            return message


def set_agent_context(
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    operation: Optional[str] = None,
):
    """
    Set agent context for logging.

    Args:
        agent_id: Agent identifier
        session_id: Session identifier
        user_id: User identifier
        operation: Current operation
    """
    if agent_id is not None:
        agent_id_context.set(agent_id)
    if session_id is not None:
        session_id_context.set(session_id)
    if user_id is not None:
        user_id_context.set(user_id)
    if operation is not None:
        operation_context.set(operation)


def clear_agent_context():
    """Clear all agent context variables."""
    agent_id_context.set(None)
    session_id_context.set(None)
    user_id_context.set(None)
    operation_context.set(None)


def get_agent_logger(name: str) -> AgentLogger:
    """
    Get an agent-aware logger.

    Args:
        name: Logger name

    Returns:
        AgentLogger instance
    """
    return AgentLogger(name)


class AgentLogContext:
    """
    Context manager for agent logging.

    Automatically sets and clears agent context.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.session_id = session_id
        self.user_id = user_id
        self.operation = operation

        # Store previous context to restore later
        self.prev_agent_id = None
        self.prev_session_id = None
        self.prev_user_id = None
        self.prev_operation = None

    def __enter__(self):
        # Store previous context
        self.prev_agent_id = agent_id_context.get()
        self.prev_session_id = session_id_context.get()
        self.prev_user_id = user_id_context.get()
        self.prev_operation = operation_context.get()

        # Set new context
        set_agent_context(self.agent_id, self.session_id, self.user_id, self.operation)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        agent_id_context.set(self.prev_agent_id)
        session_id_context.set(self.prev_session_id)
        user_id_context.set(self.prev_user_id)
        operation_context.set(self.prev_operation)


# Convenience function for creating context managers
def agent_log_context(
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    operation: Optional[str] = None,
) -> AgentLogContext:
    """
    Create an agent logging context manager.

    Args:
        agent_id: Agent identifier
        session_id: Session identifier
        user_id: User identifier
        operation: Current operation

    Returns:
        AgentLogContext instance
    """
    return AgentLogContext(agent_id, session_id, user_id, operation)
