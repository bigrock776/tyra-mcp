"""
Agent Session Management System.

Handles session lifecycle, agent authentication, and multi-agent coordination.
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from src.core.cache.redis_cache import CacheLevel, RedisCache
from src.core.utils.config import load_config
from src.core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentSession:
    """Represents an active agent session."""

    session_id: str
    agent_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = datetime.utcnow()

    def is_expired(self, timeout_seconds: int) -> bool:
        """Check if session has expired."""
        expiry_time = self.last_activity + timedelta(seconds=timeout_seconds)
        return datetime.utcnow() > expiry_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata,
            "context": self.context,
            "is_active": self.is_active
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSession":
        """Create session from dictionary."""
        return cls(
            session_id=data["session_id"],
            agent_id=data["agent_id"],
            user_id=data.get("user_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            metadata=data.get("metadata", {}),
            context=data.get("context", {}),
            is_active=data.get("is_active", True)
        )


class AgentSessionManager:
    """
    Manages agent sessions, authentication, and multi-agent coordination.

    Features:
    - Session lifecycle management
    - Agent authentication and authorization
    - Cross-agent session isolation
    - Session persistence in Redis
    - Automatic session cleanup
    - Multi-agent coordination
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.agent_configs = self.config.get("agents", {})
        self.shared_config = self.config.get("shared", {})

        # Session storage
        self.active_sessions: Dict[str, AgentSession] = {}
        self.agent_sessions: Dict[str, Set[str]] = {}  # agent_id -> session_ids

        # Configuration
        self.session_timeout = self.shared_config.get("session_timeout", 3600)
        self.max_concurrent_sessions = self.shared_config.get("max_concurrent_sessions", 10)
        self.isolation_level = self.shared_config.get("memory_isolation_level", "session")

        # Cache for session persistence
        self.cache: Optional[RedisCache] = None

        # Cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False

        logger.info(f"AgentSessionManager initialized with {len(self.agent_configs)} agent configs")

    async def initialize(self):
        """Initialize the session manager."""
        try:
            # Initialize Redis cache for session persistence
            self.cache = RedisCache()
            await self.cache.initialize()

            # Load existing sessions from cache
            await self._load_sessions_from_cache()

            # Start cleanup task
            self.is_running = True
            self.cleanup_task = asyncio.create_task(self._cleanup_task())

            logger.info("AgentSessionManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AgentSessionManager: {e}")
            raise

    async def create_session(
        self,
        agent_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentSession:
        """
        Create a new agent session.

        Args:
            agent_id: ID of the agent
            user_id: Optional user ID for multi-user systems
            metadata: Optional session metadata
            context: Optional session context

        Returns:
            Created AgentSession

        Raises:
            ValueError: If agent not configured or session limit exceeded
        """
        # Validate agent
        if agent_id not in self.agent_configs:
            raise ValueError(f"Unknown agent ID: {agent_id}")

        # Check session limits for agent
        current_sessions = len(self.agent_sessions.get(agent_id, set()))
        if current_sessions >= self.max_concurrent_sessions:
            raise ValueError(f"Maximum concurrent sessions exceeded for agent {agent_id}")

        # Generate session ID
        session_id = f"{agent_id}_{uuid.uuid4().hex[:8]}"

        # Create session
        session = AgentSession(
            session_id=session_id,
            agent_id=agent_id,
            user_id=user_id,
            metadata=metadata or {},
            context=context or {}
        )

        # Store session
        self.active_sessions[session_id] = session

        # Track agent sessions
        if agent_id not in self.agent_sessions:
            self.agent_sessions[agent_id] = set()
        self.agent_sessions[agent_id].add(session_id)

        # Persist to cache
        await self._persist_session(session)

        logger.info(f"Created session {session_id} for agent {agent_id}")
        return session

    async def get_session(self, session_id: str) -> Optional[AgentSession]:
        """
        Get an active session by ID.

        Args:
            session_id: Session identifier

        Returns:
            AgentSession if found and active, None otherwise
        """
        session = self.active_sessions.get(session_id)

        if session and session.is_active:
            # Check if expired
            if session.is_expired(self.session_timeout):
                await self.end_session(session_id)
                return None

            # Update activity
            session.update_activity()
            await self._persist_session(session)
            return session

        return None

    async def update_session_context(
        self,
        session_id: str,
        context_updates: Dict[str, Any]
    ) -> bool:
        """
        Update session context.

        Args:
            session_id: Session identifier
            context_updates: Context updates to apply

        Returns:
            True if updated successfully, False otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        session.context.update(context_updates)
        session.update_activity()
        await self._persist_session(session)

        logger.debug(f"Updated context for session {session_id}")
        return True

    async def end_session(self, session_id: str) -> bool:
        """
        End an active session.

        Args:
            session_id: Session identifier

        Returns:
            True if ended successfully, False if session not found
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return False

        # Mark as inactive
        session.is_active = False

        # Remove from tracking
        agent_id = session.agent_id
        if agent_id in self.agent_sessions:
            self.agent_sessions[agent_id].discard(session_id)
            if not self.agent_sessions[agent_id]:
                del self.agent_sessions[agent_id]

        # Remove from active sessions
        del self.active_sessions[session_id]

        # Remove from cache
        if self.cache:
            await self.cache.delete(f"session:{session_id}")\n        \n        logger.info(f"Ended session {session_id} for agent {agent_id}")
        return True

    async def get_agent_sessions(self, agent_id: str) -> List[AgentSession]:
        """
        Get all active sessions for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of active sessions for the agent
        """
        if agent_id not in self.agent_sessions:
            return []

        sessions = []
        for session_id in self.agent_sessions[agent_id].copy():
            session = await self.get_session(session_id)
            if session:
                sessions.append(session)

        return sessions

    def get_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent configuration or None if not found
        """
        return self.agent_configs.get(agent_id)

    def is_agent_configured(self, agent_id: str) -> bool:
        """Check if an agent is configured."""
        return agent_id in self.agent_configs

    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session management statistics.

        Returns:
            Dictionary with session statistics
        """
        total_sessions = len(self.active_sessions)
        agent_counts = {
            agent_id: len(sessions)
            for agent_id, sessions in self.agent_sessions.items()
        }

        return {
            "total_active_sessions": total_sessions,
            "sessions_by_agent": agent_counts,
            "configured_agents": list(self.agent_configs.keys()),
            "session_timeout": self.session_timeout,
            "max_concurrent_sessions": self.max_concurrent_sessions
        }

    @asynccontextmanager
    async def session_context(self, session_id: str):
        """
        Context manager for session operations.

        Automatically updates session activity and handles errors.
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found or expired")

        try:
            yield session
        except Exception as e:
            logger.error(f"Error in session {session_id}: {e}")
            raise
        finally:
            # Ensure activity is updated
            session.update_activity()
            await self._persist_session(session)

    async def _persist_session(self, session: AgentSession):
        """Persist session to cache."""
        if self.cache:
            cache_key = f"session:{session.session_id}"
            await self.cache.set(
                cache_key,
                session.to_dict(),
                ttl=self.session_timeout,
                level=CacheLevel.L2_SEARCH
            )

    async def _load_sessions_from_cache(self):
        """Load existing sessions from cache."""
        if not self.cache:
            return

        try:
            # This would require a cache.scan method to find all session keys
            # For now, we'll start with empty sessions
            logger.info("Session cache loaded (starting fresh)")

        except Exception as e:
            logger.warning(f"Failed to load sessions from cache: {e}")

    async def _cleanup_task(self):
        """Background task to clean up expired sessions."""
        while self.is_running:
            try:
                await self._cleanup_expired_sessions()
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in session cleanup task: {e}")
                await asyncio.sleep(60)

    async def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        expired_sessions = []

        for session_id, session in self.active_sessions.items():
            if session.is_expired(self.session_timeout):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self.end_session(session_id)

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def close(self):
        """Close the session manager and cleanup resources."""
        self.is_running = False

        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # End all active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.end_session(session_id)

        if self.cache:
            await self.cache.close()

        logger.info("AgentSessionManager closed")


# Global session manager instance
_session_manager: Optional[AgentSessionManager] = None


async def get_session_manager() -> AgentSessionManager:
    """Get the global session manager instance."""
    global _session_manager

    if _session_manager is None:
        _session_manager = AgentSessionManager()
        await _session_manager.initialize()

    return _session_manager


async def create_agent_session(
    agent_id: str,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> AgentSession:
    """Convenience function to create an agent session."""
    manager = await get_session_manager()
    return await manager.create_session(agent_id, user_id, metadata)


async def get_agent_session(session_id: str) -> Optional[AgentSession]:
    """Convenience function to get an agent session."""
    manager = await get_session_manager()
    return await manager.get_session(session_id)
