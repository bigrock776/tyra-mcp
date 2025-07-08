"""
High-level client library for interacting with the Tyra memory system.

Provides a simplified interface for agents to store, retrieve, and analyze
memories with built-in error handling, caching, and performance optimization.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aiohttp

from ..core.cache import CacheLevel, RedisCache
from ..core.observability import get_memory_metrics, get_tracer
from ..core.utils.config import get_settings
from ..core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryRequest:
    """Request for storing memory."""

    content: str
    agent_id: str = "tyra"
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    extract_entities: bool = True
    chunk_content: bool = False


@dataclass
class SearchRequest:
    """Request for searching memories."""

    query: str
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    top_k: int = 10
    min_confidence: float = 0.0
    search_type: str = "hybrid"  # "vector", "graph", "hybrid"
    include_analysis: bool = True


@dataclass
class MemoryResult:
    """Result from memory operations."""

    success: bool
    memory_id: Optional[str] = None
    chunk_ids: List[str] = None
    entities_created: int = 0
    relationships_created: int = 0
    processing_time: Dict[str, float] = None
    error: Optional[str] = None


@dataclass
class SearchResult:
    """Result from search operations."""

    success: bool
    query: str
    results: List[Dict[str, Any]] = None
    total_results: int = 0
    search_type: str = "hybrid"
    hallucination_analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class MemoryClient:
    """
    High-level client for the Tyra memory system.

    Features:
    - Simplified agent interface
    - Automatic error handling and retries
    - Built-in caching for performance
    - Observability and metrics integration
    - Session management
    - Batch operations
    - Streaming search results
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize components
        self.tracer = get_tracer()
        self.metrics = get_memory_metrics()
        self.cache: Optional[RedisCache] = None

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

        # Configuration
        self.settings = get_settings()

        # Performance tracking
        self._request_count = 0
        self._error_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

    async def initialize(self, cache_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the memory client."""
        try:
            # Initialize cache if enabled
            if self.settings.cache.get("enabled", True):
                self.cache = RedisCache()
                await self.cache.initialize(cache_config)

            # Create HTTP session
            await self._get_session()

            logger.info(
                "Memory client initialized",
                base_url=self.base_url,
                cache_enabled=bool(self.cache),
            )

        except Exception as e:
            logger.error("Failed to initialize memory client", error=str(e))
            raise

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            async with self._session_lock:
                if self.session is None or self.session.closed:
                    headers = {"Content-Type": "application/json"}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"

                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    self.session = aiohttp.ClientSession(
                        headers=headers,
                        timeout=timeout,
                        connector=aiohttp.TCPConnector(limit=20, limit_per_host=10),
                    )

        return self.session

    async def store_memory(
        self,
        content: str,
        agent_id: str = "tyra",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extract_entities: bool = True,
        chunk_content: bool = False,
    ) -> MemoryResult:
        """Store a memory in the system."""
        async with self.tracer.trace_memory_operation("store", agent_id):
            start_time = time.time()

            try:
                request = MemoryRequest(
                    content=content,
                    agent_id=agent_id,
                    session_id=session_id,
                    metadata=metadata,
                    extract_entities=extract_entities,
                    chunk_content=chunk_content,
                )

                # Make API request
                response_data = await self._make_request(
                    "POST", "/v1/memory/store", data=asdict(request)
                )

                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                await self.metrics.record_memory_operation(
                    "store",
                    duration_ms,
                    success=True,
                    metadata={
                        "agent_id": agent_id,
                        "chunks": len(response_data.get("chunk_ids", [])),
                    },
                )

                return MemoryResult(
                    success=True,
                    memory_id=response_data.get("memory_id"),
                    chunk_ids=response_data.get("chunk_ids", []),
                    entities_created=response_data.get("entities_created", 0),
                    relationships_created=response_data.get("relationships_created", 0),
                    processing_time=response_data.get("processing_time", {}),
                )

            except Exception as e:
                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                await self.metrics.record_memory_operation(
                    "store", duration_ms, success=False, metadata={"agent_id": agent_id}
                )

                logger.error(
                    "Memory storage failed",
                    agent_id=agent_id,
                    content_length=len(content),
                    error=str(e),
                )

                return MemoryResult(success=False, error=str(e))

    async def search_memories(
        self,
        query: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 10,
        min_confidence: float = 0.0,
        search_type: str = "hybrid",
        include_analysis: bool = True,
    ) -> SearchResult:
        """Search memories with advanced filtering."""
        async with self.tracer.trace_memory_operation("search", agent_id):
            start_time = time.time()

            try:
                # Check cache first
                cache_key = self._build_search_cache_key(
                    query, agent_id, session_id, top_k, search_type
                )

                if self.cache:
                    cached_result = await self.cache.get(
                        cache_key, level=CacheLevel.L2_SEARCH
                    )
                    if cached_result:
                        self._cache_hits += 1
                        return SearchResult(**cached_result)

                self._cache_misses += 1

                request = SearchRequest(
                    query=query,
                    agent_id=agent_id,
                    session_id=session_id,
                    top_k=top_k,
                    min_confidence=min_confidence,
                    search_type=search_type,
                    include_analysis=include_analysis,
                )

                # Make API request
                response_data = await self._make_request(
                    "POST", "/v1/memory/search", data=asdict(request)
                )

                # Create result
                result = SearchResult(
                    success=True,
                    query=query,
                    results=response_data.get("results", []),
                    total_results=response_data.get("total_results", 0),
                    search_type=search_type,
                    hallucination_analysis=response_data.get("hallucination_analysis"),
                )

                # Cache the result
                if self.cache:
                    await self.cache.set(
                        cache_key, asdict(result), level=CacheLevel.L2_SEARCH
                    )

                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                await self.metrics.record_memory_operation(
                    "search",
                    duration_ms,
                    success=True,
                    metadata={
                        "agent_id": agent_id or "unknown",
                        "results": len(result.results),
                        "search_type": search_type,
                    },
                )

                return result

            except Exception as e:
                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                await self.metrics.record_memory_operation(
                    "search",
                    duration_ms,
                    success=False,
                    metadata={"agent_id": agent_id or "unknown"},
                )

                logger.error(
                    "Memory search failed",
                    query=query[:100] + "..." if len(query) > 100 else query,
                    agent_id=agent_id,
                    error=str(e),
                )

                return SearchResult(success=False, query=query, error=str(e))

    async def analyze_response(
        self,
        response: str,
        query: Optional[str] = None,
        retrieved_memories: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Analyze a response for hallucinations and confidence."""
        async with self.tracer.trace_hallucination_check(
            len(response), len(retrieved_memories or [])
        ):
            try:
                request_data = {
                    "response": response,
                    "query": query,
                    "retrieved_memories": retrieved_memories or [],
                }

                response_data = await self._make_request(
                    "POST", "/v1/memory/analyze", data=request_data
                )

                return response_data

            except Exception as e:
                logger.error(
                    "Response analysis failed",
                    response_length=len(response),
                    error=str(e),
                )
                return {"success": False, "error": str(e)}

    async def get_memory_stats(
        self,
        agent_id: Optional[str] = None,
        include_performance: bool = True,
        include_recommendations: bool = True,
    ) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        try:
            params = {}
            if agent_id:
                params["agent_id"] = agent_id
            if not include_performance:
                params["include_performance"] = False
            if not include_recommendations:
                params["include_recommendations"] = False

            return await self._make_request("GET", "/v1/memory/stats", params=params)

        except Exception as e:
            logger.error("Failed to get memory stats", error=str(e))
            return {"success": False, "error": str(e)}

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        try:
            response_data = await self._make_request(
                "DELETE", f"/v1/memory/{memory_id}"
            )

            return response_data.get("success", False)

        except Exception as e:
            logger.error("Memory deletion failed", memory_id=memory_id, error=str(e))
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the memory system."""
        try:
            return await self._make_request("GET", "/v1/health")
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "client_errors": self._error_count,
            }

    async def batch_store_memories(
        self, memories: List[Dict[str, Any]], agent_id: str = "tyra"
    ) -> List[MemoryResult]:
        """Store multiple memories efficiently."""
        results = []

        # Process in batches to avoid overwhelming the server
        batch_size = 10
        for i in range(0, len(memories), batch_size):
            batch = memories[i : i + batch_size]

            # Process batch concurrently
            tasks = []
            for memory_data in batch:
                task = self.store_memory(
                    content=memory_data["content"],
                    agent_id=agent_id,
                    session_id=memory_data.get("session_id"),
                    metadata=memory_data.get("metadata"),
                    extract_entities=memory_data.get("extract_entities", True),
                    chunk_content=memory_data.get("chunk_content", False),
                )
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    results.append(MemoryResult(success=False, error=str(result)))
                else:
                    results.append(result)

        logger.info(
            "Batch memory storage completed",
            total_memories=len(memories),
            successful=sum(1 for r in results if r.success),
            failed=sum(1 for r in results if not r.success),
        )

        return results

    async def stream_search_results(
        self, query: str, agent_id: Optional[str] = None, batch_size: int = 5
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream search results for large result sets."""
        offset = 0

        while True:
            # Search with pagination
            result = await self.search_memories(
                query=query, agent_id=agent_id, top_k=batch_size
            )

            if not result.success or not result.results:
                break

            for memory in result.results:
                yield memory

            # Check if we got fewer results than requested (end of results)
            if len(result.results) < batch_size:
                break

            offset += batch_size

    def _build_search_cache_key(
        self,
        query: str,
        agent_id: Optional[str],
        session_id: Optional[str],
        top_k: int,
        search_type: str,
    ) -> str:
        """Build cache key for search requests."""
        import hashlib

        key_parts = [
            query,
            agent_id or "all",
            session_id or "all",
            str(top_k),
            search_type,
        ]

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}{endpoint}"
        session = await self._get_session()

        for attempt in range(self.max_retries + 1):
            try:
                self._request_count += 1

                async with session.request(
                    method, url, json=data, params=params
                ) as response:
                    response.raise_for_status()
                    return await response.json()

            except Exception as e:
                if attempt == self.max_retries:
                    self._error_count += 1
                    raise

                # Exponential backoff
                await asyncio.sleep(2**attempt)

    async def close(self) -> None:
        """Close the memory client."""
        if self.session and not self.session.closed:
            await self.session.close()

        if self.cache:
            await self.cache.close()

        logger.info(
            "Memory client closed",
            total_requests=self._request_count,
            error_count=self._error_count,
            cache_hit_rate=self._cache_hits
            / max(self._cache_hits + self._cache_misses, 1),
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get client performance statistics."""
        total_cache_requests = self._cache_hits + self._cache_misses

        return {
            "requests": {
                "total": self._request_count,
                "errors": self._error_count,
                "error_rate": self._error_count / max(self._request_count, 1),
            },
            "cache": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": self._cache_hits / max(total_cache_requests, 1),
                "enabled": bool(self.cache),
            },
            "configuration": {
                "base_url": self.base_url,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            },
        }


class AgentMemoryInterface:
    """
    Simplified interface specifically designed for AI agents.

    Provides high-level methods that abstract away complexity
    and focus on common agent use cases.
    """

    def __init__(self, client: MemoryClient, agent_id: str):
        self.client = client
        self.agent_id = agent_id
        self.session_id = None

    def set_session(self, session_id: str) -> None:
        """Set session ID for memory isolation."""
        self.session_id = session_id

    async def remember(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        extract_entities: bool = True,
    ) -> bool:
        """Store a memory (simplified interface)."""
        result = await self.client.store_memory(
            content=content,
            agent_id=self.agent_id,
            session_id=self.session_id,
            metadata=context,
            extract_entities=extract_entities,
        )

        return result.success

    async def recall(
        self, query: str, limit: int = 5, min_confidence: float = 0.6
    ) -> List[str]:
        """Recall memories (simplified interface)."""
        result = await self.client.search_memories(
            query=query,
            agent_id=self.agent_id,
            session_id=self.session_id,
            top_k=limit,
            min_confidence=min_confidence,
        )

        if result.success:
            return [item["content"] for item in result.results]

        return []

    async def check_confidence(self, response: str, query: str) -> float:
        """Check confidence in a response (simplified interface)."""
        analysis = await self.client.analyze_response(response, query)

        if analysis.get("success"):
            return analysis.get("analysis", {}).get("overall_confidence", 0.0) / 100.0

        return 0.0

    async def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of agent's memories."""
        stats = await self.client.get_memory_stats(agent_id=self.agent_id)

        if stats.get("success"):
            return {
                "total_memories": stats.get("memory_stats", {}).get(
                    "total_memories", 0
                ),
                "recent_activity": stats.get("memory_stats", {}).get(
                    "recent_activity", 0
                ),
                "confidence_levels": stats.get("confidence_distribution", {}),
                "health_score": stats.get("health_score", 0),
            }

        return {"error": "Failed to get memory summary"}
