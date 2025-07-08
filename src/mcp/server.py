"""
Advanced MCP server for Tyra's memory system.

Provides Model Context Protocol (MCP) tools for memory storage, retrieval,
analytics, and adaptive learning capabilities with comprehensive error handling
and performance monitoring.
"""

import asyncio
import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    GetPromptRequest,
    GetPromptResult,
    ListToolsRequest,
    Prompt,
    PromptMessage,
    Role,
    TextContent,
    Tool,
)

from ..core.adaptation.learning_engine import LearningEngine
from ..core.analytics.performance_tracker import MetricType, PerformanceTracker
from ..core.memory.manager import MemoryManager, MemorySearchRequest, MemoryStoreRequest
from ..core.rag.hallucination_detector import HallucinationDetector
from ..core.utils.simple_config import get_setting, get_settings
from ..core.utils.simple_logger import get_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class TyraMemoryServer:
    """
    Advanced MCP server for Tyra's memory system.

    Features:
    - Memory storage and retrieval tools
    - Advanced search with hallucination detection
    - Performance analytics and monitoring
    - Adaptive learning and optimization
    - Multi-agent memory isolation
    - Comprehensive error handling
    """

    def __init__(self):
        self.server = Server("tyra-memory-server")

        # Core components
        self.memory_manager: Optional[MemoryManager] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        self.learning_engine: Optional[LearningEngine] = None
        self.hallucination_detector: Optional[HallucinationDetector] = None

        # Server state
        self._initialized = False
        self._total_requests = 0
        self._successful_requests = 0
        self._start_time = datetime.utcnow()

        # Register handlers
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register MCP protocol handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available memory tools."""
            return [
                Tool(
                    name="store_memory",
                    description="Store information in memory with optional metadata and entity extraction",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The content to store in memory",
                            },
                            "agent_id": {
                                "type": "string",
                                "description": "ID of the agent storing the memory (e.g., 'tyra', 'claude', 'archon')",
                                "default": "tyra",
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session ID for grouping related memories",
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Additional metadata to store with the memory",
                                "additionalProperties": True,
                            },
                            "extract_entities": {
                                "type": "boolean",
                                "description": "Whether to extract entities and relationships",
                                "default": True,
                            },
                            "chunk_content": {
                                "type": "boolean",
                                "description": "Whether to chunk large content",
                                "default": False,
                            },
                        },
                        "required": ["content"],
                    },
                ),
                Tool(
                    name="search_memory",
                    description="Search memories with advanced filtering and confidence scoring",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for finding relevant memories",
                            },
                            "agent_id": {
                                "type": "string",
                                "description": "Filter by agent ID (e.g., 'tyra', 'claude', 'archon')",
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Filter by session ID",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 100,
                            },
                            "min_confidence": {
                                "type": "number",
                                "description": "Minimum confidence score (0.0-1.0)",
                                "default": 0.0,
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "search_type": {
                                "type": "string",
                                "description": "Type of search to perform",
                                "enum": ["vector", "graph", "hybrid"],
                                "default": "hybrid",
                            },
                            "include_analysis": {
                                "type": "boolean",
                                "description": "Whether to include hallucination analysis",
                                "default": True,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="analyze_response",
                    description="Analyze a response for hallucinations and confidence scoring",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "response": {
                                "type": "string",
                                "description": "The response text to analyze",
                            },
                            "query": {
                                "type": "string",
                                "description": "Original query that generated the response",
                            },
                            "retrieved_memories": {
                                "type": "array",
                                "description": "Memory chunks used to generate the response",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "id": {"type": "string"},
                                        "metadata": {"type": "object"},
                                    },
                                },
                            },
                        },
                        "required": ["response"],
                    },
                ),
                Tool(
                    name="get_memory_stats",
                    description="Get comprehensive memory system statistics and health metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Filter stats by agent ID",
                            },
                            "include_performance": {
                                "type": "boolean",
                                "description": "Include performance analytics",
                                "default": True,
                            },
                            "include_recommendations": {
                                "type": "boolean",
                                "description": "Include optimization recommendations",
                                "default": True,
                            },
                        },
                    },
                ),
                Tool(
                    name="get_learning_insights",
                    description="Get insights from the adaptive learning system",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Filter insights by category",
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of days to look back",
                                "default": 7,
                                "minimum": 1,
                                "maximum": 90,
                            },
                        },
                    },
                ),
                Tool(
                    name="delete_memory",
                    description="Delete a specific memory by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "ID of the memory to delete",
                            }
                        },
                        "required": ["memory_id"],
                    },
                ),
                Tool(
                    name="health_check",
                    description="Perform comprehensive system health check",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "detailed": {
                                "type": "boolean",
                                "description": "Include detailed component health",
                                "default": False,
                            }
                        },
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Dict[str, Any]
        ) -> CallToolResult:
            """Handle tool execution with comprehensive error handling."""
            start_time = datetime.utcnow()
            self._total_requests += 1

            try:
                # Ensure system is initialized
                if not self._initialized:
                    await self._initialize_components()

                # Route to appropriate handler
                if name == "store_memory":
                    result = await self._handle_store_memory(arguments)
                elif name == "search_memory":
                    result = await self._handle_search_memory(arguments)
                elif name == "analyze_response":
                    result = await self._handle_analyze_response(arguments)
                elif name == "get_memory_stats":
                    result = await self._handle_get_memory_stats(arguments)
                elif name == "get_learning_insights":
                    result = await self._handle_get_learning_insights(arguments)
                elif name == "delete_memory":
                    result = await self._handle_delete_memory(arguments)
                elif name == "health_check":
                    result = await self._handle_health_check(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")

                # Record successful request
                self._successful_requests += 1

                # Track performance
                response_time = (datetime.utcnow() - start_time).total_seconds()
                if self.performance_tracker:
                    await self.performance_tracker.record_metric(
                        MetricType.RESPONSE_TIME,
                        response_time,
                        context={"tool": name, "success": True},
                    )

                return CallToolResult(
                    content=[
                        TextContent(type="text", text=json.dumps(result, indent=2))
                    ]
                )

            except Exception as e:
                # Log error
                error_msg = f"Tool '{name}' failed: {str(e)}"
                logger.error(
                    "Tool execution failed",
                    tool=name,
                    arguments=arguments,
                    error=str(e),
                    traceback=traceback.format_exc(),
                )

                # Track error metrics
                response_time = (datetime.utcnow() - start_time).total_seconds()
                if self.performance_tracker:
                    await self.performance_tracker.record_metric(
                        MetricType.RESPONSE_TIME,
                        response_time,
                        context={"tool": name, "success": False, "error": str(e)},
                    )

                # Return error response
                error_result = {
                    "success": False,
                    "error": error_msg,
                    "timestamp": datetime.utcnow().isoformat(),
                    "tool": name,
                }

                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=json.dumps(error_result, indent=2)
                        )
                    ]
                )

    async def _initialize_components(self) -> None:
        """Initialize all server components."""
        try:
            logger.info("Initializing Tyra memory server components...")

            # Initialize memory manager
            self.memory_manager = MemoryManager()
            await self.memory_manager.initialize()

            # Initialize performance tracker
            self.performance_tracker = PerformanceTracker()

            # Initialize hallucination detector
            if self.memory_manager.embedding_provider:
                self.hallucination_detector = HallucinationDetector(
                    self.memory_manager.embedding_provider
                )

            # Initialize learning engine
            self.learning_engine = LearningEngine(self.performance_tracker)

            # Start learning cycle in background
            asyncio.create_task(self.learning_engine.start_learning_cycle())

            self._initialized = True
            logger.info("Tyra memory server components initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize server components", error=str(e))
            raise

    async def _handle_store_memory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory storage requests."""
        try:
            request = MemoryStoreRequest(
                content=arguments["content"],
                agent_id=arguments.get("agent_id", "tyra"),
                session_id=arguments.get("session_id"),
                metadata=arguments.get("metadata"),
                extract_entities=arguments.get("extract_entities", True),
                chunk_content=arguments.get("chunk_content", False),
            )

            result = await self.memory_manager.store_memory(request)

            return {
                "success": True,
                "memory_id": result.memory_id,
                "chunk_ids": result.chunk_ids,
                "entities_created": len(result.entities_created),
                "relationships_created": len(result.relationships_created),
                "processing_time": {
                    "embedding": result.embedding_time,
                    "storage": result.storage_time,
                    "graph": result.graph_time,
                    "total": result.total_time,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Memory storage failed", arguments=arguments, error=str(e))
            raise

    async def _handle_search_memory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory search requests."""
        try:
            request = MemorySearchRequest(
                query=arguments["query"],
                agent_id=arguments.get("agent_id"),
                session_id=arguments.get("session_id"),
                top_k=arguments.get("top_k", 10),
                min_confidence=arguments.get("min_confidence", 0.0),
                search_type=arguments.get("search_type", "hybrid"),
                include_graph=arguments.get("search_type", "hybrid")
                in ["graph", "hybrid"],
            )

            # Perform search
            search_results = await self.memory_manager.search_memory(request)

            # Prepare results
            results = []
            retrieved_chunks = []

            for result in search_results:
                result_data = {
                    "id": result.id,
                    "content": result.content,
                    "score": result.score,
                    "confidence": result.confidence,
                    "source_type": result.source_type,
                    "metadata": result.metadata,
                }

                if result.rerank_explanation:
                    result_data["rerank_explanation"] = result.rerank_explanation

                if result.entities:
                    result_data["entities"] = [
                        {
                            "id": entity.id,
                            "name": entity.name,
                            "type": entity.entity_type,
                            "confidence": entity.confidence,
                        }
                        for entity in result.entities
                    ]

                results.append(result_data)

                # Prepare for hallucination analysis
                retrieved_chunks.append(
                    {
                        "content": result.content,
                        "id": result.id,
                        "metadata": result.metadata,
                    }
                )

            response = {
                "success": True,
                "query": request.query,
                "results": results,
                "total_results": len(results),
                "search_type": request.search_type,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Add hallucination analysis if requested
            if (
                arguments.get("include_analysis", True)
                and self.hallucination_detector
                and results
            ):
                # Create a summary response from top results
                summary_content = "\n".join([r["content"] for r in results[:3]])

                analysis = await self.hallucination_detector.analyze_response(
                    summary_content, retrieved_chunks, request.query
                )

                response["hallucination_analysis"] = {
                    "confidence": analysis.overall_confidence,
                    "confidence_level": analysis.confidence_level.value,
                    "hallucination_flag": analysis.hallucination_flag,
                    "safe_to_act_on": analysis.safe_to_act_on,
                    "evidence_count": analysis.evidence_count,
                    "reasoning": analysis.reasoning,
                    "warnings": analysis.warnings,
                }

            return response

        except Exception as e:
            logger.error("Memory search failed", arguments=arguments, error=str(e))
            raise

    async def _handle_analyze_response(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle response analysis for hallucination detection."""
        try:
            if not self.hallucination_detector:
                return {
                    "success": False,
                    "error": "Hallucination detector not available",
                }

            response_text = arguments["response"]
            query = arguments.get("query", "")
            retrieved_memories = arguments.get("retrieved_memories", [])

            # Perform analysis
            analysis = await self.hallucination_detector.analyze_response(
                response_text, retrieved_memories, query
            )

            return {
                "success": True,
                "response_text": response_text,
                "analysis": {
                    "overall_confidence": analysis.overall_confidence,
                    "confidence_level": analysis.confidence_level.value,
                    "confidence_emoji": self.hallucination_detector.get_confidence_emoji(
                        analysis.confidence_level
                    ),
                    "hallucination_flag": analysis.hallucination_flag,
                    "safe_to_act_on": analysis.safe_to_act_on,
                    "grounding_score": analysis.grounding_score,
                    "evidence_count": analysis.evidence_count,
                    "reasoning": analysis.reasoning,
                    "warnings": analysis.warnings,
                    "analysis_time": analysis.analysis_time,
                },
                "evidence": [
                    {
                        "source_chunk": evidence.source_chunk,
                        "similarity_score": evidence.similarity_score,
                        "relevance_score": evidence.relevance_score,
                        "confidence": evidence.confidence,
                        "chunk_id": evidence.chunk_id,
                    }
                    for evidence in analysis.evidence[:5]  # Top 5 evidence pieces
                ],
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Response analysis failed", arguments=arguments, error=str(e))
            raise

    async def _handle_get_memory_stats(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle memory statistics requests."""
        try:
            # Get memory manager stats
            memory_stats = await self.memory_manager.get_stats()

            # Get health check
            health_status = await self.memory_manager.health_check()

            response = {
                "success": True,
                "memory_stats": memory_stats,
                "health_status": health_status,
                "server_stats": {
                    "total_requests": self._total_requests,
                    "successful_requests": self._successful_requests,
                    "success_rate": self._successful_requests
                    / max(self._total_requests, 1),
                    "uptime_seconds": (
                        datetime.utcnow() - self._start_time
                    ).total_seconds(),
                    "initialized": self._initialized,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Add performance analytics if requested
            if arguments.get("include_performance", True) and self.performance_tracker:
                performance_summary = (
                    await self.performance_tracker.get_performance_summary(days=7)
                )
                response["performance_analytics"] = performance_summary

            # Add optimization recommendations if requested
            if arguments.get("include_recommendations", True) and self.learning_engine:
                recommendations = await self.learning_engine.get_recommendations()
                response["optimization_recommendations"] = [
                    {
                        "category": rec.category,
                        "priority": rec.priority,
                        "title": rec.title,
                        "description": rec.description,
                        "expected_impact": rec.expected_impact,
                        "confidence": rec.confidence,
                    }
                    for rec in recommendations[:5]  # Top 5 recommendations
                ]

            return response

        except Exception as e:
            logger.error(
                "Memory stats request failed", arguments=arguments, error=str(e)
            )
            raise

    async def _handle_get_learning_insights(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle learning insights requests."""
        try:
            if not self.learning_engine:
                return {"success": False, "error": "Learning engine not available"}

            category = arguments.get("category")
            insights = await self.learning_engine.get_learning_insights(category)

            # Get experiment history
            days = arguments.get("days", 7)
            experiments = await self.learning_engine.get_experiment_history(days)

            return {
                "success": True,
                "insights": [
                    {
                        "category": insight.category,
                        "insight": insight.insight,
                        "confidence": insight.confidence,
                        "actionable": insight.actionable,
                        "impact_estimate": insight.impact_estimate,
                        "timestamp": insight.timestamp.isoformat(),
                    }
                    for insight in insights
                ],
                "recent_experiments": [
                    {
                        "id": exp.id,
                        "type": exp.adaptation_type.value,
                        "status": exp.status.value,
                        "success": exp.success,
                        "improvements": exp.improvement,
                        "confidence": exp.confidence,
                        "start_time": exp.start_time.isoformat(),
                    }
                    for exp in experiments
                ],
                "learning_stats": self.learning_engine.get_learning_stats(),
                "period_days": days,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(
                "Learning insights request failed", arguments=arguments, error=str(e)
            )
            raise

    async def _handle_delete_memory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory deletion requests."""
        try:
            memory_id = arguments["memory_id"]
            success = await self.memory_manager.delete_memory(memory_id)

            return {
                "success": success,
                "memory_id": memory_id,
                "deleted": success,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Memory deletion failed", arguments=arguments, error=str(e))
            raise

    async def _handle_health_check(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check requests."""
        try:
            detailed = arguments.get("detailed", False)

            # Basic health check
            health_status = {
                "success": True,
                "status": "healthy",
                "initialized": self._initialized,
                "uptime_seconds": (
                    datetime.utcnow() - self._start_time
                ).total_seconds(),
                "total_requests": self._total_requests,
                "success_rate": self._successful_requests
                / max(self._total_requests, 1),
                "timestamp": datetime.utcnow().isoformat(),
            }

            if detailed and self._initialized:
                # Detailed component health
                components = {}

                if self.memory_manager:
                    components["memory_manager"] = (
                        await self.memory_manager.health_check()
                    )

                if self.performance_tracker:
                    components["performance_tracker"] = {
                        "status": "healthy",
                        "stats": self.performance_tracker.get_analytics_stats(),
                    }

                if self.learning_engine:
                    components["learning_engine"] = {
                        "status": "healthy",
                        "stats": self.learning_engine.get_learning_stats(),
                    }

                health_status["components"] = components

            return health_status

        except Exception as e:
            logger.error("Health check failed", arguments=arguments, error=str(e))
            return {
                "success": False,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def run(self) -> None:
        """Run the MCP server."""
        try:
            logger.info("Starting Tyra Memory MCP Server...")

            # Pre-initialize components
            await self._initialize_components()

            logger.info("Tyra Memory MCP Server ready for connections")

            # Run the server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="tyra-memory-server",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=None, experimental_capabilities=None
                        ),
                    ),
                )

        except Exception as e:
            logger.error("Server startup failed", error=str(e))
            raise
        finally:
            # Cleanup
            if self.memory_manager:
                await self.memory_manager.close()

            logger.info("Tyra Memory MCP Server stopped")


async def main():
    """Main entry point for the server."""
    server = TyraMemoryServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
