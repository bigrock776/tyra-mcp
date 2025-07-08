"""
MCP tool definitions for Tyra Memory Server.

This module defines the individual MCP tools that provide memory operations
for agents. Tools are modular and reusable components that can be combined
in different server configurations.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from mcp.types import Tool


class MemoryTools:
    """Collection of memory-related MCP tools."""

    @staticmethod
    def get_store_memory_tool() -> Tool:
        """Get the store_memory tool definition."""
        return Tool(
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
        )

    @staticmethod
    def get_search_memory_tool() -> Tool:
        """Get the search_memory tool definition."""
        return Tool(
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
        )

    @staticmethod
    def get_analyze_response_tool() -> Tool:
        """Get the analyze_response tool definition."""
        return Tool(
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
        )

    @staticmethod
    def get_memory_stats_tool() -> Tool:
        """Get the get_memory_stats tool definition."""
        return Tool(
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
        )

    @staticmethod
    def get_learning_insights_tool() -> Tool:
        """Get the get_learning_insights tool definition."""
        return Tool(
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
        )

    @staticmethod
    def get_delete_memory_tool() -> Tool:
        """Get the delete_memory tool definition."""
        return Tool(
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
        )

    @staticmethod
    def get_health_check_tool() -> Tool:
        """Get the health_check tool definition."""
        return Tool(
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
        )

    @classmethod
    def get_all_tools(cls) -> List[Tool]:
        """Get all available memory tools."""
        return [
            cls.get_store_memory_tool(),
            cls.get_search_memory_tool(),
            cls.get_analyze_response_tool(),
            cls.get_memory_stats_tool(),
            cls.get_learning_insights_tool(),
            cls.get_delete_memory_tool(),
            cls.get_health_check_tool(),
        ]


class AnalyticsTools:
    """Collection of analytics and monitoring tools."""

    @staticmethod
    def get_performance_metrics_tool() -> Tool:
        """Get the performance_metrics tool definition."""
        return Tool(
            name="get_performance_metrics",
            description="Get detailed performance metrics and analytics",
            inputSchema={
                "type": "object",
                "properties": {
                    "metric_type": {
                        "type": "string",
                        "description": "Type of metrics to retrieve",
                        "enum": [
                            "latency",
                            "throughput",
                            "error_rate",
                            "cache_hit_rate",
                            "all",
                        ],
                        "default": "all",
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Time range for metrics",
                        "enum": ["1h", "6h", "24h", "7d"],
                        "default": "24h",
                    },
                    "include_trends": {
                        "type": "boolean",
                        "description": "Include trend analysis",
                        "default": True,
                    },
                },
            },
        )

    @staticmethod
    def get_system_health_tool() -> Tool:
        """Get the system_health tool definition."""
        return Tool(
            name="get_system_health",
            description="Get comprehensive system health and resource utilization",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_predictions": {
                        "type": "boolean",
                        "description": "Include resource usage predictions",
                        "default": False,
                    },
                    "detailed_breakdown": {
                        "type": "boolean",
                        "description": "Include detailed component breakdown",
                        "default": True,
                    },
                },
            },
        )

    @classmethod
    def get_all_tools(cls) -> List[Tool]:
        """Get all available analytics tools."""
        return [cls.get_performance_metrics_tool(), cls.get_system_health_tool()]


class AdminTools:
    """Collection of administrative tools."""

    @staticmethod
    def get_optimize_config_tool() -> Tool:
        """Get the optimize_config tool definition."""
        return Tool(
            name="optimize_config",
            description="Trigger configuration optimization based on current performance",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "description": "Optimization strategy to use",
                        "enum": [
                            "performance",
                            "efficiency",
                            "reliability",
                            "balanced",
                        ],
                        "default": "balanced",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Only analyze without applying changes",
                        "default": True,
                    },
                },
            },
        )

    @staticmethod
    def get_clear_cache_tool() -> Tool:
        """Get the clear_cache tool definition."""
        return Tool(
            name="clear_cache",
            description="Clear various system caches",
            inputSchema={
                "type": "object",
                "properties": {
                    "cache_type": {
                        "type": "string",
                        "description": "Type of cache to clear",
                        "enum": ["embeddings", "search", "rerank", "all"],
                        "default": "all",
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Confirmation to proceed with cache clearing",
                        "default": False,
                    },
                },
            },
        )

    @classmethod
    def get_all_tools(cls) -> List[Tool]:
        """Get all available admin tools."""
        return [cls.get_optimize_config_tool(), cls.get_clear_cache_tool()]


def get_all_available_tools() -> List[Tool]:
    """Get all available MCP tools across all categories."""
    tools = []
    tools.extend(MemoryTools.get_all_tools())
    tools.extend(AnalyticsTools.get_all_tools())
    tools.extend(AdminTools.get_all_tools())
    return tools


def get_tools_by_category(category: str) -> List[Tool]:
    """Get tools by category name."""
    category = category.lower()

    if category == "memory":
        return MemoryTools.get_all_tools()
    elif category == "analytics":
        return AnalyticsTools.get_all_tools()
    elif category == "admin":
        return AdminTools.get_all_tools()
    else:
        raise ValueError(f"Unknown tool category: {category}")


def get_tool_by_name(name: str) -> Optional[Tool]:
    """Get a specific tool by name."""
    all_tools = get_all_available_tools()
    for tool in all_tools:
        if tool.name == name:
            return tool
    return None
