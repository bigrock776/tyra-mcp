"""
MCP (Model Context Protocol) server implementation for Tyra Memory Server.

This module provides the MCP server interface for memory operations including:
- Memory storage and retrieval
- Advanced search capabilities
- Hallucination detection
- Performance analytics
- Learning insights
"""

from .server import TyraMemoryServer, main

__all__ = ["TyraMemoryServer", "main"]
