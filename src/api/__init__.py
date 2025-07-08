"""
FastAPI application for the Tyra memory system.

Provides REST API endpoints for all memory operations, complementing the MCP server
with standard HTTP interfaces for broader integration.
"""

from .app import create_app, get_app

__all__ = ["create_app", "get_app"]
