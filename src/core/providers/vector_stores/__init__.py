"""
Vector store providers module for Tyra MCP Memory Server.

This module provides various vector database implementations with a pluggable
architecture for easy swapping of vector storage backends.
"""

from .pgvector import PgVectorStore

# Provider registry for dynamic loading
VECTOR_STORE_PROVIDERS = {
    "pgvector": PgVectorStore,
    "postgresql": PgVectorStore,
}

__all__ = ["PgVectorStore", "VECTOR_STORE_PROVIDERS"]
