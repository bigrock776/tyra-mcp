"""
Embedding providers module for Tyra MCP Memory Server.

This module provides various embedding implementations with a pluggable
architecture for easy swapping of embedding models.
"""

from .huggingface import HuggingFaceEmbeddingProvider

# Provider registry for dynamic loading
EMBEDDING_PROVIDERS = {
    "huggingface": HuggingFaceEmbeddingProvider,
    "e5-large": HuggingFaceEmbeddingProvider,
    "minilm": HuggingFaceEmbeddingProvider,
}

__all__ = ["HuggingFaceEmbeddingProvider", "EMBEDDING_PROVIDERS"]
