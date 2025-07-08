"""
Reranker providers module for Tyra MCP Memory Server.

This module provides various reranking implementations with a pluggable
architecture for easy swapping of reranking models.
"""

from .cross_encoder import CrossEncoderReranker
from .registry import RerankerProviderRegistry, create_reranker, get_available_providers

# Legacy dictionary for backwards compatibility
RERANKER_PROVIDERS = {
    "cross_encoder": CrossEncoderReranker,
    "ms-marco": CrossEncoderReranker,
}

__all__ = [
    "CrossEncoderReranker",
    "RerankerProviderRegistry",
    "get_available_providers",
    "create_reranker",
    "RERANKER_PROVIDERS",
]
