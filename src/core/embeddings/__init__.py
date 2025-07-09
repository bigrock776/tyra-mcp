"""
Core embeddings module for centralized embedding operations.

This module provides a unified interface for all embedding operations,
coordinating between different providers and implementing shared functionality.
"""

from .manager import EmbeddingManager
from .models import EmbeddingRequest, EmbeddingResponse

__all__ = ["EmbeddingManager", "EmbeddingRequest", "EmbeddingResponse"]