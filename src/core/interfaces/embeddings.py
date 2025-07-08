"""
Abstract interface for embedding providers.

This module defines the standard interface that all embedding providers must implement,
enabling easy swapping of embedding models without changing core logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the embedding provider with configuration.

        Args:
            config: Provider-specific configuration dictionary
        """
        pass

    @abstractmethod
    async def embed_texts(
        self, texts: List[str], batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors as numpy arrays
        """
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Some models use different embeddings for queries vs documents.

        Args:
            query: Query text to embed

        Returns:
            Query embedding as numpy array
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Return the embedding dimension.

        Returns:
            Number of dimensions in the embedding vectors
        """
        pass

    @abstractmethod
    def supports_gpu(self) -> bool:
        """
        Check if this provider supports GPU acceleration.

        Returns:
            True if GPU is supported and available
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name/identifier of the embedding model.

        Returns:
            Model name string
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the embedding provider.

        Returns:
            Dictionary with health status information
        """
        pass

    async def embed_single(self, text: str) -> np.ndarray:
        """
        Convenience method to embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        embeddings = await self.embed_texts([text], batch_size=1)
        return embeddings[0]

    async def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between 0 and 1
        """
        emb1 = await self.embed_single(text1)
        emb2 = await self.embed_single(text2)

        # Compute cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


class EmbeddingProviderError(Exception):
    """Base exception for embedding provider errors."""

    pass


class EmbeddingInitializationError(EmbeddingProviderError):
    """Raised when embedding provider initialization fails."""

    pass


class EmbeddingGenerationError(EmbeddingProviderError):
    """Raised when embedding generation fails."""

    pass


class EmbeddingConfigurationError(EmbeddingProviderError):
    """Raised when embedding provider configuration is invalid."""

    pass
