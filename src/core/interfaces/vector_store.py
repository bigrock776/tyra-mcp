"""
Abstract interface for vector storage systems.

This module defines the standard interface that all vector storage providers must implement,
enabling easy swapping of vector databases without changing core logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""

    id: str
    score: float
    metadata: Dict[str, Any]
    content: str
    embedding: Optional[np.ndarray] = None


@dataclass
class VectorDocument:
    """Document to be stored in vector database."""

    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class VectorStore(ABC):
    """Abstract base class for vector storage systems."""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the vector store with configuration.

        Args:
            config: Store-specific configuration dictionary
        """
        pass

    @abstractmethod
    async def store_documents(self, documents: List[VectorDocument]) -> List[str]:
        """
        Store multiple documents in the vector database.

        Args:
            documents: List of documents to store

        Returns:
            List of document IDs that were stored
        """
        pass

    @abstractmethod
    async def store_document(self, document: VectorDocument) -> str:
        """
        Store a single document in the vector database.

        Args:
            document: Document to store

        Returns:
            Document ID that was stored
        """
        pass

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """
        Search for similar documents using vector similarity.

        Args:
            query_embedding: Query vector to search with
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            min_score: Minimum similarity score threshold

        Returns:
            List of search results ordered by similarity score
        """
        pass

    @abstractmethod
    async def hybrid_search(
        self,
        query_embedding: np.ndarray,
        text_query: Optional[str] = None,
        top_k: int = 10,
        vector_weight: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid search combining vector and text search.

        Args:
            query_embedding: Query vector for semantic search
            text_query: Optional text query for keyword search
            top_k: Maximum number of results to return
            vector_weight: Weight for vector search (1 - vector_weight for text)
            filters: Optional metadata filters

        Returns:
            List of search results ordered by combined score
        """
        pass

    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """
        Retrieve a document by ID.

        Args:
            document_id: ID of document to retrieve

        Returns:
            Document if found, None otherwise
        """
        pass

    @abstractmethod
    async def update_document(self, document: VectorDocument) -> bool:
        """
        Update an existing document.

        Args:
            document: Updated document

        Returns:
            True if document was updated, False if not found
        """
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by ID.

        Args:
            document_id: ID of document to delete

        Returns:
            True if document was deleted, False if not found
        """
        pass

    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> int:
        """
        Delete multiple documents by ID.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            Number of documents actually deleted
        """
        pass

    @abstractmethod
    async def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents in the store.

        Args:
            filters: Optional metadata filters

        Returns:
            Number of documents matching filters
        """
        pass

    @abstractmethod
    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorDocument]:
        """
        List documents with pagination.

        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            filters: Optional metadata filters

        Returns:
            List of documents
        """
        pass

    @abstractmethod
    async def create_index(self, index_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create or rebuild vector index for better search performance.

        Args:
            index_params: Optional index-specific parameters

        Returns:
            True if index was created successfully
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the vector store.

        Returns:
            Dictionary with health status information
        """
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics (doc count, index size, etc.)
        """
        pass

    async def search_by_ids(self, document_ids: List[str]) -> List[VectorDocument]:
        """
        Retrieve multiple documents by their IDs.

        Args:
            document_ids: List of document IDs to retrieve

        Returns:
            List of found documents (may be shorter than input list)
        """
        documents = []
        for doc_id in document_ids:
            doc = await self.get_document(doc_id)
            if doc:
                documents.append(doc)
        return documents


class VectorStoreError(Exception):
    """Base exception for vector store errors."""

    pass


class VectorStoreInitializationError(VectorStoreError):
    """Raised when vector store initialization fails."""

    pass


class VectorStoreOperationError(VectorStoreError):
    """Raised when vector store operation fails."""

    pass


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when vector store configuration is invalid."""

    pass
