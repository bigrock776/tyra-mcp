"""
Advanced retrieval system with hybrid search and semantic reranking.

Provides MCP-compatible retrieval with vector search, keyword matching,
and graph traversal for comprehensive memory retrieval.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..interfaces.embeddings import EmbeddingProvider
from ..interfaces.graph_engine import Entity, GraphEngine, Relationship
from ..interfaces.reranker import Reranker, RerankingCandidate, RerankingResult
from ..interfaces.vector_store import VectorSearchResult, VectorStore
from ..utils.logger import get_logger
from ..utils.registry import ProviderType, get_provider

logger = get_logger(__name__)


class RetrievalStrategy(Enum):
    """Retrieval strategy types."""

    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID_WEIGHTED = "hybrid_weighted"
    ADAPTIVE = "adaptive"
    MULTIMODAL = "multimodal"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""

    strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE
    vector_weight: float = 0.7
    graph_weight: float = 0.3
    max_results: int = 20
    min_score_threshold: float = 0.1
    enable_reranking: bool = True
    rerank_top_k: int = 10
    enable_diversity: bool = True
    diversity_penalty: float = 0.1


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""

    content: str
    score: float
    metadata: Dict[str, Any]
    source_type: str  # "vector", "graph", "hybrid"
    timestamp: datetime
    chunk_id: Optional[str] = None
    entity_id: Optional[str] = None


class AdvancedRetriever:
    """
    Advanced retrieval system with multiple strategies.

    Combines vector search, graph traversal, and hybrid approaches
    for comprehensive memory retrieval.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        """Initialize retriever with configuration."""
        self.config = config or RetrievalConfig()
        self.embedding_provider = None
        self.vector_store = None
        self.graph_engine = None
        self.reranker = None

        # Initialize providers
        self._init_providers()

        logger.info(
            f"AdvancedRetriever initialized with strategy: {self.config.strategy}"
        )

    def _init_providers(self):
        """Initialize all provider instances."""
        try:
            # Get embedding provider with fallback
            self.embedding_provider = get_provider(
                ProviderType.EMBEDDING, "huggingface"
            )

            # Get vector store
            self.vector_store = get_provider(ProviderType.VECTOR_STORE, "pgvector")

            # Get graph engine
            self.graph_engine = get_provider(ProviderType.GRAPH_ENGINE, "memgraph")

            # Get reranker if enabled
            if self.config.enable_reranking:
                self.reranker = get_provider(ProviderType.RERANKER, "cross_encoder")

        except Exception as e:
            logger.error(f"Failed to initialize providers: {e}")
            raise

    async def retrieve(
        self,
        query: str,
        max_results: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant content using configured strategy.

        Args:
            query: Search query text
            max_results: Maximum number of results to return
            filters: Optional filters for search

        Returns:
            List of retrieval results sorted by relevance
        """
        max_results = max_results or self.config.max_results

        try:
            if self.config.strategy == RetrievalStrategy.VECTOR_ONLY:
                results = await self._vector_search(query, max_results, filters)
            elif self.config.strategy == RetrievalStrategy.GRAPH_ONLY:
                results = await self._graph_search(query, max_results, filters)
            elif self.config.strategy == RetrievalStrategy.HYBRID_WEIGHTED:
                results = await self._hybrid_search(query, max_results, filters)
            elif self.config.strategy == RetrievalStrategy.ADAPTIVE:
                results = await self._adaptive_search(query, max_results, filters)
            else:
                # Fallback to vector search
                results = await self._vector_search(query, max_results, filters)

            # Apply diversity penalty if enabled
            if self.config.enable_diversity:
                results = self._apply_diversity_penalty(results)

            # Rerank if enabled
            if self.config.enable_reranking and self.reranker:
                results = await self._rerank_results(query, results)

            # Filter by minimum score threshold
            results = [r for r in results if r.score >= self.config.min_score_threshold]

            return results[:max_results]

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise

    async def _vector_search(
        self, query: str, max_results: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform vector-only search."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_provider.embed_text(query)

            # Search vector store
            vector_results = await self.vector_store.search(
                query_embedding=query_embedding, limit=max_results, filters=filters
            )

            # Convert to RetrievalResult format
            results = []
            for result in vector_results:
                retrieval_result = RetrievalResult(
                    content=result.content,
                    score=result.score,
                    metadata=result.metadata,
                    source_type="vector",
                    timestamp=datetime.now(),
                    chunk_id=result.id,
                )
                results.append(retrieval_result)

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _graph_search(
        self, query: str, max_results: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform graph-only search."""
        try:
            # Extract entities from query
            entities = await self._extract_entities(query)

            if not entities:
                logger.warning("No entities found in query for graph search")
                return []

            # Search graph for related content
            graph_results = []
            for entity in entities:
                related = await self.graph_engine.get_related_entities(
                    entity_id=entity, max_depth=2, limit=max_results // len(entities)
                )
                graph_results.extend(related)

            # Convert to RetrievalResult format
            results = []
            for entity in graph_results:
                # Score based on entity importance and relationships
                score = self._calculate_graph_score(entity, query)

                retrieval_result = RetrievalResult(
                    content=entity.properties.get("content", ""),
                    score=score,
                    metadata=entity.properties,
                    source_type="graph",
                    timestamp=datetime.now(),
                    entity_id=entity.id,
                )
                results.append(retrieval_result)

            return sorted(results, key=lambda x: x.score, reverse=True)

        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    async def _hybrid_search(
        self, query: str, max_results: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform hybrid vector + graph search."""
        try:
            # Run both searches concurrently
            vector_task = self._vector_search(query, max_results, filters)
            graph_task = self._graph_search(query, max_results, filters)

            vector_results, graph_results = await asyncio.gather(
                vector_task, graph_task, return_exceptions=True
            )

            # Handle exceptions
            if isinstance(vector_results, Exception):
                logger.error(f"Vector search failed in hybrid: {vector_results}")
                vector_results = []
            if isinstance(graph_results, Exception):
                logger.error(f"Graph search failed in hybrid: {graph_results}")
                graph_results = []

            # Merge and reweight results
            merged_results = self._merge_hybrid_results(vector_results, graph_results)

            return sorted(merged_results, key=lambda x: x.score, reverse=True)

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    async def _adaptive_search(
        self, query: str, max_results: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Adaptive search that chooses strategy based on query.

        Uses heuristics to determine best approach.
        """
        try:
            # Analyze query to determine best strategy
            query_analysis = self._analyze_query(query)

            if query_analysis["entity_rich"]:
                # Use graph search for entity-rich queries
                return await self._graph_search(query, max_results, filters)
            elif query_analysis["semantic_complex"]:
                # Use hybrid for complex semantic queries
                return await self._hybrid_search(query, max_results, filters)
            else:
                # Default to vector search
                return await self._vector_search(query, max_results, filters)

        except Exception as e:
            logger.error(f"Adaptive search failed: {e}")
            # Fallback to vector search
            return await self._vector_search(query, max_results, filters)

    def _merge_hybrid_results(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Merge and reweight hybrid search results."""
        merged = []

        # Reweight vector results
        for result in vector_results:
            result.score *= self.config.vector_weight
            result.source_type = "hybrid_vector"
            merged.append(result)

        # Reweight graph results
        for result in graph_results:
            result.score *= self.config.graph_weight
            result.source_type = "hybrid_graph"
            merged.append(result)

        # Remove duplicates based on content similarity
        merged = self._deduplicate_results(merged)

        return merged

    def _apply_diversity_penalty(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Apply diversity penalty to reduce redundant results."""
        if len(results) <= 1:
            return results

        diverse_results = [results[0]]  # Always include top result

        for result in results[1:]:
            # Calculate similarity to already selected results
            max_similarity = 0.0
            for selected in diverse_results:
                similarity = self._calculate_content_similarity(
                    result.content, selected.content
                )
                max_similarity = max(max_similarity, similarity)

            # Apply diversity penalty
            penalty = max_similarity * self.config.diversity_penalty
            result.score *= 1.0 - penalty
            diverse_results.append(result)

        return sorted(diverse_results, key=lambda x: x.score, reverse=True)

    async def _rerank_results(
        self, query: str, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank results using cross-encoder."""
        if not results or not self.reranker:
            return results

        try:
            # Prepare candidates for reranking
            candidates = []
            for i, result in enumerate(results[: self.config.rerank_top_k]):
                candidate = RerankingCandidate(
                    id=str(i),
                    text=result.content,
                    score=result.score,
                    metadata=result.metadata,
                )
                candidates.append(candidate)

            # Rerank
            reranked = await self.reranker.rerank(query, candidates)

            # Update scores and reorder
            score_map = {r.id: r.score for r in reranked.candidates}

            for i, result in enumerate(results[: self.config.rerank_top_k]):
                if str(i) in score_map:
                    result.score = score_map[str(i)]

            # Keep non-reranked results at the end
            reranked_results = results[: self.config.rerank_top_k]
            remaining_results = results[self.config.rerank_top_k :]

            return (
                sorted(reranked_results, key=lambda x: x.score, reverse=True)
                + remaining_results
            )

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results

    async def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query text."""
        # Simplified entity extraction - could be enhanced with NLP
        # For now, use basic keyword extraction
        words = query.lower().split()
        # Filter common words and return potential entities
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        entities = [word for word in words if word not in stop_words and len(word) > 2]
        return entities[:5]  # Limit to top 5 potential entities

    def _calculate_graph_score(self, entity: Entity, query: str) -> float:
        """Calculate relevance score for graph entity."""
        # Simplified scoring based on property matching
        score = 0.0
        query_words = set(query.lower().split())

        # Check entity properties for query word matches
        for prop_value in entity.properties.values():
            if isinstance(prop_value, str):
                prop_words = set(prop_value.lower().split())
                matches = len(query_words.intersection(prop_words))
                score += matches * 0.1

        # Boost score based on entity importance (relationship count)
        importance = entity.properties.get("relationship_count", 0)
        score += min(importance * 0.05, 0.3)

        return min(score, 1.0)

    def _analyze_query(self, query: str) -> Dict[str, bool]:
        """Analyze query to determine best retrieval strategy."""
        words = query.lower().split()

        # Check for entity indicators
        entity_indicators = {"who", "what", "where", "when", "which"}
        entity_rich = any(word in entity_indicators for word in words)

        # Check for semantic complexity
        semantic_complex = len(words) > 10 or any(
            word in ["relationship", "connection", "related", "similar", "compare"]
            for word in words
        )

        return {"entity_rich": entity_rich, "semantic_complex": semantic_complex}

    def _deduplicate_results(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Remove duplicate results based on content similarity."""
        if len(results) <= 1:
            return results

        unique_results = []
        for result in results:
            is_duplicate = False
            for unique in unique_results:
                similarity = self._calculate_content_similarity(
                    result.content, unique.content
                )
                if similarity > 0.9:  # 90% similarity threshold
                    is_duplicate = True
                    # Keep the higher scoring result
                    if result.score > unique.score:
                        unique_results.remove(unique)
                        unique_results.append(result)
                    break

            if not is_duplicate:
                unique_results.append(result)

        return unique_results

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        # Simple Jaccard similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0
