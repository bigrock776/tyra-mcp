"""
Abstract interface for reranking systems.

This module defines the standard interface that all reranking providers must implement,
enabling easy swapping of reranking models without changing core logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RerankerType(Enum):
    """Types of reranking models."""

    CROSS_ENCODER = "cross_encoder"
    LLM_BASED = "llm_based"
    NEURAL = "neural"
    STATISTICAL = "statistical"


@dataclass
class RerankingCandidate:
    """A candidate document for reranking."""

    id: str
    content: str
    original_score: float
    metadata: Dict[str, Any]


@dataclass
class RerankingResult:
    """Result from reranking operation."""

    id: str
    content: str
    original_score: float
    rerank_score: float
    final_score: float
    metadata: Dict[str, Any]
    explanation: Optional[str] = None


class Reranker(ABC):
    """Abstract base class for reranking systems."""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the reranker with configuration.

        Args:
            config: Reranker-specific configuration dictionary
        """
        pass

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: List[RerankingCandidate],
        top_k: Optional[int] = None,
    ) -> List[RerankingResult]:
        """
        Rerank a list of candidate documents for a given query.

        Args:
            query: The search query
            candidates: List of candidate documents to rerank
            top_k: Optional limit on number of results to return

        Returns:
            List of reranking results ordered by final score (descending)
        """
        pass

    @abstractmethod
    async def score_pair(self, query: str, document: str) -> float:
        """
        Score the relevance between a query and a single document.

        Args:
            query: The search query
            document: The document content

        Returns:
            Relevance score (typically between 0 and 1)
        """
        pass

    @abstractmethod
    def get_reranker_type(self) -> RerankerType:
        """
        Get the type of this reranker.

        Returns:
            RerankerType enum value
        """
        pass

    @abstractmethod
    def supports_batch_reranking(self) -> bool:
        """
        Check if this reranker supports efficient batch processing.

        Returns:
            True if batch processing is supported
        """
        pass

    @abstractmethod
    def get_max_candidates(self) -> int:
        """
        Get the maximum number of candidates this reranker can handle.

        Returns:
            Maximum number of candidates
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the reranker.

        Returns:
            Dictionary with health status information
        """
        pass

    async def rerank_with_fusion(
        self,
        query: str,
        candidates: List[RerankingCandidate],
        fusion_method: str = "rrf",  # "rrf" (Reciprocal Rank Fusion), "weighted_sum"
        top_k: Optional[int] = None,
    ) -> List[RerankingResult]:
        """
        Rerank candidates using score fusion with original scores.

        Args:
            query: The search query
            candidates: List of candidate documents to rerank
            fusion_method: Method to combine original and rerank scores
            top_k: Optional limit on number of results to return

        Returns:
            List of reranking results with fused scores
        """
        # Get rerank scores
        rerank_results = await self.rerank(query, candidates, top_k=None)

        # Apply fusion
        if fusion_method == "rrf":
            # Reciprocal Rank Fusion
            k = 60  # RRF parameter

            # Sort by original scores and assign ranks
            orig_sorted = sorted(
                candidates, key=lambda x: x.original_score, reverse=True
            )
            orig_ranks = {doc.id: i + 1 for i, doc in enumerate(orig_sorted)}

            # Sort by rerank scores and assign ranks
            rerank_sorted = sorted(
                rerank_results, key=lambda x: x.rerank_score, reverse=True
            )
            rerank_ranks = {doc.id: i + 1 for i, doc in enumerate(rerank_sorted)}

            # Compute RRF scores
            for result in rerank_results:
                orig_rank = orig_ranks.get(result.id, len(candidates) + 1)
                rerank_rank = rerank_ranks.get(result.id, len(rerank_results) + 1)
                result.final_score = 1 / (k + orig_rank) + 1 / (k + rerank_rank)

        elif fusion_method == "weighted_sum":
            # Weighted sum of normalized scores
            alpha = 0.3  # Weight for original score
            beta = 0.7  # Weight for rerank score

            # Normalize scores to [0, 1]
            if candidates:
                max_orig = max(c.original_score for c in candidates)
                min_orig = min(c.original_score for c in candidates)
                orig_range = max_orig - min_orig if max_orig > min_orig else 1

            if rerank_results:
                max_rerank = max(r.rerank_score for r in rerank_results)
                min_rerank = min(r.rerank_score for r in rerank_results)
                rerank_range = max_rerank - min_rerank if max_rerank > min_rerank else 1

            for result in rerank_results:
                norm_orig = (result.original_score - min_orig) / orig_range
                norm_rerank = (result.rerank_score - min_rerank) / rerank_range
                result.final_score = alpha * norm_orig + beta * norm_rerank

        # Sort by final score and apply top_k
        rerank_results.sort(key=lambda x: x.final_score, reverse=True)
        if top_k:
            rerank_results = rerank_results[:top_k]

        return rerank_results

    async def batch_score_pairs(self, query: str, documents: List[str]) -> List[float]:
        """
        Score multiple query-document pairs efficiently.

        Args:
            query: The search query
            documents: List of document contents

        Returns:
            List of relevance scores in same order as documents
        """
        if self.supports_batch_reranking():
            # Use batch reranking if supported
            candidates = [
                RerankingCandidate(
                    id=str(i), content=doc, original_score=0.0, metadata={}
                )
                for i, doc in enumerate(documents)
            ]
            results = await self.rerank(query, candidates)
            # Return scores in original order
            score_map = {int(r.id): r.rerank_score for r in results}
            return [score_map.get(i, 0.0) for i in range(len(documents))]
        else:
            # Fall back to individual scoring
            scores = []
            for doc in documents:
                score = await self.score_pair(query, doc)
                scores.append(score)
            return scores


class RerankerError(Exception):
    """Base exception for reranker errors."""

    pass


class RerankerInitializationError(RerankerError):
    """Raised when reranker initialization fails."""

    pass


class RerankerOperationError(RerankerError):
    """Raised when reranking operation fails."""

    pass


class RerankerConfigurationError(RerankerError):
    """Raised when reranker configuration is invalid."""

    pass
