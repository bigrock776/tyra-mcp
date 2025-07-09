"""Reranking system for improving retrieval accuracy."""

import asyncio
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from ...utils.logger import get_logger
from ...utils.circuit_breaker import circuit_breaker
from ..interfaces.reranker import Reranker as RerankerInterface
from ..observability.tracing import trace_method
from ..cache.redis_cache import CacheManager

logger = get_logger(__name__)


class CrossEncoderReranker(RerankerInterface):
    """Cross-encoder based reranking implementation."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_manager: Optional[CacheManager] = None,
        cache_ttl: int = 3600
    ):
        """Initialize cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name
            device: Device to use (cuda/cpu/auto)
            batch_size: Batch size for processing
            cache_manager: Optional cache for results
            cache_ttl: Cache TTL in seconds
        """
        self.model_name = model_name
        self.device = device or self._detect_device()
        self.batch_size = batch_size
        self.cache_manager = cache_manager
        self.cache_ttl = cache_ttl
        self.model: Optional[CrossEncoder] = None
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the reranker model."""
        try:
            # Override with config values if provided
            self.model_name = config.get('model_name', self.model_name)
            self.device = config.get('device', self.device)
            self.batch_size = config.get('batch_size', self.batch_size)
            
            # Load model
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512
            )
            
            logger.info(f"Cross-encoder reranker initialized with model '{self.model_name}' on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            raise
            
    async def close(self) -> None:
        """Clean up resources."""
        # Cross-encoder doesn't need explicit cleanup
        self.model = None
        
    @trace_method("rerank")
    @circuit_breaker(failure_threshold=3, recovery_timeout=30)
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[float]:
        """Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Optional limit on results (not used in scoring)
            
        Returns:
            List of relevance scores for each document
        """
        if not self.model:
            raise RuntimeError("Reranker not initialized")
            
        if not documents:
            return []
            
        # Check cache
        cache_key = self._get_cache_key(query, documents)
        if self.cache_manager:
            cached = await self.cache_manager.get(cache_key)
            if cached:
                logger.debug("Reranking cache hit")
                return cached
                
        # Prepare query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score in batches
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            
            # Run scoring in thread pool to avoid blocking
            scores = await asyncio.get_event_loop().run_in_executor(
                None,
                self._score_batch,
                batch
            )
            all_scores.extend(scores)
            
        # Normalize scores to 0-1 range
        scores_array = np.array(all_scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score > min_score:
            normalized_scores = (scores_array - min_score) / (max_score - min_score)
        else:
            normalized_scores = scores_array
            
        scores_list = normalized_scores.tolist()
        
        # Cache results
        if self.cache_manager:
            await self.cache_manager.set(cache_key, scores_list, ttl=self.cache_ttl)
            
        return scores_list
        
    async def rerank_with_scores(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Rerank and return indices with scores.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top results to return
            
        Returns:
            List of (index, score) tuples sorted by score
        """
        scores = await self.rerank(query, documents, top_k)
        
        # Create index-score pairs
        indexed_scores = list(enumerate(scores))
        
        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to top_k if specified
        if top_k:
            indexed_scores = indexed_scores[:top_k]
            
        return indexed_scores
        
    def _score_batch(self, pairs: List[List[str]]) -> List[float]:
        """Score a batch of query-document pairs."""
        if not self.model:
            raise RuntimeError("Model not initialized")
            
        # Use cross-encoder to score pairs
        scores = self.model.predict(pairs, convert_to_numpy=True)
        
        # Apply sigmoid to get probabilities
        scores = 1 / (1 + np.exp(-scores))
        
        return scores.tolist()
        
    def _detect_device(self) -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
            
    def _get_cache_key(self, query: str, documents: List[str]) -> str:
        """Generate cache key for query and documents."""
        # Use first 100 chars of each document for key
        doc_fingerprints = [doc[:100] for doc in documents[:5]]  # Limit to first 5
        key_parts = [query] + doc_fingerprints
        return f"rerank:{hash(tuple(key_parts))}"
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranker model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'batch_size': self.batch_size,
            'initialized': self.model is not None
        }


# Import the vLLM reranker
from .vllm_reranker import VLLMReranker

# Alias for backward compatibility
LLMReranker = VLLMReranker


class HybridReranker(RerankerInterface):
    """Hybrid reranker combining multiple reranking strategies."""
    
    def __init__(
        self,
        rerankers: List[RerankerInterface],
        weights: Optional[List[float]] = None
    ):
        """Initialize hybrid reranker.
        
        Args:
            rerankers: List of reranker instances
            weights: Optional weights for each reranker
        """
        self.rerankers = rerankers
        self.weights = weights or [1.0 / len(rerankers)] * len(rerankers)
        
        if len(self.weights) != len(self.rerankers):
            raise ValueError("Number of weights must match number of rerankers")
            
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize all rerankers."""
        for i, reranker in enumerate(self.rerankers):
            await reranker.initialize(config.get(f'reranker_{i}', {}))
            
    async def close(self) -> None:
        """Close all rerankers."""
        for reranker in self.rerankers:
            await reranker.close()
            
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[float]:
        """Rerank using weighted combination of rerankers."""
        # Get scores from all rerankers
        all_scores = await asyncio.gather(*[
            reranker.rerank(query, documents, top_k)
            for reranker in self.rerankers
        ])
        
        # Combine scores with weights
        combined_scores = []
        for doc_idx in range(len(documents)):
            weighted_score = sum(
                scores[doc_idx] * weight
                for scores, weight in zip(all_scores, self.weights)
            )
            combined_scores.append(weighted_score)
            
        return combined_scores