"""Scoring system for RAG responses and confidence calculation."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...utils.logger import get_logger
from ..memory.models import ConfidenceLevel, MemorySearchResult
from ..observability.tracing import trace_method

logger = get_logger(__name__)


class ConfidenceScorer:
    """Calculate confidence scores for RAG responses."""
    
    def __init__(
        self,
        rock_solid_threshold: float = 95.0,
        high_threshold: float = 80.0,
        fuzzy_threshold: float = 60.0,
        source_weight: float = 0.4,
        rerank_weight: float = 0.3,
        hallucination_weight: float = 0.3
    ):
        """Initialize confidence scorer.
        
        Args:
            rock_solid_threshold: Minimum score for rock solid confidence
            high_threshold: Minimum score for high confidence
            fuzzy_threshold: Minimum score for fuzzy confidence
            source_weight: Weight for source relevance scores
            rerank_weight: Weight for reranking scores
            hallucination_weight: Weight for hallucination detection
        """
        self.rock_solid_threshold = rock_solid_threshold
        self.high_threshold = high_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.source_weight = source_weight
        self.rerank_weight = rerank_weight
        self.hallucination_weight = hallucination_weight
        
        # Normalize weights
        total_weight = source_weight + rerank_weight + hallucination_weight
        self.source_weight /= total_weight
        self.rerank_weight /= total_weight
        self.hallucination_weight /= total_weight
        
    @trace_method("calculate_confidence")
    def calculate_confidence(
        self,
        source_scores: List[float],
        rerank_scores: Optional[List[float]] = None,
        hallucination_score: Optional[float] = None,
        source_count: Optional[int] = None
    ) -> Tuple[float, ConfidenceLevel]:
        """Calculate overall confidence score and level.
        
        Args:
            source_scores: Relevance scores from source retrieval
            rerank_scores: Optional reranking scores
            hallucination_score: Optional hallucination detection score (0-1, higher is better)
            source_count: Optional number of relevant sources found
            
        Returns:
            Tuple of (confidence_score, confidence_level)
        """
        if not source_scores:
            return 0.0, ConfidenceLevel.LOW
            
        # Calculate source relevance component
        source_component = self._calculate_source_component(source_scores, source_count)
        
        # Calculate reranking component
        if rerank_scores:
            rerank_component = self._calculate_rerank_component(rerank_scores)
        else:
            # If no reranking, use source scores
            rerank_component = source_component
            
        # Calculate hallucination component
        if hallucination_score is not None:
            hallucination_component = hallucination_score * 100
        else:
            # If no hallucination check, assume moderate confidence
            hallucination_component = 70.0
            
        # Combine components
        confidence_score = (
            source_component * self.source_weight +
            rerank_component * self.rerank_weight +
            hallucination_component * self.hallucination_weight
        )
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(confidence_score)
        
        logger.debug(
            f"Confidence calculation: source={source_component:.1f}, "
            f"rerank={rerank_component:.1f}, hallucination={hallucination_component:.1f}, "
            f"final={confidence_score:.1f} ({confidence_level})"
        )
        
        return confidence_score, confidence_level
        
    def _calculate_source_component(
        self,
        scores: List[float],
        source_count: Optional[int] = None
    ) -> float:
        """Calculate source relevance component."""
        if not scores:
            return 0.0
            
        # Consider top scores
        top_scores = sorted(scores, reverse=True)[:3]
        avg_top_score = np.mean(top_scores) * 100
        
        # Boost if multiple high-quality sources
        if len(top_scores) >= 3 and all(s > 0.8 for s in top_scores):
            avg_top_score = min(avg_top_score * 1.1, 100)
            
        # Penalty for too few sources
        if source_count and source_count < 3:
            avg_top_score *= (0.7 + 0.1 * source_count)
            
        return avg_top_score
        
    def _calculate_rerank_component(self, scores: List[float]) -> float:
        """Calculate reranking component."""
        if not scores:
            return 0.0
            
        # Consider top reranked scores
        top_scores = sorted(scores, reverse=True)[:3]
        return np.mean(top_scores) * 100
        
    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Determine confidence level from score."""
        if score >= self.rock_solid_threshold:
            return ConfidenceLevel.ROCK_SOLID
        elif score >= self.high_threshold:
            return ConfidenceLevel.HIGH
        elif score >= self.fuzzy_threshold:
            return ConfidenceLevel.FUZZY
        else:
            return ConfidenceLevel.LOW
            
    def is_safe_for_action(
        self,
        confidence_score: float,
        confidence_level: ConfidenceLevel,
        action_type: str = "general"
    ) -> bool:
        """Determine if response is safe for automated action.
        
        Args:
            confidence_score: Calculated confidence score
            confidence_level: Confidence level
            action_type: Type of action (general, trading, etc.)
            
        Returns:
            True if safe for action
        """
        # Trading requires rock solid confidence
        if action_type == "trading":
            return confidence_level == ConfidenceLevel.ROCK_SOLID
            
        # General actions need at least high confidence
        return confidence_level in [ConfidenceLevel.ROCK_SOLID, ConfidenceLevel.HIGH]
        
    def score_memory_results(
        self,
        results: List[MemorySearchResult],
        query_length: int
    ) -> Dict[str, float]:
        """Score memory search results for quality metrics.
        
        Args:
            results: Memory search results
            query_length: Length of the original query
            
        Returns:
            Dictionary of quality metrics
        """
        if not results:
            return {
                'coverage': 0.0,
                'diversity': 0.0,
                'recency': 0.0,
                'relevance': 0.0
            }
            
        # Calculate coverage (how well results cover the query)
        total_text_length = sum(len(r.memory.text) for r in results)
        coverage = min(total_text_length / (query_length * 10), 1.0)
        
        # Calculate diversity (variety in results)
        unique_sources = len(set(r.source for r in results))
        diversity = unique_sources / len(results)
        
        # Calculate recency (how recent the memories are)
        now = datetime.utcnow()
        recency_scores = []
        for result in results:
            age_days = (now - result.memory.created_at).days
            recency_score = 1.0 / (1.0 + age_days / 30)  # Decay over 30 days
            recency_scores.append(recency_score)
        recency = np.mean(recency_scores)
        
        # Calculate average relevance
        relevance = np.mean([r.score for r in results])
        
        return {
            'coverage': coverage,
            'diversity': diversity,
            'recency': recency,
            'relevance': relevance
        }
        
    def adjust_confidence_for_context(
        self,
        base_confidence: float,
        context: Dict[str, Any]
    ) -> float:
        """Adjust confidence based on contextual factors.
        
        Args:
            base_confidence: Base confidence score
            context: Additional context (e.g., user history, domain)
            
        Returns:
            Adjusted confidence score
        """
        adjusted = base_confidence
        
        # Boost for frequently accessed memories
        if context.get('access_count', 0) > 10:
            adjusted = min(adjusted * 1.05, 100)
            
        # Penalty for controversial topics
        if context.get('is_controversial', False):
            adjusted *= 0.9
            
        # Boost for verified sources
        if context.get('has_verified_sources', False):
            adjusted = min(adjusted * 1.1, 100)
            
        # Penalty for outdated information
        if context.get('is_outdated', False):
            adjusted *= 0.8
            
        return adjusted
        
    def generate_confidence_report(
        self,
        confidence_score: float,
        confidence_level: ConfidenceLevel,
        components: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate detailed confidence report.
        
        Args:
            confidence_score: Overall confidence score
            confidence_level: Confidence level
            components: Component scores
            metadata: Optional additional metadata
            
        Returns:
            Detailed confidence report
        """
        report = {
            'confidence_score': confidence_score,
            'confidence_level': confidence_level.value,
            'is_rock_solid': confidence_level == ConfidenceLevel.ROCK_SOLID,
            'components': components,
            'thresholds': {
                'rock_solid': self.rock_solid_threshold,
                'high': self.high_threshold,
                'fuzzy': self.fuzzy_threshold
            },
            'weights': {
                'source': self.source_weight,
                'rerank': self.rerank_weight,
                'hallucination': self.hallucination_weight
            }
        }
        
        if metadata:
            report['metadata'] = metadata
            
        return report