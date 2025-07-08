"""
Advanced hallucination detection and confidence scoring system.

Provides sophisticated grounding analysis, confidence scoring, and safety assessment
for generated responses using multiple validation techniques.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..interfaces.embeddings import EmbeddingProvider
from ..utils.config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for responses."""

    ROCK_SOLID = "rock_solid"  # 95+ - 💪 Safe for automated actions
    HIGH = "high"  # 80-94 - 🧠 Generally reliable
    FUZZY = "fuzzy"  # 60-79 - 🤔 Needs verification
    LOW = "low"  # 0-59 - ⚠️ Not confident


@dataclass
class GroundingEvidence:
    """Evidence supporting a claim in the response."""

    source_chunk: str
    similarity_score: float
    relevance_score: float
    confidence: float
    chunk_id: str
    metadata: Dict[str, Any]


@dataclass
class HallucinationAnalysis:
    """Complete hallucination analysis result."""

    overall_confidence: float
    confidence_level: ConfidenceLevel
    hallucination_flag: bool
    safe_to_act_on: bool
    grounding_score: float
    evidence_count: int
    evidence: List[GroundingEvidence]
    reasoning: str
    warnings: List[str]
    analysis_time: float


class HallucinationDetector:
    """
    Advanced hallucination detection using grounding analysis.

    Features:
    - Multi-level confidence scoring
    - Semantic grounding verification
    - Evidence collection and ranking
    - Trading-safe thresholds
    - Comprehensive analysis reporting
    - Performance optimization
    """

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.settings = get_settings()

        # Load thresholds from configuration
        hallucination_config = self.settings.rag.hallucination
        self.confidence_thresholds = {
            ConfidenceLevel.ROCK_SOLID: hallucination_config.confidence_levels.get(
                "rock_solid", 95
            ),
            ConfidenceLevel.HIGH: hallucination_config.confidence_levels.get(
                "high", 80
            ),
            ConfidenceLevel.FUZZY: hallucination_config.confidence_levels.get(
                "fuzzy", 60
            ),
            ConfidenceLevel.LOW: hallucination_config.confidence_levels.get("low", 0),
        }

        self.hallucination_threshold = hallucination_config.get("threshold", 75)
        self.trading_safe_threshold = 95  # Higher threshold for trading actions

        # Performance tracking
        self._total_analyses: int = 0
        self._total_time: float = 0.0
        self._cache: Dict[str, HallucinationAnalysis] = {}
        self._max_cache_size: int = 1000

    async def analyze_response(
        self,
        response: str,
        retrieved_chunks: List[Dict[str, Any]],
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> HallucinationAnalysis:
        """
        Perform comprehensive hallucination analysis on a response.

        Args:
            response: The generated response to analyze
            retrieved_chunks: Memory chunks used to generate the response
            query: Original query (optional, for additional context)
            context: Additional context for analysis

        Returns:
            Complete hallucination analysis
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._get_cache_key(response, retrieved_chunks)
            if cache_key in self._cache:
                logger.debug("Using cached hallucination analysis")
                return self._cache[cache_key]

            # Extract claims from response
            claims = await self._extract_claims(response)

            # Analyze grounding for each claim
            evidence_list = []
            grounding_scores = []

            for claim in claims:
                evidence, grounding_score = await self._analyze_claim_grounding(
                    claim, retrieved_chunks, query
                )
                evidence_list.extend(evidence)
                grounding_scores.append(grounding_score)

            # Calculate overall metrics
            overall_grounding = np.mean(grounding_scores) if grounding_scores else 0.0
            confidence_score = self._calculate_confidence_score(
                overall_grounding, evidence_list, len(claims)
            )

            # Determine confidence level
            confidence_level = self._get_confidence_level(confidence_score)

            # Check for hallucination
            hallucination_flag = confidence_score < (
                self.hallucination_threshold / 100.0
            )

            # Safety assessment
            safe_to_act_on = confidence_score >= (self.trading_safe_threshold / 100.0)

            # Generate reasoning and warnings
            reasoning = self._generate_reasoning(
                confidence_score, overall_grounding, evidence_list, claims
            )

            warnings = self._generate_warnings(
                confidence_score, overall_grounding, evidence_list
            )

            # Create analysis result
            analysis = HallucinationAnalysis(
                overall_confidence=confidence_score * 100,  # Convert to percentage
                confidence_level=confidence_level,
                hallucination_flag=hallucination_flag,
                safe_to_act_on=safe_to_act_on,
                grounding_score=overall_grounding * 100,
                evidence_count=len(evidence_list),
                evidence=evidence_list,
                reasoning=reasoning,
                warnings=warnings,
                analysis_time=time.time() - start_time,
            )

            # Update cache
            self._cache[cache_key] = analysis
            if len(self._cache) > self._max_cache_size:
                # Remove oldest entries
                oldest_keys = list(self._cache.keys())[: -self._max_cache_size // 2]
                for key in oldest_keys:
                    del self._cache[key]

            # Update performance tracking
            self._total_analyses += 1
            self._total_time += analysis.analysis_time

            logger.debug(
                "Hallucination analysis completed",
                confidence=analysis.overall_confidence,
                level=analysis.confidence_level.value,
                hallucination=analysis.hallucination_flag,
                evidence_count=analysis.evidence_count,
                time=analysis.analysis_time,
            )

            return analysis

        except Exception as e:
            logger.error(
                "Hallucination analysis failed",
                response_length=len(response),
                chunks_count=len(retrieved_chunks),
                error=str(e),
            )

            # Return conservative analysis on error
            return HallucinationAnalysis(
                overall_confidence=0.0,
                confidence_level=ConfidenceLevel.LOW,
                hallucination_flag=True,
                safe_to_act_on=False,
                grounding_score=0.0,
                evidence_count=0,
                evidence=[],
                reasoning=f"Analysis failed due to error: {str(e)}",
                warnings=["Analysis error - treat response as unreliable"],
                analysis_time=time.time() - start_time,
            )

    async def _extract_claims(self, response: str) -> List[str]:
        """Extract factual claims from the response."""
        # Simplified claim extraction - split by sentences
        # In production, use more sophisticated NLP techniques

        import re

        # Split into sentences
        sentences = re.split(r"[.!?]+", response)

        # Filter out very short sentences and clean up
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum length for a meaningful claim
                claims.append(sentence)

        return claims

    async def _analyze_claim_grounding(
        self,
        claim: str,
        retrieved_chunks: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> Tuple[List[GroundingEvidence], float]:
        """Analyze how well a claim is grounded in the retrieved chunks."""
        try:
            # Generate embedding for the claim
            claim_embedding = await self.embedding_provider.embed_query(claim)

            evidence_list = []
            similarity_scores = []

            # Compare claim against each chunk
            for chunk in retrieved_chunks:
                chunk_content = chunk.get("content", "")
                if not chunk_content:
                    continue

                # Generate chunk embedding
                chunk_embedding = await self.embedding_provider.embed_single(
                    chunk_content
                )

                # Calculate semantic similarity
                similarity = self._calculate_cosine_similarity(
                    claim_embedding, chunk_embedding
                )
                similarity_scores.append(similarity)

                # Calculate relevance score (considering context)
                relevance = self._calculate_relevance_score(
                    claim, chunk_content, similarity, query
                )

                # Calculate confidence for this evidence
                confidence = self._calculate_evidence_confidence(similarity, relevance)

                # Create evidence object
                evidence = GroundingEvidence(
                    source_chunk=(
                        chunk_content[:200] + "..."
                        if len(chunk_content) > 200
                        else chunk_content
                    ),
                    similarity_score=similarity,
                    relevance_score=relevance,
                    confidence=confidence,
                    chunk_id=chunk.get("id", "unknown"),
                    metadata=chunk.get("metadata", {}),
                )

                evidence_list.append(evidence)

            # Sort evidence by confidence
            evidence_list.sort(key=lambda x: x.confidence, reverse=True)

            # Calculate overall grounding score
            if similarity_scores:
                # Use the best similarity score, weighted by top evidence
                grounding_score = max(similarity_scores)
                if len(evidence_list) >= 2:
                    # Boost score if multiple sources support the claim
                    top_scores = sorted(similarity_scores, reverse=True)[:3]
                    grounding_score = np.mean(top_scores)
            else:
                grounding_score = 0.0

            return evidence_list, grounding_score

        except Exception as e:
            logger.error(
                "Claim grounding analysis failed", claim=claim[:50], error=str(e)
            )
            return [], 0.0

    def _calculate_cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

            # Ensure result is in valid range
            return max(0.0, min(1.0, float(similarity)))

        except Exception as e:
            logger.error("Cosine similarity calculation failed", error=str(e))
            return 0.0

    def _calculate_relevance_score(
        self,
        claim: str,
        chunk_content: str,
        similarity: float,
        query: Optional[str] = None,
    ) -> float:
        """Calculate relevance score considering multiple factors."""
        try:
            relevance = similarity  # Base on semantic similarity

            # Boost for exact keyword matches
            claim_words = set(claim.lower().split())
            chunk_words = set(chunk_content.lower().split())
            overlap = len(claim_words & chunk_words) / max(len(claim_words), 1)
            relevance += 0.1 * overlap

            # Boost for query relevance if available
            if query:
                query_words = set(query.lower().split())
                query_chunk_overlap = len(query_words & chunk_words) / max(
                    len(query_words), 1
                )
                relevance += 0.05 * query_chunk_overlap

            # Penalty for very long chunks (diluted information)
            if len(chunk_content) > 1000:
                relevance *= 0.95

            return max(0.0, min(1.0, relevance))

        except Exception as e:
            logger.error("Relevance calculation failed", error=str(e))
            return similarity

    def _calculate_evidence_confidence(
        self, similarity: float, relevance: float
    ) -> float:
        """Calculate confidence score for a piece of evidence."""
        # Weighted combination of similarity and relevance
        confidence = 0.7 * similarity + 0.3 * relevance

        # Apply confidence curve (sigmoid-like)
        # This makes the function more sensitive around the decision boundary
        confidence = 1 / (1 + np.exp(-10 * (confidence - 0.5)))

        return float(confidence)

    def _calculate_confidence_score(
        self,
        grounding_score: float,
        evidence_list: List[GroundingEvidence],
        claim_count: int,
    ) -> float:
        """Calculate overall confidence score for the response."""
        try:
            base_confidence = grounding_score

            # Adjust based on evidence quality
            if evidence_list:
                # Consider evidence distribution
                high_confidence_evidence = sum(
                    1 for e in evidence_list if e.confidence > 0.8
                )
                evidence_boost = min(0.2, high_confidence_evidence * 0.05)
                base_confidence += evidence_boost

                # Penalty for conflicting evidence
                evidence_variance = np.var([e.confidence for e in evidence_list[:5]])
                if evidence_variance > 0.1:
                    base_confidence *= 0.95

            # Adjust based on claim count
            if claim_count > 5:
                # Penalty for responses with too many claims (harder to verify)
                base_confidence *= max(0.8, 1.0 - (claim_count - 5) * 0.02)

            return max(0.0, min(1.0, base_confidence))

        except Exception as e:
            logger.error("Confidence calculation failed", error=str(e))
            return 0.0

    def _get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level based on score."""
        confidence_pct = confidence_score * 100

        if confidence_pct >= self.confidence_thresholds[ConfidenceLevel.ROCK_SOLID]:
            return ConfidenceLevel.ROCK_SOLID
        elif confidence_pct >= self.confidence_thresholds[ConfidenceLevel.HIGH]:
            return ConfidenceLevel.HIGH
        elif confidence_pct >= self.confidence_thresholds[ConfidenceLevel.FUZZY]:
            return ConfidenceLevel.FUZZY
        else:
            return ConfidenceLevel.LOW

    def _generate_reasoning(
        self,
        confidence_score: float,
        grounding_score: float,
        evidence_list: List[GroundingEvidence],
        claims: List[str],
    ) -> str:
        """Generate human-readable reasoning for the confidence score."""
        try:
            confidence_pct = confidence_score * 100
            grounding_pct = grounding_score * 100

            reasoning_parts = []

            # Overall assessment
            if confidence_pct >= 95:
                reasoning_parts.append(
                    "Response is highly confident and well-grounded."
                )
            elif confidence_pct >= 80:
                reasoning_parts.append(
                    "Response shows good confidence with solid grounding."
                )
            elif confidence_pct >= 60:
                reasoning_parts.append(
                    "Response has moderate confidence but may need verification."
                )
            else:
                reasoning_parts.append(
                    "Response has low confidence and should be treated cautiously."
                )

            # Grounding details
            reasoning_parts.append(f"Semantic grounding score: {grounding_pct:.1f}%.")

            # Evidence summary
            if evidence_list:
                high_conf_evidence = sum(1 for e in evidence_list if e.confidence > 0.8)
                reasoning_parts.append(
                    f"Found {len(evidence_list)} pieces of evidence, "
                    f"{high_conf_evidence} with high confidence."
                )
            else:
                reasoning_parts.append(
                    "No supporting evidence found in retrieved chunks."
                )

            # Claims analysis
            reasoning_parts.append(f"Response contains {len(claims)} factual claims.")

            return " ".join(reasoning_parts)

        except Exception as e:
            logger.error("Reasoning generation failed", error=str(e))
            return "Unable to generate reasoning due to analysis error."

    def _generate_warnings(
        self,
        confidence_score: float,
        grounding_score: float,
        evidence_list: List[GroundingEvidence],
    ) -> List[str]:
        """Generate warnings based on analysis results."""
        warnings = []

        confidence_pct = confidence_score * 100
        grounding_pct = grounding_score * 100

        # Confidence-based warnings
        if confidence_pct < 60:
            warnings.append("Low confidence - verify information before use")

        if confidence_pct < self.trading_safe_threshold:
            warnings.append("Not suitable for automated trading actions")

        # Grounding-based warnings
        if grounding_pct < 50:
            warnings.append("Poor grounding in source material")

        # Evidence-based warnings
        if not evidence_list:
            warnings.append("No supporting evidence found")
        elif len([e for e in evidence_list if e.confidence > 0.7]) == 0:
            warnings.append("No high-confidence evidence found")

        # Consistency warnings
        if evidence_list:
            confidences = [e.confidence for e in evidence_list[:5]]
            if np.std(confidences) > 0.3:
                warnings.append("Inconsistent evidence quality")

        return warnings

    def _get_cache_key(
        self, response: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """Generate cache key for response and chunks."""
        import hashlib

        # Create hash from response and chunk IDs
        chunk_ids = [chunk.get("id", "") for chunk in retrieved_chunks]
        content = f"{response}|{'|'.join(sorted(chunk_ids))}"

        return hashlib.md5(content.encode()).hexdigest()

    def get_confidence_emoji(self, level: ConfidenceLevel) -> str:
        """Get emoji representation for confidence level."""
        emoji_map = {
            ConfidenceLevel.ROCK_SOLID: "💪",
            ConfidenceLevel.HIGH: "🧠",
            ConfidenceLevel.FUZZY: "🤔",
            ConfidenceLevel.LOW: "⚠️",
        }
        return emoji_map.get(level, "❓")

    def format_confidence_display(self, analysis: HallucinationAnalysis) -> str:
        """Format confidence for user display."""
        emoji = self.get_confidence_emoji(analysis.confidence_level)
        return f"{emoji} {analysis.confidence_level.value.replace('_', ' ').title()} ({analysis.overall_confidence:.1f}%)"

    async def batch_analyze(
        self,
        responses: List[str],
        retrieved_chunks_list: List[List[Dict[str, Any]]],
        queries: Optional[List[str]] = None,
    ) -> List[HallucinationAnalysis]:
        """Analyze multiple responses in batch for efficiency."""
        tasks = []

        for i, (response, chunks) in enumerate(zip(responses, retrieved_chunks_list)):
            query = queries[i] if queries and i < len(queries) else None
            task = self.analyze_response(response, chunks, query)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time = self._total_time / max(self._total_analyses, 1)

        return {
            "total_analyses": self._total_analyses,
            "total_time": self._total_time,
            "average_time_per_analysis": avg_time,
            "analyses_per_second": 1 / max(avg_time, 0.001),
            "cache_size": len(self._cache),
            "cache_hit_potential": len(self._cache) / max(self._total_analyses, 1),
            "configuration": {
                "hallucination_threshold": self.hallucination_threshold,
                "trading_safe_threshold": self.trading_safe_threshold,
                "confidence_thresholds": {
                    level.value: threshold
                    for level, threshold in self.confidence_thresholds.items()
                },
            },
        }

    async def clear_cache(self) -> None:
        """Clear the analysis cache."""
        cache_size = len(self._cache)
        self._cache.clear()

        logger.info("Hallucination detector cache cleared", cleared_entries=cache_size)
