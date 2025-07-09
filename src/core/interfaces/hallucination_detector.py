"""Abstract interface for hallucination detection implementations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field


class HallucinationResult(BaseModel):
    """Result from hallucination detection."""
    has_hallucination: bool = Field(..., description="Whether hallucination was detected")
    confidence_score: float = Field(..., ge=0.0, le=100.0, description="Confidence in the response (0-100)")
    grounding_score: float = Field(..., ge=0.0, le=1.0, description="How well grounded the response is (0-1)")
    problematic_statements: List[str] = Field(default_factory=list, description="Statements that may be hallucinated")
    explanation: str = Field("", description="Explanation of the detection result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional detection metadata")


class HallucinationDetector(ABC):
    """Abstract base class for hallucination detection."""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the hallucination detector.
        
        Args:
            config: Configuration dictionary
        """
        pass
        
    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass
        
    @abstractmethod
    async def detect(
        self,
        query: str,
        response: str,
        sources: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> HallucinationResult:
        """Detect hallucinations in a response.
        
        Args:
            query: Original query
            response: Generated response to check
            sources: Source documents used for generation
            context: Optional additional context
            
        Returns:
            HallucinationResult with detection details
        """
        pass
        
    @abstractmethod
    async def check_statement(
        self,
        statement: str,
        sources: List[str]
    ) -> Tuple[bool, float]:
        """Check if a single statement is grounded in sources.
        
        Args:
            statement: Statement to check
            sources: Source documents
            
        Returns:
            Tuple of (is_grounded, confidence)
        """
        pass
        
    async def batch_detect(
        self,
        queries: List[str],
        responses: List[str],
        sources_list: List[List[str]],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[HallucinationResult]:
        """Detect hallucinations in multiple responses.
        
        Args:
            queries: List of queries
            responses: List of responses
            sources_list: List of source lists
            contexts: Optional list of contexts
            
        Returns:
            List of HallucinationResult
        """
        # Default implementation processes sequentially
        results = []
        contexts = contexts or [None] * len(queries)
        
        for query, response, sources, context in zip(queries, responses, sources_list, contexts):
            result = await self.detect(query, response, sources, context)
            results.append(result)
            
        return results
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get detector capabilities and configuration.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            'supports_batch': True,
            'supports_streaming': False,
            'max_response_length': 10000,
            'max_sources': 50
        }