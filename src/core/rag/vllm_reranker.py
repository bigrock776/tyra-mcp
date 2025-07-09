"""vLLM-based reranking implementation for high-performance document reranking."""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

from ...utils.logger import get_logger
from ...utils.circuit_breaker import circuit_breaker
from ..interfaces.reranker import Reranker as RerankerInterface
from ..observability.tracing import trace_method
from ..cache.cache_manager import CacheManager

logger = get_logger(__name__)


class VLLMReranker(RerankerInterface):
    """vLLM-based reranker using local LLM for document relevance scoring."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        api_key: str = "dummy-key",
        max_tokens: int = 10,
        temperature: float = 0.0,
        timeout: int = 30,
        batch_size: int = 8,
        cache_manager: Optional[CacheManager] = None,
        cache_ttl: int = 3600
    ):
        """Initialize vLLM reranker.
        
        Args:
            base_url: vLLM server base URL
            model_name: Model name for reranking
            api_key: API key (dummy for local vLLM)
            max_tokens: Maximum tokens for response
            temperature: Temperature for generation
            timeout: Request timeout in seconds
            batch_size: Batch size for parallel processing
            cache_manager: Optional cache for results
            cache_ttl: Cache TTL in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.batch_size = batch_size
        self.cache_manager = cache_manager
        self.cache_ttl = cache_ttl
        
        # Session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Performance tracking
        self._total_requests = 0
        self._total_latency = 0.0
        self._error_count = 0
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the vLLM reranker."""
        try:
            # Override with config values if provided
            self.base_url = config.get('base_url', self.base_url).rstrip('/')
            self.model_name = config.get('model_name', self.model_name)
            self.api_key = config.get('api_key', self.api_key)
            self.max_tokens = config.get('max_tokens', self.max_tokens)
            self.temperature = config.get('temperature', self.temperature)
            self.timeout = config.get('timeout', self.timeout)
            self.batch_size = config.get('batch_size', self.batch_size)
            
            # Create HTTP session with connection pooling
            connector = aiohttp.TCPConnector(
                limit=50,  # Total connection pool size
                limit_per_host=20,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
            )
            
            timeout_config = aiohttp.ClientTimeout(total=self.timeout)
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout_config,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
            )
            
            # Test connection
            await self._test_connection()
            
            logger.info(
                f"vLLM reranker initialized",
                base_url=self.base_url,
                model_name=self.model_name,
                batch_size=self.batch_size
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM reranker: {e}")
            raise
            
    async def close(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None
            
    async def _test_connection(self) -> None:
        """Test connection to vLLM server."""
        if not self._session:
            raise RuntimeError("Session not initialized")
            
        try:
            # Simple test prompt
            test_prompt = "Test connection. Respond with: OK"
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": test_prompt}
                ],
                "max_tokens": 5,
                "temperature": 0.0
            }
            
            async with self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(f"vLLM connection test failed: {response.status} - {text}")
                    
                result = await response.json()
                logger.debug("vLLM connection test successful")
                
        except Exception as e:
            logger.error(f"vLLM connection test failed: {e}")
            raise
            
    @trace_method("vllm_rerank")
    @circuit_breaker(failure_threshold=3, recovery_timeout=60)
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[float]:
        """Rerank documents using vLLM scoring.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Optional limit on results (used for optimization)
            
        Returns:
            List of relevance scores for each document
        """
        if not documents:
            return []
            
        if not self._session:
            raise RuntimeError("vLLM reranker not initialized")
            
        # Check cache
        cache_key = self._get_cache_key(query, documents)
        if self.cache_manager:
            cached = await self.cache_manager.get(cache_key, level="rerank")
            if cached:
                logger.debug("vLLM reranking cache hit")
                return cached
                
        start_time = time.time()
        
        try:
            # Process documents in batches for better performance
            all_scores = []
            
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                batch_scores = await self._score_batch(query, batch)
                all_scores.extend(batch_scores)
                
            # Normalize scores to 0-1 range
            scores_array = np.array(all_scores)
            if len(scores_array) > 1:
                min_score = scores_array.min()
                max_score = scores_array.max()
                
                if max_score > min_score:
                    normalized_scores = (scores_array - min_score) / (max_score - min_score)
                else:
                    normalized_scores = scores_array
            else:
                normalized_scores = scores_array
                
            scores_list = normalized_scores.tolist()
            
            # Track performance
            self._total_requests += 1
            latency = time.time() - start_time
            self._total_latency += latency
            
            # Cache results
            if self.cache_manager:
                await self.cache_manager.set(
                    cache_key, 
                    scores_list, 
                    ttl=self.cache_ttl,
                    level="rerank"
                )
                
            logger.debug(
                f"vLLM reranking completed",
                query_length=len(query),
                documents_count=len(documents),
                latency=latency,
                avg_score=np.mean(scores_list)
            )
            
            return scores_list
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"vLLM reranking failed: {e}")
            raise
            
    async def _score_batch(self, query: str, documents: List[str]) -> List[float]:
        """Score a batch of documents using vLLM."""
        if not self._session:
            raise RuntimeError("Session not initialized")
            
        # Create scoring tasks for parallel processing
        tasks = []
        for doc in documents:
            task = self._score_single_document(query, doc)
            tasks.append(task)
            
        # Execute in parallel with concurrency limit
        semaphore = asyncio.Semaphore(min(self.batch_size, 5))
        
        async def limited_score(task):
            async with semaphore:
                return await task
                
        scores = await asyncio.gather(*[limited_score(task) for task in tasks])
        return scores
        
    async def _score_single_document(self, query: str, document: str) -> float:
        """Score a single document using vLLM."""
        # Create relevance scoring prompt
        prompt = self._create_scoring_prompt(query, document)
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a document relevance scorer. Rate document relevance on a scale of 0-10."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": ["\n", ".", "Explanation:", "Reason:"]
        }
        
        try:
            async with self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.warning(f"vLLM scoring request failed: {response.status} - {text}")
                    return 0.5  # Default score
                    
                result = await response.json()
                
                # Extract score from response
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "5")
                score = self._extract_score(content.strip())
                
                # Normalize to 0-1 range
                return min(max(score / 10.0, 0.0), 1.0)
                
        except asyncio.TimeoutError:
            logger.warning("vLLM scoring timeout")
            return 0.5
        except Exception as e:
            logger.warning(f"vLLM scoring error: {e}")
            return 0.5
            
    def _create_scoring_prompt(self, query: str, document: str) -> str:
        """Create relevance scoring prompt."""
        # Truncate document if too long
        max_doc_length = 1000
        if len(document) > max_doc_length:
            document = document[:max_doc_length] + "..."
            
        return f"""Query: "{query}"

Document: "{document}"

Rate how relevant this document is to answering the query on a scale of 0-10:
- 0: Completely irrelevant
- 5: Somewhat related
- 10: Perfectly relevant

Score:"""
        
    def _extract_score(self, content: str) -> float:
        """Extract numerical score from LLM response."""
        try:
            # Try to find a number in the response
            import re
            
            # Look for patterns like "8", "8.5", "8/10", "Score: 8"
            patterns = [
                r'(\d+\.?\d*)/10',  # "8/10" or "8.5/10"
                r'(\d+\.?\d*)',     # Any number
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    score = float(match.group(1))
                    # Handle different scales
                    if score > 10:
                        score = score / 10.0  # Handle percentages
                    return min(max(score, 0.0), 10.0)
                    
            # Default to middle score if no number found
            return 5.0
            
        except (ValueError, TypeError):
            logger.warning(f"Could not extract score from: {content}")
            return 5.0
            
    async def rerank_with_scores(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Rerank and return indices with scores."""
        scores = await self.rerank(query, documents, top_k)
        
        # Create index-score pairs
        indexed_scores = list(enumerate(scores))
        
        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to top_k if specified
        if top_k:
            indexed_scores = indexed_scores[:top_k]
            
        return indexed_scores
        
    def _get_cache_key(self, query: str, documents: List[str]) -> str:
        """Generate cache key for query and documents."""
        # Use first 100 chars of each document for key
        doc_fingerprints = [doc[:50] for doc in documents[:3]]  # Limit to first 3
        key_parts = [query[:100]] + doc_fingerprints
        return f"vllm_rerank:{hash(tuple(key_parts))}"
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the vLLM reranker."""
        return {
            'type': 'vllm_reranker',
            'base_url': self.base_url,
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'initialized': self._session is not None,
            'performance': {
                'total_requests': self._total_requests,
                'avg_latency': self._total_latency / max(1, self._total_requests),
                'error_count': self._error_count,
                'error_rate': self._error_count / max(1, self._total_requests)
            }
        }
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on vLLM reranker."""
        try:
            if not self._session:
                return {"status": "unhealthy", "error": "Not initialized"}
                
            # Test with simple query
            test_query = "test"
            test_doc = "This is a test document."
            
            start_time = time.time()
            score = await self._score_single_document(test_query, test_doc)
            latency = time.time() - start_time
            
            return {
                "status": "healthy",
                "model": self.model_name,
                "base_url": self.base_url,
                "test_score": score,
                "test_latency": latency,
                "performance": self.get_model_info()["performance"]
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model_name,
                "base_url": self.base_url
            }