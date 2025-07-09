"""
Central embedding manager that coordinates all embedding operations.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..providers.embeddings.registry import get_embedding_provider
from ..cache.cache_manager import CacheManager
from ..utils.config import get_config
from .models import (
    EmbeddingRequest,
    EmbeddingResponse,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
)

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Central manager for all embedding operations.
    
    Coordinates between different embedding providers, handles caching,
    fallback strategies, and batch processing.
    """

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager
        self.config = get_config()
        self._primary_provider = None
        self._fallback_provider = None
        
    async def initialize(self):
        """Initialize embedding providers."""
        try:
            # Get primary provider
            primary_name = self.config.get("embeddings.primary.provider", "huggingface")
            self._primary_provider = await get_embedding_provider(primary_name)
            
            # Get fallback provider
            fallback_name = self.config.get("embeddings.fallback.provider", "huggingface_fallback")
            self._fallback_provider = await get_embedding_provider(fallback_name)
            
            logger.info(f"Embedding manager initialized with primary: {primary_name}, fallback: {fallback_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding providers: {e}")
            raise

    async def generate_embedding(
        self,
        request: Union[str, EmbeddingRequest]
    ) -> EmbeddingResponse:
        """Generate embedding for a single text."""
        if isinstance(request, str):
            request = EmbeddingRequest(text=request)
            
        start_time = time.time()
        
        # Check cache first
        if self.cache_manager and request.cache_key:
            cached_embedding = await self.cache_manager.get_embedding(request.cache_key)
            if cached_embedding is not None:
                return EmbeddingResponse(
                    embedding=cached_embedding,
                    model="cached",
                    dimensions=len(cached_embedding),
                    cached=True,
                    processing_time=time.time() - start_time
                )
        
        # Try primary provider
        try:
            embedding = await self._primary_provider.generate_embedding(request.text)
            model_name = self._primary_provider.model_name
            
        except Exception as e:
            logger.warning(f"Primary embedding provider failed: {e}, trying fallback")
            
            # Try fallback provider
            try:
                embedding = await self._fallback_provider.generate_embedding(request.text)
                model_name = self._fallback_provider.model_name
                
            except Exception as e2:
                logger.error(f"Both embedding providers failed. Primary: {e}, Fallback: {e2}")
                raise RuntimeError("All embedding providers failed")
        
        # Normalize if requested
        if request.normalize:
            embedding = embedding / np.linalg.norm(embedding)
        
        # Cache the result
        if self.cache_manager and request.cache_key:
            await self.cache_manager.cache_embedding(request.cache_key, embedding)
        
        return EmbeddingResponse(
            embedding=embedding,
            model=model_name,
            dimensions=len(embedding),
            cached=False,
            processing_time=time.time() - start_time,
            metadata=request.metadata
        )

    async def generate_batch_embeddings(
        self,
        request: Union[List[str], BatchEmbeddingRequest]
    ) -> BatchEmbeddingResponse:
        """Generate embeddings for multiple texts."""
        if isinstance(request, list):
            request = BatchEmbeddingRequest(texts=request)
            
        start_time = time.time()
        batch_size = request.batch_size or self.config.get("embeddings.batch_size", 32)
        
        embeddings = []
        cached_count = 0
        
        # Process in batches
        for i in range(0, len(request.texts), batch_size):
            batch_texts = request.texts[i:i + batch_size]
            batch_requests = [
                EmbeddingRequest(
                    text=text,
                    model=request.model,
                    normalize=request.normalize,
                    cache_key=f"batch_{hash(text)}" if self.cache_manager else None
                )
                for text in batch_texts
            ]
            
            # Generate embeddings for batch
            batch_embeddings = await asyncio.gather(*[
                self.generate_embedding(req) for req in batch_requests
            ])
            
            # Extract embeddings and count cached
            for response in batch_embeddings:
                embeddings.append(response.embedding)
                if response.cached:
                    cached_count += 1
        
        # Get model name from first embedding (they should all be the same)
        model_name = "unknown"
        if embeddings:
            # Try to get model from a non-cached response
            for i, text in enumerate(request.texts):
                req = EmbeddingRequest(text=text, model=request.model, normalize=request.normalize)
                try:
                    if self._primary_provider:
                        model_name = self._primary_provider.model_name
                        break
                except:
                    if self._fallback_provider:
                        model_name = self._fallback_provider.model_name
                        break
        
        return BatchEmbeddingResponse(
            embeddings=embeddings,
            model=model_name,
            dimensions=len(embeddings[0]) if embeddings else 0,
            processing_time=time.time() - start_time,
            cached_count=cached_count,
            total_count=len(request.texts),
            metadata=request.metadata
        )

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about available embedding models."""
        info = {
            "primary": None,
            "fallback": None,
            "cache_enabled": self.cache_manager is not None
        }
        
        if self._primary_provider:
            info["primary"] = {
                "name": self._primary_provider.model_name,
                "dimensions": getattr(self._primary_provider, "dimensions", "unknown"),
                "device": getattr(self._primary_provider, "device", "unknown")
            }
            
        if self._fallback_provider:
            info["fallback"] = {
                "name": self._fallback_provider.model_name,
                "dimensions": getattr(self._fallback_provider, "dimensions", "unknown"),
                "device": getattr(self._fallback_provider, "device", "unknown")
            }
            
        return info

    async def health_check(self) -> Dict[str, Any]:
        """Check health of embedding providers."""
        health = {
            "primary": {"status": "unknown", "error": None},
            "fallback": {"status": "unknown", "error": None},
            "cache": {"status": "unknown", "error": None}
        }
        
        # Check primary provider
        if self._primary_provider:
            try:
                test_embedding = await self._primary_provider.generate_embedding("test")
                health["primary"]["status"] = "healthy" if test_embedding is not None else "unhealthy"
            except Exception as e:
                health["primary"]["status"] = "unhealthy"
                health["primary"]["error"] = str(e)
        
        # Check fallback provider
        if self._fallback_provider:
            try:
                test_embedding = await self._fallback_provider.generate_embedding("test")
                health["fallback"]["status"] = "healthy" if test_embedding is not None else "unhealthy"
            except Exception as e:
                health["fallback"]["status"] = "unhealthy"
                health["fallback"]["error"] = str(e)
        
        # Check cache
        if self.cache_manager:
            try:
                await self.cache_manager.health_check()
                health["cache"]["status"] = "healthy"
            except Exception as e:
                health["cache"]["status"] = "unhealthy"
                health["cache"]["error"] = str(e)
        else:
            health["cache"]["status"] = "disabled"
            
        return health