"""Main embedder implementation with intelligent fallback support."""

import asyncio
from typing import List, Optional, Dict, Any

import numpy as np
import torch

from ....utils.logger import get_logger
from ....utils.config import settings
from ...interfaces.embeddings import EmbeddingProvider
from ...observability.tracing import trace_method
from .registry import get_embedding_provider

logger = get_logger(__name__)


class Embedder:
    """Intelligent embedder with automatic fallback and optimization."""
    
    def __init__(
        self,
        primary_provider: Optional[str] = None,
        fallback_provider: Optional[str] = None,
        batch_size: int = 32,
        max_retries: int = 3,
        device: Optional[str] = None
    ):
        """Initialize embedder with primary and fallback providers.
        
        Args:
            primary_provider: Name of primary embedding provider
            fallback_provider: Name of fallback provider
            batch_size: Batch size for processing
            max_retries: Maximum retry attempts
            device: Device preference (cuda/cpu/auto)
        """
        # Get provider names from config if not specified
        self.primary_provider_name = primary_provider or settings.embeddings.primary.provider
        self.fallback_provider_name = fallback_provider or settings.embeddings.fallback.provider
        
        # Initialize providers
        self.primary_provider: Optional[EmbeddingProvider] = None
        self.fallback_provider: Optional[EmbeddingProvider] = None
        
        # Settings
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.device = device or self._detect_device()
        
        # Statistics
        self.fallback_count = 0
        self.total_embeddings = 0
        
    async def initialize(self) -> None:
        """Initialize embedding providers."""
        # Initialize primary provider
        try:
            self.primary_provider = get_embedding_provider(self.primary_provider_name)
            primary_config = self._get_provider_config(self.primary_provider_name)
            primary_config['device'] = self.device
            await self.primary_provider.initialize(primary_config)
            logger.info(f"Primary embedding provider '{self.primary_provider_name}' initialized")
        except Exception as e:
            logger.error(f"Failed to initialize primary provider: {e}")
            
        # Initialize fallback provider
        try:
            self.fallback_provider = get_embedding_provider(self.fallback_provider_name)
            fallback_config = self._get_provider_config(self.fallback_provider_name)
            # Force fallback to CPU to save GPU memory
            fallback_config['device'] = 'cpu'
            await self.fallback_provider.initialize(fallback_config)
            logger.info(f"Fallback embedding provider '{self.fallback_provider_name}' initialized")
        except Exception as e:
            logger.error(f"Failed to initialize fallback provider: {e}")
            
        if not self.primary_provider and not self.fallback_provider:
            raise RuntimeError("No embedding providers available")
            
    async def close(self) -> None:
        """Clean up providers."""
        if self.primary_provider:
            await self.primary_provider.close()
        if self.fallback_provider:
            await self.fallback_provider.close()
            
    @trace_method("embed_texts")
    async def embed_texts(
        self,
        texts: List[str],
        use_fallback: bool = False
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with automatic fallback.
        
        Args:
            texts: List of texts to embed
            use_fallback: Force use of fallback provider
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        self.total_embeddings += len(texts)
        
        # Choose provider
        provider = self._select_provider(use_fallback)
        
        # Process in batches
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._embed_batch_with_retry(batch, provider)
            embeddings.extend(batch_embeddings)
            
        return embeddings
        
    @trace_method("embed_query")
    async def embed_query(
        self,
        query: str,
        use_fallback: bool = False
    ) -> np.ndarray:
        """Generate embedding for a single query.
        
        Args:
            query: Query text
            use_fallback: Force use of fallback provider
            
        Returns:
            Embedding vector
        """
        provider = self._select_provider(use_fallback)
        
        # Try primary first
        for attempt in range(self.max_retries):
            try:
                return await provider.embed_query(query)
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                
                # Try fallback on last attempt
                if attempt == self.max_retries - 1 and provider == self.primary_provider and self.fallback_provider:
                    logger.info("Switching to fallback provider for query embedding")
                    self.fallback_count += 1
                    return await self.fallback_provider.embed_query(query)
                    
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                
        raise RuntimeError(f"Failed to generate embedding after {self.max_retries} attempts")
        
    async def _embed_batch_with_retry(
        self,
        texts: List[str],
        provider: EmbeddingProvider
    ) -> List[np.ndarray]:
        """Embed a batch of texts with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return await provider.embed_texts(texts, batch_size=min(len(texts), self.batch_size))
            except Exception as e:
                logger.warning(f"Batch embedding attempt {attempt + 1} failed: {e}")
                
                # Try fallback on last attempt
                if attempt == self.max_retries - 1 and provider == self.primary_provider and self.fallback_provider:
                    logger.info(f"Switching to fallback provider for batch of {len(texts)} texts")
                    self.fallback_count += len(texts)
                    return await self.fallback_provider.embed_texts(texts, batch_size=min(len(texts), self.batch_size))
                    
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                
        raise RuntimeError(f"Failed to embed batch after {self.max_retries} attempts")
        
    def _select_provider(self, use_fallback: bool) -> EmbeddingProvider:
        """Select appropriate provider based on conditions."""
        if use_fallback and self.fallback_provider:
            return self.fallback_provider
            
        if self.primary_provider and self._check_provider_health(self.primary_provider):
            return self.primary_provider
            
        if self.fallback_provider:
            logger.warning("Primary provider unhealthy, using fallback")
            return self.fallback_provider
            
        raise RuntimeError("No healthy embedding providers available")
        
    def _check_provider_health(self, provider: EmbeddingProvider) -> bool:
        """Check if provider is healthy and has resources."""
        try:
            # Check GPU memory if using GPU
            if provider.supports_gpu() and self.device == 'cuda':
                if torch.cuda.is_available():
                    # Check available memory (keep 20% buffer)
                    free_memory = torch.cuda.mem_get_info()[0]
                    total_memory = torch.cuda.mem_get_info()[1]
                    if free_memory / total_memory < 0.2:
                        logger.warning("Low GPU memory, provider may fail")
                        return False
                        
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
            
    def _detect_device(self) -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            logger.info("CUDA available, using GPU for embeddings")
            return 'cuda'
        else:
            logger.info("CUDA not available, using CPU for embeddings")
            return 'cpu'
            
    def _get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        # Get configuration from settings
        if provider_name == settings.embeddings.primary.provider:
            config = {
                'model_name': settings.embeddings.primary.model_name,
                'device': self.device,
                'batch_size': settings.embeddings.primary.batch_size or self.batch_size,
                'max_length': getattr(settings.embeddings.primary, 'max_length', 512),
                'normalize_embeddings': getattr(settings.embeddings.primary, 'normalize_embeddings', True)
            }
            
            # Add local model path if specified
            if hasattr(settings.embeddings.primary, 'model_path') and settings.embeddings.primary.model_path:
                config['model_path'] = settings.embeddings.primary.model_path
                config['use_local_files'] = getattr(settings.embeddings.primary, 'use_local_files', True)
                
            return config
            
        elif provider_name == settings.embeddings.fallback.provider:
            config = {
                'model_name': settings.embeddings.fallback.model_name,
                'device': 'cpu',  # Always use CPU for fallback
                'batch_size': getattr(settings.embeddings.fallback, 'batch_size', self.batch_size),
                'max_length': getattr(settings.embeddings.fallback, 'max_length', 384),
                'normalize_embeddings': getattr(settings.embeddings.fallback, 'normalize_embeddings', True)
            }
            
            # Add local model path if specified
            if hasattr(settings.embeddings.fallback, 'model_path') and settings.embeddings.fallback.model_path:
                config['model_path'] = settings.embeddings.fallback.model_path
                config['use_local_files'] = getattr(settings.embeddings.fallback, 'use_local_files', True)
                
            return config
        else:
            return {}
            
    def get_dimension(self) -> int:
        """Get embedding dimension from active provider."""
        if self.primary_provider:
            return self.primary_provider.get_dimension()
        elif self.fallback_provider:
            return self.fallback_provider.get_dimension()
        else:
            raise RuntimeError("No providers initialized")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get embedder statistics."""
        return {
            'total_embeddings': self.total_embeddings,
            'fallback_count': self.fallback_count,
            'fallback_rate': self.fallback_count / max(1, self.total_embeddings),
            'primary_provider': self.primary_provider_name,
            'fallback_provider': self.fallback_provider_name,
            'device': self.device
        }
        
    async def warm_up(self) -> None:
        """Warm up models by running test embeddings."""
        test_texts = ["This is a test sentence for warming up the embedding model."]
        
        try:
            # Warm up primary
            if self.primary_provider:
                await self.primary_provider.embed_texts(test_texts)
                logger.info("Primary provider warmed up")
                
            # Warm up fallback
            if self.fallback_provider:
                await self.fallback_provider.embed_texts(test_texts)
                logger.info("Fallback provider warmed up")
                
        except Exception as e:
            logger.warning(f"Error during warm-up: {e}")


# Singleton instance
_embedder_instance: Optional[Embedder] = None


async def get_embedder() -> Embedder:
    """Get or create the singleton embedder instance."""
    global _embedder_instance
    
    if _embedder_instance is None:
        _embedder_instance = Embedder()
        await _embedder_instance.initialize()
        await _embedder_instance.warm_up()
        
    return _embedder_instance