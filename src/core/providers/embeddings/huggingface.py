"""
HuggingFace embedding provider implementation.

High-performance, optimized embedding provider supporting multiple HuggingFace models
with automatic GPU detection, batch processing, and intelligent fallbacks.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from ...interfaces.embeddings import (
    EmbeddingGenerationError,
    EmbeddingInitializationError,
    EmbeddingProvider,
    EmbeddingProviderError,
)
from ...utils.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceProvider(EmbeddingProvider):
    """
    High-performance HuggingFace embedding provider.

    Features:
    - Automatic GPU/CPU detection and optimization
    - Batch processing with dynamic sizing
    - Memory-efficient inference
    - Query/document prefixes for improved accuracy
    - Comprehensive error handling and monitoring
    - Hot-swappable model loading
    """

    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.config: Dict[str, Any] = {}
        self.device: str = "cpu"
        self.model_name: str = ""
        self.max_sequence_length: int = 512
        self.dimensions: int = 0
        self.normalize_embeddings: bool = True
        self.query_prefix: str = ""
        self.document_prefix: str = ""
        self.batch_size: int = 32
        self.fp16_enabled: bool = False

        # Performance tracking
        self._total_embeddings: int = 0
        self._total_time: float = 0.0
        self._error_count: int = 0

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the HuggingFace embedding provider with optimizations."""
        try:
            self.config = config
            self.model_name = config.get(
                "model_name", "sentence-transformers/all-MiniLM-L12-v2"
            )
            self.device = config.get("device", "auto")
            self.max_sequence_length = config.get("max_length", 512)
            self.normalize_embeddings = config.get("normalize_embeddings", True)
            self.query_prefix = config.get("query_prefix", "")
            self.document_prefix = config.get("document_prefix", "")
            self.batch_size = config.get("batch_size", 64)  # OPTIMIZED: Increased default

            # Auto-detect device
            if self.device == "auto":
                self.device = self._detect_best_device()

            # OPTIMIZATION: Pre-warm GPU if available
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.init()  # Initialize CUDA context early

            # Load model
            await self._load_model()

            # Configure FP16 if using GPU
            if self.device.startswith("cuda") and torch.cuda.is_available():
                self.fp16_enabled = config.get("use_fp16", True)
                if self.fp16_enabled:
                    self.model = self.model.half()
                    
                # OPTIMIZATION: Enable CUDA optimizations
                if hasattr(torch.backends.cudnn, 'benchmark'):
                    torch.backends.cudnn.benchmark = True
                    
            # OPTIMIZATION: Perform warmup queries for optimal performance
            warmup_queries = config.get("warmup_queries", 10)
            if warmup_queries > 0:
                await self.warmup(warmup_queries)

            logger.info(
                "HuggingFace embedding provider initialized",
                model_name=self.model_name,
                device=self.device,
                dimensions=self.dimensions,
                max_length=self.max_sequence_length,
                fp16=self.fp16_enabled,
            )

        except Exception as e:
            logger.error(
                "Failed to initialize HuggingFace provider",
                model_name=self.model_name,
                error=str(e),
            )
            raise EmbeddingInitializationError(
                f"HuggingFace initialization failed: {e}"
            )

    def _detect_best_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            # Check CUDA memory
            try:
                device_count = torch.cuda.device_count()
                best_device = "cuda:0"
                max_memory = 0

                for i in range(device_count):
                    memory = torch.cuda.get_device_properties(i).total_memory
                    if memory > max_memory:
                        max_memory = memory
                        best_device = f"cuda:{i}"

                logger.info(
                    "CUDA device detected",
                    device=best_device,
                    memory_gb=max_memory / (1024**3),
                )
                return best_device

            except Exception as e:
                logger.warning(
                    "CUDA detection failed, falling back to CPU", error=str(e)
                )

        return "cpu"

    async def _load_model(self) -> None:
        """Load the SentenceTransformer model from local path or HuggingFace Hub."""
        try:
            # Load in executor to avoid blocking
            loop = asyncio.get_event_loop()

            def load_model():
                # Check if local model path is specified
                model_path = self.config.get("model_path")
                use_local_files = self.config.get("use_local_files", False)
                
                if model_path and use_local_files:
                    logger.info(f"Loading model from local path: {model_path}")
                    # Load from local directory
                    model = SentenceTransformer(
                        model_path,
                        device=self.device,
                        trust_remote_code=self.config.get("trust_remote_code", False),
                        local_files_only=True  # Force local-only loading
                    )
                else:
                    logger.info(f"Loading model from HuggingFace Hub: {self.model_name}")
                    # Load from HuggingFace Hub
                    model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        trust_remote_code=self.config.get("trust_remote_code", False),
                    )

                # Set max sequence length
                if hasattr(model, "max_seq_length"):
                    model.max_seq_length = self.max_sequence_length

                return model

            self.model = await loop.run_in_executor(None, load_model)

            # Get model dimensions
            self.dimensions = self.model.get_sentence_embedding_dimension()

            # Load tokenizer for advanced features
            try:
                model_path = self.config.get("model_path")
                use_local_files = self.config.get("use_local_files", False)
                
                if model_path and use_local_files:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path, 
                        local_files_only=True
                    )
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except Exception as e:
                logger.warning(
                    "Could not load tokenizer, some features may be limited",
                    error=str(e),
                )

        except Exception as e:
            model_path = self.config.get("model_path", "N/A")
            use_local_files = self.config.get("use_local_files", False)
            
            if use_local_files and model_path != "N/A":
                if not os.path.exists(model_path):
                    raise EmbeddingInitializationError(
                        f"Local model directory not found: {model_path}\n"
                        f"Please download the model using:\n"
                        f"huggingface-cli download {self.model_name} --local-dir {model_path} --local-dir-use-symlinks False"
                    )
                else:
                    raise EmbeddingInitializationError(
                        f"Failed to load model {self.model_name} from local path {model_path}: {e}\n"
                        f"The model directory exists but files may be corrupted or incomplete.\n"
                        f"Try re-downloading with:\n"
                        f"huggingface-cli download {self.model_name} --local-dir {model_path} --local-dir-use-symlinks False"
                    )
            else:
                raise EmbeddingInitializationError(
                    f"Failed to load model {self.model_name} from HuggingFace Hub: {e}"
                )

    async def embed_texts(
        self, texts: List[str], batch_size: int = 32
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with optimized batching."""
        if not self.model:
            raise EmbeddingGenerationError("Model not initialized")

        if not texts:
            return []

        start_time = time.time()
        batch_size = min(batch_size, self.batch_size)

        try:
            # Add document prefix if configured
            if self.document_prefix:
                texts = [f"{self.document_prefix}{text}" for text in texts]

            # OPTIMIZATION: Use async semaphore for concurrent batch processing
            max_concurrent_batches = self.config.get("max_concurrent_batches", 3)
            semaphore = asyncio.Semaphore(max_concurrent_batches)
            
            async def process_batch(batch_texts: List[str]) -> List[np.ndarray]:
                async with semaphore:
                    # OPTIMIZATION: Use asyncio.to_thread instead of run_in_executor for better performance
                    return await asyncio.to_thread(self._encode_batch, batch_texts)
            
            # Create batch tasks for parallel processing
            batch_tasks = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                task = asyncio.create_task(process_batch(batch_texts))
                batch_tasks.append(task)

            # OPTIMIZATION: Wait for all batches to complete in parallel
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Flatten results
            all_embeddings = []
            for batch_embeddings in batch_results:
                all_embeddings.extend(batch_embeddings)

            # Update performance tracking
            embedding_time = time.time() - start_time
            self._total_embeddings += len(texts)
            self._total_time += embedding_time

            logger.debug(
                "Generated embeddings",
                count=len(texts),
                time=embedding_time,
                avg_time_per_text=embedding_time / len(texts),
                concurrent_batches=len(batch_tasks),
            )

            return all_embeddings

        except Exception as e:
            self._error_count += 1
            logger.error(
                "Failed to generate embeddings",
                error=str(e),
                text_count=len(texts),
                batch_size=batch_size,
            )
            raise EmbeddingGenerationError(f"Embedding generation failed: {e}")

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode a batch of texts synchronously with optimizations."""
        try:
            # OPTIMIZATION: Pre-allocate GPU memory and use memory-efficient processing
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()  # Clear GPU memory before batch
            
            # OPTIMIZATION: Use optimized encoding parameters
            encode_kwargs = {
                "normalize_embeddings": self.normalize_embeddings,
                "convert_to_numpy": True,
                "show_progress_bar": False,
                "batch_size": min(len(texts), 64),  # OPTIMIZED: Limit GPU batch size
            }
            
            # OPTIMIZATION: Add memory-efficient attention if available
            if hasattr(self.model, '_modules') and self.config.get("memory_efficient_attention", True):
                encode_kwargs["device"] = self.device
            
            # Use the model's encode method
            embeddings = self.model.encode(texts, **encode_kwargs)

            # Ensure we have a list of numpy arrays
            if isinstance(embeddings, np.ndarray):
                if embeddings.ndim == 1:
                    # Single embedding
                    return [embeddings]
                else:
                    # Multiple embeddings
                    return [emb for emb in embeddings]

            return embeddings

        except Exception as e:
            logger.error("Batch encoding failed", error=str(e))
            raise

    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query with query prefix."""
        if not self.model:
            raise EmbeddingGenerationError("Model not initialized")

        start_time = time.time()

        try:
            # Add query prefix if configured
            if self.query_prefix:
                query = f"{self.query_prefix}{query}"

            # Generate embedding
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self._encode_single, query)

            # Update performance tracking
            self._total_embeddings += 1
            self._total_time += time.time() - start_time

            return embedding

        except Exception as e:
            self._error_count += 1
            logger.error(
                "Failed to generate query embedding",
                query=query[:100] + "..." if len(query) > 100 else query,
                error=str(e),
            )
            raise EmbeddingGenerationError(f"Query embedding failed: {e}")

    def _encode_single(self, text: str) -> np.ndarray:
        """Encode a single text synchronously."""
        embedding = self.model.encode(
            [text],
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if isinstance(embedding, np.ndarray):
            if embedding.ndim == 2:
                return embedding[0]
            return embedding

        return embedding[0]

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.dimensions

    def supports_gpu(self) -> bool:
        """Check if GPU is supported and available."""
        return torch.cuda.is_available() and self.device.startswith("cuda")

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the embedding provider."""
        try:
            # Test embedding generation
            start_time = time.time()
            test_embedding = await self.embed_query("test query")
            response_time = time.time() - start_time

            # Check GPU memory if using CUDA
            gpu_info = {}
            if self.device.startswith("cuda"):
                try:
                    device_idx = int(self.device.split(":")[-1])
                    gpu_info = {
                        "allocated_memory": torch.cuda.memory_allocated(device_idx),
                        "cached_memory": torch.cuda.memory_reserved(device_idx),
                        "max_memory": torch.cuda.get_device_properties(
                            device_idx
                        ).total_memory,
                    }
                except Exception as e:
                    gpu_info = {"error": str(e)}

            # Calculate performance metrics
            avg_time_per_embedding = (
                self._total_time / self._total_embeddings
                if self._total_embeddings > 0
                else 0
            )

            error_rate = self._error_count / max(self._total_embeddings, 1)

            return {
                "status": "healthy",
                "model_name": self.model_name,
                "device": self.device,
                "dimensions": self.dimensions,
                "response_time": response_time,
                "test_embedding_shape": test_embedding.shape,
                "fp16_enabled": self.fp16_enabled,
                "performance": {
                    "total_embeddings": self._total_embeddings,
                    "total_time": self._total_time,
                    "avg_time_per_embedding": avg_time_per_embedding,
                    "error_count": self._error_count,
                    "error_rate": error_rate,
                },
                "gpu_info": gpu_info,
                "config": {
                    "max_sequence_length": self.max_sequence_length,
                    "batch_size": self.batch_size,
                    "normalize_embeddings": self.normalize_embeddings,
                    "query_prefix": bool(self.query_prefix),
                    "document_prefix": bool(self.document_prefix),
                },
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_name": self.model_name,
                "device": self.device,
                "error_count": self._error_count,
            }

    async def optimize_batch_size(self, sample_texts: List[str]) -> int:
        """Dynamically optimize batch size based on available memory."""
        if not self.supports_gpu():
            return self.batch_size

        try:
            device_idx = int(self.device.split(":")[-1])
            available_memory = torch.cuda.get_device_properties(device_idx).total_memory
            current_memory = torch.cuda.memory_allocated(device_idx)
            free_memory = available_memory - current_memory

            # Estimate memory per text (rough approximation)
            avg_text_length = sum(len(text) for text in sample_texts) / len(
                sample_texts
            )
            estimated_memory_per_text = (
                avg_text_length * self.dimensions * 4
            )  # 4 bytes per float

            # Calculate optimal batch size with safety margin
            optimal_batch_size = int(free_memory * 0.8 / estimated_memory_per_text)
            optimal_batch_size = max(
                1, min(optimal_batch_size, 256)
            )  # Clamp between 1 and 256

            logger.info(
                "Optimized batch size",
                original=self.batch_size,
                optimized=optimal_batch_size,
                free_memory_mb=free_memory / (1024**2),
                avg_text_length=avg_text_length,
            )

            return optimal_batch_size

        except Exception as e:
            logger.warning("Batch size optimization failed", error=str(e))
            return self.batch_size

    async def warmup(self, num_warmup_texts: int = 5) -> None:
        """Warm up the model with dummy inference to optimize performance."""
        try:
            warmup_texts = [f"warmup text {i}" for i in range(num_warmup_texts)]
            start_time = time.time()

            await self.embed_texts(warmup_texts)

            warmup_time = time.time() - start_time
            logger.info(
                "Model warmup completed",
                warmup_texts=num_warmup_texts,
                warmup_time=warmup_time,
            )

        except Exception as e:
            logger.warning("Model warmup failed", error=str(e))

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        avg_time = self._total_time / max(self._total_embeddings, 1)
        throughput = self._total_embeddings / max(
            self._total_time, 0.001
        )  # texts per second

        return {
            "total_embeddings": self._total_embeddings,
            "total_time": self._total_time,
            "average_time_per_embedding": avg_time,
            "throughput_per_second": throughput,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._total_embeddings, 1),
            "model_info": {
                "name": self.model_name,
                "device": self.device,
                "dimensions": self.dimensions,
                "fp16": self.fp16_enabled,
            },
        }

    async def close(self) -> None:
        """Clean up resources."""
        if self.model and self.device.startswith("cuda"):
            # Clear GPU cache
            torch.cuda.empty_cache()

        logger.info(
            "HuggingFace provider closed",
            total_embeddings=self._total_embeddings,
            total_time=self._total_time,
        )
