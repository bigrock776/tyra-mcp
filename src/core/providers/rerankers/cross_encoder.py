"""
Cross-encoder reranker implementation.

High-performance reranking using cross-encoder models for improved relevance scoring
with batch processing, caching, and comprehensive monitoring.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from ...interfaces.reranker import (
    Reranker,
    RerankerError,
    RerankerInitializationError,
    RerankerOperationError,
    RerankerType,
    RerankingCandidate,
    RerankingResult,
)
from ...utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker(Reranker):
    """
    High-performance cross-encoder reranker.

    Features:
    - Optimized cross-encoder model inference
    - GPU/CPU auto-detection and optimization
    - Batch processing with dynamic sizing
    - Score caching and normalization
    - Comprehensive performance monitoring
    - Memory-efficient processing
    """

    def __init__(self):
        self.model: Optional[CrossEncoder] = None
        self.config: Dict[str, Any] = {}
        self.device: str = "cpu"
        self.model_name: str = ""
        self.max_sequence_length: int = 512
        self.batch_size: int = 16
        self.fp16_enabled: bool = False
        self.score_range: Tuple[float, float] = (0.0, 1.0)
        self.normalization: str = "sigmoid"

        # Performance tracking
        self._total_pairs: int = 0
        self._total_time: float = 0.0
        self._error_count: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Score cache for repeated query-document pairs
        self._score_cache: Dict[str, float] = {}
        self._max_cache_size: int = 10000

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the cross-encoder reranker."""
        try:
            self.config = config
            self.model_name = config.get(
                "model_name", "cross-encoder/ms-marco-MiniLM-L-12-v2"
            )
            self.device = config.get("device", "auto")
            self.max_sequence_length = config.get("max_length", 512)
            self.batch_size = config.get("batch_size", 16)
            self.score_range = tuple(config.get("score_range", [0.0, 1.0]))
            self.normalization = config.get("normalization", "sigmoid")

            # Auto-detect device
            if self.device == "auto":
                self.device = self._detect_best_device()

            # Load model
            await self._load_model()

            # Configure FP16 if using GPU
            if self.device.startswith("cuda") and torch.cuda.is_available():
                self.fp16_enabled = config.get("use_fp16", True)
                if self.fp16_enabled and hasattr(self.model.model, "half"):
                    self.model.model = self.model.model.half()

            logger.info(
                "Cross-encoder reranker initialized",
                model_name=self.model_name,
                device=self.device,
                max_length=self.max_sequence_length,
                batch_size=self.batch_size,
                fp16=self.fp16_enabled,
            )

        except Exception as e:
            logger.error(
                "Failed to initialize cross-encoder reranker",
                model_name=self.model_name,
                error=str(e),
            )
            raise RerankerInitializationError(
                f"Cross-encoder initialization failed: {e}"
            )

    def _detect_best_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
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
                    "CUDA device detected for reranker",
                    device=best_device,
                    memory_gb=max_memory / (1024**3),
                )
                return best_device

            except Exception as e:
                logger.warning(
                    "CUDA detection failed for reranker, using CPU", error=str(e)
                )

        return "cpu"

    async def _load_model(self) -> None:
        """Load the cross-encoder model from local path or HuggingFace Hub."""
        try:
            # Load in executor to avoid blocking
            loop = asyncio.get_event_loop()

            def load_model():
                # Check if local model path is specified
                model_path = self.config.get("model_path")
                use_local_files = self.config.get("use_local_files", False)
                
                if model_path and use_local_files:
                    logger.info(f"Loading cross-encoder from local path: {model_path}")
                    # Load from local directory
                    model = CrossEncoder(
                        model_path,
                        device=self.device,
                        trust_remote_code=self.config.get("trust_remote_code", False),
                        local_files_only=True  # Force local-only loading
                    )
                else:
                    logger.info(f"Loading cross-encoder from HuggingFace Hub: {self.model_name}")
                    # Load from HuggingFace Hub
                    model = CrossEncoder(
                        self.model_name,
                        device=self.device,
                        trust_remote_code=self.config.get("trust_remote_code", False),
                    )

                # Set max sequence length
                if hasattr(model, "max_length"):
                    model.max_length = self.max_sequence_length

                return model

            self.model = await loop.run_in_executor(None, load_model)

            logger.info(
                "Cross-encoder model loaded",
                model_name=self.model_name,
                device=self.device,
                local_path=self.config.get("model_path", "N/A"),
            )

        except Exception as e:
            model_path = self.config.get("model_path", "N/A")
            use_local_files = self.config.get("use_local_files", False)
            
            if use_local_files and model_path != "N/A":
                if not os.path.exists(model_path):
                    raise RerankerInitializationError(
                        f"Local cross-encoder directory not found: {model_path}\n"
                        f"Please download the model using:\n"
                        f"huggingface-cli download {self.model_name} --local-dir {model_path} --local-dir-use-symlinks False"
                    )
                else:
                    raise RerankerInitializationError(
                        f"Failed to load cross-encoder {self.model_name} from local path {model_path}: {e}\n"
                        f"The model directory exists but files may be corrupted or incomplete.\n"
                        f"Try re-downloading with:\n"
                        f"huggingface-cli download {self.model_name} --local-dir {model_path} --local-dir-use-symlinks False"
                    )
            else:
                raise RerankerInitializationError(
                    f"Failed to load cross-encoder {self.model_name} from HuggingFace Hub: {e}"
                )

    async def rerank(
        self,
        query: str,
        candidates: List[RerankingCandidate],
        top_k: Optional[int] = None,
    ) -> List[RerankingResult]:
        """Rerank candidates using cross-encoder scoring."""
        if not self.model:
            raise RerankerOperationError("Model not initialized")

        if not candidates:
            return []

        start_time = time.time()

        try:
            # Prepare query-document pairs
            pairs = [(query, candidate.content) for candidate in candidates]

            # Check cache for existing scores
            cached_scores = {}
            uncached_pairs = []
            uncached_indices = []

            for i, (q, doc) in enumerate(pairs):
                cache_key = self._get_cache_key(q, doc)
                if cache_key in self._score_cache:
                    cached_scores[i] = self._score_cache[cache_key]
                    self._cache_hits += 1
                else:
                    uncached_pairs.append((q, doc))
                    uncached_indices.append(i)
                    self._cache_misses += 1

            # Compute scores for uncached pairs
            uncached_scores = []
            if uncached_pairs:
                uncached_scores = await self._predict_scores(uncached_pairs)

                # Update cache
                for idx, score in zip(uncached_indices, uncached_scores):
                    cache_key = self._get_cache_key(pairs[idx][0], pairs[idx][1])
                    self._score_cache[cache_key] = score

                # Prune cache if too large
                if len(self._score_cache) > self._max_cache_size:
                    # Remove oldest entries (simple FIFO)
                    keys_to_remove = list(self._score_cache.keys())[
                        : -self._max_cache_size // 2
                    ]
                    for key in keys_to_remove:
                        del self._score_cache[key]

            # Combine cached and computed scores
            all_scores = [0.0] * len(pairs)
            uncached_score_idx = 0

            for i in range(len(pairs)):
                if i in cached_scores:
                    all_scores[i] = cached_scores[i]
                else:
                    all_scores[i] = uncached_scores[uncached_score_idx]
                    uncached_score_idx += 1

            # Create reranking results
            results = []
            for i, (candidate, score) in enumerate(zip(candidates, all_scores)):
                # Normalize score if needed
                normalized_score = self._normalize_score(score)

                results.append(
                    RerankingResult(
                        id=candidate.id,
                        content=candidate.content,
                        original_score=candidate.original_score,
                        rerank_score=normalized_score,
                        final_score=normalized_score,  # Will be adjusted by fusion if needed
                        metadata=candidate.metadata,
                        explanation=f"Cross-encoder score: {normalized_score:.4f}",
                    )
                )

            # Sort by rerank score
            results.sort(key=lambda x: x.rerank_score, reverse=True)

            # Apply top_k limit
            if top_k:
                results = results[:top_k]

            # Update performance tracking
            rerank_time = time.time() - start_time
            self._total_pairs += len(candidates)
            self._total_time += rerank_time

            logger.debug(
                "Reranking completed",
                candidates=len(candidates),
                cached_scores=len(cached_scores),
                computed_scores=len(uncached_pairs),
                time=rerank_time,
                avg_time_per_pair=rerank_time / len(candidates),
            )

            return results

        except Exception as e:
            self._error_count += 1
            logger.error(
                "Reranking failed",
                query=query[:100] + "..." if len(query) > 100 else query,
                candidates=len(candidates),
                error=str(e),
            )
            raise RerankerOperationError(f"Reranking failed: {e}")

    async def _predict_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Predict scores for query-document pairs."""
        if not pairs:
            return []

        try:
            # Process in batches
            all_scores = []

            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]

                # Predict scores in executor
                loop = asyncio.get_event_loop()
                batch_scores = await loop.run_in_executor(
                    None, self._predict_batch, batch_pairs
                )

                all_scores.extend(batch_scores)

                # Log progress for large batches
                if len(pairs) > 50 and (i + self.batch_size) % 50 == 0:
                    logger.debug(
                        "Reranking progress",
                        processed=min(i + self.batch_size, len(pairs)),
                        total=len(pairs),
                    )

            return all_scores

        except Exception as e:
            logger.error("Score prediction failed", error=str(e))
            raise

    def _predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Predict scores for a batch of pairs synchronously."""
        try:
            # Convert to list of lists format expected by CrossEncoder
            formatted_pairs = [[query, doc] for query, doc in pairs]

            # Predict scores
            scores = self.model.predict(formatted_pairs, show_progress_bar=False)

            # Ensure we have a list of floats
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = [float(scores)]

            return [float(score) for score in scores]

        except Exception as e:
            logger.error("Batch prediction failed", error=str(e))
            raise

    def _normalize_score(self, score: float) -> float:
        """Normalize score based on configuration."""
        if self.normalization == "sigmoid":
            # Apply sigmoid to map to [0, 1]
            normalized = 1 / (1 + np.exp(-score))
        elif self.normalization == "minmax":
            # Linear scaling to [0, 1] based on expected range
            min_val, max_val = self.score_range
            normalized = (score - min_val) / (max_val - min_val)
            normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
        elif self.normalization == "softmax":
            # Single value softmax (not very useful, but included for completeness)
            normalized = np.exp(score) / (1 + np.exp(score))
        else:
            # No normalization
            normalized = score

        return float(normalized)

    def _get_cache_key(self, query: str, document: str) -> str:
        """Generate cache key for query-document pair."""
        import hashlib

        content = f"{query}|||{document}"
        return hashlib.md5(content.encode()).hexdigest()

    async def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        if not self.model:
            raise RerankerOperationError("Model not initialized")

        try:
            candidate = RerankingCandidate(
                id="single", content=document, original_score=0.0, metadata={}
            )

            results = await self.rerank(query, [candidate], top_k=1)

            return results[0].rerank_score if results else 0.0

        except Exception as e:
            self._error_count += 1
            logger.error(
                "Single pair scoring failed",
                query=query[:50] + "..." if len(query) > 50 else query,
                error=str(e),
            )
            raise RerankerOperationError(f"Pair scoring failed: {e}")

    def get_reranker_type(self) -> RerankerType:
        """Get the reranker type."""
        return RerankerType.CROSS_ENCODER

    def supports_batch_reranking(self) -> bool:
        """Check if batch reranking is supported."""
        return True

    def get_max_candidates(self) -> int:
        """Get maximum number of candidates that can be processed."""
        # Based on typical GPU memory constraints
        if self.device.startswith("cuda"):
            return 1000  # Conservative estimate for GPU
        else:
            return 500  # Conservative estimate for CPU

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the reranker."""
        try:
            # Test reranking with dummy data
            start_time = time.time()

            test_candidates = [
                RerankingCandidate(
                    id="test1",
                    content="This is a test document",
                    original_score=0.5,
                    metadata={},
                ),
                RerankingCandidate(
                    id="test2",
                    content="Another test document",
                    original_score=0.3,
                    metadata={},
                ),
            ]

            results = await self.rerank("test query", test_candidates)
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
            avg_time_per_pair = self._total_time / max(self._total_pairs, 1)

            cache_hit_rate = self._cache_hits / max(
                self._cache_hits + self._cache_misses, 1
            )

            error_rate = self._error_count / max(self._total_pairs, 1)

            return {
                "status": "healthy",
                "model_name": self.model_name,
                "device": self.device,
                "response_time": response_time,
                "test_results": len(results),
                "fp16_enabled": self.fp16_enabled,
                "performance": {
                    "total_pairs": self._total_pairs,
                    "total_time": self._total_time,
                    "avg_time_per_pair": avg_time_per_pair,
                    "error_count": self._error_count,
                    "error_rate": error_rate,
                    "cache_hit_rate": cache_hit_rate,
                    "cache_size": len(self._score_cache),
                },
                "gpu_info": gpu_info,
                "config": {
                    "max_sequence_length": self.max_sequence_length,
                    "batch_size": self.batch_size,
                    "score_range": self.score_range,
                    "normalization": self.normalization,
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

    async def optimize_batch_size(self, sample_pairs: List[Tuple[str, str]]) -> int:
        """Dynamically optimize batch size based on available memory."""
        if not self.device.startswith("cuda"):
            return self.batch_size

        try:
            device_idx = int(self.device.split(":")[-1])
            available_memory = torch.cuda.get_device_properties(device_idx).total_memory
            current_memory = torch.cuda.memory_allocated(device_idx)
            free_memory = available_memory - current_memory

            # Estimate memory per pair (rough approximation)
            avg_text_length = sum(len(q) + len(d) for q, d in sample_pairs) / len(
                sample_pairs
            )
            estimated_memory_per_pair = avg_text_length * 4 * 2  # Rough estimate

            # Calculate optimal batch size with safety margin
            optimal_batch_size = int(free_memory * 0.6 / estimated_memory_per_pair)
            optimal_batch_size = max(
                1, min(optimal_batch_size, 128)
            )  # Clamp between 1 and 128

            logger.info(
                "Optimized reranker batch size",
                original=self.batch_size,
                optimized=optimal_batch_size,
                free_memory_mb=free_memory / (1024**2),
            )

            return optimal_batch_size

        except Exception as e:
            logger.warning("Reranker batch size optimization failed", error=str(e))
            return self.batch_size

    async def warmup(self, num_warmup_pairs: int = 5) -> None:
        """Warm up the model for optimal performance."""
        try:
            warmup_pairs = [
                (f"warmup query {i}", f"warmup document {i}")
                for i in range(num_warmup_pairs)
            ]

            start_time = time.time()
            await self._predict_scores(warmup_pairs)
            warmup_time = time.time() - start_time

            logger.info(
                "Reranker warmup completed",
                warmup_pairs=num_warmup_pairs,
                warmup_time=warmup_time,
            )

        except Exception as e:
            logger.warning("Reranker warmup failed", error=str(e))

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        avg_time = self._total_time / max(self._total_pairs, 1)
        throughput = self._total_pairs / max(
            self._total_time, 0.001
        )  # pairs per second
        cache_hit_rate = self._cache_hits / max(
            self._cache_hits + self._cache_misses, 1
        )

        return {
            "total_pairs": self._total_pairs,
            "total_time": self._total_time,
            "average_time_per_pair": avg_time,
            "throughput_per_second": throughput,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._total_pairs, 1),
            "cache_statistics": {
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self._score_cache),
                "max_cache_size": self._max_cache_size,
            },
            "model_info": {
                "name": self.model_name,
                "device": self.device,
                "batch_size": self.batch_size,
                "fp16": self.fp16_enabled,
                "normalization": self.normalization,
            },
        }

    async def clear_cache(self) -> None:
        """Clear the score cache."""
        cache_size = len(self._score_cache)
        self._score_cache.clear()

        logger.info("Reranker cache cleared", cleared_entries=cache_size)

    async def close(self) -> None:
        """Clean up resources."""
        if self.model and self.device.startswith("cuda"):
            # Clear GPU cache
            torch.cuda.empty_cache()

        await self.clear_cache()

        logger.info(
            "Cross-encoder reranker closed",
            total_pairs=self._total_pairs,
            total_time=self._total_time,
            cache_hit_rate=self._cache_hits
            / max(self._cache_hits + self._cache_misses, 1),
        )
