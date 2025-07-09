"""
Embedding models and data structures.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np


@dataclass
class EmbeddingRequest:
    """Request for generating embeddings."""
    
    text: str
    model: Optional[str] = None
    normalize: bool = True
    cache_key: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""
    
    embedding: np.ndarray
    model: str
    dimensions: int
    cached: bool = False
    processing_time: float = 0.0
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.dimensions == 0:
            self.dimensions = len(self.embedding)


@dataclass
class BatchEmbeddingRequest:
    """Request for batch embedding generation."""
    
    texts: List[str]
    model: Optional[str] = None
    normalize: bool = True
    batch_size: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchEmbeddingResponse:
    """Response from batch embedding generation."""
    
    embeddings: List[np.ndarray]
    model: str
    dimensions: int
    processing_time: float = 0.0
    cached_count: int = 0
    total_count: int = 0
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.total_count == 0:
            self.total_count = len(self.embeddings)
        if self.dimensions == 0 and self.embeddings:
            self.dimensions = len(self.embeddings[0])