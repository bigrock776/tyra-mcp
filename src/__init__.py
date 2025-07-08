"""
Tyra MCP Memory Server

Advanced RAG-powered Model Context Protocol memory server with self-learning capabilities.
Combines vector search, knowledge graphs, and hallucination detection for genius-tier memory.
"""

__version__ = "1.0.0"
__author__ = "Tyra AI System"
__description__ = (
    "Advanced RAG-powered MCP memory server with self-learning capabilities"
)

from .core.utils.simple_logger import get_logger

logger = get_logger(__name__)

from .core.interfaces.embeddings import EmbeddingProvider
from .core.interfaces.graph_engine import GraphEngine
from .core.interfaces.vector_store import VectorStore

# Export main components for easy importing
from .core.memory.manager import MemoryManager

__all__ = [
    "MemoryManager",
    "EmbeddingProvider",
    "VectorStore",
    "GraphEngine",
    "__version__",
    "__author__",
    "__description__",
]
