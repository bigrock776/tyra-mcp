"""Pydantic models for memory operations."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict, field_validator


class ConfidenceLevel(str, Enum):
    """Confidence level categories for responses."""
    ROCK_SOLID = "rock_solid"  # 95%+ - Safe for automated actions
    HIGH = "high"              # 80-94% - Generally reliable
    FUZZY = "fuzzy"            # 60-79% - Needs verification
    LOW = "low"                # <60% - Not confident


class Memory(BaseModel):
    """Base memory model."""
    model_config = ConfigDict(json_encoders={UUID: str, datetime: lambda v: v.isoformat()})
    
    id: UUID = Field(default_factory=uuid4)
    text: str = Field(..., description="Memory content")
    agent_id: Optional[str] = Field(None, description="Agent that created this memory")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(1, description="Memory version for tracking updates")
    expires_at: Optional[datetime] = Field(None, description="Optional expiration time")
    
    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Validate text is not empty."""
        if not v or not v.strip():
            raise ValueError("Memory text cannot be empty")
        return v.strip()


class MemoryWithEmbedding(Memory):
    """Memory with embedding vector."""
    embedding: List[float] = Field(..., description="Embedding vector")
    embedding_model: str = Field(..., description="Model used to generate embedding")
    
    @field_validator('embedding')
    @classmethod
    def embedding_not_empty(cls, v: List[float]) -> List[float]:
        """Validate embedding is not empty."""
        if not v:
            raise ValueError("Embedding cannot be empty")
        return v


class MemorySearchResult(BaseModel):
    """Search result with relevance score."""
    memory: Memory
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    distance: Optional[float] = Field(None, description="Vector distance")
    rerank_score: Optional[float] = Field(None, description="Reranking score if applied")
    source: str = Field("vector", description="Search source: vector, text, or hybrid")


class RetrievalContext(BaseModel):
    """Context for memory retrieval operations."""
    query: str = Field(..., description="Search query")
    agent_id: Optional[str] = Field(None, description="Filter by agent")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
    include_expired: bool = Field(False, description="Include expired memories")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum relevance score")
    rerank: bool = Field(True, description="Apply reranking")
    
    
class MemoryStats(BaseModel):
    """Memory system statistics."""
    total_memories: int = Field(..., ge=0)
    unique_agents: int = Field(..., ge=0)
    total_size_mb: float = Field(..., ge=0)
    oldest_memory: Optional[datetime] = None
    newest_memory: Optional[datetime] = None
    avg_text_length: float = Field(..., ge=0)
    expired_count: int = Field(0, ge=0)
    
    
class SaveMemoryRequest(BaseModel):
    """Request to save a memory."""
    text: str = Field(..., description="Memory content")
    agent_id: Optional[str] = Field(None, description="Agent identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    expires_in_hours: Optional[int] = Field(None, ge=1, description="Hours until expiration")
    
    
class SaveMemoryResponse(BaseModel):
    """Response from saving a memory."""
    id: UUID = Field(..., description="Memory ID")
    success: bool = Field(True)
    message: str = Field("Memory saved successfully")
    
    
class SearchMemoriesRequest(BaseModel):
    """Request to search memories."""
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=100)
    agent_id: Optional[str] = Field(None, description="Filter by agent")
    filters: Dict[str, Any] = Field(default_factory=dict)
    use_reranking: bool = Field(True)
    check_hallucination: bool = Field(False)
    min_confidence: float = Field(0.0, ge=0.0, le=100.0)
    
    
class SearchMemoriesResponse(BaseModel):
    """Response from searching memories."""
    results: List[MemorySearchResult] = Field(default_factory=list)
    total_found: int = Field(..., ge=0)
    search_time_ms: float = Field(..., ge=0)
    confidence_score: float = Field(..., ge=0.0, le=100.0)
    confidence_level: ConfidenceLevel
    hallucination_detected: bool = Field(False)
    safe_to_act_on: bool = Field(False)
    
    @property
    def is_rock_solid(self) -> bool:
        """Check if confidence is rock solid (95%+)."""
        return self.confidence_level == ConfidenceLevel.ROCK_SOLID
        
        
class GetAllMemoriesRequest(BaseModel):
    """Request to get all memories."""
    agent_id: Optional[str] = Field(None, description="Filter by agent")
    include_expired: bool = Field(False)
    offset: int = Field(0, ge=0)
    limit: int = Field(100, ge=1, le=1000)
    
    
class GetAllMemoriesResponse(BaseModel):
    """Response from getting all memories."""
    memories: List[Memory] = Field(default_factory=list)
    total: int = Field(..., ge=0)
    offset: int = Field(..., ge=0)
    has_more: bool = Field(False)
    
    
class UpdateMemoryRequest(BaseModel):
    """Request to update a memory."""
    id: UUID = Field(..., description="Memory ID to update")
    text: Optional[str] = Field(None, description="New text content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="New metadata")
    expires_in_hours: Optional[int] = Field(None, ge=1)
    
    
class DeleteMemoryRequest(BaseModel):
    """Request to delete a memory."""
    id: UUID = Field(..., description="Memory ID to delete")
    
    
class MemoryHealthReport(BaseModel):
    """Health report for memory system."""
    status: str = Field("healthy", description="Overall health status")
    total_memories: int = Field(..., ge=0)
    stale_memories: int = Field(0, ge=0)
    redundant_memories: int = Field(0, ge=0)
    low_confidence_memories: int = Field(0, ge=0)
    recommendations: List[str] = Field(default_factory=list)
    last_cleanup: Optional[datetime] = None
    
    
class MemoryAnalytics(BaseModel):
    """Analytics for memory usage patterns."""
    most_accessed_topics: List[Dict[str, Any]] = Field(default_factory=list)
    agent_activity: Dict[str, int] = Field(default_factory=dict)
    temporal_patterns: Dict[str, Any] = Field(default_factory=dict)
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    performance_stats: Dict[str, float] = Field(default_factory=dict)