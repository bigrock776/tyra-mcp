"""
Pydantic schemas for document ingestion requests and responses.

Comprehensive validation schemas for the enhanced document ingestion system
supporting multiple file types with dynamic chunking and metadata tracking.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, validator


class IngestRequest(BaseModel):
    """Request schema for document ingestion."""
    
    source_type: Literal["file", "url", "base64"] = Field(
        ...,
        description="Type of source content: local file, URL, or base64 encoded"
    )
    
    file_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Original filename with extension"
    )
    
    file_type: Literal["pdf", "docx", "pptx", "txt", "md", "html", "json", "csv", "epub"] = Field(
        ...,
        description="File type determining the parsing strategy"
    )
    
    content: Optional[str] = Field(
        None,
        description="Base64 encoded content or plain text content"
    )
    
    file_url: Optional[HttpUrl] = Field(
        None,
        description="URL to fetch content from (when source_type is 'url')"
    )
    
    source_agent: Optional[str] = Field(
        default="tyra",
        description="Agent ID that initiated the ingestion"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for grouping related ingestions"
    )
    
    description: Optional[str] = Field(
        default="",
        max_length=1000,
        description="Human-readable description of the document"
    )
    
    chunking_strategy: Optional[str] = Field(
        default="auto",
        description="Chunking strategy: auto, paragraph, semantic, slide, line, token"
    )
    
    chunk_size: Optional[int] = Field(
        default=512,
        ge=50,
        le=4096,
        description="Maximum chunk size in characters"
    )
    
    chunk_overlap: Optional[int] = Field(
        default=50,
        ge=0,
        le=200,
        description="Overlap between chunks in characters"
    )
    
    enable_llm_context: Optional[bool] = Field(
        default=True,
        description="Whether to enhance chunks with LLM context before embedding"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata to store with the document"
    )

    @validator("content")
    def validate_content_by_source_type(cls, v, values):
        """Validate content field based on source_type."""
        source_type = values.get("source_type")
        
        if source_type == "base64" and not v:
            raise ValueError("content is required when source_type is 'base64'")
        elif source_type == "file" and v:
            raise ValueError("content should not be provided when source_type is 'file'")
        
        return v

    @validator("file_url")
    def validate_url_by_source_type(cls, v, values):
        """Validate file_url field based on source_type."""
        source_type = values.get("source_type")
        
        if source_type == "url" and not v:
            raise ValueError("file_url is required when source_type is 'url'")
        elif source_type != "url" and v:
            raise ValueError("file_url should only be provided when source_type is 'url'")
        
        return v


class ChunkMetadata(BaseModel):
    """Metadata for an individual chunk."""
    
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    chunk_index: int
    chunk_text: str
    tokens: int
    chunk_type: str  # paragraph, slide, semantic, line, etc.
    source_page: Optional[int] = None
    source_section: Optional[str] = None
    enhanced_context: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    hallucination_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentMetadata(BaseModel):
    """Comprehensive metadata for the ingested document."""
    
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str
    file_type: str
    file_size: Optional[int] = None
    source_type: str
    source_agent: str
    session_id: Optional[str] = None
    description: str
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    total_chunks: int
    total_tokens: int
    processing_time: float
    embedding_time: float
    storage_time: float
    graph_time: float
    llm_context_enabled: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ingestion_method: str = "enhanced_api"
    
    # File-specific metadata
    page_count: Optional[int] = None
    slide_count: Optional[int] = None
    table_count: Optional[int] = None
    image_count: Optional[int] = None
    
    # Quality metrics
    avg_confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    avg_hallucination_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    parsing_warnings: List[str] = Field(default_factory=list)
    
    # Additional metadata
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionWarning(BaseModel):
    """Warning or error during ingestion."""
    
    type: Literal["warning", "error", "info"]
    message: str
    chunk_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class IngestResponse(BaseModel):
    """Response schema for document ingestion."""
    
    status: Literal["success", "partial_success", "failed"]
    doc_id: str
    summary: str
    chunks_ingested: int
    total_chunks_attempted: int
    processing_time: float
    
    # Detailed metadata
    document_metadata: DocumentMetadata
    chunks_metadata: List[ChunkMetadata]
    
    # Quality and safety metrics
    avg_confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    avg_hallucination_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    safety_flags: List[str] = Field(default_factory=list)
    
    # Warnings and errors
    warnings: List[IngestionWarning] = Field(default_factory=list)
    
    # Graph relationships created
    entities_created: List[str] = Field(default_factory=list)
    relationships_created: List[str] = Field(default_factory=list)
    
    # Performance metrics
    embedding_time: float
    storage_time: float
    graph_time: float
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IngestionProgress(BaseModel):
    """Progress tracking for long-running ingestion jobs."""
    
    job_id: str
    doc_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    progress_percentage: float = Field(ge=0.0, le=100.0)
    current_chunk: int
    total_chunks: int
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class BatchIngestRequest(BaseModel):
    """Request schema for batch document ingestion."""
    
    documents: List[IngestRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of documents to ingest (max 100)"
    )
    
    batch_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the batch"
    )
    
    source_agent: Optional[str] = Field(
        default="tyra",
        description="Agent ID that initiated the batch ingestion"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for grouping the batch"
    )
    
    max_concurrent: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of concurrent ingestion jobs"
    )
    
    priority: Optional[Literal["low", "normal", "high"]] = Field(
        default="normal",
        description="Processing priority for the batch"
    )


class BatchIngestResponse(BaseModel):
    """Response schema for batch document ingestion."""
    
    batch_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    total_documents: int
    successful_ingestions: int
    failed_ingestions: int
    
    # Individual results
    results: List[Union[IngestResponse, IngestionWarning]] = Field(default_factory=list)
    
    # Batch metrics
    total_processing_time: float
    avg_processing_time: float
    total_chunks_ingested: int
    
    # Progress tracking
    progress_percentage: float = Field(ge=0.0, le=100.0)
    estimated_completion: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class SupportedFormats(BaseModel):
    """Information about supported file formats."""
    
    format: str
    extensions: List[str]
    description: str
    max_file_size: Optional[str] = None
    chunking_strategies: List[str]
    features: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class IngestionCapabilities(BaseModel):
    """Available ingestion capabilities and formats."""
    
    supported_formats: List[SupportedFormats]
    chunking_strategies: List[str]
    max_file_size: str
    max_batch_size: int
    concurrent_limit: int
    features: List[str]
    version: str = "1.0.0"