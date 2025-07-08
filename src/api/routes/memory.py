"""
Memory management API endpoints.

Provides REST API for memory CRUD operations, versioning,
and metadata management.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...core.memory.manager import MemoryManager
from ...core.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Request/Response Models
class MemoryStoreRequest(BaseModel):
    """Request to store a new memory."""

    content: str = Field(..., description="Memory content text")
    agent_id: str = Field(default="tyra", description="Agent that created the memory")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional metadata"
    )
    extract_entities: bool = Field(
        default=True, description="Extract entities from content"
    )
    chunk_content: bool = Field(default=False, description="Chunk large content")


class MemoryUpdateRequest(BaseModel):
    """Request to update an existing memory."""

    text: Optional[str] = Field(None, description="Updated memory content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    tags: Optional[List[str]] = Field(None, description="Updated tags")


class MemoryStoreResult(BaseModel):
    """Result from storing a memory."""

    success: bool = Field(..., description="Whether operation succeeded")
    memory_id: Optional[str] = Field(None, description="Created memory ID")
    chunk_ids: List[str] = Field(default=[], description="Created chunk IDs")
    entities_created: int = Field(default=0, description="Number of entities created")
    relationships_created: int = Field(
        default=0, description="Number of relationships created"
    )
    processing_time: Dict[str, float] = Field(
        default={}, description="Processing timing breakdown"
    )
    error: Optional[str] = Field(None, description="Error message if failed")


class MemorySearchRequest(BaseModel):
    """Request for searching memories."""

    query: str = Field(..., description="Search query")
    agent_id: Optional[str] = Field(None, description="Agent filter")
    session_id: Optional[str] = Field(None, description="Session filter")
    top_k: int = Field(default=10, description="Number of results to return")
    min_confidence: float = Field(default=0.0, description="Minimum confidence score")
    search_type: str = Field(
        default="hybrid", description="Search type: vector, graph, or hybrid"
    )
    include_analysis: bool = Field(
        default=True, description="Include hallucination analysis"
    )


class MemorySearchResult(BaseModel):
    """Result from searching memories."""

    success: bool = Field(..., description="Whether search succeeded")
    query: str = Field(..., description="Original query")
    results: List[Dict[str, Any]] = Field(default=[], description="Search results")
    total_results: int = Field(default=0, description="Total number of results")
    search_type: str = Field(default="hybrid", description="Search type used")
    hallucination_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Hallucination analysis results"
    )
    error: Optional[str] = Field(None, description="Error message if failed")


class AnalysisRequest(BaseModel):
    """Request for analyzing a response."""

    response: str = Field(..., description="Response text to analyze")
    query: Optional[str] = Field(None, description="Original query")
    retrieved_memories: Optional[List[Dict[str, Any]]] = Field(
        default=[], description="Retrieved memory chunks"
    )


class MemoryStatsResponse(BaseModel):
    """Memory system statistics response."""

    success: bool = Field(..., description="Whether request succeeded")
    memory_stats: Dict[str, Any] = Field(default={}, description="Memory statistics")
    performance_stats: Dict[str, Any] = Field(
        default={}, description="Performance statistics"
    )
    health_score: float = Field(default=0.0, description="Overall system health score")
    recommendations: List[str] = Field(default=[], description="System recommendations")
    error: Optional[str] = Field(None, description="Error message if failed")


# Import the dependency from app
from ..app import get_memory_manager, get_request_context


@router.post("/store", response_model=MemoryStoreResult)
async def store_memory(
    request: MemoryStoreRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    context: Dict[str, Any] = Depends(get_request_context),
):
    """
    Store a new memory in the system.

    Creates a new memory with embeddings, entity extraction, and graph relationships.
    """
    try:
        # Store memory using the memory manager
        result = await memory_manager.store_memory(
            text=request.content,
            agent_id=request.agent_id,
            session_id=request.session_id,
            metadata=request.metadata,
            extract_entities=request.extract_entities,
            chunk_content=request.chunk_content,
        )

        return MemoryStoreResult(
            success=True,
            memory_id=result.get("memory_id"),
            chunk_ids=result.get("chunk_ids", []),
            entities_created=result.get("entities_created", 0),
            relationships_created=result.get("relationships_created", 0),
            processing_time=result.get("processing_time", {}),
        )

    except Exception as e:
        logger.error(f"Failed to store memory: {e}", extra=context)
        return MemoryStoreResult(success=False, error=str(e))


@router.post("/search", response_model=MemorySearchResult)
async def search_memories(
    request: MemorySearchRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    context: Dict[str, Any] = Depends(get_request_context),
):
    """
    Search memories using advanced RAG techniques.

    Supports vector search, graph traversal, and hybrid approaches with
    reranking and hallucination detection.
    """
    try:
        # Search memories using the memory manager
        result = await memory_manager.search_memories(
            query=request.query,
            agent_id=request.agent_id,
            session_id=request.session_id,
            top_k=request.top_k,
            min_confidence=request.min_confidence,
            search_type=request.search_type,
            include_analysis=request.include_analysis,
        )

        return MemorySearchResult(
            success=True,
            query=request.query,
            results=result.get("results", []),
            total_results=result.get("total_results", 0),
            search_type=request.search_type,
            hallucination_analysis=result.get("hallucination_analysis"),
        )

    except Exception as e:
        logger.error(f"Failed to search memories: {e}", extra=context)
        return MemorySearchResult(success=False, query=request.query, error=str(e))


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_response(
    request: AnalysisRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    context: Dict[str, Any] = Depends(get_request_context),
):
    """
    Analyze a response for hallucinations and confidence.

    Uses advanced hallucination detection to assess response quality
    against retrieved memory chunks.
    """
    try:
        # Analyze response using the memory manager
        result = await memory_manager.analyze_response(
            response=request.response,
            query=request.query,
            retrieved_memories=request.retrieved_memories,
        )

        return {"success": True, "analysis": result}

    except Exception as e:
        logger.error(f"Failed to analyze response: {e}", extra=context)
        return {"success": False, "error": str(e)}


@router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(
    agent_id: Optional[str] = Query(None, description="Filter stats by agent ID"),
    include_performance: bool = Query(True, description="Include performance metrics"),
    include_recommendations: bool = Query(
        True, description="Include system recommendations"
    ),
    memory_manager: MemoryManager = Depends(get_memory_manager),
    context: Dict[str, Any] = Depends(get_request_context),
):
    """
    Get comprehensive memory system statistics.

    Returns memory counts, performance metrics, and system health information.
    """
    try:
        # Get stats from memory manager
        result = await memory_manager.get_stats(
            agent_id=agent_id,
            include_performance=include_performance,
            include_recommendations=include_recommendations,
        )

        return MemoryStatsResponse(
            success=True,
            memory_stats=result.get("memory_stats", {}),
            performance_stats=result.get("performance_stats", {}),
            health_score=result.get("health_score", 0.0),
            recommendations=result.get("recommendations", []),
        )

    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}", extra=context)
        return MemoryStatsResponse(success=False, error=str(e))


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    context: Dict[str, Any] = Depends(get_request_context),
):
    """
    Delete a specific memory by ID.

    Removes the memory and all associated data from the system.
    """
    try:
        # Delete memory using the memory manager
        success = await memory_manager.delete_memory(memory_id)

        if success:
            return {
                "success": True,
                "message": f"Memory {memory_id} deleted successfully",
            }
        else:
            raise HTTPException(status_code=404, detail="Memory not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete memory {memory_id}: {e}", extra=context)
        return {"success": False, "error": str(e)}
