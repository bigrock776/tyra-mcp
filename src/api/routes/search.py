"""
Search API endpoints.

Provides various search strategies including vector similarity,
keyword search, hybrid search, and advanced filtering.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...core.memory.models import Memory
from ...core.search.searcher import Searcher
from ...core.utils.logger import get_logger
from ...core.utils.registry import ProviderType, get_provider

logger = get_logger(__name__)

router = APIRouter()


# Enums
class SearchStrategy(str, Enum):
    """Available search strategies."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"


class SortOrder(str, Enum):
    """Sort order for results."""

    RELEVANCE = "relevance"
    CREATED_ASC = "created_asc"
    CREATED_DESC = "created_desc"
    UPDATED_ASC = "updated_asc"
    UPDATED_DESC = "updated_desc"


# Request/Response Models
class SearchRequest(BaseModel):
    """Search request parameters."""

    query: str = Field(..., description="Search query text")
    strategy: SearchStrategy = Field(
        SearchStrategy.HYBRID, description="Search strategy to use"
    )
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")
    filters: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional filters"
    )
    sort: SortOrder = Field(SortOrder.RELEVANCE, description="Sort order for results")
    include_metadata: bool = Field(True, description="Include metadata in results")
    min_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum relevance score"
    )


class SearchResult(BaseModel):
    """Individual search result."""

    memory_id: str = Field(..., description="Memory ID")
    text: str = Field(..., description="Memory text content")
    score: float = Field(..., description="Relevance score (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(
        default={}, description="Memory metadata"
    )
    highlights: Optional[List[str]] = Field(default=[], description="Text highlights")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")
    agent_id: Optional[str] = Field(None, description="Associated agent ID")


class SearchResponse(BaseModel):
    """Search response with results and metadata."""

    query: str = Field(..., description="Original query")
    strategy: SearchStrategy = Field(..., description="Strategy used")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total matching results")
    search_time_ms: float = Field(..., description="Search execution time")
    has_more: bool = Field(..., description="Whether more results exist")


class SimilaritySearchRequest(BaseModel):
    """Find similar memories request."""

    memory_id: str = Field(..., description="Memory ID to find similar to")
    limit: int = Field(10, ge=1, le=50, description="Maximum results")
    min_similarity: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    exclude_self: bool = Field(True, description="Exclude the source memory")


class BatchSearchRequest(BaseModel):
    """Batch search request."""

    queries: List[str] = Field(..., description="List of queries to search")
    strategy: SearchStrategy = Field(
        SearchStrategy.HYBRID, description="Search strategy"
    )
    limit_per_query: int = Field(5, ge=1, le=20, description="Results per query")
    deduplicate: bool = Field(
        True, description="Remove duplicate results across queries"
    )


# Dependencies
async def get_searcher() -> Searcher:
    """Get searcher instance."""
    try:
        return get_provider(ProviderType.SEARCHER, "default")
    except Exception as e:
        logger.error(f"Failed to get searcher: {e}")
        raise HTTPException(status_code=500, detail="Searcher unavailable")


@router.post("/", response_model=SearchResponse)
async def search_memories(
    request: SearchRequest, searcher: Searcher = Depends(get_searcher)
):
    """
    Search memories using various strategies.

    Supports vector similarity, keyword matching, and hybrid approaches.
    Results are ranked by relevance and can be filtered.
    """
    try:
        import time

        start_time = time.time()

        # Perform search
        results = await searcher.search(
            query=request.query,
            strategy=request.strategy,
            limit=request.limit,
            offset=request.offset,
            filters=request.filters,
            min_score=request.min_score,
        )

        # Count total results
        total_results = await searcher.count_results(
            query=request.query, strategy=request.strategy, filters=request.filters
        )

        # Convert to response format
        search_results = []
        for result in results:
            search_results.append(
                SearchResult(
                    memory_id=result["memory_id"],
                    text=result["text"],
                    score=result["score"],
                    metadata=(
                        result.get("metadata", {}) if request.include_metadata else {}
                    ),
                    highlights=result.get("highlights", []),
                    created_at=result["created_at"],
                    updated_at=result["updated_at"],
                    agent_id=result.get("agent_id"),
                )
            )

        # Sort results if needed
        if request.sort != SortOrder.RELEVANCE:
            search_results = _sort_results(search_results, request.sort)

        search_time = (time.time() - start_time) * 1000

        return SearchResponse(
            query=request.query,
            strategy=request.strategy,
            results=search_results,
            total_results=total_results,
            search_time_ms=search_time,
            has_more=(request.offset + request.limit) < total_results,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similar", response_model=SearchResponse)
async def find_similar_memories(
    request: SimilaritySearchRequest, searcher: Searcher = Depends(get_searcher)
):
    """
    Find memories similar to a given memory.

    Uses vector similarity to find related content.
    """
    try:
        import time

        start_time = time.time()

        # Find similar memories
        results = await searcher.find_similar(
            memory_id=request.memory_id,
            limit=request.limit,
            min_similarity=request.min_similarity,
            exclude_self=request.exclude_self,
        )

        # Convert to response format
        search_results = []
        for result in results:
            search_results.append(
                SearchResult(
                    memory_id=result["memory_id"],
                    text=result["text"],
                    score=result["similarity"],
                    metadata=result.get("metadata", {}),
                    highlights=[],
                    created_at=result["created_at"],
                    updated_at=result["updated_at"],
                    agent_id=result.get("agent_id"),
                )
            )

        search_time = (time.time() - start_time) * 1000

        return SearchResponse(
            query=f"similar_to:{request.memory_id}",
            strategy=SearchStrategy.VECTOR,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time,
            has_more=False,
        )

    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=Dict[str, SearchResponse])
async def batch_search(
    request: BatchSearchRequest, searcher: Searcher = Depends(get_searcher)
):
    """
    Perform multiple searches in a single request.

    Executes searches in parallel and optionally deduplicates results.
    """
    try:
        import asyncio
        import time

        start_time = time.time()

        # Execute searches in parallel
        search_tasks = []
        for query in request.queries:
            task = searcher.search(
                query=query, strategy=request.strategy, limit=request.limit_per_query
            )
            search_tasks.append(task)

        results = await asyncio.gather(*search_tasks)

        # Process results
        batch_results = {}
        seen_ids = set() if request.deduplicate else None

        for query, query_results in zip(request.queries, results):
            # Filter duplicates if requested
            if request.deduplicate:
                filtered_results = []
                for result in query_results:
                    if result["memory_id"] not in seen_ids:
                        filtered_results.append(result)
                        seen_ids.add(result["memory_id"])
                query_results = filtered_results

            # Convert to response format
            search_results = []
            for result in query_results:
                search_results.append(
                    SearchResult(
                        memory_id=result["memory_id"],
                        text=result["text"],
                        score=result["score"],
                        metadata=result.get("metadata", {}),
                        highlights=result.get("highlights", []),
                        created_at=result["created_at"],
                        updated_at=result["updated_at"],
                        agent_id=result.get("agent_id"),
                    )
                )

            search_time = (time.time() - start_time) * 1000

            batch_results[query] = SearchResponse(
                query=query,
                strategy=request.strategy,
                results=search_results,
                total_results=len(search_results),
                search_time_ms=search_time,
                has_more=False,
            )

        return batch_results

    except Exception as e:
        logger.error(f"Batch search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggest")
async def search_suggestions(
    prefix: str = Query(..., description="Search prefix"),
    limit: int = Query(5, ge=1, le=20, description="Maximum suggestions"),
    searcher: Searcher = Depends(get_searcher),
):
    """
    Get search suggestions based on prefix.

    Returns autocomplete suggestions for search queries.
    """
    try:
        suggestions = await searcher.get_suggestions(prefix=prefix, limit=limit)

        return {"prefix": prefix, "suggestions": suggestions}

    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/facets")
async def get_search_facets(
    query: Optional[str] = Query(None, description="Optional query to filter facets"),
    searcher: Searcher = Depends(get_searcher),
):
    """
    Get available search facets.

    Returns aggregated information about tags, agents, and dates
    that can be used for filtering.
    """
    try:
        facets = await searcher.get_facets(query=query)

        return {"query": query, "facets": facets}

    except Exception as e:
        logger.error(f"Failed to get facets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced")
async def advanced_search(
    query: str = Query(..., description="Search query"),
    vector_weight: float = Query(
        0.7, ge=0.0, le=1.0, description="Vector search weight"
    ),
    keyword_weight: float = Query(
        0.3, ge=0.0, le=1.0, description="Keyword search weight"
    ),
    rerank: bool = Query(True, description="Apply reranking"),
    expand_query: bool = Query(False, description="Expand query with synonyms"),
    temporal_boost: bool = Query(False, description="Boost recent results"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    searcher: Searcher = Depends(get_searcher),
):
    """
    Advanced search with fine-grained control.

    Allows customization of search weights, reranking, and other parameters.
    """
    try:
        import time

        start_time = time.time()

        # Perform advanced search
        results = await searcher.advanced_search(
            query=query,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            rerank=rerank,
            expand_query=expand_query,
            temporal_boost=temporal_boost,
            limit=limit,
        )

        # Convert to response format
        search_results = []
        for result in results:
            search_results.append(
                SearchResult(
                    memory_id=result["memory_id"],
                    text=result["text"],
                    score=result["score"],
                    metadata=result.get("metadata", {}),
                    highlights=result.get("highlights", []),
                    created_at=result["created_at"],
                    updated_at=result["updated_at"],
                    agent_id=result.get("agent_id"),
                )
            )

        search_time = (time.time() - start_time) * 1000

        return SearchResponse(
            query=query,
            strategy=SearchStrategy.HYBRID,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time,
            has_more=False,
        )

    except Exception as e:
        logger.error(f"Advanced search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _sort_results(
    results: List[SearchResult], sort_order: SortOrder
) -> List[SearchResult]:
    """Sort search results by specified order."""
    if sort_order == SortOrder.CREATED_ASC:
        return sorted(results, key=lambda x: x.created_at)
    elif sort_order == SortOrder.CREATED_DESC:
        return sorted(results, key=lambda x: x.created_at, reverse=True)
    elif sort_order == SortOrder.UPDATED_ASC:
        return sorted(results, key=lambda x: x.updated_at)
    elif sort_order == SortOrder.UPDATED_DESC:
        return sorted(results, key=lambda x: x.updated_at, reverse=True)
    else:
        return results
