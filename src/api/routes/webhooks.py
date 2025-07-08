"""
Webhook endpoints for external integrations.

Provides webhook endpoints for n8n, document ingestion, and event streaming.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.dependencies import get_memory_manager
from src.core.agents.agent_logger import agent_log_context, get_agent_logger
from src.core.agents.session_manager import get_session_manager
from src.core.memory.manager import MemoryManager

logger = get_agent_logger(__name__)
router = APIRouter(prefix="/v1/webhooks", tags=["webhooks"])


# Request/Response Models
class WebhookEvent(BaseModel):
    """Base webhook event model."""

    event_type: str = Field(..., description="Type of webhook event")
    source: str = Field(..., description="Source system/application")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(..., description="Event data")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class N8nWebhookRequest(BaseModel):
    """N8n webhook request model."""

    workflow_id: str = Field(..., description="N8n workflow ID")
    execution_id: str = Field(..., description="N8n execution ID")
    node_name: str = Field(..., description="Node name that triggered webhook")
    data: Dict[str, Any] = Field(..., description="Workflow data")
    agent_id: Optional[str] = Field("n8n", description="Agent ID for memory operations")
    session_id: Optional[str] = Field(None, description="Session ID")


class DocumentIngestionRequest(BaseModel):
    """Document ingestion request model."""

    content: str = Field(..., description="Document content")
    title: Optional[str] = Field(None, description="Document title")
    source: str = Field(..., description="Document source")
    agent_id: str = Field(..., description="Agent ID for document")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    chunk_content: bool = Field(True, description="Whether to chunk large content")
    extract_entities: bool = Field(True, description="Whether to extract entities")


class BatchMemoryRequest(BaseModel):
    """Batch memory processing request."""

    memories: List[Dict[str, Any]] = Field(
        ..., description="List of memories to process"
    )
    agent_id: str = Field(..., description="Agent ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    process_async: bool = Field(True, description="Process asynchronously")


class WebhookResponse(BaseModel):
    """Standard webhook response."""

    success: bool
    message: str
    event_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Optional[Dict[str, Any]] = None


# N8n Integration Endpoints
@router.post("/n8n/memory-store", response_model=WebhookResponse)
async def n8n_memory_store(
    request: N8nWebhookRequest,
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """
    N8n webhook for storing memories from workflows.

    Allows n8n workflows to store processed data as memories.
    """
    event_id = str(uuid.uuid4())

    with agent_log_context(
        agent_id=request.agent_id,
        session_id=request.session_id,
        operation="n8n_memory_store",
    ):
        try:
            # Extract content from n8n data
            content = (
                request.data.get("content")
                or request.data.get("text")
                or str(request.data)
            )

            if not content:
                raise HTTPException(
                    status_code=400, detail="No content found in webhook data"
                )

            # Store memory
            result = await memory_manager.store_memory(
                text=content,
                agent_id=request.agent_id,
                session_id=request.session_id,
                metadata={
                    **request.data.get("metadata", {}),
                    "source": "n8n",
                    "workflow_id": request.workflow_id,
                    "execution_id": request.execution_id,
                    "node_name": request.node_name,
                    "webhook_event_id": event_id,
                },
                extract_entities=True,
                chunk_content=len(content) > 1000,
            )

            logger.log_agent_action(
                "n8n_memory_store",
                "success" if result.get("success") else "error",
                details={
                    "workflow_id": request.workflow_id,
                    "memory_id": result.get("memory_id"),
                    "content_length": len(content),
                },
            )

            return WebhookResponse(
                success=result.get("success", False),
                message=(
                    f"Memory stored successfully"
                    if result.get("success")
                    else "Failed to store memory"
                ),
                event_id=event_id,
                data={
                    "memory_id": result.get("memory_id"),
                    "entities_created": result.get("entities_created", 0),
                    "processing_time": result.get("processing_time", {}),
                },
            )

        except Exception as e:
            logger.error(f"N8n memory store webhook failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/n8n/memory-search", response_model=WebhookResponse)
async def n8n_memory_search(
    request: N8nWebhookRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """
    N8n webhook for searching memories.

    Allows n8n workflows to search for relevant memories.
    """
    event_id = str(uuid.uuid4())

    with agent_log_context(
        agent_id=request.agent_id,
        session_id=request.session_id,
        operation="n8n_memory_search",
    ):
        try:
            # Extract search query from n8n data
            query = (
                request.data.get("query")
                or request.data.get("search")
                or request.data.get("text")
            )

            if not query:
                raise HTTPException(
                    status_code=400, detail="No query found in webhook data"
                )

            # Search memories
            result = await memory_manager.search_memories(
                query=query,
                agent_id=request.agent_id,
                session_id=request.session_id,
                top_k=request.data.get("top_k", 10),
                min_confidence=request.data.get("min_confidence", 0.3),
                rerank=request.data.get("rerank", True),
                include_analysis=request.data.get("include_analysis", True),
            )

            logger.log_memory_operation(
                "search",
                query=query,
                result_count=len(result.get("results", [])),
                processing_time=result.get("processing_time", {}).get("total"),
            )

            return WebhookResponse(
                success=result.get("success", False),
                message=f"Found {len(result.get('results', []))} memories",
                event_id=event_id,
                data={
                    "results": result.get("results", []),
                    "total_results": result.get("total_results", 0),
                    "hallucination_analysis": result.get("hallucination_analysis"),
                },
            )

        except Exception as e:
            logger.error(f"N8n memory search webhook failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Document Ingestion Endpoints
@router.post("/ingest/document", response_model=WebhookResponse)
async def ingest_document(
    request: DocumentIngestionRequest,
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """
    Ingest a document into the memory system.

    Supports large documents with automatic chunking and entity extraction.
    """
    event_id = str(uuid.uuid4())

    with agent_log_context(agent_id=request.agent_id, operation="document_ingestion"):
        try:
            # Store document as memory
            result = await memory_manager.store_memory(
                text=request.content,
                agent_id=request.agent_id,
                metadata={
                    **request.metadata,
                    "source": request.source,
                    "title": request.title,
                    "ingestion_event_id": event_id,
                    "document_type": "ingested_document",
                    "content_length": len(request.content),
                },
                extract_entities=request.extract_entities,
                chunk_content=request.chunk_content,
            )

            logger.log_agent_action(
                "document_ingestion",
                "success" if result.get("success") else "error",
                details={
                    "source": request.source,
                    "title": request.title,
                    "content_length": len(request.content),
                    "chunks_created": len(result.get("chunk_ids", [])),
                    "entities_created": result.get("entities_created", 0),
                },
            )

            return WebhookResponse(
                success=result.get("success", False),
                message=f"Document ingested successfully",
                event_id=event_id,
                data={
                    "memory_id": result.get("memory_id"),
                    "chunk_ids": result.get("chunk_ids", []),
                    "entities_created": result.get("entities_created", 0),
                    "relationships_created": result.get("relationships_created", 0),
                },
            )

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/batch", response_model=WebhookResponse)
async def ingest_batch_memories(
    request: BatchMemoryRequest,
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """
    Batch ingest multiple memories.

    Efficiently processes multiple memories in a single request.
    """
    event_id = str(uuid.uuid4())

    with agent_log_context(
        agent_id=request.agent_id,
        session_id=request.session_id,
        operation="batch_ingestion",
    ):
        try:
            if request.process_async:
                # Process asynchronously in background
                background_tasks.add_task(
                    _process_batch_memories,
                    request.memories,
                    request.agent_id,
                    request.session_id,
                    event_id,
                    memory_manager,
                )

                return WebhookResponse(
                    success=True,
                    message=f"Batch processing started for {len(request.memories)} memories",
                    event_id=event_id,
                    data={
                        "batch_size": len(request.memories),
                        "processing_async": True,
                    },
                )

            else:
                # Process synchronously
                results = await _process_batch_memories(
                    request.memories,
                    request.agent_id,
                    request.session_id,
                    event_id,
                    memory_manager,
                )

                return WebhookResponse(
                    success=True,
                    message=f"Processed {len(results['successful'])} of {len(request.memories)} memories",
                    event_id=event_id,
                    data=results,
                )

        except Exception as e:
            logger.error(f"Batch ingestion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Event Streaming Endpoints
@router.post("/events/memory-update")
async def handle_memory_update_event(
    request: Request, background_tasks: BackgroundTasks
):
    """
    Handle memory update events from external systems.

    Processes real-time memory updates and triggers appropriate actions.
    """
    event_id = str(uuid.uuid4())

    try:
        # Parse event data
        event_data = await request.json()
        event = WebhookEvent(**event_data)

        # Process event based on type
        if event.event_type == "memory_created":
            await _handle_memory_created_event(event, event_id)
        elif event.event_type == "memory_updated":
            await _handle_memory_updated_event(event, event_id)
        elif event.event_type == "memory_deleted":
            await _handle_memory_deleted_event(event, event_id)
        else:
            logger.warning(f"Unknown event type: {event.event_type}")

        return JSONResponse(
            {
                "success": True,
                "message": f"Event {event.event_type} processed",
                "event_id": event_id,
            }
        )

    except Exception as e:
        logger.error(f"Event processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/notify")
async def send_webhook_notification(
    url: str, payload: Dict[str, Any], background_tasks: BackgroundTasks
):
    """
    Send webhook notification to external system.

    Allows the memory server to notify external systems of events.
    """
    event_id = str(uuid.uuid4())

    try:
        # Send notification asynchronously
        background_tasks.add_task(_send_webhook_notification, url, payload, event_id)

        return JSONResponse(
            {
                "success": True,
                "message": "Webhook notification queued",
                "event_id": event_id,
            }
        )

    except Exception as e:
        logger.error(f"Webhook notification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper Functions
async def _process_batch_memories(
    memories: List[Dict[str, Any]],
    agent_id: str,
    session_id: Optional[str],
    event_id: str,
    memory_manager: MemoryManager,
) -> Dict[str, Any]:
    """Process a batch of memories."""
    successful = []
    failed = []

    for i, memory_data in enumerate(memories):
        try:
            result = await memory_manager.store_memory(
                text=memory_data.get("content", ""),
                agent_id=agent_id,
                session_id=session_id,
                metadata={
                    **memory_data.get("metadata", {}),
                    "batch_event_id": event_id,
                    "batch_index": i,
                },
                extract_entities=memory_data.get("extract_entities", True),
                chunk_content=memory_data.get("chunk_content", True),
            )

            if result.get("success"):
                successful.append(
                    {
                        "index": i,
                        "memory_id": result.get("memory_id"),
                        "entities_created": result.get("entities_created", 0),
                    }
                )
            else:
                failed.append(
                    {"index": i, "error": result.get("error", "Unknown error")}
                )

        except Exception as e:
            failed.append({"index": i, "error": str(e)})

    logger.log_agent_action(
        "batch_processing",
        "completed",
        details={
            "total_memories": len(memories),
            "successful": len(successful),
            "failed": len(failed),
            "event_id": event_id,
        },
    )

    return {
        "successful": successful,
        "failed": failed,
        "total_processed": len(memories),
        "success_rate": len(successful) / len(memories) if memories else 0,
    }


async def _handle_memory_created_event(event: WebhookEvent, event_id: str):
    """Handle memory created event."""
    logger.info(f"Processing memory created event: {event_id}")
    # Add custom logic for memory creation events
    pass


async def _handle_memory_updated_event(event: WebhookEvent, event_id: str):
    """Handle memory updated event."""
    logger.info(f"Processing memory updated event: {event_id}")
    # Add custom logic for memory update events
    pass


async def _handle_memory_deleted_event(event: WebhookEvent, event_id: str):
    """Handle memory deleted event."""
    logger.info(f"Processing memory deleted event: {event_id}")
    # Add custom logic for memory deletion events
    pass


async def _send_webhook_notification(url: str, payload: Dict[str, Any], event_id: str):
    """Send webhook notification to external URL."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json={
                    **payload,
                    "event_id": event_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "tyra-memory-server",
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                logger.info(f"Webhook notification sent successfully: {event_id}")
            else:
                logger.warning(
                    f"Webhook notification failed with status {response.status_code}: {event_id}"
                )

    except Exception as e:
        logger.error(f"Failed to send webhook notification {event_id}: {e}")


# Health check for webhooks
@router.get("/health")
async def webhook_health():
    """Health check for webhook endpoints."""
    return {
        "status": "healthy",
        "service": "webhooks",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "n8n_memory_store": "/v1/webhooks/n8n/memory-store",
            "n8n_memory_search": "/v1/webhooks/n8n/memory-search",
            "document_ingestion": "/v1/webhooks/ingest/document",
            "batch_ingestion": "/v1/webhooks/ingest/batch",
            "event_streaming": "/v1/webhooks/events/memory-update",
            "notifications": "/v1/webhooks/notify",
        },
    }
