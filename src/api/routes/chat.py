"""
Chat API endpoints.

Provides conversational interfaces with memory integration,
context management, and streaming responses.
"""

import asyncio
import json
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...core.memory.manager import MemoryManager
from ...core.rag.reranker import Reranker
from ...core.search.searcher import Searcher
from ...core.utils.logger import get_logger
from ...core.utils.registry import ProviderType, get_provider

logger = get_logger(__name__)

router = APIRouter()


# Enums
class ChatModel(str, Enum):
    """Available chat models."""
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


class ResponseFormat(str, Enum):
    """Response format options."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


# Request/Response Models
class Message(BaseModel):
    """Chat message."""
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    timestamp: Optional[datetime] = Field(None, description="Message timestamp")


class ChatRequest(BaseModel):
    """Chat completion request."""
    messages: List[Message] = Field(..., description="Conversation messages")
    model: ChatModel = Field(ChatModel.BALANCED, description="Model to use")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Response temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="Maximum response tokens")
    stream: bool = Field(False, description="Stream the response")
    use_memory: bool = Field(True, description="Use memory system for context")
    memory_limit: int = Field(10, ge=0, le=50, description="Maximum memories to retrieve")
    agent_id: Optional[str] = Field(None, description="Agent ID for context")
    format: ResponseFormat = Field(ResponseFormat.TEXT, description="Response format")


class ChatResponse(BaseModel):
    """Chat completion response."""
    id: str = Field(..., description="Response ID")
    model: str = Field(..., description="Model used")
    created: datetime = Field(..., description="Creation timestamp")
    message: Message = Field(..., description="Response message")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    memories_used: List[str] = Field(default=[], description="Memory IDs used for context")
    confidence: float = Field(..., description="Response confidence score")


class ConversationRequest(BaseModel):
    """Conversation management request."""
    agent_id: str = Field(..., description="Agent ID")
    title: Optional[str] = Field(None, description="Conversation title")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Conversation metadata")


class ConversationResponse(BaseModel):
    """Conversation information."""
    id: str = Field(..., description="Conversation ID")
    agent_id: str = Field(..., description="Agent ID")
    title: Optional[str] = Field(None, description="Conversation title")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    message_count: int = Field(..., description="Number of messages")
    metadata: Dict[str, Any] = Field(default={}, description="Conversation metadata")


# Dependencies
async def get_memory_manager() -> MemoryManager:
    """Get memory manager instance."""
    try:
        return get_provider(ProviderType.MEMORY_MANAGER, "default")
    except Exception as e:
        logger.error(f"Failed to get memory manager: {e}")
        raise HTTPException(status_code=500, detail="Memory manager unavailable")


async def get_searcher() -> Searcher:
    """Get searcher instance."""
    try:
        return get_provider(ProviderType.SEARCHER, "default")
    except Exception as e:
        logger.error(f"Failed to get searcher: {e}")
        raise HTTPException(status_code=500, detail="Searcher unavailable")


@router.post("/completions", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    searcher: Searcher = Depends(get_searcher)
):
    """
    Create a chat completion.

    Generates a response based on the conversation history,
    optionally using the memory system for context.
    """
    try:
        import uuid

        # Extract user message
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # Retrieve relevant memories if enabled
        memories_used = []
        context = ""

        if request.use_memory and request.memory_limit > 0:
            # Search for relevant memories
            search_results = await searcher.search(
                query=user_message,
                strategy="hybrid",
                limit=request.memory_limit,
                filters={"agent_id": request.agent_id} if request.agent_id else None
            )

            # Build context from memories
            if search_results:
                context_parts = []
                for result in search_results:
                    context_parts.append(result["text"])
                    memories_used.append(result["memory_id"])

                context = "\n\n".join(context_parts)

        # Generate response (placeholder - would integrate with LLM)
        response_content = _generate_response(
            messages=request.messages,
            context=context,
            model=request.model,
            temperature=request.temperature
        )

        # Format response based on requested format
        if request.format == ResponseFormat.JSON:
            response_content = json.dumps({"response": response_content})
        elif request.format == ResponseFormat.MARKDOWN:
            response_content = f"**Assistant**: {response_content}"

        # Calculate token usage (placeholder)
        usage = {
            "prompt_tokens": sum(len(msg.content.split()) for msg in request.messages),
            "completion_tokens": len(response_content.split()),
            "total_tokens": 0
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        # Create response
        response = ChatResponse(
            id=str(uuid.uuid4()),
            model=request.model,
            created=datetime.utcnow(),
            message=Message(
                role="assistant",
                content=response_content,
                timestamp=datetime.utcnow()
            ),
            usage=usage,
            memories_used=memories_used,
            confidence=0.85  # Placeholder confidence
        )

        # Store the interaction as memory if agent_id is provided
        if request.agent_id:
            await memory_manager.store_memory(
                memory_id=str(uuid.uuid4()),
                text=f"User: {user_message}\nAssistant: {response_content}",
                metadata={
                    "type": "chat_interaction",
                    "agent_id": request.agent_id,
                    "model": request.model,
                    "timestamp": datetime.utcnow().isoformat()
                },
                agent_id=request.agent_id
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/completions/stream")
async def create_chat_completion_stream(
    request: ChatRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    searcher: Searcher = Depends(get_searcher)
):
    """
    Create a streaming chat completion.

    Streams the response as Server-Sent Events (SSE).
    """
    if not request.stream:
        raise HTTPException(status_code=400, detail="Stream parameter must be true")

    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            import uuid

            # Extract user message
            user_message = None
            for msg in reversed(request.messages):
                if msg.role == "user":
                    user_message = msg.content
                    break

            if not user_message:
                yield f"data: {json.dumps({'error': 'No user message found'})}\n\n"
                return

            # Retrieve relevant memories
            context = ""
            if request.use_memory and request.memory_limit > 0:
                search_results = await searcher.search(
                    query=user_message,
                    strategy="hybrid",
                    limit=request.memory_limit,
                    filters={"agent_id": request.agent_id} if request.agent_id else None
                )

                if search_results:
                    context_parts = [result["text"] for result in search_results]
                    context = "\n\n".join(context_parts)

            # Stream response tokens (placeholder)
            response_id = str(uuid.uuid4())
            tokens = _generate_response(
                messages=request.messages,
                context=context,
                model=request.model,
                temperature=request.temperature
            ).split()

            # Send initial message
            yield f"data: {json.dumps({'id': response_id, 'model': request.model, 'created': datetime.utcnow().isoformat()})}\n\n"

            # Stream tokens
            for i, token in enumerate(tokens):
                chunk = {
                    "id": response_id,
                    "choices": [{
                        "delta": {"content": token + " "},
                        "index": 0
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)  # Simulate streaming delay

            # Send done message
            yield f"data: {json.dumps({'id': response_id, 'choices': [{'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    Create a new conversation.

    Initializes a conversation session for tracking chat history.
    """
    try:
        import uuid

        conversation_id = str(uuid.uuid4())

        # Store conversation metadata
        await memory_manager.store_memory(
            memory_id=conversation_id,
            text=f"Conversation started: {request.title or 'Untitled'}",
            metadata={
                "type": "conversation_metadata",
                "agent_id": request.agent_id,
                "title": request.title,
                "created_at": datetime.utcnow().isoformat(),
                **request.metadata or {}
            },
            agent_id=request.agent_id
        )

        return ConversationResponse(
            id=conversation_id,
            agent_id=request.agent_id,
            title=request.title,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            message_count=0,
            metadata=request.metadata or {}
        )

    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    Get conversation details.

    Returns information about a specific conversation.
    """
    try:
        # Get conversation metadata
        conversation = await memory_manager.get_memory(conversation_id)

        if not conversation or conversation.metadata.get("type") != "conversation_metadata":
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Count messages in conversation
        messages = await memory_manager.list_memories(
            filters={
                "metadata.conversation_id": conversation_id,
                "metadata.type": "chat_interaction"
            }
        )

        return ConversationResponse(
            id=conversation_id,
            agent_id=conversation.metadata["agent_id"],
            title=conversation.metadata.get("title"),
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=len(messages[0]) if messages else 0,
            metadata={k: v for k, v in conversation.metadata.items()
                     if k not in ["type", "agent_id", "title"]}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum messages to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    Get messages from a conversation.

    Returns the chat history for a conversation.
    """
    try:
        # Get messages
        messages, total = await memory_manager.list_memories(
            filters={
                "metadata.conversation_id": conversation_id,
                "metadata.type": "chat_interaction"
            },
            page=offset // limit + 1,
            page_size=limit
        )

        # Parse messages
        chat_messages = []
        for memory in messages:
            # Extract user and assistant messages from stored format
            parts = memory.text.split("\nAssistant: ", 1)
            if len(parts) == 2:
                user_part = parts[0].replace("User: ", "")
                assistant_part = parts[1]

                chat_messages.extend([
                    {
                        "role": "user",
                        "content": user_part,
                        "timestamp": memory.created_at
                    },
                    {
                        "role": "assistant",
                        "content": assistant_part,
                        "timestamp": memory.created_at
                    }
                ])

        return {
            "conversation_id": conversation_id,
            "messages": chat_messages,
            "total": total * 2,  # Each memory contains 2 messages
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Failed to get messages for conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    Delete a conversation.

    Removes the conversation and all associated messages.
    """
    try:
        # Get all messages in conversation
        messages, _ = await memory_manager.list_memories(
            filters={
                "metadata.conversation_id": conversation_id,
                "metadata.type": "chat_interaction"
            }
        )

        # Delete all messages
        for message in messages:
            await memory_manager.delete_memory(message.id, hard_delete=True)

        # Delete conversation metadata
        await memory_manager.delete_memory(conversation_id, hard_delete=True)

        return {
            "conversation_id": conversation_id,
            "deleted_messages": len(messages),
            "message": "Conversation deleted successfully"
        }

    except Exception as e:
        logger.error(f"Failed to delete conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """
    List available chat models.

    Returns information about available models and their capabilities.
    """
    models = [
        {
            "id": ChatModel.FAST,
            "name": "Fast Model",
            "description": "Quick responses with good accuracy",
            "max_tokens": 2048,
            "supports_streaming": True,
            "supports_memory": True
        },
        {
            "id": ChatModel.BALANCED,
            "name": "Balanced Model",
            "description": "Balanced speed and accuracy",
            "max_tokens": 4096,
            "supports_streaming": True,
            "supports_memory": True
        },
        {
            "id": ChatModel.ACCURATE,
            "name": "Accurate Model",
            "description": "Highest accuracy, slower responses",
            "max_tokens": 4096,
            "supports_streaming": True,
            "supports_memory": True
        }
    ]

    return {"models": models}


def _generate_response(
    messages: List[Message],
    context: str,
    model: ChatModel,
    temperature: float
) -> str:
    """Generate a response (placeholder for LLM integration)."""
    # This is a placeholder - in production, this would call an actual LLM

    # Extract the last user message
    user_message = ""
    for msg in reversed(messages):
        if msg.role == "user":
            user_message = msg.content
            break

    # Generate a simple response based on the model
    if model == ChatModel.FAST:
        response = f"Quick response to: {user_message[:50]}..."
    elif model == ChatModel.ACCURATE:
        response = f"Detailed and accurate response to your query about {user_message[:30]}..."
    else:
        response = f"Balanced response regarding: {user_message[:40]}..."

    if context:
        response += f"\n\nBased on relevant context from memory system."

    return response
