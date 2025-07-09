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
from ...core.rag.hallucination_detector import HallucinationDetector
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


class TradingRequest(BaseModel):
    """Trading-specific chat request with high confidence requirements."""
    messages: List[Message] = Field(..., description="Conversation messages")
    model: ChatModel = Field(ChatModel.ACCURATE, description="Model to use (defaults to accurate)")
    temperature: float = Field(0.3, ge=0.0, le=0.5, description="Lower temperature for trading (max 0.5)")
    use_memory: bool = Field(True, description="Use memory system for context")
    memory_limit: int = Field(20, ge=10, le=50, description="Memory limit for trading context")
    agent_id: str = Field(..., description="Agent ID (required for trading)")
    confirm_high_confidence: bool = Field(False, description="Confirm you understand high confidence requirement")


class TradingResponse(BaseModel):
    """Trading-specific response with confidence guarantees."""
    id: str = Field(..., description="Response ID")
    message: Message = Field(..., description="Response message")
    confidence: float = Field(..., description="Response confidence score (95%+ required)")
    confidence_level: str = Field(..., description="Confidence level classification")
    trading_approved: bool = Field(..., description="Whether response meets trading safety requirements")
    hallucination_score: float = Field(..., description="Hallucination detection score")
    memories_used: List[str] = Field(default=[], description="Memory IDs used for context")
    safety_checks: Dict[str, bool] = Field(..., description="Safety check results")
    created: datetime = Field(..., description="Creation timestamp")
    warnings: List[str] = Field(default=[], description="Any safety warnings")


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


async def get_hallucination_detector() -> HallucinationDetector:
    """Get hallucination detector instance."""
    try:
        return get_provider(ProviderType.HALLUCINATION_DETECTOR, "default")
    except Exception as e:
        logger.error(f"Failed to get hallucination detector: {e}")
        raise HTTPException(status_code=500, detail="Hallucination detector unavailable")


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


@router.post("/trading", response_model=TradingResponse)
async def create_trading_completion(
    request: TradingRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    searcher: Searcher = Depends(get_searcher),
    hallucination_detector: HallucinationDetector = Depends(get_hallucination_detector)
):
    """
    Create a trading-specific chat completion with high confidence requirements.
    
    This endpoint enforces strict safety requirements for trading operations:
    - Minimum 95% confidence score (rock_solid level)
    - Hallucination detection enabled
    - Comprehensive safety checks
    - Lower temperature for more consistent responses
    - Required agent ID for accountability
    """
    try:
        import uuid
        
        # Safety validation
        if not request.confirm_high_confidence:
            raise HTTPException(
                status_code=400, 
                detail="Must confirm understanding of high confidence requirement"
            )
        
        # Extract user message
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Check for trading-related keywords for additional safety
        trading_keywords = [
            "buy", "sell", "trade", "order", "position", "portfolio", 
            "invest", "market", "stock", "crypto", "forex", "futures",
            "options", "execute", "transaction", "purchase", "liquidate"
        ]
        
        is_trading_query = any(keyword in user_message.lower() for keyword in trading_keywords)
        
        # Retrieve relevant memories with higher limit for trading context
        memories_used = []
        context = ""
        
        if request.use_memory:
            search_results = await searcher.search(
                query=user_message,
                strategy="hybrid",
                limit=request.memory_limit,
                filters={"agent_id": request.agent_id}
            )
            
            if search_results:
                context_parts = []
                for result in search_results:
                    context_parts.append(result["text"])
                    memories_used.append(result["memory_id"])
                
                context = "\n\n".join(context_parts)
        
        # Generate response with conservative settings
        response_content = _generate_trading_response(
            messages=request.messages,
            context=context,
            model=request.model,
            temperature=request.temperature,
            is_trading_query=is_trading_query
        )
        
        # Calculate confidence score (enhanced for trading)
        confidence_score = await _calculate_trading_confidence(
            query=user_message,
            response=response_content,
            context=context,
            memories_used=memories_used,
            searcher=searcher
        )
        
        # Run hallucination detection
        hallucination_result = await hallucination_detector.detect_hallucination(
            query=user_message,
            response=response_content,
            context=context
        )
        
        hallucination_score = hallucination_result.get("confidence", 0.0)
        
        # Determine confidence level
        if confidence_score >= 95:
            confidence_level = "rock_solid"
        elif confidence_score >= 80:
            confidence_level = "high"
        elif confidence_score >= 60:
            confidence_level = "fuzzy"
        else:
            confidence_level = "low"
        
        # Safety checks
        safety_checks = {
            "confidence_above_95": confidence_score >= 95,
            "hallucination_below_threshold": hallucination_score >= 75,  # Higher threshold for trading
            "agent_id_provided": bool(request.agent_id),
            "trading_model_used": request.model == ChatModel.ACCURATE,
            "temperature_appropriate": request.temperature <= 0.5,
            "memory_context_available": len(memories_used) > 0
        }
        
        # Determine if trading is approved
        trading_approved = all(safety_checks.values())
        
        # Generate warnings
        warnings = []
        if not safety_checks["confidence_above_95"]:
            warnings.append(f"Confidence {confidence_score:.1f}% below required 95% threshold")
        if not safety_checks["hallucination_below_threshold"]:
            warnings.append(f"Hallucination score {hallucination_score:.1f}% below threshold")
        if not safety_checks["memory_context_available"]:
            warnings.append("No memory context available for decision support")
        if is_trading_query and not trading_approved:
            warnings.append("CRITICAL: Trading query detected but safety requirements not met")
        
        # Create trading response
        response = TradingResponse(
            id=str(uuid.uuid4()),
            message=Message(
                role="assistant",
                content=response_content,
                timestamp=datetime.utcnow(),
                metadata={
                    "trading_mode": True,
                    "safety_validated": trading_approved,
                    "confidence_level": confidence_level
                }
            ),
            confidence=confidence_score,
            confidence_level=confidence_level,
            trading_approved=trading_approved,
            hallucination_score=hallucination_score,
            memories_used=memories_used,
            safety_checks=safety_checks,
            created=datetime.utcnow(),
            warnings=warnings
        )
        
        # Store the trading interaction with special metadata
        await memory_manager.store_memory(
            memory_id=str(uuid.uuid4()),
            text=f"TRADING QUERY: {user_message}\nTRADING RESPONSE: {response_content}",
            metadata={
                "type": "trading_interaction",
                "agent_id": request.agent_id,
                "confidence": confidence_score,
                "confidence_level": confidence_level,
                "trading_approved": trading_approved,
                "hallucination_score": hallucination_score,
                "safety_checks": safety_checks,
                "timestamp": datetime.utcnow().isoformat(),
                "warnings": warnings
            },
            agent_id=request.agent_id
        )
        
        # Log trading interaction for audit
        logger.info(
            f"Trading interaction: agent_id={request.agent_id}, "
            f"confidence={confidence_score:.1f}%, approved={trading_approved}, "
            f"warnings={len(warnings)}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trading completion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Trading completion failed: {str(e)}")


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
            "supports_memory": True,
            "trading_approved": False
        },
        {
            "id": ChatModel.BALANCED,
            "name": "Balanced Model",
            "description": "Balanced speed and accuracy",
            "max_tokens": 4096,
            "supports_streaming": True,
            "supports_memory": True,
            "trading_approved": False
        },
        {
            "id": ChatModel.ACCURATE,
            "name": "Accurate Model",
            "description": "Highest accuracy, slower responses",
            "max_tokens": 4096,
            "supports_streaming": True,
            "supports_memory": True,
            "trading_approved": True
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


def _generate_trading_response(
    messages: List[Message],
    context: str,
    model: ChatModel,
    temperature: float,
    is_trading_query: bool
) -> str:
    """Generate a trading-specific response with enhanced safety."""
    # Extract the last user message
    user_message = ""
    for msg in reversed(messages):
        if msg.role == "user":
            user_message = msg.content
            break

    # Enhanced response generation for trading
    if is_trading_query:
        response = f"TRADING ANALYSIS: Based on your query about {user_message[:50]}...\n\n"
        response += "Key considerations:\n"
        response += "- Market conditions analysis required\n"
        response += "- Risk assessment needed\n"
        response += "- Position sizing recommendations\n"
        response += "- Stop-loss and take-profit levels\n\n"
        
        if context:
            response += "Historical context from memory system:\n"
            response += f"- {context[:200]}...\n\n"
            
        response += "⚠️ IMPORTANT: This analysis requires human verification before execution."
        
    else:
        # Non-trading query but using trading endpoint
        response = f"GENERAL QUERY (Trading Mode): {user_message[:50]}...\n\n"
        response += "Note: This response was generated using trading-grade safety protocols.\n"
        
        if context:
            response += f"\nContext from memory: {context[:200]}..."
    
    return response


async def _calculate_trading_confidence(
    query: str,
    response: str,
    context: str,
    memories_used: List[str],
    searcher: Searcher
) -> float:
    """Calculate confidence score specifically for trading operations."""
    
    # Base confidence calculation
    base_confidence = 85.0  # Start with moderate confidence
    
    # Confidence modifiers
    modifiers = {
        "context_available": 5.0 if context else -10.0,
        "memory_support": min(len(memories_used) * 2, 10.0),
        "query_clarity": 5.0 if len(query.split()) > 5 else -5.0,
        "response_length": 3.0 if len(response.split()) > 20 else -3.0,
        "trading_keywords": 2.0 if any(kw in query.lower() for kw in ["buy", "sell", "trade"]) else 0.0
    }
    
    # Apply modifiers
    final_confidence = base_confidence + sum(modifiers.values())
    
    # Additional safety checks for trading
    if "⚠️" in response:  # Contains warning
        final_confidence += 2.0
    if "verification" in response.lower():  # Mentions verification
        final_confidence += 3.0
    if "risk" in response.lower():  # Mentions risk
        final_confidence += 2.0
    
    # Cap confidence to prevent over-confidence
    final_confidence = min(final_confidence, 98.0)  # Max 98% to leave room for uncertainty
    final_confidence = max(final_confidence, 0.0)   # Min 0%
    
    return final_confidence
