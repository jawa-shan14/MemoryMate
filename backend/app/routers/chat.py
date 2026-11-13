"""Chat router — main conversational endpoint."""
import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse, ConversationOut, MessageOut
from app.services.conversation_service import ConversationService

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Send a message and receive an AI response with memory context."""
    try:
        svc = ConversationService(db)
        return await svc.chat(request)
    except Exception as e:
        logger.error(
            "chat_error",
            error_type=type(e).__name__,
            status_code=getattr(e, "status_code", None),
        )
        raise HTTPException(
            status_code=502,
            detail="Chat request failed. Check server logs for sanitized diagnostics.",
        ) from e


@router.get("/conversations/{user_id}")
async def list_conversations(
    user_id: str,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List all conversations for a user."""
    svc = ConversationService(db)
    return await svc.get_conversations(user_id, limit=limit, offset=offset)


@router.get("/conversations/{user_id}/{conversation_id}/messages")
async def get_messages(
    user_id: str,
    conversation_id: str,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Get messages for a specific conversation."""
    svc = ConversationService(db)
    return await svc.get_conversation_messages(conversation_id, user_id, limit=limit)


@router.delete("/conversations/{user_id}/{conversation_id}")
async def delete_conversation(
    user_id: str,
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation and all its messages."""
    svc = ConversationService(db)
    deleted = await svc.delete_conversation(conversation_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True, "conversation_id": conversation_id}
