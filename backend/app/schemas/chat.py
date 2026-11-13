"""Chat request/response schemas."""
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import Optional


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_id: str = Field(default="default-user")
    conversation_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=10000)
    use_memory: bool = True
    use_graph: bool = True


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    response: str
    retrieved_memories: list[dict] = []
    retrieved_graph_context: list[dict] = []
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None


class ConversationOut(BaseModel):
    id: str
    title: str
    summary: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime
    metadata_: dict = {}

    class Config:
        from_attributes = True
