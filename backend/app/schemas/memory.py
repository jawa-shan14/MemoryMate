"""Memory schemas."""
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import Optional


class MemoryOut(BaseModel):
    id: str
    memory_type: str
    content: str
    confidence: float
    tags: list[str] = []
    source_conversation_id: Optional[str] = None
    is_active: bool
    access_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MemorySearchRequest(BaseModel):
    user_id: str
    query: str
    memory_types: list[str] = Field(default_factory=lambda: ["episodic", "semantic", "consolidated"])
    top_k: int = Field(default=10, ge=1, le=100)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MemoryUpdateRequest(BaseModel):
    content: Optional[str] = None
    is_active: Optional[bool] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class MemoryStats(BaseModel):
    total_memories: int
    episodic_count: int
    semantic_count: int
    consolidated_count: int
    avg_confidence: float
