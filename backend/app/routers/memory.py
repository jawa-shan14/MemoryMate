"""Memory router — CRUD for user memories."""
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas.memory import MemoryOut, MemorySearchRequest, MemoryUpdateRequest, MemoryStats
from app.services.memory_service import MemoryService

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/{user_id}", response_model=list[MemoryOut])
async def list_memories(
    user_id: str,
    memory_type: str | None = Query(None, description="Filter by type: episodic|semantic|consolidated"),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List all memories for a user."""
    svc = MemoryService(db)
    memories = await svc.get_user_memories(user_id, memory_type=memory_type, limit=limit, offset=offset)
    return [
        MemoryOut(
            id=str(m.id),
            memory_type=m.memory_type.value,
            content=m.content,
            confidence=m.confidence,
            tags=m.tags or [],
            source_conversation_id=m.source_conversation_id,
            is_active=m.is_active,
            access_count=m.access_count,
            created_at=m.created_at,
            updated_at=m.updated_at,
        )
        for m in memories
    ]


@router.post("/search", response_model=list[dict])
async def search_memories(
    request: MemorySearchRequest,
    db: AsyncSession = Depends(get_db),
):
    """Semantic search for relevant memories."""
    svc = MemoryService(db)
    from app.models.memory import MemoryType
    types = [MemoryType(t) for t in request.memory_types]
    return await svc.retrieve_relevant_memories(
        user_id=request.user_id,
        query=request.query,
        top_k=request.top_k,
        memory_types=types,
        min_confidence=request.min_confidence,
    )


@router.get("/{user_id}/stats", response_model=MemoryStats)
async def memory_stats(user_id: str, db: AsyncSession = Depends(get_db)):
    """Get memory statistics for a user."""
    svc = MemoryService(db)
    stats = await svc.get_memory_stats(user_id)
    return MemoryStats(**stats)


@router.patch("/{user_id}/{memory_id}", response_model=MemoryOut)
async def update_memory(
    user_id: str,
    memory_id: str,
    request: MemoryUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update a memory's content or confidence."""
    svc = MemoryService(db)
    updated = await svc.update_memory(
        memory_id=memory_id,
        user_id=user_id,
        content=request.content,
        confidence=request.confidence,
        is_active=request.is_active,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryOut(
        id=str(updated.id),
        memory_type=updated.memory_type.value,
        content=updated.content,
        confidence=updated.confidence,
        tags=updated.tags or [],
        source_conversation_id=updated.source_conversation_id,
        is_active=updated.is_active,
        access_count=updated.access_count,
        created_at=updated.created_at,
        updated_at=updated.updated_at,
    )


@router.delete("/{user_id}/{memory_id}")
async def delete_memory(
    user_id: str,
    memory_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Soft-delete a memory."""
    svc = MemoryService(db)
    deleted = await svc.delete_memory(memory_id=memory_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True, "memory_id": memory_id}
