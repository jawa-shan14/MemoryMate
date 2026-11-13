"""Graph router — knowledge graph visualization and stats."""
import structlog
from fastapi import APIRouter, Depends, Query
from app.services.graph_service import get_graph_service

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/{user_id}")
async def get_user_graph(user_id: str):
    """Get full knowledge graph for a user (nodes + edges for visualization)."""
    graph_svc = get_graph_service()
    return await graph_svc.get_user_graph(user_id)


@router.get("/{user_id}/stats")
async def graph_stats(user_id: str):
    """Get graph statistics for a user."""
    graph_svc = get_graph_service()
    return await graph_svc.get_stats(user_id)


@router.get("/{user_id}/context")
async def retrieve_graph_context(
    user_id: str,
    query: str = Query(..., description="Search query for relevant graph context"),
):
    """Retrieve graph context relevant to a query."""
    graph_svc = get_graph_service()
    return await graph_svc.retrieve_context(user_id=user_id, query=query)
