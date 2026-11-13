"""Health check router."""
import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.database import get_db
from app.services.graph_service import get_graph_service

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Check health of all backend components."""
    status = {"status": "ok", "components": {}}

    # Database check (SQLite for local dev, PostgreSQL optional)
    try:
        await db.execute(text("SELECT 1"))
        status["components"]["database"] = "ok"
    except Exception as e:
        status["components"]["database"] = f"error: {str(e)}"
        status["status"] = "degraded"

    # Neo4j check
    try:
        graph_svc = get_graph_service()
        neo4j_ok = await graph_svc._is_available()
        status["components"]["neo4j"] = "ok" if neo4j_ok else "unavailable"
    except Exception as e:
        logger.warning("neo4j_health_check_failed", error_type=type(e).__name__)
        status["components"]["neo4j"] = "unavailable"

    return status


@router.get("/ready")
async def readiness():
    """Simple readiness probe."""
    return {"ready": True}
