"""Database engine and session management."""
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

try:
    import asyncpg  # noqa: F401
except ImportError:
    if settings.database_url.startswith(("postgresql", "postgres")):
        logger.warning("asyncpg_missing_fallback_to_sqlite", database_url=settings.database_url)
        settings.database_url = "sqlite+aiosqlite:///./memorymate.db"
        settings.database_url_sync = "sqlite:///./memorymate.db"


class Base(DeclarativeBase):
    pass


engine_kwargs = {
    "echo": settings.environment == "development",
    "connect_args": {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
}
if settings.database_url.startswith("sqlite"):
    engine_kwargs["poolclass"] = NullPool
else:
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20

engine = create_async_engine(settings.database_url, **engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Create all tables. Called on startup."""
    try:
        # Import all models to register them before create_all
        from app.models import user, conversation, memory, personalization  # noqa: F401
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified")
    except Exception:
        logger.warning("Database unavailable during startup; continuing without schema initialization")
