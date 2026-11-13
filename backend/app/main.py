"""Application entrypoint for the MemoryMate backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db
from app.routers import chat, evaluation, graph, health, memory

settings = get_settings()

app = FastAPI(
    title="MemoryMate",
    version="0.1.0",
    description="AI assistant with agentic long-term memory, GraphRAG and personalization.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router_module in (health, chat, memory, graph, evaluation):
    app.include_router(router_module.router)


@app.on_event("startup")
async def startup_event() -> None:
    await init_db()


@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    return {"service": "MemoryMate", "status": "ok"}
