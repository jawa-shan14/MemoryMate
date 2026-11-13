# MemoryMate — AI Assistant with Agentic Long-Term Memory, GraphRAG and Personalization

> A portfolio-grade AI research project demonstrating measurable comparisons between memory architectures for long-term conversational AI.

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       Next.js Frontend                          │
│  Chat │ Memory Explorer │ Knowledge Graph │ Eval Lab │ Settings │
└───────────────────────────┬─────────────────────────────────────┘
                            │ REST API
┌───────────────────────────▼─────────────────────────────────────┐
│                      FastAPI Backend                            │
│                                                                 │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ LLM Service│  │ Memory       │  │ Graph Service          │  │
│  │ (NVIDIA    │  │ Service      │  │ (Neo4j GraphRAG)       │  │
│  │ Nemotron)  │  │              │  │                        │  │
│  └────────────┘  │ ┌──────────┐│  └────────────────────────┘  │
│                  │ │Episodic  ││                                │
│  ┌────────────┐  │ │Memory    ││  ┌────────────────────────┐  │
│  │ Embedding  │  │ ├──────────┤│  │ Consolidation Service  │  │
│  │ Service    │  │ │Semantic  ││  │ (Scheduled background) │  │
│  │(all-MiniLM)│  │ │Memory    ││  └────────────────────────┘  │
│  └────────────┘  │ ├──────────┤│                                │
│                  │ │Long-Term ││  ┌────────────────────────┐  │
│                  │ │Memory    ││  │ Personalization Service│  │
│                  │ └──────────┘│  └────────────────────────┘  │
│                  └──────────────┘                                │
└───────┬────────────────┬────────────────────────────────────────┘
        │                │
┌───────▼──────┐  ┌──────▼──────────┐
│ PostgreSQL   │  │     Neo4j       │
│ + pgvector   │  │  Knowledge      │
│              │  │   Graph DB      │
│ • Users      │  │ • Entities      │
│ • Convos     │  │ • Relationships │
│ • Messages   │  │ • Multi-hop     │
│ • Memories   │  │   queries       │
│ • Embeddings │  └─────────────────┘
└──────────────┘
```

## Three Memory Layers

| Layer | Storage | Purpose |
|-------|---------|---------|
| **Episodic** | SQLite JSON vectors locally; PostgreSQL optionally | Conversation events and their real sentence-transformer embeddings |
| **Semantic** | SQLite JSON vectors locally; PostgreSQL optionally | Nemotron-extracted facts, preferences, and superseded values |
| **Consolidated** | SQLite JSON vectors locally; PostgreSQL optionally | Long-term distilled memories from episodic |

Memory retrieval computes cosine similarity against the stored real embeddings; it does not generate fallback/random vectors. Semantic facts use duplicate detection and supersede an active fact when Nemotron identifies the same subject and predicate with a changed value. Conversation messages and semantic/episodic memory are stored in the database, so conversation IDs can be resumed in later requests or server sessions.

## Evaluation Framework

Three configurations tested head-to-head on a synthetic persona benchmark:

| Config | Memory | Retrieval | Expected Strength |
|--------|--------|-----------|-------------------|
| **Baseline** | None | None | Reference point |
| **VectorRAG** | Episodic + Semantic | pgvector cosine similarity | Factual and preference recall |
| **GraphRAG** | All layers | Neo4j + pgvector hybrid | Multi-hop reasoning |

Metrics: Factual Recall · Preference Recall · Temporal Reasoning · Multi-hop · Contradiction Handling · Retrieval Precision/Recall · Latency · Token Consumption

---

## Quick Start

### Prerequisites

- Python 3.11–3.13 (3.14 is not yet fully supported by all native wheels on Windows; use 3.12/3.13 if you hit build-tool errors)
- Node.js 18+
- SQLite is the local-development default; PostgreSQL 15+ with pgvector is optional
- Neo4j Community Edition 5.x (optional — GraphRAG features degrade gracefully without it)
- NVIDIA API key from [build.nvidia.com](https://build.nvidia.com)
- On Windows, install Visual Studio Build Tools if you want to compile native Python packages from source

### 1. Database Setup

**Local development:** no database server is required. The sample environment uses `sqlite+aiosqlite:///./memorymate.db`.

**Optional PostgreSQL:**
```sql
CREATE USER memorymate WITH PASSWORD 'memorymate';
CREATE DATABASE memorymate OWNER memorymate;
\c memorymate
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

**Neo4j (optional):** Run Neo4j Community Edition and set `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` in `backend/.env` to the values for your local database. Use a unique local password; do not commit `.env` or use the sample placeholder as a real credential.

> Or use Docker Compose (requires Docker Desktop):
> ```bash
> docker compose up -d
> ```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# or: source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env — add your NVIDIA_API_KEY

# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs at: http://localhost:8000/docs

Run backend tests from `backend/`:
```bash
python -m pytest -q tests
```

The Neo4j integration test uses the configured local database and skips when it cannot connect. Configure `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` in `backend/.env`, then run the same test command to exercise real node/relationship writes, reads, multi-hop retrieval, and hybrid retrieval.

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

App at: http://localhost:3000

---

## Project Structure

```
MemoryMate/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app + lifespan
│   │   ├── config.py                  # Pydantic settings
│   │   ├── database.py                # Async SQLAlchemy engine
│   │   ├── models/                    # ORM models
│   │   │   ├── user.py
│   │   │   ├── conversation.py
│   │   │   ├── memory.py
│   │   │   └── personalization.py
│   │   ├── schemas/                   # Pydantic I/O schemas
│   │   ├── routers/                   # FastAPI route handlers
│   │   │   ├── chat.py
│   │   │   ├── memory.py
│   │   │   ├── graph.py
│   │   │   ├── personalization.py
│   │   │   └── evaluation.py
│   │   └── services/
│   │       ├── llm_service.py         # NVIDIA Nemotron client
│   │       ├── conversation_service.py # Main chat pipeline
│   │       ├── memory_service.py      # 3-layer memory management
│   │       ├── embedding_service.py   # all-MiniLM-L6-v2
│   │       ├── graph_service.py       # Neo4j GraphRAG
│   │       ├── consolidation_service.py # Background memory consolidation
│   │       ├── personalization_service.py # Preference tracking
│   │       └── evaluation_service.py  # Benchmark runner
│   ├── evaluation/
│   │   ├── datasets/default.json      # 15-question synthetic persona
│   │   └── results/                   # JSON result files per run
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── src/
│       ├── app/                       # Next.js App Router
│       │   ├── chat/page.tsx          # Main chat interface
│       │   ├── memory/page.tsx        # Memory Explorer
│       │   ├── graph/page.tsx         # Knowledge Graph viz
│       │   ├── personalization/page.tsx
│       │   ├── evaluation/page.tsx    # Eval Lab + charts
│       │   └── settings/page.tsx
│       ├── components/
│       │   ├── Navigation.tsx
│       │   └── Providers.tsx
│       ├── lib/
│       │   ├── api.ts                 # Typed API client
│       │   └── utils.ts
│       └── stores/useStore.ts         # Zustand state
├── docker-compose.yml
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send message, get response with memory |
| `GET` | `/api/chat/conversations` | List user conversations |
| `GET` | `/api/chat/conversations/{id}/messages` | Get conversation messages |
| `GET` | `/api/memory/` | List user memories |
| `POST` | `/api/memory/search` | Semantic memory search |
| `GET` | `/api/memory/stats` | Memory statistics |
| `DELETE` | `/api/memory/{id}` | Soft-delete a memory |
| `POST` | `/api/memory/consolidate` | Trigger consolidation |
| `GET` | `/api/graph/` | Full graph for visualization |
| `GET` | `/api/graph/stats` | Graph statistics |
| `GET` | `/api/graph/context?query=...` | Retrieve graph context |
| `POST` | `/api/personalization/feedback` | Record message feedback |
| `GET` | `/api/personalization/preferences` | Get user preferences |
| `POST` | `/api/evaluation/run` | Start a benchmark run |
| `GET` | `/api/evaluation/results` | List past results |
| `GET` | `/health` | Health check |

Full interactive docs: http://localhost:8000/docs

---

## Evaluation Results

Evaluation results are stored in `backend/evaluation/results/` as JSON files.

To run the full benchmark suite:
```bash
# From the Evaluation Lab UI: select each config and click Run
# Or via API:
curl -X POST http://localhost:8000/api/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"config_name": "no_memory", "dataset_name": "default"}'

curl -X POST http://localhost:8000/api/evaluation/run \
  -d '{"config_name": "vector_memory", "dataset_name": "default"}'

curl -X POST http://localhost:8000/api/evaluation/run \
  -d '{"config_name": "graph_memory", "dataset_name": "default"}'
```

Results include per-question answers, keyword match scores, latency, and token consumption — categorized into: factual recall, preference recall, temporal recall, multi-hop, and relationship recall.

---

## Research Contribution

The core contribution of this project is a **reproducible comparison of three memory architectures** for long-term conversational AI:

1. **Without memory** — demonstrates baseline LLM limitations (no recall across sessions)
2. **With vector memory** — shows significant improvement in factual and preference recall
3. **With graph + vector memory** — shows additional gains specifically in multi-hop and relational reasoning

All experiments use a fixed synthetic persona (Alex Chen), deterministic question set, and documented keyword-based scoring rubric. Results can be regenerated by running the evaluation endpoints.

---

## Notes on Graceful Degradation

- **Without NVIDIA API key**: Chat will fail with an authentication error. The UI, memory explorer, and evaluation UI remain functional.
- **Without PostgreSQL**: Local development continues with SQLite.
- **Without Neo4j**: Graph features are skipped. `graph.available = false` is returned. Vector memory continues to function.

---

## License

MIT © 2026
