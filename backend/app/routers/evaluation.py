"""Evaluation router — run and retrieve evaluation results."""
import uuid
import asyncio
import structlog
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.schemas.evaluation import EvalRunRequest, EvalRunResult, MetricResult

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# In-memory store for eval runs (use DB in production)
_eval_runs: dict[str, EvalRunResult] = {}


async def _run_evaluation(run_id: str, request: EvalRunRequest):
    """Execute an evaluation run asynchronously."""
    run = _eval_runs[run_id]
    try:
        from app.services.llm_service import get_llm_service
        from app.services.memory_service import MemoryService
        from app.services.graph_service import get_graph_service
        from app.database import AsyncSessionLocal
        import time

        # Load dataset
        dataset = _get_eval_dataset(request.dataset_name)
        questions = dataset[:request.max_questions] if request.max_questions else dataset

        total_tokens = 0
        total_latency = 0.0
        correct = 0
        errors = []

        llm_svc = get_llm_service()

        for i, item in enumerate(questions):
            try:
                question = item["question"]
                expected = item.get("expected_answer", "")

                # Build messages based on config
                messages = [{"role": "user", "content": question}]

                if request.config_name in ("vector_memory", "graph_memory"):
                    async with AsyncSessionLocal() as db:
                        memory_svc = MemoryService(db)
                        memories = await memory_svc.retrieve_relevant_memories(
                            user_id=request.user_id, query=question, top_k=3
                        )
                    if memories:
                        mem_block = "\n".join(f"- {m['content']}" for m in memories)
                        messages.insert(0, {
                            "role": "system",
                            "content": f"Relevant memories:\n{mem_block}",
                        })

                if request.config_name == "graph_memory":
                    graph_svc = get_graph_service()
                    graph_ctx = await graph_svc.retrieve_context(
                        user_id=request.user_id, query=question
                    )
                    if graph_ctx.get("summary"):
                        messages[0]["content"] += f"\n\nGraph context: {graph_ctx['summary']}"

                start = time.monotonic()
                result = await llm_svc.chat(messages, temperature=0.1, max_tokens=256)
                latency = (time.monotonic() - start) * 1000
                total_latency += latency
                if result.get("tokens_used"):
                    total_tokens += result["tokens_used"]

                # Simple correctness check
                answer = result["content"].lower()
                if expected and any(kw.lower() in answer for kw in expected.split()[:5]):
                    correct += 1

            except Exception as e:
                errors.append(f"Q{i}: {str(e)}")
                logger.error("eval_question_error", run_id=run_id, question_idx=i, error=str(e))

        n = len(questions)
        accuracy = correct / n if n > 0 else 0.0
        avg_latency = total_latency / n if n > 0 else 0.0

        run.metrics = [
            MetricResult(metric_name="accuracy", value=round(accuracy, 4)),
            MetricResult(metric_name="avg_latency_ms", value=round(avg_latency, 1)),
            MetricResult(metric_name="total_tokens", value=float(total_tokens)),
        ]
        run.total_questions = n
        run.avg_latency_ms = avg_latency
        run.total_tokens = total_tokens
        run.errors = errors
        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)

    except Exception as e:
        run.status = "failed"
        run.errors = [str(e)]
        run.completed_at = datetime.now(timezone.utc)
        logger.error("eval_run_failed", run_id=run_id, error=str(e))

    _eval_runs[run_id] = run


def _get_eval_dataset(dataset_name: str) -> list[dict]:
    """Return evaluation dataset. Extend with real datasets as needed."""
    default_dataset = [
        {
            "question": "What is the capital of France?",
            "expected_answer": "Paris",
        },
        {
            "question": "Who wrote the theory of relativity?",
            "expected_answer": "Albert Einstein",
        },
        {
            "question": "What programming language is known for its use in data science?",
            "expected_answer": "Python",
        },
        {
            "question": "What is machine learning?",
            "expected_answer": "machine learning artificial intelligence",
        },
        {
            "question": "What does API stand for?",
            "expected_answer": "Application Programming Interface",
        },
    ]
    return default_dataset


@router.post("/run", response_model=EvalRunResult)
async def start_eval_run(request: EvalRunRequest, background_tasks: BackgroundTasks):
    """Start an evaluation run in the background."""
    run_id = str(uuid.uuid4())
    run = EvalRunResult(
        run_id=run_id,
        config_name=request.config_name,
        dataset_name=request.dataset_name,
        started_at=datetime.now(timezone.utc),
        completed_at=None,
        status="running",
    )
    _eval_runs[run_id] = run
    background_tasks.add_task(_run_evaluation, run_id, request)
    return run


@router.get("/run/{run_id}", response_model=EvalRunResult)
async def get_eval_run(run_id: str):
    """Get evaluation run status and results."""
    run = _eval_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run


@router.get("/runs")
async def list_eval_runs():
    """List all evaluation runs."""
    return list(_eval_runs.values())
