"""Evaluation schemas."""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class EvalRunRequest(BaseModel):
    config_name: str  # "no_memory" | "vector_memory" | "graph_memory"
    dataset_name: str = "default"
    user_id: str = "eval-user"
    max_questions: Optional[int] = None


class MetricResult(BaseModel):
    metric_name: str
    value: float
    details: dict = {}


class EvalRunResult(BaseModel):
    run_id: str
    config_name: str
    dataset_name: str
    started_at: datetime
    completed_at: Optional[datetime]
    status: str
    metrics: list[MetricResult] = []
    total_questions: int = 0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    errors: list[str] = []
