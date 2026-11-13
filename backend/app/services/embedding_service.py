"""Embedding service using FastEmbed's ONNX runtime."""
import numpy as np
import structlog
from math import isfinite
from pathlib import Path
from tempfile import gettempdir
from app.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class EmbeddingService:
    """Generates text embeddings using a local sentence-transformer model."""

    def __init__(self):
        self._model = None
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self.cache_dir = str(Path(gettempdir()) / "memorymate-fastembed")

    def _load_model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "Embedding model dependency missing: install fastembed to enable memory retrieval."
                ) from exc
            logger.info("loading_embedding_model", model=self.model_name)
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
            )
            logger.info("embedding_model_loaded", model=self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a list of floats."""
        model = self._load_model()
        embedding = next(model.embed([text]))
        return self._normalize_embedding(embedding.tolist())

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts at once. More efficient than calling embed() in a loop."""
        if not texts:
            return []
        model = self._load_model()
        embeddings = model.embed(texts, batch_size=32)
        return [self._normalize_embedding(item.tolist()) for item in embeddings]

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        validated = self._validate_embedding(embedding)
        vector = np.asarray(validated, dtype=np.float64)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return validated
        return self._validate_embedding((vector / norm).tolist())

    def _validate_embedding(self, embedding: list[float]) -> list[float]:
        if len(embedding) != self.dimension:
            raise ValueError(
                f"Embedding model returned {len(embedding)} dimensions; expected {self.dimension}."
            )
        if not all(isfinite(value) for value in embedding):
            raise ValueError("Embedding model returned non-finite values.")
        return embedding

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        va = np.asarray(a, dtype=np.float64)
        vb = np.asarray(b, dtype=np.float64)
        if va.ndim != 1 or vb.ndim != 1 or va.shape != vb.shape:
            raise ValueError("Embedding vectors must be one-dimensional and have matching dimensions.")
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
