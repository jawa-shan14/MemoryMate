"""Memory service — manages episodic, semantic and consolidated memories."""
import json
import re
import structlog
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.memory import Memory, MemoryType
from app.models.user import User
from app.services.embedding_service import get_embedding_service
from app.services.llm_service import get_llm_service
from app.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

PROFILE_QUERY_PATTERN = re.compile(
    r"\b(?:who am i|what do you know about me|tell me about myself|describe me)\b",
    re.IGNORECASE,
)
PROFILE_MEMORY_QUERY = "Tell me the name, college, field of study, and interests of the user."
PROFILE_SIMILARITY_THRESHOLD = 0.20


class MemoryService:
    """Manages all memory layers for a user."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedding_svc = get_embedding_service()
        self.llm_svc = None

    async def _resolve_user_id(self, user_id: str) -> str:
        result = await self.db.execute(
            select(User.id).where((User.id == user_id) | (User.username == user_id))
        )
        return result.scalar_one_or_none() or user_id

    @staticmethod
    def _parse_facts(response: str) -> list[dict[str, str]]:
        match = re.search(r"\{.*\}|\[.*\]", response, flags=re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict):
            parsed = parsed.get("facts", [])
        if not isinstance(parsed, list):
            return []

        facts = []
        for item in parsed[:10]:
            if isinstance(item, str):
                content = item.strip()
                fact = {"content": content}
            elif isinstance(item, dict):
                content = str(item.get("content", "")).strip()
                fact = {
                    key: str(item.get(key, "")).strip()
                    for key in ("content", "subject", "predicate", "value")
                }
            else:
                continue
            if content:
                facts.append(fact)
        return facts

    async def _active_semantic_memories(self, user_id: str) -> list[Memory]:
        result = await self.db.execute(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.memory_type == MemoryType.semantic,
                Memory.is_active.is_(True),
            )
        )
        return list(result.scalars().all())

    # ── Episodic Memory ──────────────────────────────────────────────────────

    async def store_episodic_memory(
        self,
        user_id: str,
        content: str,
        conversation_id: str | None = None,
        message_ids: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> Memory:
        """Store a conversational event as an episodic memory with embedding."""
        user_id = await self._resolve_user_id(user_id)
        embedding = self.embedding_svc.embed(content)
        memory = Memory(
            user_id=user_id,
            memory_type=MemoryType.episodic,
            content=content,
            embedding=embedding,
            source_conversation_id=conversation_id if conversation_id else None,
            source_message_ids=message_ids or [],
            tags=tags or [],
            confidence=1.0,
        )
        self.db.add(memory)
        await self.db.commit()
        await self.db.refresh(memory)
        logger.info("episodic_memory_stored", user_id=user_id, memory_id=str(memory.id))
        return memory

    # ── Semantic Memory ──────────────────────────────────────────────────────

    async def extract_and_store_semantic_memories(
        self, user_id: str, conversation_text: str, conversation_id: str
    ) -> list[Memory]:
        """Use LLM to extract facts/preferences from conversation and store them."""
        system = """You are a memory extraction specialist. Extract factual statements, preferences, interests, and goals explicitly stated by the user.
Return ONLY JSON in this format: {"facts":[{"content":"User prefers Python","subject":"user","predicate":"prefers","value":"Python"}]}.
Focus on: personal info, preferences, goals, skills, interests, relationships, work/projects.
Be specific and avoid vague statements. Maximum 10 facts. If nothing notable, return {"facts": []}.
Use the same subject and predicate for facts that refer to the same property so changed values can replace outdated facts.
Do not treat assistant statements as facts about the user unless the user confirmed them."""

        user_id = await self._resolve_user_id(user_id)
        if self.llm_svc is None:
            self.llm_svc = get_llm_service()

        prompt = f"Extract memorable facts from this conversation:\n\n{conversation_text}"
        try:
            response = await self.llm_svc.extract_structured(prompt, system)
            facts = self._parse_facts(response)
        except Exception as e:
            logger.error(
                "semantic_extraction_error",
                error_type=type(e).__name__,
                status_code=getattr(e, "status_code", None),
            )
            return []

        stored = []
        changed = False
        active_memories = await self._active_semantic_memories(user_id)
        for fact in facts:
            content = fact["content"]
            embedding = self.embedding_svc.embed(content)
            subject = fact.get("subject", "").casefold()
            predicate = fact.get("predicate", "").casefold()
            value = fact.get("value", "").casefold()

            matching_property = None
            if subject and predicate and value:
                matching_property = next(
                    (
                        memory for memory in active_memories
                        if str((memory.metadata_ or {}).get("subject", "")).casefold() == subject
                        and str((memory.metadata_ or {}).get("predicate", "")).casefold() == predicate
                    ),
                    None,
                )

            if matching_property:
                old_value = str((matching_property.metadata_ or {}).get("value", "")).casefold()
                if value and old_value == value:
                    matching_property.confidence = min(1.0, matching_property.confidence + 0.05)
                    changed = True
                    continue
                matching_property.is_active = False
                matching_property.tags = list(matching_property.tags or []) + ["superseded"]
                changed = True

            duplicate = None
            for existing in active_memories:
                if not existing.is_active or existing is matching_property or not existing.embedding:
                    continue
                similarity = self.embedding_svc.cosine_similarity(embedding, existing.embedding)
                if similarity >= 0.92:
                    duplicate = existing
                    break
            if duplicate:
                duplicate.confidence = min(1.0, duplicate.confidence + 0.05)
                changed = True
                continue

            memory = Memory(
                user_id=user_id,
                memory_type=MemoryType.semantic,
                content=content,
                embedding=embedding,
                source_conversation_id=conversation_id,
                confidence=0.85,
                tags=["extracted"] + (["updated"] if matching_property else []),
                metadata_={
                    key: fact[key]
                    for key in ("subject", "predicate", "value")
                    if fact.get(key)
                },
            )
            self.db.add(memory)
            stored.append(memory)
            active_memories.append(memory)
            changed = True
        if changed:
            await self.db.commit()
        logger.info("semantic_memories_stored", user_id=user_id, count=len(stored))
        return stored

    # ── Retrieval ────────────────────────────────────────────────────────────

    async def retrieve_relevant_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        memory_types: list[MemoryType] | None = None,
        min_confidence: float = 0.0,
    ) -> list[dict]:
        """Retrieve memories most relevant to the query using embedding similarity."""
        user_id = await self._resolve_user_id(user_id)
        profile_query = PROFILE_QUERY_PATTERN.search(query) is not None
        retrieval_query = PROFILE_MEMORY_QUERY if profile_query else query
        query_embedding = self.embedding_svc.embed(retrieval_query)
        types = memory_types or [MemoryType.episodic, MemoryType.semantic, MemoryType.consolidated]
        if profile_query and memory_types is None:
            types = [MemoryType.semantic]

        result = await self.db.execute(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.memory_type.in_(types),
                Memory.is_active.is_(True),
                Memory.embedding.isnot(None),
            )
        )
        all_memories = result.scalars().all()

        # Compute similarities in Python (pgvector ops require native vector column)
        scored = []
        for mem in all_memories:
            if mem.embedding:
                sim = self.embedding_svc.cosine_similarity(query_embedding, mem.embedding)
                scored.append((sim, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [
            item for item in scored
            if item[0] >= self._similarity_threshold(item[1], profile_query)
            and item[1].confidence >= min_confidence
        ][:top_k]

        # Update access count
        for _, mem in top:
            mem.access_count += 1
            mem.last_accessed = datetime.now(timezone.utc)
        if top:
            await self.db.commit()

        return [
            {
                "id": str(m.id),
                "content": m.content,
                "type": m.memory_type.value,
                "confidence": m.confidence,
                "similarity": round(sim, 4),
                "created_at": m.created_at.isoformat(),
                "tags": m.tags,
            }
            for sim, m in top
        ]

    @staticmethod
    def _similarity_threshold(memory: Memory, profile_query: bool) -> float:
        fact_threshold = min(
            settings.memory_similarity_threshold,
            settings.memory_fact_similarity_threshold,
        )
        if profile_query and memory.memory_type == MemoryType.semantic:
            return min(fact_threshold, PROFILE_SIMILARITY_THRESHOLD)
        if memory.memory_type in (MemoryType.episodic, MemoryType.semantic):
            return fact_threshold
        return settings.memory_similarity_threshold

    async def retrieve_hybrid_context(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        graph_context: dict | None = None,
    ) -> dict:
        """Return vector memories and graph facts as distinct, traceable context sources."""
        memories = await self.retrieve_relevant_memories(
            user_id=user_id,
            query=query,
            top_k=top_k,
            min_confidence=0.0,
        )
        graph_context = graph_context or {"entities": [], "relationships": [], "summary": ""}
        return {
            "memories": memories,
            "entities": graph_context.get("entities", []),
            "relationships": graph_context.get("relationships", []),
            "summary": graph_context.get("summary", ""),
        }

    async def _find_similar_memory(
        self, user_id: str, text: str, memory_type: MemoryType, threshold: float = 0.90
    ) -> Memory | None:
        """Find an existing memory that is very similar to text."""
        user_id = await self._resolve_user_id(user_id)
        embedding = self.embedding_svc.embed(text)
        result = await self.db.execute(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.memory_type == memory_type,
                Memory.is_active.is_(True),
            )
        )
        for mem in result.scalars():
            if mem.embedding:
                sim = self.embedding_svc.cosine_similarity(embedding, mem.embedding)
                if sim >= threshold:
                    return mem
        return None

    async def get_user_memories(
        self, user_id: str, memory_type: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[Memory]:
        user_id = await self._resolve_user_id(user_id)
        query = select(Memory).where(
            Memory.user_id == user_id,
            Memory.is_active.is_(True),
        ).order_by(Memory.created_at.desc()).limit(limit).offset(offset)
        if memory_type:
            query = query.where(Memory.memory_type == MemoryType(memory_type))
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_memory_stats(self, user_id: str) -> dict:
        user_id = await self._resolve_user_id(user_id)
        result = await self.db.execute(
            select(Memory.memory_type, func.count(Memory.id), func.avg(Memory.confidence))
            .where(Memory.user_id == user_id, Memory.is_active.is_(True))
            .group_by(Memory.memory_type)
        )
        rows = result.all()
        stats = {
            "total_memories": 0,
            "episodic_count": 0,
            "semantic_count": 0,
            "consolidated_count": 0,
            "avg_confidence": 0.0,
        }
        total_conf = 0.0
        for mtype, count, avg_conf in rows:
            stats[f"{mtype.value}_count"] = count
            stats["total_memories"] += count
            total_conf += (avg_conf or 0.0) * count
        if stats["total_memories"] > 0:
            stats["avg_confidence"] = round(total_conf / stats["total_memories"], 3)
        return stats

    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        user_id = await self._resolve_user_id(user_id)
        result = await self.db.execute(
            select(Memory).where(
                Memory.id == memory_id,
                Memory.user_id == user_id,
            )
        )
        mem = result.scalar_one_or_none()
        if not mem:
            return False
        mem.is_active = False
        await self.db.commit()
        return True

    async def update_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str | None = None,
        confidence: float | None = None,
        is_active: bool | None = None,
    ) -> Memory | None:
        user_id = await self._resolve_user_id(user_id)
        result = await self.db.execute(
            select(Memory).where(
                Memory.id == memory_id,
                Memory.user_id == user_id,
            )
        )
        mem = result.scalar_one_or_none()
        if not mem:
            return None
        if content is not None:
            mem.content = content
            mem.embedding = self.embedding_svc.embed(content)
            if mem.memory_type == MemoryType.semantic:
                metadata = dict(mem.metadata_ or {})
                for key in ("subject", "predicate", "value"):
                    metadata.pop(key, None)
                mem.metadata_ = metadata
        if confidence is not None:
            mem.confidence = confidence
        if is_active is not None:
            mem.is_active = is_active
        await self.db.commit()
        await self.db.refresh(mem)
        return mem
