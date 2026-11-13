import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.conversation import Conversation, Message, MessageRole
from app.models.memory import Memory, MemoryType
from app.models.user import User
from app.routers.memory import list_memories
from app.services.conversation_service import ConversationService
from app.services.embedding_service import EmbeddingService
from app.services.memory_service import MemoryService


class FixedEmbeddingService:
    def embed(self, text):
        return [0.0, 1.0] if "java" in text.lower() else [1.0, 0.0]

    @staticmethod
    def cosine_similarity(left, right):
        return EmbeddingService().cosine_similarity(left, right)


class QueuedLLMService:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def extract_structured(self, prompt, system):
        return next(self.responses)


@pytest.mark.asyncio
async def test_conversation_history_is_available_in_a_new_session(session_factory):
    async with session_factory() as session:
        user = User(username="persistent-user", display_name="Persistent User")
        session.add(user)
        await session.flush()
        conversation = Conversation(user_id=user.id, title="History")
        session.add(conversation)
        await session.flush()
        created_at = datetime.now(timezone.utc)
        session.add_all([
            Message(
                conversation_id=conversation.id,
                role=MessageRole.user,
                content="Remember my preferred language.",
                created_at=created_at,
            ),
            Message(
                conversation_id=conversation.id,
                role=MessageRole.assistant,
                content="I will remember.",
                created_at=created_at + timedelta(microseconds=1),
            ),
        ])
        await session.commit()
        conversation_id = conversation.id

    async with session_factory() as new_session:
        history = await ConversationService(new_session).get_conversation_messages(
            conversation_id,
            "persistent-user",
        )

    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[0]["content"] == "Remember my preferred language."


@pytest.mark.asyncio
async def test_memory_list_includes_its_source_conversation(session_factory):
    async with session_factory() as session:
        user = User(username="memory-source-user", display_name="Memory Source User")
        session.add(user)
        await session.flush()
        conversation = Conversation(user_id=user.id, title="Source conversation")
        session.add(conversation)
        await session.flush()
        memory = Memory(
            user_id=user.id,
            memory_type=MemoryType.episodic,
            content="A remembered event",
            source_conversation_id=conversation.id,
        )
        session.add(memory)
        await session.commit()

        result = await list_memories(
            user.username, memory_type=None, limit=200, offset=0, db=session
        )

    assert len(result) == 1
    assert result[0].source_conversation_id == str(conversation.id)


def test_cosine_similarity_is_bounded_and_rejects_mismatched_vectors():
    embeddings = EmbeddingService()
    assert embeddings.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert embeddings.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert embeddings.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="matching dimensions"):
        embeddings.cosine_similarity([1.0], [1.0, 0.0])


def test_real_embedding_model_produces_semantically_useful_embeddings():
    embeddings = EmbeddingService()
    preference = embeddings.embed("I enjoy programming in Python.")
    paraphrase = embeddings.embed("Python is my favorite programming language.")
    unrelated = embeddings.embed("The weather is cloudy today.")

    assert len(preference) == embeddings.dimension
    assert embeddings.cosine_similarity(preference, paraphrase) > embeddings.cosine_similarity(
        preference, unrelated
    )


@pytest.mark.asyncio
async def test_retrieval_uses_real_vector_scores_threshold_and_confidence(session_factory, monkeypatch):
    monkeypatch.setattr("app.services.memory_service.settings.memory_similarity_threshold", 0.75)
    monkeypatch.setattr("app.services.memory_service.settings.memory_fact_similarity_threshold", 0.75)
    async with session_factory() as session:
        user = User(username="vector-user", display_name="Vector User")
        session.add(user)
        await session.flush()
        session.add_all([
            Memory(
                user_id=user.id,
                memory_type=MemoryType.semantic,
                content="high similarity",
                embedding=[1.0, 0.0],
                confidence=0.9,
            ),
            Memory(
                user_id=user.id,
                memory_type=MemoryType.semantic,
                content="below threshold",
                embedding=[0.6, 0.8],
                confidence=0.95,
            ),
            Memory(
                user_id=user.id,
                memory_type=MemoryType.semantic,
                content="low confidence",
                embedding=[0.99, 0.01],
                confidence=0.2,
            ),
        ])
        await session.commit()
        service = MemoryService(session)
        service.embedding_svc = FixedEmbeddingService()

        results = await service.retrieve_relevant_memories(
            user.username,
            "Python",
            min_confidence=0.5,
        )
        hybrid = await service.retrieve_hybrid_context(
            user.username,
            "Python",
            graph_context={
                "entities": [{"name": "Python"}],
                "relationships": [{"source": "User", "relationship": "USES", "target": "Python"}],
                "summary": "User uses Python",
            },
        )

    assert [memory["content"] for memory in results] == ["high similarity"]
    assert results[0]["similarity"] == pytest.approx(1.0)
    assert hybrid["memories"][0]["content"] == "high similarity"
    assert hybrid["entities"] == [{"name": "Python"}]
    assert hybrid["relationships"][0]["relationship"] == "USES"


@pytest.mark.asyncio
async def test_profile_and_episodic_queries_retrieve_below_default_threshold(
    session_factory, monkeypatch
):
    monkeypatch.setattr("app.services.memory_service.settings.memory_similarity_threshold", 0.55)
    monkeypatch.setattr("app.services.memory_service.settings.memory_fact_similarity_threshold", 0.35)

    class RecordingEmbeddingService:
        def __init__(self):
            self.queries = []

        def embed(self, text):
            self.queries.append(text)
            return [1.0, 0.0]

        @staticmethod
        def cosine_similarity(left, right):
            return EmbeddingService().cosine_similarity(left, right)

    async with session_factory() as session:
        user = User(username="cross-chat-user", display_name="Cross Chat User")
        session.add(user)
        await session.flush()
        session.add_all([
            Memory(
                user_id=user.id,
                memory_type=MemoryType.semantic,
                content="User's name is Aria",
                embedding=[0.25, 0.9682458366],
            ),
            Memory(
                user_id=user.id,
                memory_type=MemoryType.episodic,
                content="An old assistant answer says there are no memories",
                embedding=[1.0, 0.0],
            ),
        ])
        project_user = User(username="episodic-project-user", display_name="Project User")
        session.add(project_user)
        await session.flush()
        session.add_all([
            Memory(
                user_id=project_user.id,
                memory_type=MemoryType.semantic,
                content="User is building Project Quartz",
                embedding=[0.4, 0.9165151389],
            ),
            Memory(
                user_id=project_user.id,
                memory_type=MemoryType.episodic,
                content="Project uses FastAPI and SQLite",
                embedding=[0.4, 0.9165151389],
            ),
        ])
        await session.commit()

        service = MemoryService(session)
        embeddings = RecordingEmbeddingService()
        service.embedding_svc = embeddings

        profile_results = await service.retrieve_relevant_memories(
            user.username, "Who am I?"
        )
        project_results = await service.retrieve_relevant_memories(
            project_user.username, "What project am I building?"
        )

    assert embeddings.queries == [
        "Tell me the name, college, field of study, and interests of the user.",
        "What project am I building?",
    ]
    assert [memory["content"] for memory in profile_results] == ["User's name is Aria"]
    assert {memory["content"] for memory in project_results} == {
        "User is building Project Quartz",
        "Project uses FastAPI and SQLite",
    }


@pytest.mark.asyncio
async def test_episodic_memory_stores_real_vector_and_resolves_username(session_factory):
    async with session_factory() as session:
        user = User(username="episodic-user", display_name="Episodic User")
        session.add(user)
        await session.commit()
        service = MemoryService(session)
        service.embedding_svc = FixedEmbeddingService()

        memory = await service.store_episodic_memory(
            "episodic-user",
            "User: I like Python.",
            tags=["test"],
        )

    assert memory.user_id == user.id
    assert memory.memory_type is MemoryType.episodic
    assert memory.embedding == [1.0, 0.0]
    assert memory.tags == ["test"]


@pytest.mark.asyncio
async def test_manual_memory_update_reembeds_and_can_deactivate(session_factory):
    async with session_factory() as session:
        user = User(username="update-user", display_name="Update User")
        session.add(user)
        await session.flush()
        memory = Memory(
            user_id=user.id,
            memory_type=MemoryType.semantic,
            content="User likes Python",
            embedding=[1.0, 0.0],
            confidence=0.7,
            metadata_={"subject": "user", "predicate": "likes", "value": "Python"},
        )
        session.add(memory)
        await session.commit()

        service = MemoryService(session)
        service.embedding_svc = FixedEmbeddingService()
        updated = await service.update_memory(
            memory_id=memory.id,
            user_id=user.username,
            content="User likes Java",
            confidence=0.95,
            is_active=False,
        )

    assert updated is not None
    assert updated.content == "User likes Java"
    assert updated.embedding == [0.0, 1.0]
    assert updated.confidence == 0.95
    assert updated.is_active is False
    assert updated.metadata_ == {}


@pytest.mark.asyncio
async def test_semantic_facts_deduplicate_and_supersede_contradictions(session_factory):
    python_fact = {
        "facts": [{
            "content": "User prefers Python",
            "subject": "user",
            "predicate": "prefers",
            "value": "Python",
        }]
    }
    java_fact = {
        "facts": [{
            "content": "User prefers Java",
            "subject": "user",
            "predicate": "prefers",
            "value": "Java",
        }]
    }
    async with session_factory() as session:
        user = User(username="semantic-user", display_name="Semantic User")
        session.add(user)
        await session.flush()
        conversation = Conversation(user_id=user.id)
        session.add(conversation)
        await session.commit()
        user_id, conversation_id = user.id, conversation.id

        service = MemoryService(session)
        service.embedding_svc = FixedEmbeddingService()
        service.llm_svc = QueuedLLMService([
            json.dumps(python_fact),
            json.dumps(python_fact),
            json.dumps(java_fact),
        ])

        first = await service.extract_and_store_semantic_memories(
            user_id, "I prefer Python.", conversation_id
        )
        assert len(first) == 1
        original_id = first[0].id

        duplicate = await service.extract_and_store_semantic_memories(
            user_id, "I still prefer Python.", conversation_id
        )
        assert duplicate == []

        replacement = await service.extract_and_store_semantic_memories(
            user_id, "I now prefer Java.", conversation_id
        )
        assert len(replacement) == 1
        assert replacement[0].content == "User prefers Java"

        memories = (
            await session.execute(
                select(Memory).where(Memory.user_id == user_id).order_by(Memory.created_at)
            )
        ).scalars().all()
        old_memory = next(memory for memory in memories if memory.id == original_id)
        assert old_memory.is_active is False
        assert "superseded" in old_memory.tags
        assert replacement[0].is_active is True
        assert replacement[0].metadata_["value"] == "Java"
