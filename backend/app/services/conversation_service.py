"""Conversation service — orchestrates the full chat pipeline."""
import uuid
import structlog
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import AsyncSessionLocal
from app.models.conversation import Conversation, Message, MessageRole
from app.models.user import User
from app.services.llm_service import get_llm_service
from app.services.memory_service import MemoryService
from app.services.graph_service import get_graph_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are MemoryMate, an intelligent AI assistant with long-term memory capabilities.
You have access to memories from past conversations with this user. Use them to provide personalized, context-aware responses.

When memories are provided:
- Reference them naturally when relevant
- Build on past conversations
- Remember user preferences and goals
- Acknowledge when you're using stored information ("Based on what you told me before...")

Be helpful, honest, and conversational. If you're uncertain about a remembered fact, say so.
Never fabricate memories you weren't given."""


class ConversationService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.llm_svc = get_llm_service()
        self.memory_svc = MemoryService(db)
        self.graph_svc = get_graph_service()

    async def _get_or_create_user(self, user_id: str) -> User:
        result = await self.db.execute(select(User).where(User.username == user_id))
        user = result.scalar_one_or_none()
        if not user:
            user = User(id=str(uuid.uuid4()), username=user_id, display_name=user_id)
            self.db.add(user)
            await self.db.commit()
            await self.db.refresh(user)
        return user

    async def _get_or_create_conversation(
        self, user: User, conversation_id: str | None
    ) -> Conversation:
        if conversation_id:
            result = await self.db.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user.id,
                )
            )
            conv = result.scalar_one_or_none()
            if conv:
                return conv
            raise ValueError("Conversation not found for this user.")
        conv = Conversation(user_id=user.id, title="New Conversation")
        self.db.add(conv)
        await self.db.commit()
        await self.db.refresh(conv)
        return conv

    async def _get_recent_messages(
        self, conversation_id: str, limit: int = 10
    ) -> list[Message]:
        result = await self.db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Main chat pipeline with memory retrieval and response generation."""
        user = await self._get_or_create_user(request.user_id)
        conv = await self._get_or_create_conversation(user, request.conversation_id)

        # Store user message
        user_msg = Message(
            conversation_id=conv.id,
            role=MessageRole.user,
            content=request.message,
        )
        self.db.add(user_msg)
        conv.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(user_msg)

        # Build context
        retrieved_memories: list[dict] = []
        graph_context: list[dict] = []

        graph_data = {"entities": [], "relationships": [], "summary": ""}
        if request.use_graph:
            graph_data = await self.graph_svc.retrieve_context(
                user_id=str(user.id),
                query=request.message,
            )

        if request.use_memory:
            hybrid_context = await self.memory_svc.retrieve_hybrid_context(
                user_id=str(user.id),
                query=request.message,
                top_k=5,
                graph_context=graph_data,
            )
            retrieved_memories = hybrid_context["memories"]
            graph_context = hybrid_context["relationships"]
        elif graph_data.get("summary"):
            graph_context = graph_data.get("relationships", [])

        # Build the message list for LLM
        recent_msgs = await self._get_recent_messages(conv.id, limit=10)
        # Exclude the message we just added (it's the last one)
        recent_msgs = [m for m in recent_msgs if m.id != user_msg.id]

        system_content = SYSTEM_PROMPT
        if retrieved_memories:
            mem_block = "\n".join(
                f"- [{m['type']}] {m['content']} (confidence: {m['confidence']:.2f})"
                for m in retrieved_memories
            )
            system_content += f"\n\n## Relevant Memories:\n{mem_block}"

        if graph_context:
            graph_block = "\n".join(
                f"- {r.get('source', '')} {r.get('relationship', '').lower().replace('_', ' ')} {r.get('target', '')}"
                for r in graph_context[:5]
            )
            system_content += f"\n\n## Knowledge Graph Context:\n{graph_block}"

        messages = [{"role": "system", "content": system_content}]
        for msg in recent_msgs[-8:]:  # Keep last 8 messages for context window
            messages.append({"role": msg.role.value, "content": msg.content})
        messages.append({"role": "user", "content": request.message})

        # Call LLM
        llm_result = await self.llm_svc.chat(messages, temperature=0.7, max_tokens=1024)
        response_text = llm_result["content"]
        tokens_used = llm_result.get("tokens_used")
        latency_ms = llm_result.get("latency_ms")

        # Store assistant message
        assistant_msg = Message(
            conversation_id=conv.id,
            role=MessageRole.assistant,
            content=response_text,
            token_count=tokens_used,
        )
        self.db.add(assistant_msg)
        await self.db.commit()
        await self.db.refresh(assistant_msg)

        # Update conversation title if first exchange
        if conv.title == "New Conversation":
            # Simple title: first 60 chars of user message
            conv.title = request.message[:60] + ("..." if len(request.message) > 60 else "")
            await self.db.commit()

        # Persist memory extraction before returning so it survives process restarts.
        conversation_text = f"User: {request.message}\nAssistant: {response_text}"
        await self._post_chat_processing(
            user_id=str(user.id),
            conversation_id=str(conv.id),
            conversation_text=conversation_text,
            message_ids=[str(user_msg.id), str(assistant_msg.id)],
        )

        return ChatResponse(
            conversation_id=str(conv.id),
            message_id=str(assistant_msg.id),
            response=response_text,
            retrieved_memories=retrieved_memories,
            retrieved_graph_context=graph_context,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )

    async def _post_chat_processing(
        self,
        user_id: str,
        conversation_id: str,
        conversation_text: str,
        message_ids: list[str],
    ):
        """Background task: store episodic memory and extract semantic facts."""
        try:
            async with AsyncSessionLocal() as db:
                memory_svc = MemoryService(db)
                await memory_svc.store_episodic_memory(
                    user_id=user_id,
                    content=conversation_text,
                    conversation_id=conversation_id,
                    message_ids=message_ids,
                    tags=["auto"],
                )
                await memory_svc.extract_and_store_semantic_memories(
                    user_id=user_id,
                    conversation_text=conversation_text,
                    conversation_id=conversation_id,
                )
            # Extract graph entities
            await self.graph_svc.extract_and_store(
                user_id=user_id,
                conversation_text=conversation_text,
                conversation_id=conversation_id,
            )
        except Exception as e:
            logger.error(
                "post_chat_processing_error",
                error_type=type(e).__name__,
                user_id=user_id,
            )

    async def get_conversations(
        self, user_id: str, limit: int = 20, offset: int = 0
    ) -> list[dict]:
        user = await self._get_or_create_user(user_id)
        result = await self.db.execute(
            select(Conversation)
            .where(Conversation.user_id == user.id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        convs = result.scalars().all()

        out = []
        for c in convs:
            count_result = await self.db.execute(
                select(func.count(Message.id)).where(Message.conversation_id == c.id)
            )
            msg_count = count_result.scalar() or 0
            out.append({
                "id": str(c.id),
                "title": c.title,
                "summary": c.summary,
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
                "message_count": msg_count,
            })
        return out

    async def get_conversation_messages(
        self, conversation_id: str, user_id: str, limit: int = 50
    ) -> list[dict]:
        user = await self._get_or_create_user(user_id)
        result = await self.db.execute(
            select(Message)
            .join(Conversation)
            .where(
                Message.conversation_id == conversation_id,
                Conversation.user_id == user.id,
            )
            .order_by(Message.created_at.asc(), Message.id.asc())
            .limit(limit)
        )
        msgs = result.scalars().all()
        return [
            {
                "id": str(m.id),
                "role": m.role.value,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
                "token_count": m.token_count,
            }
            for m in msgs
        ]

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        user = await self._get_or_create_user(user_id)
        result = await self.db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user.id,
            )
        )
        conv = result.scalar_one_or_none()
        if not conv:
            return False
        await self.db.delete(conv)
        await self.db.commit()
        return True
