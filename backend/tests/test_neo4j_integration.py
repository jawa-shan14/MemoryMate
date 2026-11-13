import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.models.memory import Memory, MemoryType
from app.models.user import User
from app.services.embedding_service import get_embedding_service
from app.services.graph_service import GraphService
from app.services.memory_service import MemoryService


@pytest_asyncio.fixture
async def available_neo4j():
    service = GraphService()
    if not await service._is_available():
        await service.close()
        pytest.skip("Neo4j is not reachable with the configured environment.")
    try:
        yield service
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_live_neo4j_write_read_multihop_and_hybrid_retrieval(
    available_neo4j,
    session_factory,
):
    graph = available_neo4j
    test_id = uuid.uuid4().hex[:12]
    user_id = f"neo4j-integration-{test_id}"
    conversation_id = str(uuid.uuid4())
    person = f"Integration Person {test_id}"
    project = f"MemoryMate Project {test_id}"
    technology = f"Python Tool {test_id}"
    query = f"{person} works on {project}, and {project} uses {technology}."

    try:
        stored = await graph.store_extracted_graph(
            user_id=user_id,
            conversation_id=conversation_id,
            entities=[
                {"name": person, "type": "Person", "properties": {}},
                {"name": project, "type": "Project", "properties": {}},
                {"name": technology, "type": "Technology", "properties": {}},
            ],
            relationships=[
                {"source": person, "relationship": "WORKS_ON", "target": project},
                {"source": project, "relationship": "USES", "target": technology},
            ],
        )
        assert len(stored["entities"]) == 3
        assert len(stored["relationships"]) == 2

        graph_data = await graph.retrieve_context(user_id, project, max_hops=2)
        assert {entity["name"] for entity in graph_data["entities"]} >= {
            person,
            project,
            technology,
        }
        edges = {
            (edge["source"], edge["relationship"], edge["target"])
            for edge in graph_data["relationships"]
        }
        assert (person, "WORKS_ON", project) in edges
        assert (project, "USES", technology) in edges

        visual_graph = await graph.get_user_graph(user_id)
        assert visual_graph["available"] is True
        assert len(visual_graph["nodes"]) == 3
        assert len(visual_graph["edges"]) == 2

        async with session_factory() as session:
            user = User(username=user_id, display_name="Neo4j Integration Test")
            session.add(user)
            await session.flush()
            embedding = get_embedding_service().embed(query)
            session.add(
                Memory(
                    user_id=user.id,
                    memory_type=MemoryType.semantic,
                    content=query,
                    embedding=embedding,
                    confidence=0.95,
                )
            )
            await session.commit()

            hybrid = await MemoryService(session).retrieve_hybrid_context(
                user_id=user_id,
                query=query,
                graph_context=await graph.retrieve_context(user_id, query, max_hops=2),
            )

        assert any(memory["content"] == query for memory in hybrid["memories"])
        assert {entity["name"] for entity in hybrid["entities"]} >= {project, technology}
        assert any(edge["relationship"] == "USES" for edge in hybrid["relationships"])
    finally:
        if graph._get_driver() is not None:
            async with graph._get_session() as session:
                result = await session.run(
                    "MATCH (n {user_id: $user_id}) DETACH DELETE n",
                    user_id=user_id,
                )
                await result.consume()
