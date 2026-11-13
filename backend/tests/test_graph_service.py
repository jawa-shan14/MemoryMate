import json

import pytest

from app.services.graph_service import GraphService


class FakeResult:
    def __init__(self, records=None):
        self.records = records or []

    async def data(self):
        return self.records


class FakeSession:
    def __init__(self, records=None):
        self.records = records
        self.queries = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def run(self, query, **parameters):
        self.queries.append((query, parameters))
        return FakeResult(self.records)


class FakeDriver:
    def __init__(self, session):
        self.test_session = session
        self.databases = []

    async def verify_connectivity(self):
        return True

    def session(self, database=None):
        self.databases.append(database)
        return self.test_session


class FakeLLM:
    async def extract_structured(self, prompt, system):
        return json.dumps({
            "entities": [
                {"name": "Python", "type": "Technology", "properties": {"kind": "language"}},
                {"name": "MemoryMate", "type": "Project", "properties": {}},
                {"name": "Bad", "type": "Topic) DETACH DELETE n //", "properties": {}},
            ],
            "relationships": [
                {"source": "MemoryMate", "relationship": "USES", "target": "Python"},
                {"source": "User", "relationship": "USES] DELETE n //", "target": "Bad"},
            ],
        })


class FakePath:
    nodes = [
        {"properties": {"name": "User"}, "labels": ["User"]},
        {"properties": {"name": "Python"}, "labels": ["Technology"]},
    ]
    relationships = [{"source": "User", "relationship": "USES", "target": "Python"}]


@pytest.mark.asyncio
async def test_graph_storage_validates_model_generated_labels_and_relationships():
    session = FakeSession()
    service = GraphService()
    service._driver = FakeDriver(session)
    service.llm_svc = FakeLLM()

    result = await service.store_extracted_graph(
        "user-1",
        "conversation-1",
        [
            {"name": "Python", "type": "Technology", "properties": {"kind": "language"}},
            {"name": "Bad", "type": "Topic) DETACH DELETE n //", "properties": {}},
        ],
        [
            {"source": "Python", "relationship": "USES", "target": "Python"},
            {"source": "Python", "relationship": "USES] DELETE n //", "target": "Bad"},
        ],
    )

    assert result["entities"][0]["name"] == "Python"
    assert len(result["entities"]) == 1
    assert result["relationships"] == [
        {"source": "Python", "relationship": "USES", "target": "Python"}
    ]
    queries = [query for query, _ in session.queries]
    assert any("MERGE (e:Technology " in query for query in queries)
    assert any("MERGE (a)-[r:USES]->(b)" in query for query in queries)
    assert all("DETACH DELETE" not in query and "DELETE n" not in query for query in queries)
    assert service._driver.databases == ["memorymate"]


@pytest.mark.asyncio
async def test_conversation_extraction_flows_into_graph_persistence():
    session = FakeSession()
    service = GraphService()
    service._driver = FakeDriver(session)
    service.llm_svc = FakeLLM()

    result = await service.extract_and_store(
        "user-1",
        "I use Python.",
        "conversation-1",
    )

    assert result["entities"][0]["name"] == "Python"
    assert len(result["relationships"]) == 1
    assert any("MERGE (a)-[r:USES]->(b)" in query for query, _ in session.queries)


@pytest.mark.asyncio
async def test_graph_retrieval_uses_bounded_multi_hop_query():
    session = FakeSession(
        records=[{
            "path_nodes": FakePath.nodes,
            "path_relationships": FakePath.relationships,
        }]
    )
    service = GraphService()
    service._driver = FakeDriver(session)

    context = await service.retrieve_context("user-1", "Python", max_hops=99)

    assert len(context["entities"]) == 2
    assert {entity["type"] for entity in context["entities"]} == {"User", "Technology"}
    assert context["relationships"] == [
        {"source": "User", "relationship": "USES", "target": "Python"}
    ]
    query = session.queries[0][0]
    assert "[*1..3]" in query
    assert "nodes(p)" in query
    assert "startNode(rel).name" in query
    assert "relationships(p)" in query
    assert context["summary"] == "User uses Python"


@pytest.mark.asyncio
async def test_user_graph_uses_stable_ids_and_entity_labels():
    session = FakeSession(records=[{
        "n_id": "node-1",
        "n_properties": {"name": "Python"},
        "n_labels": ["Technology"],
        "m_id": "node-2",
        "m_properties": {"name": "MemoryMate"},
        "m_labels": ["Project"],
        "relationship_type": "USES",
    }])
    service = GraphService()
    service._driver = FakeDriver(session)

    graph = await service.get_user_graph("user-1")

    assert graph == {
        "nodes": [
            {"id": "node-1", "label": "Python", "type": "Technology"},
            {"id": "node-2", "label": "MemoryMate", "type": "Project"},
        ],
        "edges": [{"source": "node-1", "target": "node-2", "label": "USES"}],
        "available": True,
    }


@pytest.mark.asyncio
async def test_graph_service_degrades_cleanly_when_neo4j_is_not_configured():
    service = GraphService()
    service._driver = None

    result = await service.retrieve_context("user-1", "Python memory", max_hops=2)

    assert result == {"entities": [], "relationships": [], "summary": ""}
