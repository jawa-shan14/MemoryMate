"""Knowledge graph service using Neo4j for entity-relationship storage and retrieval."""
import json
import re
import structlog
try:
    from neo4j import AsyncGraphDatabase
except ImportError:  # pragma: no cover - optional dependency
    AsyncGraphDatabase = None
from app.config import get_settings
from app.services.llm_service import get_llm_service

logger = structlog.get_logger(__name__)
settings = get_settings()

EXTRACT_PROMPT_SYSTEM = """You are a knowledge graph extraction specialist. Extract entities and relationships from the conversation.
Return ONLY valid JSON in this exact format:
{
  "entities": [
    {"name": "entity name", "type": "Person|Organization|Technology|Topic|Project|Location|Concept", "properties": {}}
  ],
  "relationships": [
    {"source": "entity name", "relationship": "RELATIONSHIP_TYPE", "target": "entity name", "properties": {}}
  ]
}
Relationship types should be uppercase with underscores: INTERESTED_IN, LEARNING, WORKING_ON, USES, KNOWS, LOCATED_IN, PART_OF, etc.
Only extract clear, factual relationships. If nothing clear exists, return {"entities": [], "relationships": []}."""
ENTITY_LABELS = {
    "Person",
    "Organization",
    "Technology",
    "Topic",
    "Project",
    "Location",
    "Concept",
    "Entity",
}
RELATIONSHIP_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class GraphService:
    """Manages entity-relationship knowledge graph in Neo4j."""

    def __init__(self):
        self._driver = None
        self.llm_svc = None

    def _get_driver(self):
        if self._driver is None:
            if AsyncGraphDatabase is None:
                logger.warning("neo4j_unavailable", reason="dependency_missing")
                return None
            try:
                self._driver = AsyncGraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_username, settings.neo4j_password)
                    if settings.neo4j_password
                    else None,
                )
                logger.info("neo4j_connected", uri=settings.neo4j_uri)
            except Exception as e:
                logger.warning("neo4j_unavailable", error_type=type(e).__name__)
                self._driver = None
        return self._driver

    def _get_session(self):
        driver = self._get_driver()
        if driver is None:
            raise RuntimeError("Neo4j driver is unavailable.")
        return driver.session(database=settings.neo4j_database)

    async def _is_available(self) -> bool:
        driver = self._get_driver()
        if not driver:
            return False
        try:
            await driver.verify_connectivity()
            return True
        except Exception:
            return False

    async def extract_and_store(
        self, user_id: str, conversation_text: str, conversation_id: str
    ) -> dict:
        """Extract entities/relationships from text and store in Neo4j."""
        if not await self._is_available():
            logger.warning("neo4j_skip_extraction", reason="not available")
            return {"entities": [], "relationships": []}

        if self.llm_svc is None:
            self.llm_svc = get_llm_service()

        prompt = f"Extract entities and relationships from this conversation:\n\n{conversation_text}"
        try:
            response = await self.llm_svc.extract_structured(prompt, EXTRACT_PROMPT_SYSTEM)
            start = response.find('{')
            end = response.rfind('}') + 1
            if start == -1 or end == 0:
                return {"entities": [], "relationships": []}
            extracted = json.loads(response[start:end])
        except Exception as e:
            logger.error(
                "graph_extraction_error",
                error_type=type(e).__name__,
                status_code=getattr(e, "status_code", None),
            )
            return {"entities": [], "relationships": []}

        return await self.store_extracted_graph(
            user_id=user_id,
            conversation_id=conversation_id,
            entities=extracted.get("entities", []),
            relationships=extracted.get("relationships", []),
        )

    async def store_extracted_graph(
        self,
        user_id: str,
        conversation_id: str,
        entities: list[dict],
        relationships: list[dict],
    ) -> dict:
        """Validate extracted facts and persist them to Neo4j."""
        if not await self._is_available():
            logger.warning("neo4j_skip_storage", reason="not available")
            return {"entities": [], "relationships": []}

        if not isinstance(entities, list):
            entities = []
        if not isinstance(relationships, list):
            relationships = []
        valid_entities = [
            entity for entity in entities
            if isinstance(entity, dict)
            and entity.get("type", "Entity") in ENTITY_LABELS
            and isinstance(entity.get("name"), str)
            and entity["name"].strip()
        ]
        valid_entity_names = {entity["name"] for entity in valid_entities}
        valid_relationships = []
        for relationship in relationships:
            if not isinstance(relationship, dict):
                continue
            rel_type = str(relationship.get("relationship", "")).upper().replace(" ", "_")
            if (
                RELATIONSHIP_PATTERN.fullmatch(rel_type)
                and isinstance(relationship.get("source"), str)
                and relationship["source"].strip()
                and isinstance(relationship.get("target"), str)
                and relationship["target"].strip()
                and relationship["source"] in valid_entity_names
                and relationship["target"] in valid_entity_names
            ):
                valid_relationships.append({**relationship, "relationship": rel_type})

        async with self._get_session() as session:
            # Always create/merge User node
            await session.run(
                "MERGE (u:User {id: $user_id}) "
                "SET u.user_id = $user_id, u.updated_at = datetime()",
                user_id=user_id,
            )
            # Create entity nodes
            for entity in valid_entities:
                entity_type = entity.get("type", "Entity")
                entity_properties = entity.get("properties")
                if not isinstance(entity_properties, dict):
                    entity_properties = {}
                await session.run(
                    f"MERGE (e:{entity_type} {{name: $name, user_id: $user_id}}) "
                    "SET e += $props, e.updated_at = datetime()",
                    name=entity["name"],
                    user_id=user_id,
                    props={
                        key: value
                        for key, value in entity_properties.items()
                        if isinstance(value, (str, int, float, bool, list))
                    },
                )
            # Create relationships
            for rel in valid_relationships:
                rel_type = rel["relationship"]
                await session.run(
                    f"MATCH (a {{name: $source, user_id: $user_id}}) "
                    f"MATCH (b {{name: $target, user_id: $user_id}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    "SET r.conversation_id = $conv_id, r.updated_at = datetime()",
                    source=rel["source"],
                    target=rel["target"],
                    user_id=user_id,
                    conv_id=conversation_id,
                )

        logger.info(
            "graph_stored",
            user_id=user_id,
            entities=len(valid_entities),
            relationships=len(valid_relationships),
        )
        return {"entities": valid_entities, "relationships": valid_relationships}

    async def retrieve_context(self, user_id: str, query: str, max_hops: int = 2) -> dict:
        """Retrieve graph context relevant to the query."""
        if not await self._is_available():
            return {"entities": [], "relationships": [], "summary": ""}

        keywords = [w.strip(".,!?;:\"'()[]{}") for w in query.lower().split()]
        keywords = [word for word in keywords if len(word) > 2][:8]
        if not keywords:
            return {"entities": [], "relationships": [], "summary": ""}
        hops = max(1, min(int(max_hops), 3))

        try:
            async with self._get_session() as session:
                results = await session.run(
                    f"MATCH p=(n {{user_id: $user_id}})-[*1..{hops}]-(m {{user_id: $user_id}}) "
                    "WHERE any(kw IN $keywords WHERE any(node IN nodes(p) "
                    "WHERE toLower(node.name) CONTAINS kw)) "
                    "RETURN [node IN nodes(p) | {properties: properties(node), "
                    "labels: labels(node)}] AS path_nodes, "
                    "[rel IN relationships(p) | {source: startNode(rel).name, "
                    "relationship: type(rel), target: endNode(rel).name}] "
                    "AS path_relationships LIMIT 50",
                    user_id=user_id,
                    keywords=keywords,
                )
                records = await results.data()

            entities_seen = set()
            entities = []
            relationships = []

            for record in records:
                path_nodes = record.get("path_nodes") or []
                path_relationships = record.get("path_relationships") or []
                if not path_nodes:
                    continue
                for node in path_nodes:
                    properties = node.get("properties", node)
                    name = properties.get("name")
                    if name and name not in entities_seen:
                        entities.append({
                            "name": name,
                            "type": next(iter(node.get("labels", ())), "Entity"),
                            "properties": dict(properties),
                        })
                        entities_seen.add(name)
                for relationship in path_relationships:
                    source = relationship.get("source", "")
                    target = relationship.get("target", "")
                    if source and target:
                        relationships.append({
                            "source": source,
                            "relationship": relationship.get("relationship", ""),
                            "target": target,
                        })

            if relationships:
                summary_parts = [
                    f"{r['source']} {r['relationship'].lower().replace('_', ' ')} {r['target']}"
                    for r in relationships[:5]
                ]
                summary = "; ".join(summary_parts)
            else:
                summary = ""

            return {"entities": entities, "relationships": relationships, "summary": summary}

        except Exception as e:
            logger.error("graph_retrieval_error", error_type=type(e).__name__)
            return {"entities": [], "relationships": [], "summary": ""}

    async def get_user_graph(self, user_id: str) -> dict:
        """Get all nodes and relationships for a user (for visualization)."""
        if not await self._is_available():
            return {"nodes": [], "edges": [], "available": False}

        try:
            async with self._get_session() as session:
                result = await session.run(
                    "MATCH (n {user_id: $user_id}) WHERE n.name IS NOT NULL "
                    "OPTIONAL MATCH (n)-[r]->(m {user_id: $user_id}) "
                    "WHERE m.name IS NOT NULL "
                    "RETURN elementId(n) AS n_id, properties(n) AS n_properties, "
                    "labels(n) AS n_labels, elementId(m) AS m_id, "
                    "properties(m) AS m_properties, labels(m) AS m_labels, "
                    "type(r) AS relationship_type LIMIT 200",
                    user_id=user_id,
                )
                records = await result.data()

            nodes = {}
            edges = []

            for record in records:
                n_id = record.get("n_id")
                n_properties = record.get("n_properties") or {}
                n_labels = record.get("n_labels") or []
                m_id = record.get("m_id")
                m_properties = record.get("m_properties") or {}
                m_labels = record.get("m_labels") or []
                relationship_type = record.get("relationship_type")

                if n_id and n_properties.get("name"):
                    nodes[n_id] = {
                        "id": n_id,
                        "label": n_properties["name"],
                        "type": n_labels[0] if n_labels else "Entity",
                    }

                if m_id and m_properties.get("name"):
                    nodes[m_id] = {
                        "id": m_id,
                        "label": m_properties["name"],
                        "type": m_labels[0] if m_labels else "Entity",
                    }

                if relationship_type and n_id and m_id:
                    edges.append({
                        "source": n_id,
                        "target": m_id,
                        "label": relationship_type,
                    })

            return {"nodes": list(nodes.values()), "edges": edges, "available": True}
        except Exception as e:
            logger.error("graph_get_error", error_type=type(e).__name__)
            return {"nodes": [], "edges": [], "available": False}

    async def get_stats(self, user_id: str) -> dict:
        if not await self._is_available():
            return {
                "total_nodes": 0,
                "total_relationships": 0,
                "entity_types": {},
                "relationship_types": {},
                "available": False,
            }
        try:
            async with self._get_session() as session:
                node_result = await session.run(
                    "MATCH (n {user_id: $user_id}) RETURN labels(n) as labels, count(*) as cnt",
                    user_id=user_id,
                )
                node_data = await node_result.data()
                rel_result = await session.run(
                    "MATCH ({user_id: $user_id})-[r]->({user_id: $user_id}) "
                    "RETURN type(r) as rel_type, count(*) as cnt",
                    user_id=user_id,
                )
                rel_data = await rel_result.data()

            entity_types = {}
            total_nodes = 0
            for row in node_data:
                label = row["labels"][0] if row["labels"] else "Unknown"
                entity_types[label] = entity_types.get(label, 0) + row["cnt"]
                total_nodes += row["cnt"]

            rel_types = {row["rel_type"]: row["cnt"] for row in rel_data}
            total_rels = sum(rel_types.values())

            return {
                "total_nodes": total_nodes,
                "total_relationships": total_rels,
                "entity_types": entity_types,
                "relationship_types": rel_types,
                "available": True,
            }
        except Exception as e:
            logger.error("graph_stats_error", error_type=type(e).__name__)
            return {
                "total_nodes": 0,
                "total_relationships": 0,
                "entity_types": {},
                "relationship_types": {},
                "available": False,
            }

    async def close(self):
        if self._driver:
            await self._driver.close()


_graph_service: GraphService | None = None


def get_graph_service() -> GraphService:
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphService()
    return _graph_service
