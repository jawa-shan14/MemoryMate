"""Knowledge graph schemas."""
from pydantic import BaseModel
from typing import Optional


class EntityOut(BaseModel):
    id: str
    name: str
    type: str
    properties: dict = {}


class RelationshipOut(BaseModel):
    source: str
    source_type: str
    relationship: str
    target: str
    target_type: str
    properties: dict = {}


class GraphContext(BaseModel):
    entities: list[EntityOut] = []
    relationships: list[RelationshipOut] = []
    summary: str = ""


class GraphStatsOut(BaseModel):
    total_nodes: int
    total_relationships: int
    entity_types: dict[str, int]
    relationship_types: dict[str, int]
