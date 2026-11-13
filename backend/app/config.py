"""Application configuration loaded from environment variables."""
from functools import lru_cache
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # NVIDIA
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"

    # Database
    # Default to a local SQLite database so the app can boot without a Postgres server.
    # Override with DATABASE_URL if you want PostgreSQL in development or production.
    database_url: str = "sqlite+aiosqlite:///./memorymate.db"
    database_url_sync: str = "sqlite:///./memorymate.db"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # App
    secret_key: str = "dev-secret-key-change-in-production"
    environment: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Memory
    max_episodic_memories: int = 1000
    max_semantic_memories: int = 500
    consolidation_interval_hours: int = 24
    memory_similarity_threshold: float = 0.55
    memory_fact_similarity_threshold: float = 0.35

    @field_validator("nvidia_api_key", mode="before")
    @classmethod
    def sanitize_nvidia_api_key(cls, value):
        if value is None:
            return ""
        value = str(value).strip()
        placeholder_values = {
            "",
            "your_nvidia_api_key_here",
            "changeme",
            "placeholder",
            "example",
        }
        if value.lower() in placeholder_values:
            return ""
        return value

    @field_validator("database_url", "database_url_sync", mode="before")
    @classmethod
    def normalize_database_urls(cls, value, info):
        if value is None:
            return "sqlite+aiosqlite:///./memorymate.db" if info.field_name == "database_url" else "sqlite:///./memorymate.db"
        value = str(value).strip()
        if not value or "******" in value or value.lower() in {"postgresql://", "postgres://"}:
            return "sqlite+aiosqlite:///./memorymate.db" if info.field_name == "database_url" else "sqlite:///./memorymate.db"
        return value

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
