"""Runtime configuration from the environment (and an optional local .env).

API keys are read here and passed to clients; they never appear in `public()`, traces, or results.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Qwen3-Embedding is trained with an instruction prefix on queries (documents get none).
DEFAULT_QUERY_INSTRUCTION = (
    "Given a question about food safety, retrieve passages from food-safety documents that answer it"
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    embedding_base_url: str | None = None
    embedding_model: str = "Qwen3-Embedding-8B"
    embedding_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str = "Qwen3.8-27B"
    llm_api_key: str | None = None
    docs_dir: Path = PROJECT_ROOT / "U0-HW-09-materials" / "documents"
    chroma_dir: Path = PROJECT_ROOT / "chroma"
    collection: str = "food_safety_corpus"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 5
    max_steps: int = 10
    embed_batch: int = 32
    tool_choice: str = "auto"
    temperature: float = 0.0
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION

    def __post_init__(self) -> None:
        if self.chunk_size < 100:
            raise ConfigError("chunk_size must be >= 100")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigError("chunk_overlap must be >= 0 and < chunk_size")
        if not 1 <= self.top_k <= 10:
            raise ConfigError("top_k must be between 1 and 10")
        if self.max_steps < 1:
            raise ConfigError("max_steps must be >= 1")
        if self.embed_batch < 1:
            raise ConfigError("embed_batch must be >= 1")

    def public(self) -> dict:
        """Configuration safe to record in results (no keys or service URLs)."""
        return {
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "collection": self.collection,
            "chunk_size_chars": self.chunk_size,
            "chunk_overlap_chars": self.chunk_overlap,
            "retrieval_top_k": self.top_k,
            "max_steps": self.max_steps,
            "temperature": self.temperature,
            "tool_choice": self.tool_choice,
            "query_instruction": self.query_instruction,
        }

    def require(self, *names: str) -> None:
        missing = [n.upper() for n in names if not getattr(self, n)]
        if missing:
            raise ConfigError("missing configuration: " + ", ".join(missing) + " (see .env.example)")


_ENV = {
    "embedding_base_url": "EMBEDDING_BASE_URL",
    "embedding_model": "EMBEDDING_MODEL",
    "embedding_api_key": "EMBEDDING_API_KEY",
    "llm_base_url": "LLM_BASE_URL",
    "llm_model": "LLM_MODEL",
    "llm_api_key": "LLM_API_KEY",
    "docs_dir": "DOCAGENT_DOCS_DIR",
    "chroma_dir": "DOCAGENT_CHROMA_DIR",
    "collection": "DOCAGENT_COLLECTION",
    "chunk_size": "DOCAGENT_CHUNK_SIZE",
    "chunk_overlap": "DOCAGENT_CHUNK_OVERLAP",
    "top_k": "DOCAGENT_TOP_K",
    "max_steps": "DOCAGENT_MAX_STEPS",
    "embed_batch": "DOCAGENT_EMBED_BATCH",
    "tool_choice": "DOCAGENT_TOOL_CHOICE",
}


def load_settings(env_file: Path | None = PROJECT_ROOT / ".env", **overrides) -> Settings:
    """Settings from environment variables (after loading `env_file` if it exists), then overrides."""
    if env_file is not None and Path(env_file).exists():
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    types = {f.name: f.type for f in fields(Settings)}
    values: dict = {}
    for name, var in _ENV.items():
        raw = os.environ.get(var)
        if raw is None or raw == "":
            continue
        kind = types[name]
        if kind == "int":
            values[name] = int(raw)
        elif kind == "Path":
            path = Path(raw)
            values[name] = path if path.is_absolute() else PROJECT_ROOT / path
        else:
            values[name] = raw
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**values)
