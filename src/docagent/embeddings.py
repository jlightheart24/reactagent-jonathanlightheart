"""Embedding clients. The agent and index depend only on the `Embedder` protocol."""
from __future__ import annotations

import time
from typing import Protocol

from .config import Settings


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAICompatibleEmbedder:
    """Qwen3-Embedding-8B through the course's OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, settings: Settings, max_attempts: int = 4) -> None:
        settings.require("embedding_base_url", "embedding_api_key")
        from openai import OpenAI

        self.model_name = settings.embedding_model
        self.query_instruction = settings.query_instruction
        self.max_attempts = max_attempts
        self._client = OpenAI(base_url=settings.embedding_base_url, api_key=settings.embedding_api_key,
                              timeout=120, max_retries=0)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._client.embeddings.create(model=self.model_name, input=texts)
                return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
            except Exception:
                if attempt == self.max_attempts:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        prompt = f"Instruct: {self.query_instruction}\nQuery: {text}" if self.query_instruction else text
        return self._embed([prompt])[0]
