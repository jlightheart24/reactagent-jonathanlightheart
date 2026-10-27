"""Typed agent actions: argument schemas, OpenAI tool definitions, and retrieval observations."""
from __future__ import annotations

import copy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .retrieval import Passage
from .sources import SOURCE_IDS, SOURCES

SourceId = Literal[SOURCE_IDS]  # type: ignore[valid-type]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SearchArgs(_Args):
    """Arguments of `search_documents`."""

    query: str | None = Field(default=None, min_length=3, max_length=500,
                              description="Natural-language search query describing the information needed. "
                                          "Required unless `contains` is given.")
    sources: list[SourceId] | None = Field(
        default=None, min_length=1, max_length=len(SOURCE_IDS),
        description="Optional: restrict the search to these source ids.")
    contains: str | None = Field(
        default=None, min_length=2, max_length=100,
        description="Optional: only return passages containing this exact, case-sensitive text, "
                    "e.g. a section number such as '2-201.12'.")
    k: int | None = Field(default=None, ge=1, le=10,
                          description="Optional: number of passages to return (default set by the application).")

    @field_validator("sources")
    @classmethod
    def _unique(cls, v):
        if v is not None and len(set(v)) != len(v):
            raise ValueError("sources must not repeat")
        return v

    @model_validator(mode="after")
    def _query_or_contains(self):
        if self.query is None:
            if self.contains is None:
                raise ValueError("query is required unless contains is given")
            self.query = self.contains  # an exact-text search embeds the text itself
        return self


class Claim(_Args):
    text: str = Field(min_length=1, max_length=2000,
                      description="One factual statement, supported by the cited passages.")
    citations: list[str] = Field(min_length=1, max_length=8,
                                 description="chunk_id values, exactly as returned by search_documents.")


class AnswerArgs(_Args):
    """Arguments of `final_answer`."""

    claims: list[Claim] = Field(min_length=1, max_length=15,
                                description="The answer as a list of claims, each with its own citations.")


class InsufficientEvidenceArgs(_Args):
    """Arguments of `insufficient_evidence`."""

    reason: str = Field(min_length=1, max_length=2000,
                        description="What the question asks that the retrieved passages do not support.")
    citations: list[str] = Field(default_factory=list, max_length=8,
                                 description="Optional chunk_ids of related passages that were consulted.")


SEARCH = "search_documents"
ANSWER = "final_answer"
INSUFFICIENT = "insufficient_evidence"

TOOL_SCHEMAS: dict[str, type[_Args]] = {SEARCH: SearchArgs, ANSWER: AnswerArgs, INSUFFICIENT: InsufficientEvidenceArgs}
TERMINAL_TOOLS = {ANSWER, INSUFFICIENT}

_DESCRIPTIONS = {
    SEARCH: "Semantic search over the indexed source documents. Returns passages with their source, "
            "one-based PDF page, and chunk_id.",
    ANSWER: "Finish with an answer. Every claim must cite chunk_ids returned by search_documents in this "
            "run and must be supported by those passages.",
    INSUFFICIENT: "Finish by reporting that the retrieved documents do not support an answer.",
}


def _inline_refs(schema: dict) -> dict:
    """Replace $ref/$defs so OpenAI-compatible servers that do not resolve references accept the schema."""
    defs = schema.get("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: resolve(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def tool_definitions() -> list[dict]:
    return [{"type": "function",
             "function": {"name": name, "description": _DESCRIPTIONS[name],
                          "parameters": _inline_refs(model.model_json_schema())}}
            for name, model in TOOL_SCHEMAS.items()]


def passage_record(p: Passage) -> dict:
    """One retrieval result as shown to the model and stored in the trace."""
    return {
        "chunk_id": p.chunk_id,
        "source_id": p.source_id,
        "source_title": p.title,
        "source_role": SOURCES[p.source_id].role if p.source_id in SOURCES else None,
        "page": p.page,
        "distance": p.distance,
        "text": p.text,
    }
