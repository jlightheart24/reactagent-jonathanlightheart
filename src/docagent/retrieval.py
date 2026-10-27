"""Semantic retrieval over the persistent Chroma collection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import Settings
from .embeddings import Embedder
from .ingest import open_client


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    source_id: str
    title: str
    page: int
    text: str
    distance: float | None = None

    def label(self) -> str:
        return f"{self.title}, PDF page {self.page}, chunk {self.chunk_id}"


class Retriever(Protocol):
    def search(self, query: str, k: int, sources: list[str] | None = None,
               contains: str | None = None) -> list[Passage]: ...


class IndexNotBuilt(RuntimeError):
    pass


class ChromaRetriever:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        client = open_client(settings.chroma_dir)
        if settings.collection not in [c.name for c in client.list_collections()]:
            raise IndexNotBuilt(f"collection {settings.collection!r} not found in {settings.chroma_dir}; "
                                "run `docagent ingest --rebuild`")
        self.collection = client.get_collection(settings.collection, embedding_function=None)
        stored_model = (self.collection.metadata or {}).get("embedding_model")
        if stored_model != embedder.model_name:
            raise IndexNotBuilt(f"index was built with {stored_model!r}, but queries use "
                                f"{embedder.model_name!r}; rebuild the index")
        self.embedder = embedder

    def search(self, query: str, k: int, sources: list[str] | None = None,
               contains: str | None = None) -> list[Passage]:
        kwargs: dict = {}
        if sources:
            kwargs["where"] = ({"source_id": sources[0]} if len(sources) == 1
                               else {"source_id": {"$in": list(sources)}})
        if contains:
            kwargs["where_document"] = {"$contains": contains}
        result = self.collection.query(query_embeddings=[self.embedder.embed_query(query)], n_results=k,
                                       include=["documents", "metadatas", "distances"], **kwargs)
        passages = []
        for cid, doc, meta, dist in zip(result["ids"][0], result["documents"][0], result["metadatas"][0],
                                        result["distances"][0]):
            passages.append(Passage(cid, meta["source_id"], meta["title"], int(meta["page"]), doc,
                                    round(float(dist), 4)))
        return passages
