"""PDF ingestion: page extraction, chunking, embedding, and persistent Chroma storage.

Every chunk lies within one physical PDF page, so its page metadata is exact. Chunk identifiers
have the form `<source_id>:p<page>:c<index>` (page one-based, index zero-based within the page).
They depend only on the source, the page text, and the chunk settings, so a rebuild with the same
settings reproduces the same identifiers.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .embeddings import Embedder
from .sources import SOURCES, Source


MIN_CHUNK_CHARS = 20  # shorter fragments (page numbers, stray headers) are not indexed


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict

    def embedding_text(self) -> str:
        """Text sent to the embedding model: a title/page header gives each chunk source context."""
        return f"{self.metadata['title']}, page {self.metadata['page']}\n\n{self.text}"


@dataclass
class SourceReport:
    source_id: str
    filename: str
    pages: int
    empty_pages: list[int]
    chunks: int


@dataclass
class IngestReport:
    collection: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    sources: list[SourceReport] = field(default_factory=list)
    chunks_total: int = 0
    chunks_added: int = 0
    chunks_already_present: int = 0
    built_at: str = ""

    def as_dict(self) -> dict:
        return {
            "collection": self.collection,
            "embedding_model": self.embedding_model,
            "chunk_size_chars": self.chunk_size,
            "chunk_overlap_chars": self.chunk_overlap,
            "pages_total": sum(s.pages for s in self.sources),
            "chunks_total": self.chunks_total,
            "chunks_added": self.chunks_added,
            "chunks_already_present": self.chunks_already_present,
            "built_at": self.built_at,
            "sources": [vars(s) for s in self.sources],
        }


def make_chunk_id(source_id: str, page: int, index: int) -> str:
    return f"{source_id}:p{page}:c{index}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """(one-based physical page number, normalized text) for every page, including empty ones."""
    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(path))
    return [(i, normalize_text(page.extract_text() or "")) for i, page in enumerate(reader.pages, start=1)]


_BREAKS = ("\n\n", "\n", ". ", "; ", " ")


def split_text(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Character spans of at most `size` that overlap by about `overlap` characters.

    A span ends at the latest natural break (paragraph, line, sentence, word) in the last 40% of
    the window, and the next span starts on a word boundary near `end - overlap`.
    """
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("require size >= 1 and 0 <= overlap < size")
    spans: list[tuple[int, int]] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            floor = start + int(size * 0.6)
            for brk in _BREAKS:
                pos = text.rfind(brk, floor, end)
                if pos != -1:
                    end = pos + len(brk)
                    break
        spans.append((start, end))
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        space = text.find(" ", nxt, end)
        nxt = space + 1 if space != -1 and overlap else nxt
        start = nxt if nxt > start else end
    return spans


def chunk_pages(source: Source, pages: list[tuple[int, str]], size: int, overlap: int,
                file_sha256: str = "") -> list[Chunk]:
    chunks: list[Chunk] = []
    for page, text in pages:
        index = 0
        for start, end in split_text(text, size, overlap):
            body = text[start:end].strip()
            if len(body) < MIN_CHUNK_CHARS:
                continue
            cid = make_chunk_id(source.source_id, page, index)
            chunks.append(Chunk(cid, body, {
                "chunk_id": cid,
                "source_id": source.source_id,
                "source_file": source.filename,
                "title": source.title,
                "page": page,
                "chunk_index": index,
                "char_start": start,
                "char_end": end,
                "file_sha256": file_sha256,
            }))
            index += 1
    return chunks


def chunk_source(source: Source, docs_dir: Path, size: int, overlap: int, verify: bool = True
                 ) -> tuple[list[Chunk], SourceReport]:
    path = Path(docs_dir) / source.filename
    if not path.exists():
        raise IngestError(f"missing source document: {path}")
    digest = sha256_file(path)
    if verify and digest != source.sha256:
        raise IngestError(f"{source.filename}: SHA-256 {digest} does not match the supplied corpus")
    pages = extract_pages(path)
    if verify and len(pages) != source.pages:
        raise IngestError(f"{source.filename}: {len(pages)} pages, expected {source.pages}")
    chunks = chunk_pages(source, pages, size, overlap, digest)
    empty = [p for p, t in pages if len(t) < MIN_CHUNK_CHARS]
    return chunks, SourceReport(source.source_id, source.filename, len(pages), empty, len(chunks))


def open_client(chroma_dir: Path):
    import chromadb

    return chromadb.PersistentClient(path=str(chroma_dir),
                                     settings=chromadb.Settings(anonymized_telemetry=False))


def build_index(settings: Settings, embedder: Embedder, rebuild: bool = False,
                sources: Mapping[str, Source] = SOURCES, verify: bool = True,
                progress: Callable[[str], None] = lambda msg: None) -> IngestReport:
    """Ingest every source into the persistent collection.

    With `rebuild`, the collection is dropped first. Without it, chunks whose ids are already
    stored are skipped, so an interrupted build can resume; the chunk settings and embedding model
    must match the stored ones.
    """
    client = open_client(settings.chroma_dir)
    expected = {"embedding_model": embedder.model_name, "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap}
    if rebuild and settings.collection in [c.name for c in client.list_collections()]:
        client.delete_collection(settings.collection)
        progress(f"dropped collection {settings.collection}")
    collection = client.get_or_create_collection(
        settings.collection, embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={**expected, "created_at": _now()},
    )
    stored = collection.metadata or {}
    mismatched = {k: (stored.get(k), v) for k, v in expected.items() if stored.get(k) != v}
    if mismatched:
        raise IngestError(f"collection settings differ from the current configuration {mismatched}; "
                          "run with --rebuild")

    report = IngestReport(settings.collection, embedder.model_name, settings.chunk_size, settings.chunk_overlap)
    present = set(collection.get(include=[])["ids"])
    for source in sources.values():
        progress(f"extracting {source.filename}")
        chunks, src_report = chunk_source(source, settings.docs_dir, settings.chunk_size,
                                          settings.chunk_overlap, verify)
        report.sources.append(src_report)
        report.chunks_total += len(chunks)
        todo = [c for c in chunks if c.chunk_id not in present]
        report.chunks_already_present += len(chunks) - len(todo)
        for i in range(0, len(todo), settings.embed_batch):
            batch = todo[i:i + settings.embed_batch]
            vectors = embedder.embed_documents([c.embedding_text() for c in batch])
            if len(vectors) != len(batch):
                raise IngestError(f"embedding service returned {len(vectors)} vectors for {len(batch)} inputs")
            collection.add(ids=[c.chunk_id for c in batch], embeddings=vectors,
                           documents=[c.text for c in batch], metadatas=[c.metadata for c in batch])
            report.chunks_added += len(batch)
            progress(f"  {source.source_id}: {min(i + len(batch), len(todo))}/{len(todo)} chunks embedded")
    report.built_at = _now()
    collection.modify(metadata={**expected, "created_at": stored.get("created_at", report.built_at),
                                "built_at": report.built_at, "chunk_count": collection.count()})
    if collection.count() != report.chunks_total:
        raise IngestError(f"collection holds {collection.count()} chunks, expected {report.chunks_total}; "
                          "run with --rebuild")
    return report


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
