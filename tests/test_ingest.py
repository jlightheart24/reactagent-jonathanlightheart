import re
from dataclasses import replace

import pytest

from docagent.config import Settings
from docagent.ingest import (IngestError, build_index, chunk_pages, chunk_source, extract_pages, make_chunk_id,
                             open_client, split_text)
from docagent.retrieval import ChromaRetriever, IndexNotBuilt
from docagent.sources import SOURCES, Source

from .conftest import HashEmbedder

DOCS = Settings().docs_dir
CHUNK_ID = re.compile(r"^[a-z0-9-]+:p[1-9]\d*:c\d+$")


def test_split_text_bounds_overlap_and_coverage():
    text = " ".join(f"word{i}" for i in range(400))
    spans = split_text(text, 200, 50)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (s1, e1), (s2, _) in zip(spans, spans[1:]):
        assert e1 - s1 <= 200
        assert s2 < e1, "consecutive chunks overlap"
        assert s2 > s1, "progress is made"
    covered = set()
    for s, e in spans:
        covered.update(range(s, e))
    assert covered == set(range(len(text)))


def test_split_text_short_and_invalid():
    assert split_text("short", 200, 50) == [(0, 5)]
    assert split_text("", 200, 50) == []
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)


def test_chunk_metadata_and_stable_ids():
    src = Source("demo", "demo.pdf", "Demo Title", "role", 2, "")
    pages = [(1, "First page text. " * 30), (2, ""), (3, "Third page has a sentence."), (4, "12")]
    chunks = chunk_pages(src, pages, 200, 40, file_sha256="abc")
    assert chunks == chunk_pages(src, pages, 200, 40, file_sha256="abc"), "ids are deterministic"
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(CHUNK_ID.match(i) for i in ids)
    assert {c.metadata["page"] for c in chunks} == {1, 3}, "blank and fragment-only pages yield no chunks"
    first = chunks[0].metadata
    assert first == {"chunk_id": "demo:p1:c0", "source_id": "demo", "source_file": "demo.pdf",
                     "title": "Demo Title", "page": 1, "chunk_index": 0, "char_start": 0,
                     "char_end": first["char_end"], "file_sha256": "abc"}
    for c in chunks:
        assert c.chunk_id == make_chunk_id("demo", c.metadata["page"], c.metadata["chunk_index"])
        assert c.embedding_text().startswith(f"Demo Title, page {c.metadata['page']}")


def test_extract_pages_uses_one_based_physical_pages(mini_corpus):
    settings, _ = mini_corpus
    pages = extract_pages(settings.docs_dir / "alpha.pdf")
    assert [p for p, _ in pages] == [1, 2, 3]
    assert "Alpha page one." in pages[0][1]
    assert pages[1][1] == ""
    assert "norovirus" in pages[2][1]


def test_chunk_source_reports_pages_and_verifies_hash(mini_corpus):
    settings, sources = mini_corpus
    chunks, report = chunk_source(sources["alpha"], settings.docs_dir, 200, 40, verify=False)
    assert report.pages == 3 and report.empty_pages == [2] and report.chunks == len(chunks)
    assert {c.metadata["page"] for c in chunks} == {1, 3}
    assert all(c.metadata["file_sha256"] for c in chunks)
    with pytest.raises(IngestError, match="SHA-256"):
        chunk_source(replace(sources["alpha"], sha256="0" * 64), settings.docs_dir, 200, 40, verify=True)
    with pytest.raises(IngestError, match="missing"):
        chunk_source(replace(sources["alpha"], filename="nope.pdf"), settings.docs_dir, 200, 40)


def test_build_index_persists_metadata_and_resumes(mini_corpus):
    settings, sources = mini_corpus
    embedder = HashEmbedder()
    report = build_index(settings, embedder, rebuild=True, sources=sources, verify=False)
    assert report.chunks_total == report.chunks_added > 3
    assert [s.source_id for s in report.sources] == ["alpha", "beta"]

    # Reopen from disk: data and metadata persisted.
    col = open_client(settings.chroma_dir).get_collection(settings.collection, embedding_function=None)
    assert col.count() == report.chunks_total
    assert col.metadata["embedding_model"] == "fake-hash-embedder"
    assert col.metadata["chunk_size"] == 200 and col.metadata["chunk_overlap"] == 40
    stored = col.get(include=["metadatas", "documents"])
    for cid, meta, doc in zip(stored["ids"], stored["metadatas"], stored["documents"]):
        assert cid == meta["chunk_id"] and CHUNK_ID.match(cid)
        assert meta["source_file"] in ("alpha.pdf", "beta.pdf")
        assert meta["page"] >= 1 and meta["title"] in ("Alpha Document", "Beta Document")
        assert doc
    ids_first = sorted(stored["ids"])

    # Resume without rebuild: nothing new to embed.
    again = build_index(settings, embedder, rebuild=False, sources=sources, verify=False)
    assert again.chunks_added == 0 and again.chunks_already_present == report.chunks_total

    # Rebuild reproduces the same identifiers.
    build_index(settings, embedder, rebuild=True, sources=sources, verify=False)
    col = open_client(settings.chroma_dir).get_collection(settings.collection, embedding_function=None)
    assert sorted(col.get(include=[])["ids"]) == ids_first


def test_build_index_rejects_changed_settings_without_rebuild(mini_corpus):
    settings, sources = mini_corpus
    build_index(settings, HashEmbedder(), rebuild=True, sources=sources, verify=False)
    with pytest.raises(IngestError, match="--rebuild"):
        build_index(replace(settings, chunk_size=300), HashEmbedder(), sources=sources, verify=False)
    report = build_index(replace(settings, chunk_size=300), HashEmbedder(), rebuild=True, sources=sources,
                         verify=False)
    assert report.chunk_size == 300


def test_chroma_retriever_returns_labeled_passages_and_filters(mini_corpus):
    settings, sources = mini_corpus
    embedder = HashEmbedder()
    build_index(settings, embedder, rebuild=True, sources=sources, verify=False)
    retriever = ChromaRetriever(settings, embedder)

    hits = retriever.search("Utah inspection summer rate ratios", 2)
    assert hits[0].chunk_id == "beta:p1:c0" and hits[0].page == 1 and hits[0].title == "Beta Document"
    assert hits[0].distance is not None

    only_alpha = retriever.search("Utah inspection summer", 5, sources=["alpha"])
    assert only_alpha and all(h.source_id == "alpha" for h in only_alpha)

    exact = retriever.search("section", 5, contains="2-201.12")
    assert exact and all("2-201.12" in h.text for h in exact)
    assert all(h.page == 3 for h in exact)


def test_chroma_retriever_requires_built_index_and_matching_model(mini_corpus):
    settings, sources = mini_corpus
    with pytest.raises(IndexNotBuilt):
        ChromaRetriever(settings, HashEmbedder())
    build_index(settings, HashEmbedder(), rebuild=True, sources=sources, verify=False)
    other = HashEmbedder()
    other.model_name = "different-model"
    with pytest.raises(IndexNotBuilt, match="rebuild"):
        ChromaRetriever(settings, other)


def test_source_catalog_matches_readme():
    assert sum(s.pages for s in SOURCES.values()) == 1110
    assert len(SOURCES) == 5
    assert all(len(s.sha256) == 64 for s in SOURCES.values())


@pytest.mark.skipif(not all((DOCS / s.filename).exists() for s in SOURCES.values()),
                    reason="supplied PDFs not present")
def test_supplied_corpus_is_complete():
    """When the supplied PDFs are present, they match the README's digests and page counts."""
    from pypdf import PdfReader

    from docagent.ingest import sha256_file

    for s in SOURCES.values():
        assert sha256_file(DOCS / s.filename) == s.sha256
        assert len(PdfReader(str(DOCS / s.filename)).pages) == s.pages
