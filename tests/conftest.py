"""Offline test doubles: a tiny PDF writer, a hashing embedder, a substitute retriever, and a scripted model."""
from __future__ import annotations

import hashlib
import itertools
import math
import re
from pathlib import Path

import pytest

from docagent.config import Settings
from docagent.llm import ModelTurn, ToolCall
from docagent.retrieval import Passage
from docagent.sources import Source


def write_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Write a minimal valid PDF; each page is a list of text lines (an empty list makes a blank page)."""
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    n = len(pages)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        ("<< /Type /Pages /Kids [" + " ".join(f"{4 + 2 * i} 0 R" for i in range(n)) + f"] /Count {n} >>").encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, lines in enumerate(pages):
        body = "BT /F1 11 Tf 14 TL 72 740 Td " + " ".join(f"({esc(line)}) Tj T*" for line in lines) + " ET"
        stream = body.encode("latin-1")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                       f"/Contents {5 + 2 * i} 0 R >>".encode())
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


class HashEmbedder:
    """Deterministic bag-of-words embedding; similar wording gives similar vectors."""

    model_name = "fake-hash-embedder"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeRetriever:
    """Substitute retrieval results: word-overlap ranking over a fixed passage list."""

    def __init__(self, passages: list[Passage], fail: Exception | None = None) -> None:
        self.passages = passages
        self.fail = fail
        self.calls: list[dict] = []

    def search(self, query, k, sources=None, contains=None):
        self.calls.append({"query": query, "k": k, "sources": sources, "contains": contains})
        if self.fail:
            raise self.fail
        words = set(re.findall(r"\w+", query.lower()))
        pool = [p for p in self.passages if (not sources or p.source_id in sources)
                and (not contains or contains in p.text)]
        pool.sort(key=lambda p: -len(words & set(re.findall(r"\w+", p.text.lower()))))
        return pool[:k]


_ids = itertools.count(1)


def call(name: str, content: str | None = None, **args) -> ModelTurn:
    """A model turn requesting one tool call with dict arguments."""
    return ModelTurn(content=content, tool_calls=[ToolCall(f"call_{next(_ids)}", name, args)])


def raw_call(name, arguments) -> ModelTurn:
    return ModelTurn(tool_calls=[ToolCall(f"call_{next(_ids)}", name, arguments)])


class ScriptedModel:
    """Returns scripted turns (or raises scripted exceptions) in order; optionally repeats the last."""

    name = "scripted-model"

    def __init__(self, script: list, repeat_last: bool = False) -> None:
        self.script = script
        self.repeat_last = repeat_last
        self.received: list[list[dict]] = []

    def complete(self, messages, tools):
        self.received.append([dict(m) for m in messages])
        idx = len(self.received) - 1
        if idx >= len(self.script):
            if not self.repeat_last:
                raise IndexError("script exhausted")
            idx = len(self.script) - 1
        item = self.script[idx]
        if isinstance(item, BaseException):
            raise item
        return item


FOOD_CODE_EXCLUSION = Passage(
    "food-code-2026:p59:c1", "food-code-2026", "FDA Food Code 2026", 59,
    "2-201.12 Exclusions and Restrictions. (A) Except when the symptom is from a noninfectious condition, "
    "EXCLUDE a FOOD EMPLOYEE if the FOOD EMPLOYEE is: (1) Symptomatic with vomiting or diarrhea.")
FOOD_CODE_REINSTATE = Passage(
    "food-code-2026:p61:c0", "food-code-2026", "FDA Food Code 2026", 61,
    "2-201.13 Removal, Adjustment, or Retention of Exclusions and Restrictions. (A)(1) ... the PERSON IN "
    "CHARGE may reinstate the FOOD EMPLOYEE if the employee is asymptomatic for at least 24 hours.")
BAD_BUG_NOROVIRUS = Passage(
    "bad-bug-book:p163:c2", "bad-bug-book", "Bad Bug Book, Second Edition", 163,
    "Norovirus infective dose is estimated to be as low as 18 viral particles. Onset of symptoms usually "
    "begins 12 to 48 hours after exposure.")
UTAH_SUMMER = Passage(
    "utah-risk-factor-study:p4:c1", "utah-risk-factor-study", "Utah Risk Factor Study, 2016-2017", 4,
    "Four line items were found to occur more frequently during summer months. Line item 22 rate ratio 1.37.")

PASSAGES = [FOOD_CODE_EXCLUSION, FOOD_CODE_REINSTATE, BAD_BUG_NOROVIRUS, UTAH_SUMMER]


@pytest.fixture
def retriever():
    return FakeRetriever(list(PASSAGES))


@pytest.fixture
def mini_corpus(tmp_path):
    """Two small PDFs with a matching source catalog (hash checks disabled in tests)."""
    docs = tmp_path / "docs"
    docs.mkdir()
    long_line = "Cold holding temperature control for time and temperature control for safety food. "
    write_pdf(docs / "alpha.pdf", [
        ["Alpha page one.", "Employees with vomiting must be excluded."],
        [],
        ["Alpha page three discusses norovirus transmission.", long_line * 2, long_line * 2,
         long_line * 2, "Section 2-201.12 applies."],
    ])
    write_pdf(docs / "beta.pdf", [["Beta page one: Utah inspection summer rate ratios."]])
    sources = {
        "alpha": Source("alpha", "alpha.pdf", "Alpha Document", "test role A", 3, ""),
        "beta": Source("beta", "beta.pdf", "Beta Document", "test role B", 1, ""),
    }
    settings = Settings(docs_dir=docs, chroma_dir=tmp_path / "chroma", collection="test_corpus",
                        chunk_size=200, chunk_overlap=40, embed_batch=3, embedding_model="fake-hash-embedder")
    return settings, sources
