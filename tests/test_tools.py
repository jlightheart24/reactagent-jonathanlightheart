import json

import pytest
from pydantic import ValidationError

from docagent.tools import (AnswerArgs, InsufficientEvidenceArgs, SearchArgs, passage_record, tool_definitions)

from .conftest import FOOD_CODE_EXCLUSION


def test_search_args_valid_and_normalized():
    args = SearchArgs.model_validate({"query": "  vomiting employee exclusion  ", "sources": ["food-code-2026"],
                                      "contains": "2-201.12", "k": 3})
    assert args.query == "vomiting employee exclusion"
    assert args.sources == ["food-code-2026"] and args.k == 3
    assert SearchArgs(query="norovirus").k is None
    assert SearchArgs(contains="2-201.12").query == "2-201.12"


@pytest.mark.parametrize("bad", [
    {},                                                   # query or contains required
    {"sources": ["bad-bug-book"]},
    {"query": "  "},                                      # blank after stripping
    {"query": "ab"},                                      # too short
    {"query": "x" * 501},                                 # too long
    {"query": "norovirus", "k": 0},
    {"query": "norovirus", "k": 11},
    {"query": "norovirus", "k": "five"},
    {"query": "norovirus", "sources": ["wikipedia"]},     # unknown source id
    {"query": "norovirus", "sources": []},
    {"query": "norovirus", "sources": ["bad-bug-book", "bad-bug-book"]},
    {"query": "norovirus", "contains": "x"},
    {"query": "norovirus", "top_k": 3},                   # unexpected field
    {"query": 42},
])
def test_search_args_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        SearchArgs.model_validate(bad)


def test_answer_args_require_cited_claims():
    AnswerArgs.model_validate({"claims": [{"text": "A claim.", "citations": ["x:p1:c0"]}]})
    for bad in ({"claims": []}, {"claims": [{"text": "A claim.", "citations": []}]},
                {"claims": [{"text": "", "citations": ["x:p1:c0"]}]}, {"answer": "text"}):
        with pytest.raises(ValidationError):
            AnswerArgs.model_validate(bad)


def test_insufficient_args():
    assert InsufficientEvidenceArgs(reason="Not in the corpus.").citations == []
    with pytest.raises(ValidationError):
        InsufficientEvidenceArgs.model_validate({"reason": ""})


def test_tool_definitions_are_self_contained():
    defs = tool_definitions()
    assert [d["function"]["name"] for d in defs] == ["search_documents", "final_answer", "insufficient_evidence"]
    text = json.dumps(defs)
    assert "$ref" not in text and "$defs" not in text
    search = defs[0]["function"]["parameters"]
    assert "query" in search["properties"] and search["additionalProperties"] is False


def test_passage_record_has_source_page_and_chunk_labels():
    rec = passage_record(FOOD_CODE_EXCLUSION)
    assert rec["chunk_id"] == "food-code-2026:p59:c1"
    assert rec["source_title"] == "FDA Food Code 2026" and rec["source_id"] == "food-code-2026"
    assert rec["page"] == 59
    assert "model code" in rec["source_role"]
    assert rec["text"].startswith("2-201.12")
