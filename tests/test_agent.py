import json

import pytest

from docagent.agent import SYSTEM_PROMPT, TERMINAL_REASONS, Agent, unsupported_numbers
from docagent.llm import ModelTurn, ToolCall

from .conftest import (BAD_BUG_NOROVIRUS, FOOD_CODE_EXCLUSION, FOOD_CODE_REINSTATE, FakeRetriever, ScriptedModel,
                       call, raw_call)

EXCL, REIN = FOOD_CODE_EXCLUSION.chunk_id, FOOD_CODE_REINSTATE.chunk_id


def answer(*claims):
    return call("final_answer", claims=[{"text": t, "citations": list(c)} for t, c in claims])


def run(script, retriever, max_steps=6, **kw):
    model = ScriptedModel(script, **kw)
    return Agent(model, retriever, max_steps=max_steps, top_k=3).run("What must the person in charge do?"), model


# --- scripted end-to-end runs -------------------------------------------------------------------------------

def test_supported_run(retriever):
    result, model = run([
        call("search_documents", content="Look up the exclusion rule.", query="exclude employee vomiting",
             sources=["food-code-2026"]),
        call("search_documents", query="reinstate employee asymptomatic 24 hours", contains="2-201.13"),
        answer(("The person in charge must exclude a food employee who is symptomatic with vomiting or diarrhea.",
                [EXCL]),
               ("The employee may be reinstated after being asymptomatic for at least 24 hours.", [REIN])),
    ], retriever)

    assert result.terminal_reason == "answer"
    assert result.steps_used == 3
    assert [c["chunk_id"] for c in result.citations] == [EXCL, REIN]
    assert result.citations[0] == {"chunk_id": EXCL, "source_id": "food-code-2026",
                                   "title": "FDA Food Code 2026", "page": 59}
    assert result.warnings == []
    assert "[food-code-2026:p59:c1]" in result.rendered_answer()

    # Retrieval observations carry passage text plus source, page, and chunk labels.
    obs = result.trace[0]["observation"]
    assert obs["status"] == "ok" and obs["result_count"] >= 1
    assert all(r["source_id"] == "food-code-2026" for r in obs["results"])
    first = obs["results"][0]
    assert {"chunk_id", "source_title", "page", "text"} <= first.keys()
    assert result.trace[0]["model"]["content"] == "Look up the exclusion rule."
    assert result.trace[0]["action"] == {"tool": "search_documents", "status": "accepted", "arguments": {
        "query": "exclude employee vomiting", "sources": ["food-code-2026"]}}
    assert result.trace[1]["observation"]["results"][0]["chunk_id"] == REIN, "contains filter passed through"
    assert retriever.calls[0]["k"] == 3, "default k from the agent configuration"

    # The model saw the observations as tool messages tied to its call ids.
    last_prompt = model.received[-1]
    assert last_prompt[0] == {"role": "system", "content": SYSTEM_PROMPT}
    tool_msgs = [m for m in last_prompt if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assistant = [m for m in last_prompt if m["role"] == "assistant"]
    assert tool_msgs[0]["tool_call_id"] == assistant[0]["tool_calls"][0]["id"]
    assert json.loads(tool_msgs[0]["content"])["results"][0]["chunk_id"].startswith("food-code-2026")


def test_insufficient_evidence_run(retriever):
    result, _ = run([
        call("search_documents", query="Utah line items summer penalties"),
        call("insufficient_evidence", reason="The corpus does not identify a current Utah adoption rule or penalties.",
             citations=["utah-risk-factor-study:p4:c1"]),
    ], retriever)
    assert result.terminal_reason == "insufficient_evidence"
    assert result.insufficient_reason.startswith("The corpus does not")
    assert result.claims == []
    assert [c["chunk_id"] for c in result.citations] == ["utah-risk-factor-study:p4:c1"]
    assert result.rendered_answer().startswith("Insufficient evidence:")


# --- citation validation ---------------------------------------------------------------------------------

def test_citation_not_returned_by_retrieval_is_rejected_then_corrected(retriever):
    fabricated = "food-code-2026:p999:c0"
    result, model = run([
        call("search_documents", query="exclude employee vomiting"),
        answer(("Exclude the employee.", [EXCL, fabricated])),
        answer(("Exclude the employee.", [EXCL])),
    ], retriever)
    assert result.terminal_reason == "answer"
    rejected = result.trace[1]
    assert rejected["action"]["status"] == "rejected"
    assert rejected["observation"]["error"] == "invalid_citation"
    assert fabricated in rejected["observation"]["detail"]
    assert EXCL not in rejected["observation"]["detail"]
    assert [c["chunk_id"] for c in result.citations] == [EXCL]


def test_citation_of_existing_chunk_not_retrieved_in_this_run_is_rejected(retriever):
    # BAD_BUG_NOROVIRUS exists in the retriever, but no search in this run returned it.
    result, _ = run([
        call("search_documents", query="exclude employee", sources=["food-code-2026"]),
        answer(("Norovirus dose is low.", [BAD_BUG_NOROVIRUS.chunk_id])),
    ], retriever, max_steps=2)
    assert result.trace[1]["observation"]["error"] == "invalid_citation"
    assert result.terminal_reason == "step_limit"
    assert result.claims == [] and result.citations == []


def test_answer_without_any_search_is_rejected(retriever):
    result, _ = run([answer(("Exclude the employee.", [EXCL]))], retriever, max_steps=1)
    assert result.trace[0]["observation"]["error"] == "invalid_citation"
    assert result.terminal_reason == "step_limit"


def test_insufficient_evidence_citations_are_validated(retriever):
    result, _ = run([
        call("search_documents", query="norovirus infective dose"),
        call("insufficient_evidence", reason="Not supported.", citations=["made:p1:c0"]),
    ], retriever, max_steps=2)
    assert result.trace[1]["observation"]["error"] == "invalid_citation"
    assert result.terminal_reason == "step_limit"


def test_insufficient_evidence_requires_a_search(retriever):
    result, _ = run([
        call("insufficient_evidence", reason="Off topic."),
        call("search_documents", query="strongest Avenger ranking"),
        call("insufficient_evidence", reason="Off topic."),
    ], retriever)
    assert result.trace[0]["observation"]["error"] == "search_required"
    assert result.terminal_reason == "insufficient_evidence" and result.steps_used == 3


# --- step limit and terminal reasons -----------------------------------------------------------------------

def test_step_limit_is_enforced(retriever):
    result, model = run([call("search_documents", query="norovirus symptoms")], retriever, max_steps=4,
                        repeat_last=True)
    assert result.terminal_reason == "step_limit"
    assert result.steps_used == 4 and len(model.received) == 4 and len(result.trace) == 4
    assert len(retriever.calls) == 4
    assert result.trace[-1]["observation"]["steps_remaining"] == 0
    assert "one step remains" in result.trace[-2]["observation"]["note"]
    assert "Step limit reached (4 steps)" in result.rendered_answer()


@pytest.mark.parametrize("max_steps", [0, -1, 2.5, True, None])
def test_step_limit_must_be_a_positive_integer(retriever, max_steps):
    with pytest.raises(ValueError):
        Agent(ScriptedModel([]), retriever, max_steps=max_steps)


def test_model_error_terminates_with_error(retriever):
    result, _ = run([call("search_documents", query="norovirus"), ConnectionError("service unreachable")], retriever)
    assert result.terminal_reason == "error"
    assert "ConnectionError" in result.error and result.steps_used == 2
    assert result.trace[-1] == {"step": 2, "model_error": "ConnectionError: service unreachable"}


def test_retrieval_error_terminates_with_error():
    failing = FakeRetriever([], fail=TimeoutError("embedding service timed out"))
    result, _ = run([call("search_documents", query="norovirus")], failing)
    assert result.terminal_reason == "error"
    assert "retrieval service error" in result.error
    assert result.trace[0]["observation"]["error"] == "retrieval_error"


def test_all_four_terminal_reasons(retriever):
    scripts = {
        "answer": [call("search_documents", query="exclude employee vomiting"), answer(("Exclude.", [EXCL]))],
        "insufficient_evidence": [call("search_documents", query="Avengers"),
                                  call("insufficient_evidence", reason="Off topic.")],
        "error": [RuntimeError("boom")],
        "step_limit": [call("search_documents", query="norovirus")] * 3,
    }
    seen = set()
    for expected, script in scripts.items():
        result, _ = run(script, FakeRetriever(list(retriever.passages)), max_steps=3)
        assert result.terminal_reason == expected
        seen.add(result.terminal_reason)
    assert seen == set(TERMINAL_REASONS)


# --- protocol and argument validation ------------------------------------------------------------------

@pytest.mark.parametrize("turn, error", [
    (ModelTurn(content="I think the answer is to exclude them."), "no_tool_call"),
    (raw_call("web_search", {"query": "norovirus"}), "unknown_tool"),
    (raw_call("search_documents", '{"query": "norovirus"'), "malformed_arguments"),
    (raw_call("search_documents", '["norovirus"]'), "malformed_arguments"),
    (raw_call("search_documents", {"query": "norovirus", "k": 50}), "invalid_arguments"),
    (raw_call("search_documents", {"query": "norovirus", "sources": ["wikipedia"]}), "invalid_arguments"),
    (raw_call("final_answer", {"claims": [{"text": "x", "citations": []}]}), "invalid_arguments"),
])
def test_invalid_actions_are_rejected_and_reported(retriever, turn, error):
    result, model = run([turn, call("search_documents", query="exclude employee vomiting"),
                         answer(("Exclude.", [EXCL]))], retriever)
    first = result.trace[0]
    assert first["action"]["status"] == "rejected"
    assert first["observation"]["error"] == error
    assert result.terminal_reason == "answer" and result.steps_used == 3, "rejections use a step, run continues"
    assert len(retriever.calls) == 1, "rejected calls are never executed"
    second_prompt = model.received[1]
    replies = [m for m in second_prompt if m["role"] in ("tool", "user")][1:]
    assert replies and all(json.loads(m["content"])["error"] == error for m in replies)
    if turn.tool_calls:
        assert {m["tool_call_id"] for m in replies} == {c.id for c in turn.tool_calls}


def test_only_first_of_several_tool_calls_is_executed(retriever):
    turn = ModelTurn(tool_calls=[ToolCall("a", "search_documents", {"query": "exclude employee vomiting"}),
                                 ToolCall("b", "search_documents", {"query": "norovirus dose"})])
    result, model = run([turn, answer(("Exclude.", [EXCL]))], retriever)
    assert result.terminal_reason == "answer" and result.steps_used == 2
    assert [c["query"] for c in retriever.calls] == ["exclude employee vomiting"]
    assert result.trace[0]["action"]["status"] == "accepted" and result.trace[0]["extra_tool_calls_rejected"] == 1
    replies = [m for m in model.received[1] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in replies] == ["a", "b"], "replies follow call order"
    assert json.loads(replies[0]["content"])["status"] == "ok"
    assert json.loads(replies[1]["content"])["error"] == "extra_tool_call"


def test_contains_only_search_uses_the_text_as_query(retriever):
    result, _ = run([call("search_documents", contains="2-201.13", sources=["food-code-2026"]),
                     answer(("Reinstate after 24 hours.", [REIN]))], retriever)
    assert retriever.calls[0]["query"] == "2-201.13" and retriever.calls[0]["contains"] == "2-201.13"
    assert result.terminal_reason == "answer"


def test_json_string_arguments_are_parsed(retriever):
    result, _ = run([raw_call("search_documents", json.dumps({"query": "exclude employee vomiting"})),
                     answer(("Exclude.", [EXCL]))], retriever)
    assert result.terminal_reason == "answer"


def test_unsupported_numbers_check():
    passage = "Reinstate after the employee is asymptomatic for at least 24 hours; 1,200 samples."
    assert unsupported_numbers("At least 24 hours.", [passage]) == []
    assert unsupported_numbers("1200 samples were taken.", [passage]) == []
    assert unsupported_numbers("At least 48 hours.", [passage]) == ["48"]
    assert unsupported_numbers("2.4 hours", [passage]) == ["2.4"]


def test_numeric_mismatch_is_flagged_as_warning(retriever):
    result, _ = run([call("search_documents", query="reinstate asymptomatic 24 hours"),
                     answer(("Reinstate after 48 hours without symptoms.", [REIN]))], retriever)
    assert result.terminal_reason == "answer"
    assert result.warnings and "48" in result.warnings[0]


def test_result_dict_is_json_serializable(retriever):
    result, _ = run([call("search_documents", query="exclude employee vomiting"), answer(("Exclude.", [EXCL]))],
                    retriever)
    data = json.loads(json.dumps(result.to_dict()))
    assert data["terminal_reason"] == "answer" and data["config"]["max_steps"] == 6
    assert data["retrieved_chunk_ids"]
