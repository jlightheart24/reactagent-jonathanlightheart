import json
from types import SimpleNamespace

import pytest

from docagent import cli
from docagent.agent import Agent
from docagent.config import ConfigError, Settings, load_settings
from docagent.llm import OpenAICompatibleChatModel, parse_text_tool_calls, split_reasoning
from docagent.queries import EXPECTED, parse_queries, render_markdown, render_summary

from .conftest import FOOD_CODE_EXCLUSION, FakeRetriever, ScriptedModel, call

ENV_VARS = ["EMBEDDING_BASE_URL", "EMBEDDING_MODEL", "EMBEDDING_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY",
            "DOCAGENT_CHUNK_SIZE", "DOCAGENT_CHUNK_OVERLAP", "DOCAGENT_TOP_K", "DOCAGENT_MAX_STEPS"]


@pytest.fixture
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_required_queries_parse_exactly(clean_env):
    queries = parse_queries()
    assert [q.query_id for q in queries] == [f"Q{i}" for i in range(1, 10)]
    assert queries[8].text == "Who is the strongest Avenger, and what evidence establishes that ranking?"
    assert queries[0].text.startswith("According to FDA Food Code 2026 sections 2-201.12(A)(1) and 2-201.13(A)(1),")
    assert "  " not in queries[5].text and "\n" not in queries[5].text
    assert set(EXPECTED) == {q.query_id for q in queries}


def test_settings_from_env_file_and_public_view(tmp_path, clean_env):
    env = tmp_path / ".env"
    env.write_text("EMBEDDING_BASE_URL=http://example/v1\nEMBEDDING_API_KEY=secret-key-1\n"
                   "LLM_API_KEY=secret-key-2\nDOCAGENT_CHUNK_SIZE=800\nDOCAGENT_MAX_STEPS=7\n")
    s = load_settings(env)
    assert s.chunk_size == 800 and s.max_steps == 7 and s.embedding_model == "Qwen3-Embedding-8B"
    public = json.dumps(s.public())
    assert "secret" not in public and "example" not in public
    assert load_settings(env, max_steps=3).max_steps == 3


@pytest.mark.parametrize("kwargs", [{"chunk_overlap": 1200}, {"top_k": 0}, {"max_steps": 0}, {"chunk_size": 10}])
def test_settings_validation(kwargs):
    with pytest.raises(ConfigError):
        Settings(**kwargs)


def test_missing_credentials_are_reported(clean_env, monkeypatch, tmp_path):
    with pytest.raises(ConfigError, match="LLM_BASE_URL"):
        OpenAICompatibleChatModel(Settings())
    # Ignore any local .env so the CLI sees no credentials and never contacts a service.
    monkeypatch.setattr(cli, "load_settings", lambda **kw: load_settings(env_file=None, **kw))
    assert cli.main(["ask", "question?"]) == 2


def test_split_reasoning_and_text_tool_calls():
    assert split_reasoning("<think>plan</think>Answer") == ("Answer", "plan")
    calls = parse_text_tool_calls('<tool_call>{"name": "search_documents", "arguments": {"query": "x y z"}}'
                                  '</tool_call>')
    assert calls[0].name == "search_documents" and calls[0].arguments == {"query": "x y z"}
    assert parse_text_tool_calls("<tool_call>{bad</tool_call>")[0].name is None


def test_openai_client_adapter_maps_tool_calls():
    settings = Settings(llm_base_url="http://localhost:1/v1", llm_api_key="k")
    model = OpenAICompatibleChatModel(settings)
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        tc = SimpleNamespace(id="c1", function=SimpleNamespace(name="search_documents", arguments='{"query": "abc"}'))
        msg = SimpleNamespace(content="<think>hmm</think>", tool_calls=[tc])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    model._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    turn = model.complete([{"role": "user", "content": "q"}], [{"type": "function"}])
    assert turn.tool_calls[0].name == "search_documents" and turn.reasoning == "hmm" and turn.content is None
    assert sent["temperature"] == 0.0 and sent["tool_choice"] == "auto" and sent["model"] == "Qwen3.8-27B"


def test_run_queries_writes_traces_and_summary(tmp_path, monkeypatch, clean_env):
    """The required-query runner records one configuration plus a JSON trace and Markdown view per query."""
    cid = FOOD_CODE_EXCLUSION.chunk_id

    class Router(ScriptedModel):
        def complete(self, messages, tools):
            self.received.append(messages)
            question = messages[1]["content"]
            if not any(m["role"] == "tool" for m in messages):
                return call("search_documents", query="exclude employee vomiting")
            if "Avenger" in question or "Utah regulation" in question:
                return call("insufficient_evidence", reason="Not in the corpus.")
            return call("final_answer", claims=[{"text": "Exclude the employee.", "citations": [cid]}])

    agent = Agent(Router([]), FakeRetriever([FOOD_CODE_EXCLUSION]), max_steps=4, top_k=2,
                  config={"embedding_model": "fake", "chunk_size_chars": 1200})
    monkeypatch.setattr(cli, "_agent", lambda settings: agent)
    out = tmp_path / "results"
    assert cli.main(["run-queries", "--out", str(out)]) == 0

    config = json.loads((out / "config.json").read_text())
    assert config["max_steps"] == 4 and config["retrieval_top_k"] == 2 and config["embedding_model"] == "fake"
    for i in range(1, 10):
        data = json.loads((out / f"Q{i}.json").read_text())
        assert data["terminal_reason"] == data["expected"]
        assert data["trace"] and data["config"]["max_steps"] == 4
        assert (out / f"Q{i}.md").read_text().startswith(f"# Q{i}:")
    summary = (out / "README.md").read_text()
    assert "| max_steps | `4` |" in summary and summary.count("| yes |") == 9


def test_render_markdown_includes_trace_and_citations(retriever):
    result = Agent(ScriptedModel([call("search_documents", query="exclude employee vomiting"),
                                  call("final_answer", claims=[{"text": "Exclude.",
                                                                "citations": [FOOD_CODE_EXCLUSION.chunk_id]}])]),
                   retriever, max_steps=3).run("q?")
    md = render_markdown(result)
    assert "### Step 1" in md and "`search_documents` (accepted)" in md
    assert "| `food-code-2026:p59:c1` | FDA Food Code 2026 | 59 |" in md
    assert render_summary([], {"a": 1}).startswith("# Required-query results")
