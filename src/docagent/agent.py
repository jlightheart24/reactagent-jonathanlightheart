"""Bounded ReAct controller.

Each step is one model call followed by at most one validated action. The application, not the
model, owns the step budget, the set of retrieved chunks, citation validation, the trace, and
the terminal reason:

- `answer`: `final_answer` accepted; every citation was returned by retrieval in this run.
- `insufficient_evidence`: `insufficient_evidence` accepted after at least one search.
- `error`: the model service or the retrieval service raised an exception.
- `step_limit`: `max_steps` model calls ended without an accepted terminal action.

Rejected actions (no tool call, unknown tool, malformed or invalid arguments, citations that
retrieval did not return) are returned to the model as observations and still use a step. When a
turn holds several tool calls, only the first is executed and the rest receive a rejection.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from pydantic import ValidationError

from .llm import ChatModel, ModelTurn, ToolCall
from .retrieval import Passage, Retriever
from .sources import catalog_text
from .tools import (ANSWER, INSUFFICIENT, SEARCH, TOOL_SCHEMAS, AnswerArgs, InsufficientEvidenceArgs,
                    SearchArgs, passage_record, tool_definitions)

TerminalReason = Literal["answer", "insufficient_evidence", "error", "step_limit"]
TERMINAL_REASONS: tuple[str, ...] = ("answer", "insufficient_evidence", "error", "step_limit")

SYSTEM_PROMPT = f"""\
You are a research agent that answers questions using only a fixed collection of food-safety source
documents. You act only through tools; the application records every step and decides when the run ends.

Sources (source_id: title. role):
{catalog_text()}

Tools:
- search_documents(query, sources?, contains?, k?): semantic search. Each result has chunk_id, source,
  page, and passage text. Use `sources` to focus on one document and `contains` for exact strings such
  as Food Code section numbers (e.g. contains="2-201.12"). Run several focused searches when a question
  has several parts or spans several documents.
- final_answer(claims): finish with an answer. Each claim is one statement plus the chunk_ids that
  support it.
- insufficient_evidence(reason, citations?): finish when the retrieved passages do not support an answer.

Rules:
1. Call exactly one tool per turn; only the first call in a turn is executed.
2. Use only information in passages returned by search_documents during this run. Do not use outside
   knowledge, even when you are confident.
3. Every claim must be directly supported by the passages it cites. Cite chunk_ids exactly as returned.
   Quote or closely paraphrase numbers, times, and conditions from the passages. Name a section number
   in a claim only if a cited passage contains it.
4. State each source's role and edition accurately: the FDA Food Code is a model code (binding only where
   adopted); the handbook is guidance based on the earlier 2017 Food Code; the Bad Bug Book is pathogen
   reference material; Appendix 3 is draft, nonbinding guidance; the Utah study reports historical
   inspection observations. For a current FDA model-code rule, the 2026 Food Code controls.
5. Search before finishing. If the passages do not answer the question (for example, the question is off
   topic or asks for a jurisdiction's current law or penalties that the documents do not contain), call
   insufficient_evidence instead of guessing. If the documents answer part of a question but its central
   request is unsupported, call insufficient_evidence and explain what is missing.
6. Answer what the question asks, in a few precise claims; do not add loosely related material.
7. The number of steps is limited. Finish with final_answer or insufficient_evidence before it runs out.
"""


@dataclass
class RunResult:
    question: str
    terminal_reason: TerminalReason
    steps_used: int
    max_steps: int
    claims: list[dict] = field(default_factory=list)
    insufficient_reason: str | None = None
    error: str | None = None
    citations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    model: str | None = None
    config: dict = field(default_factory=dict)
    started_at: str = ""
    duration_s: float = 0.0

    def rendered_answer(self) -> str:
        if self.terminal_reason == "answer":
            return "\n".join(f"- {c['text']} [{'; '.join(c['citations'])}]" for c in self.claims)
        if self.terminal_reason == "insufficient_evidence":
            return f"Insufficient evidence: {self.insufficient_reason}"
        if self.terminal_reason == "error":
            return f"Error: {self.error}"
        return f"Step limit reached ({self.max_steps} steps) without an answer."

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "terminal_reason": self.terminal_reason,
            "steps_used": self.steps_used,
            "max_steps": self.max_steps,
            "final_result": self.rendered_answer(),
            "claims": self.claims,
            "insufficient_reason": self.insufficient_reason,
            "error": self.error,
            "citations": self.citations,
            "warnings": self.warnings,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "model": self.model,
            "config": self.config,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "trace": self.trace,
        }


class Rejected(Exception):
    def __init__(self, error: str, detail: str) -> None:
        super().__init__(detail)
        self.error, self.detail = error, detail


def _validation_detail(err: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or '(arguments)'}: {e['msg']}"
                     for e in err.errors(include_url=False))


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def unsupported_numbers(claim: str, passages: list[str]) -> list[str]:
    """Numbers in a claim that appear in none of its cited passages (a cheap support check)."""
    evidence = " ".join(passages).replace(",", "")
    found = []
    for num in _NUMBER.findall(claim):
        plain = num.replace(",", "")
        if not re.search(rf"(?<![\d.]){re.escape(plain)}(?![\d])", evidence):
            found.append(num)
    return found


class Agent:
    def __init__(self, model: ChatModel, retriever: Retriever, max_steps: int = 10, top_k: int = 5,
                 config: dict | None = None) -> None:
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError("max_steps must be an integer >= 1")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        self.model = model
        self.retriever = retriever
        self.max_steps = max_steps
        self.top_k = top_k
        self.config = dict(config or {})
        self.tools = tool_definitions()

    def run(self, question: str) -> RunResult:
        started, t0 = datetime.now(timezone.utc).isoformat(timespec="seconds"), time.monotonic()
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": question}]
        retrieved: dict[str, Passage] = {}
        trace: list[dict] = []
        result = RunResult(question, "step_limit", 0, self.max_steps, model=getattr(self.model, "name", None),
                           config={**self.config, "max_steps": self.max_steps, "retrieval_top_k": self.top_k},
                           started_at=started)
        searches = 0
        cited: list[str] = []

        for step in range(1, self.max_steps + 1):
            result.steps_used = step
            remaining = self.max_steps - step
            try:
                turn = self.model.complete(messages, self.tools)
            except Exception as e:
                trace.append({"step": step, "model_error": f"{type(e).__name__}: {e}"})
                result.terminal_reason, result.error = "error", f"model service error: {type(e).__name__}: {e}"
                break
            entry = {"step": step, "model": _turn_record(turn)}
            trace.append(entry)
            messages.append(_assistant_message(turn))

            if not turn.tool_calls:
                obs = self._rejection("no_tool_call", "respond with exactly one tool call", remaining)
                entry["action"] = {"tool": None, "status": "rejected"}
                entry["observation"] = obs
                self._reply(messages, [], obs)
                continue
            # Only the first call in a turn is executed; any extras are answered with a rejection
            # (after the first call's observation, so tool replies follow the call order).
            call, extras = turn.tool_calls[0], turn.tool_calls[1:]
            extra_obs = self._rejection("extra_tool_call", "only the first tool call in a turn is executed; "
                                        "send one tool call per turn", remaining)
            if extras:
                entry["extra_tool_calls_rejected"] = len(extras)
            try:
                name, args = self._validate_call(call)
                if name == SEARCH and isinstance(args, SearchArgs):
                    try:
                        passages = self.retriever.search(args.query, args.k or self.top_k, args.sources,
                                                         args.contains)
                    except Exception as e:
                        entry["action"] = {"tool": name, "arguments": args.model_dump(exclude_none=True),
                                           "status": "error"}
                        entry["observation"] = {"status": "error", "error": "retrieval_error",
                                                "detail": f"{type(e).__name__}: {e}"}
                        result.terminal_reason = "error"
                        result.error = f"retrieval service error: {type(e).__name__}: {e}"
                        break
                    searches += 1
                    for p in passages:
                        retrieved.setdefault(p.chunk_id, p)
                    obs = {"status": "ok", "result_count": len(passages),
                           "results": [passage_record(p) for p in passages], "steps_remaining": remaining}
                    if not passages:
                        obs["note"] = "no passages matched; try a different query or remove filters"
                    if remaining == 1:
                        obs["note"] = ("one step remains: call final_answer or insufficient_evidence now")
                    entry["action"] = {"tool": name, "arguments": args.model_dump(exclude_none=True),
                                       "status": "accepted"}
                    entry["observation"] = obs
                    self._reply(messages, [call], obs)
                    self._reply(messages, extras or None, extra_obs)
                    continue

                if name == ANSWER and isinstance(args, AnswerArgs):
                    self._check_citations([c for claim in args.claims for c in claim.citations], retrieved)
                    entry["action"] = {"tool": name, "arguments": args.model_dump(), "status": "accepted"}
                    entry["observation"] = {"status": "accepted", "terminal_reason": "answer"}
                    result.terminal_reason = "answer"
                    result.claims = [c.model_dump() for c in args.claims]
                    for claim in args.claims:
                        missing = unsupported_numbers(claim.text, [retrieved[c].text for c in claim.citations])
                        if missing:
                            result.warnings.append(f"claim {claim.text[:60]!r} has numbers not found in its "
                                                   f"cited passages: {', '.join(missing)}")
                    cited = [c for claim in args.claims for c in claim.citations]
                    break

                if name == INSUFFICIENT and isinstance(args, InsufficientEvidenceArgs):
                    if searches == 0:
                        raise Rejected("search_required",
                                       "search the documents before concluding that evidence is insufficient")
                    self._check_citations(args.citations, retrieved)
                    entry["action"] = {"tool": name, "arguments": args.model_dump(), "status": "accepted"}
                    entry["observation"] = {"status": "accepted", "terminal_reason": "insufficient_evidence"}
                    result.terminal_reason = "insufficient_evidence"
                    result.insufficient_reason = args.reason
                    cited = list(args.citations)
                    break
                raise AssertionError(f"unhandled tool {name}")
            except Rejected as r:
                obs = self._rejection(r.error, r.detail, remaining)
                entry["action"] = {"tool": call.name, "arguments": _raw_args(call.arguments), "status": "rejected"}
                entry["observation"] = obs
                self._reply(messages, [call], obs)
                self._reply(messages, extras or None, extra_obs)
                continue

        if result.terminal_reason in ("answer", "insufficient_evidence"):
            seen: list[str] = []
            for cid in cited:
                if cid not in seen:
                    seen.append(cid)
            result.citations = [{"chunk_id": cid, "source_id": retrieved[cid].source_id,
                                 "title": retrieved[cid].title, "page": retrieved[cid].page} for cid in seen]
        result.retrieved_chunk_ids = list(retrieved)
        result.trace = trace
        result.duration_s = round(time.monotonic() - t0, 2)
        return result

    # --- validation ---

    def _validate_call(self, call: ToolCall):
        if call.name not in TOOL_SCHEMAS:
            raise Rejected("unknown_tool", f"unknown tool {call.name!r}; use one of {', '.join(TOOL_SCHEMAS)}")
        raw = call.arguments
        if raw is None or raw == "":
            raw = {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as e:
                raise Rejected("malformed_arguments", f"arguments are not valid JSON: {e}") from None
        if not isinstance(raw, dict):
            raise Rejected("malformed_arguments", "arguments must be a JSON object")
        try:
            return call.name, TOOL_SCHEMAS[call.name].model_validate(raw)
        except ValidationError as e:
            raise Rejected("invalid_arguments", _validation_detail(e)) from None

    @staticmethod
    def _check_citations(citations: list[str], retrieved: dict[str, Passage]) -> None:
        unknown = sorted({c for c in citations if c not in retrieved})
        if unknown:
            raise Rejected("invalid_citation",
                           f"these chunk_ids were not returned by search_documents in this run: "
                           f"{', '.join(unknown)}. Cite only returned chunk_ids, or search again.")

    # --- messages ---

    @staticmethod
    def _rejection(error: str, detail: str, remaining: int) -> dict:
        obs = {"status": "rejected", "error": error, "detail": detail, "steps_remaining": remaining}
        if remaining == 1:
            obs["note"] = "one step remains: call final_answer or insufficient_evidence now"
        return obs

    @staticmethod
    def _reply(messages: list[dict], calls: list[ToolCall] | None, obs: dict) -> None:
        """Tool messages for `calls`; with calls=[] (a turn without any call) a user message instead."""
        content = json.dumps(obs, ensure_ascii=False)
        if calls is None:
            return
        if calls:
            for call in calls:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        else:
            messages.append({"role": "user", "content": content})


def _raw_args(arguments):
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _turn_record(turn: ModelTurn) -> dict:
    record = {"content": turn.content, "reasoning": turn.reasoning,
              "tool_calls": [{"id": c.id, "name": c.name, "arguments": _raw_args(c.arguments)}
                             for c in turn.tool_calls]}
    if turn.parsed_from_text:
        record["parsed_from_text"] = True
    return record


def _assistant_message(turn: ModelTurn) -> dict:
    msg: dict = {"role": "assistant", "content": turn.content or ""}
    if turn.tool_calls:
        msg["tool_calls"] = [{
            "id": c.id, "type": "function",
            "function": {"name": c.name or "",
                         "arguments": c.arguments if isinstance(c.arguments, str)
                         else json.dumps(c.arguments or {})},
        } for c in turn.tool_calls]
    return msg
