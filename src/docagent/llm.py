"""Chat-model interface and the OpenAI-compatible client for the course LLM service.

The agent depends only on `ChatModel`; tests substitute a scripted model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str | None
    arguments: str | dict | None  # raw JSON text from the service, or an already-parsed object


@dataclass(frozen=True)
class ModelTurn:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None
    parsed_from_text: bool = False  # tool call recovered from message text (see parse_text_tool_calls)


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelTurn: ...


_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_TOOL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def split_reasoning(content: str | None) -> tuple[str | None, str | None]:
    """Separate inline <think>...</think> reasoning from the visible content."""
    if not content:
        return None, None
    thoughts = [m.strip() for m in _THINK.findall(content)]
    visible = _THINK.sub("", content).strip()
    return (visible or None), ("\n".join(thoughts) or None)


def parse_text_tool_calls(content: str | None) -> list[ToolCall]:
    """Recover `<tool_call>{"name": ..., "arguments": ...}</tool_call>` blocks that a service
    returned as text instead of structured tool calls."""
    calls = []
    for i, raw in enumerate(_TOOL_TAG.findall(content or "")):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            calls.append(ToolCall(f"text_call_{i}", None, raw))
            continue
        calls.append(ToolCall(f"text_call_{i}", obj.get("name"), obj.get("arguments")))
    return calls


class OpenAICompatibleChatModel:
    def __init__(self, settings: Settings) -> None:
        settings.require("llm_base_url", "llm_api_key")
        from openai import OpenAI

        self.name = settings.llm_model
        self.temperature = settings.temperature
        self.tool_choice = settings.tool_choice
        self._client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                              timeout=300, max_retries=2)

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelTurn:
        kwargs = {"tool_choice": self.tool_choice} if self.tool_choice not in ("", "none-sent") else {}
        response = self._client.chat.completions.create(
            model=self.name, messages=messages, tools=tools, temperature=self.temperature, **kwargs)
        msg = response.choices[0].message
        content, inline_reasoning = split_reasoning(msg.content)
        reasoning = getattr(msg, "reasoning_content", None) or inline_reasoning
        calls = [ToolCall(tc.id, tc.function.name, tc.function.arguments) for tc in (msg.tool_calls or [])]
        from_text = False
        if not calls and content:
            calls = parse_text_tool_calls(content)
            from_text = bool(calls)
            if from_text:
                content = _TOOL_TAG.sub("", content).strip() or None
        return ModelTurn(content, calls, reasoning, from_text)
