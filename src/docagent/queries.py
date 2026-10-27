"""Required-query parsing and result rendering."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .agent import RunResult

DEFAULT_QUERIES = Path(__file__).resolve().parents[2] / "U0-HW-09-materials" / "U0-HW-09-required-queries.md"

# Q1-Q7 should be answered from the sources; Q8-Q9 should end with insufficient evidence.
EXPECTED = {f"Q{i}": "answer" for i in range(1, 8)} | {"Q8": "insufficient_evidence", "Q9": "insufficient_evidence"}

_HEADING = re.compile(r"^## (Q\d+): (.+?)\s*$")


@dataclass(frozen=True)
class Query:
    query_id: str
    title: str
    text: str


def parse_queries(path: Path = DEFAULT_QUERIES) -> list[Query]:
    """Each `## Qn: Title` section's text, exactly as written; hard-wrapped lines are joined with spaces."""
    queries: list[Query] = []
    current: tuple[str, str] | None = None
    lines: list[str] = []

    def flush():
        if current:
            text = " ".join(line.strip() for line in lines if line.strip())
            queries.append(Query(current[0], current[1], text))

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            current, lines = (m.group(1), m.group(2)), []
        elif line.startswith("#"):
            flush()
            current, lines = None, []
        elif current:
            lines.append(line)
    flush()
    return queries


def _excerpt(text: str, n: int = 400) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " …"


def render_markdown(result: RunResult, query: Query | None = None) -> str:
    title = f"{query.query_id}: {query.title}" if query else "Agent run"
    out = [f"# {title}", "", f"**Question (as run):** {result.question}", "",
           f"- Terminal reason: `{result.terminal_reason}`"]
    if query and query.query_id in EXPECTED:
        ok = "matches" if EXPECTED[query.query_id] == result.terminal_reason else "DOES NOT match"
        out.append(f"- Expected outcome: `{EXPECTED[query.query_id]}` ({ok})")
    out += [f"- Steps used: {result.steps_used} of {result.max_steps}",
            f"- Model: `{result.model}`; duration {result.duration_s} s", "", "## Final result", ""]
    out.append(result.rendered_answer())
    if result.citations:
        out += ["", "## Citations", "", "| Chunk ID | Source | PDF page |", "|---|---|---:|"]
        out += [f"| `{c['chunk_id']}` | {c['title']} | {c['page']} |" for c in result.citations]
    if result.warnings:
        out += ["", "## Support warnings", ""] + [f"- {w}" for w in result.warnings]
    out += ["", "## Action-observation trace", ""]
    for entry in result.trace:
        out.append(f"### Step {entry['step']}")
        if "model_error" in entry:
            out += ["", f"Model error: `{entry['model_error']}`", ""]
            continue
        model = entry["model"]
        if model.get("reasoning"):
            out += ["", "**Thought:** " + _excerpt(model["reasoning"], 1500)]
        if model.get("content"):
            out += ["", "**Model text:** " + _excerpt(model["content"], 1500)]
        action = entry.get("action", {})
        out += ["", f"**Action:** `{action.get('tool')}` ({action.get('status')})"]
        if "arguments" in action:
            out += ["", "```json", _json(action["arguments"]), "```"]
        obs = entry.get("observation", {})
        out += ["", f"**Observation:** status `{obs.get('status')}`"]
        if obs.get("status") in ("rejected", "error"):
            out.append(f"— `{obs.get('error')}`: {obs.get('detail')}")
        for r in obs.get("results", []):
            out.append(f"- `{r['chunk_id']}` ({r['source_title']}, p. {r['page']}, distance {r['distance']}): "
                       f"{_excerpt(r['text'])}")
        if obs.get("note"):
            out.append(f"- Note: {obs['note']}")
        out.append("")
    out.append("Full passage text for every observation is in the JSON trace next to this file.")
    return "\n".join(out) + "\n"


def render_summary(rows: list[tuple[Query, RunResult]], config: dict) -> str:
    out = ["# Required-query results", "", "## Recorded configuration", "", "| Setting | Value |", "|---|---|"]
    out += [f"| {k} | `{v}` |" for k, v in config.items()]
    out += ["", "## Outcomes", "", "| Query | Expected | Terminal reason | Steps | Citations | Match |",
            "|---|---|---|---:|---:|---|"]
    for q, r in rows:
        exp = EXPECTED.get(q.query_id, "")
        out.append(f"| [{q.query_id}]({q.query_id}.md) | {exp} | {r.terminal_reason} | {r.steps_used} | "
                   f"{len(r.citations)} | {'yes' if exp == r.terminal_reason else 'NO'} |")
    return "\n".join(out) + "\n"


def _json(obj) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False)
