# U0-HW-09: ReAct Agent With Vector Retrieval

A bounded ReAct agent that answers food-safety questions using only five supplied source documents
(1,110 PDF pages). The documents are chunked, embedded with **Qwen3-Embedding-8B**, and stored in a
persistent **ChromaDB** collection. The agent's only way to read the corpus is a validated, typed
retrieval tool. It ends with a cited answer, an insufficient-evidence result, an error, or the step limit,
and every run records a complete action-observation trace.

## Requirements

- **Python 3.12** or newer.
- The course model services on `golem`, reached over the Utah Tech Tailnet (Tailscale, login server
  `https://tailscale.utahtech.dev`). Use `http://golem:4000/v1` for both services; the full
  `golem.cs.utahtech.edu` hostname does not resolve over the Tailnet.
- The supplied PDFs in `U0-HW-09-materials/documents/`. They are git-ignored and never committed. Extract the
  materials package into the project root so the PDFs land at that path, or set `DOCAGENT_DOCS_DIR`.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # dependency installation
cp .env.example .env             # then replace both placeholder API keys with your credential
```

`.env.example` holds the same service settings as `U0-HW-09-materials/model-service.env.example`.
`.env` is git-ignored. The application never prints, logs, or records keys or service URLs.

## Commands

| Task | Command |
|---|---|
| Rebuild the vector store from scratch | `python -m docagent ingest --rebuild` |
| Resume an interrupted build | `python -m docagent ingest` |
| Show stored index settings | `python -m docagent info` |
| Retrieval only (no LLM) | `python -m docagent search "norovirus infective dose" --source bad-bug-book` |
| Ask one question | `python -m docagent ask "..." -v --trace run.json` |
| Run the nine required queries | `python -m docagent run-queries` |
| Run the automated tests (offline) | `python -m pytest` |
| Build the PDF report from `report/report.html` | `scripts/build_report.sh` |

A clean build reads all 1,110 pages and embeds 3,030 chunks; it took about 15 minutes on the course service. The build checks each
PDF's SHA-256 digest and page count against the materials README before indexing, and it writes a summary
to `results/index-manifest.json`. `run-queries` writes to `results/required-queries/`, and a manual claim-support review is in
`results/required-queries/REVIEW.md`:

- `config.json`: the single recorded configuration.
- `README.md`: an outcome summary.
- `Qn.json`: the full result and trace for each query, including complete passage text.
- `Qn.md`: a readable view of each trace.

The step limit and retrieval count can be overridden for a run with `--max-steps N` and `-k N`, or through
`DOCAGENT_MAX_STEPS` and `DOCAGENT_TOP_K`.

## Recorded configuration (defaults)

| Setting | Value |
|---|---|
| Embedding model | Qwen3-Embedding-8B (queries use the Qwen3 instruction prefix; documents do not) |
| Chat model | Qwen3.8-Flash-Next, temperature 0 (the materials name Qwen3.8-27B, but the assigned key is not authorized for it) |
| Chunk size / overlap | 1,200 / 200 characters, within a single PDF page |
| Retrieval result count (k) | 5 per search (the model may request 1–10) |
| Model/tool step limit | 10 steps; one step is one model call plus at most one action |
| Distance | cosine (HNSW) |

## Design

```
question ─► model ─► controller ─┬─ search_documents ─► Chroma (Qwen3 query embedding) ─► observation ─┐
              ▲                  ├─ rejected action  ─► error observation ─────────────────────────────┤
              └──────────────────┴──────────────────────────────────────────────────────────────────────┘
                                 ├─ final_answer (citations ⊆ retrieved)  ─► terminal: answer
                                 ├─ insufficient_evidence (after a search) ─► terminal: insufficient_evidence
                                 ├─ model or retrieval exception           ─► terminal: error
                                 └─ max_steps model calls used             ─► terminal: step_limit
```

- **Ingestion** (`src/docagent/ingest.py`): `pypdf` extracts text page by page. Each page is split
  into overlapping chunks that break at paragraphs, lines, sentences, or words, and no chunk crosses
  a page. Each chunk stores:
  - `source_id`, `source_file`, and `title`;
  - `page`, the one-based physical PDF page, not the printed page label;
  - `chunk_index`, plus character offsets;
  - the file digest;
  - a stable `chunk_id` of the form `<source_id>:p<page>:c<index>` (for example `food-code-2026:p59:c1`).

  The text sent for embedding is prefixed with the title and page. The stored document is the raw chunk.
  The collection metadata records the embedding model and chunk settings. A later build or query with
  different settings is refused until you run `--rebuild`.
- **Retrieval tool** (`src/docagent/tools.py`, `retrieval.py`): `search_documents(query, sources?, contains?, k?)`
  is validated by Pydantic. Unknown fields, unknown source ids, out-of-range `k`, and blank queries are all
  rejected. `contains` is an exact-text filter, useful for Food Code section numbers. Each observation
  returns the passage text with `chunk_id`, source title and role, page, and distance.
- **Controller** (`src/docagent/agent.py`): the application owns several things the model does not:
  - **The step budget.** The default limit is 10 steps, configurable.
  - **Retrieved chunks.** It keeps the set of chunk IDs that retrieval returned during the run.
  - **The trace.** Each entry records the model's thought and text, the action, and the observation.
  - **The terminal reason.**

  Each turn must contain exactly one tool call. The following are returned to the model as observations,
  and each still costs a step:
  - a turn with no tool call (when a turn has several calls, only the first runs and the rest are rejected);
  - an unknown tool;
  - malformed JSON or arguments that fail validation;
  - a citation of any `chunk_id` that retrieval did not return in this run;
  - `insufficient_evidence` before any search.

  On an accepted answer, a support check also flags any number in a claim that does not appear in that
  claim's cited passages. These flags are recorded as warnings.
- **Source roles** (`src/docagent/sources.py`): every observation carries the source's role. The system
  prompt states each role:
  - Food Code 2026: model code;
  - handbook: guidance based on the 2017 Food Code;
  - Bad Bug Book: pathogen reference;
  - Appendix 3: draft, nonbinding guidance;
  - Utah study: historical observations.

## Tests

`python -m pytest` runs 70 tests without any model or embedding service. They use a hashing embedder,
substitute retrieval results, a scripted chat model, and generated PDFs. They cover:

- ingestion metadata: source, one-based page, and stable chunk IDs; persistence; resume; rebuild; and
  refusal when settings change;
- tool-argument validation and source-labeled retrieval results;
- rejection of citations that retrieval did not return;
- step-limit enforcement;
- all four terminal reasons;
- scripted supported and insufficient-evidence runs;
- the required-query runner.

If the supplied PDFs are present, one more test verifies their digests and page counts.

## Repository hygiene

`.gitignore` excludes `.env`, the supplied PDFs, the generated `chroma/` store, virtual environments, and
caches.
