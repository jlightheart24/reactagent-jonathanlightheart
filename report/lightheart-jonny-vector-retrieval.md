# U0-HW-09: ReAct Agent With Vector Retrieval

Jonny Lightheart · CS 4300 · October 2026

Repository: <https://github.com/jlightheart24/reactagent-jonathanlightheart>

## Task

The agent answers food-safety questions using only five supplied documents (1,110 PDF pages): the FDA Food Code 2026, the FDA *Employee Health and Personal Hygiene Handbook*, the *Bad Bug Book* (Second Edition), FDA Appendix 3 on bacterial pathogen growth and inactivation, and the Utah Risk Factor Study 2016–2017. Each run ends with one of four outcomes: an answer whose claims cite retrieved passages, an insufficient-evidence result when the corpus does not support an answer, an error, or the step limit.

## Tools

- `search_documents(query, sources?, contains?, k?)` runs semantic search over the Chroma index. Pydantic validates the arguments: the query must be 3–500 characters after trimming; `sources` must be known source IDs without repeats; `contains` is an exact-text filter (useful for section numbers such as `2-201.12`); `k` must be 1–10; and unknown fields are rejected. Each observation returns the passage text with its `chunk_id`, source title, source role, one-based PDF page, and cosine distance.
- `final_answer(claims)` ends the run with an answer. Each claim is one statement with its own citation list.
- `insufficient_evidence(reason, citations?)` ends the run when the evidence does not support an answer. It is accepted only after at least one search.

## Vector-store design

- **Extraction.** `pypdf` extracts text page by page and normalizes whitespace. Ten pages have no extractable text (covers, blank pages, and image-only pages), so they produce no chunks. The build checks every file's SHA-256 digest and page count against the materials README before indexing.
- **Chunking.** Chunks are 1,200 characters with a 200-character overlap. Each chunk stays within one physical page and ends at the latest paragraph, line, sentence, or word break, giving 3,030 chunks in all.
- **Embedding.** Qwen3-Embedding-8B, called through the course's OpenAI-compatible endpoint. The text embedded for each document chunk is prefixed with its title and page. Queries use the Qwen3 instruction format.
- **Storage.** A persistent ChromaDB collection using cosine HNSW. The collection metadata records the embedding model, chunk settings, build time, and chunk count. A build or query whose settings differ from those recorded is refused until you run `python -m docagent ingest --rebuild`. An interrupted build can be resumed.

## Source attribution

Each chunk's metadata holds its source ID, file name, title, one-based physical PDF page, chunk index, character offsets, and a stable identifier `<source_id>:p<page>:c<index>`. The controller keeps the set of chunk IDs that retrieval returned during the current run. A final answer or insufficient-evidence result is rejected (and sent back to the model as an observation) if any citation is outside that set. Accepted citations are resolved to title and page from the stored metadata, not from model output. In addition, a numeric support check flags any number in a claim that does not appear in that claim's cited passages. Matching citations establishes where a chunk came from. Whether each claim actually follows from its passages was reviewed by hand (see Required-query outcomes).

## Controller

The ReAct loop is bounded. One step is one model call plus at most one validated action, and the default limit is 10 steps (configurable). Each of the following uses a step and is returned to the model as an observation: a turn with no tool call, an unknown tool, malformed JSON, arguments that fail validation, and an invalid citation. When a turn contains several tool calls, only the first is executed and the rest are rejected. Model-service and retrieval-service exceptions end the run with the reason `error`. The trace records, for every step, the model's reasoning and text, the requested tool call, the action status, and the full observation.

## Recorded configuration

| Setting | Value |
|---|---|
| Embedding model | Qwen3-Embedding-8B |
| Chat model | Qwen3.8-Flash-Next, temperature 0. The supplied configuration names Qwen3.8-27B, but the assigned API key returned HTTP 403 for that model. |
| Chunk size / overlap | 1,200 / 200 characters |
| Retrieval result count | 5 per search (default) |
| Model/tool step limit | 10 |
| Index | 3,030 chunks from 1,110 pages (10 pages had no extractable text) |

## Test results

`python -m pytest`: **70 passed**, with no live services. The tests cover the following:

- Ingestion metadata: source, one-based physical page, and chunk-ID stability across rebuilds; persistence after reopening; resuming; refusal when settings change; and hash checks.
- Validation of retrieval-tool arguments (13 invalid cases) and source-labeled results from Chroma and substitute retrievers.
- Rejection of citations that retrieval did not return, including a chunk that exists in the corpus but was not returned in that run.
- Step-limit enforcement and validation of the step-limit value.
- All four terminal reasons.
- Scripted supported and insufficient-evidence runs.
- Protocol rejections and the required-query runner.

## Required-query outcomes

| Query | Expected | Terminal reason | Steps | Claims supported by cited passages |
|---|---|---|---|---|
| Q1 Food Code section lookup | answer | answer | 6 | Yes, except the final claim (see note) |
| Q2 Employee illness procedure | answer | answer | 5 | Yes |
| Q3 Norovirus reference | answer | answer | 6 | Yes |
| Q4 Pathogen growth limits | answer | answer | 2 | Yes (all Table 3-A values verified) |
| Q5 Utah inspection findings | answer | answer | 6 | Yes |
| Q6 Cross-document scenario | answer | answer | 8 | Yes |
| Q7 Source authority and version | answer | answer | 8 | Yes |
| Q8 Utah jurisdiction | insufficient_evidence | insufficient_evidence | 5 | Correctly declined |
| Q9 Off-topic | insufficient_evidence | insufficient_evidence | 2 | Correctly declined |

All nine queries ran once under the recorded configuration, and every one ended with its expected terminal reason. No query was rerun. Full traces are in `results/required-queries/`: one JSON file and one Markdown view per query, plus `REVIEW.md`.

I checked each claim against the stored text of its cited chunk, concentrating on numbers, table values, and section attributions. Examples:

- Q4: all fourteen Table 3-A values for *Salmonella* spp. and *L. monocytogenes* match the flattened table on Appendix 3, PDF page 3, and no row is shifted.
- Q5: the four summer line items (22, 28, 33, 36) and their rate ratios (1.37, 1.15, 1.50, 1.23) match the Utah study, PDF page 4.
- Q6: the answer keeps the Bad Bug Book's infective dose (1 to 10 particles) separate from the Food Code Annex figure (as few as 18 particles). It treats vomiting as a symptom, not as a norovirus diagnosis.
- Q7: the answer states that the 2026 Food Code controls the current model-code rule, and that the handbook, adapted from the 2017 Food Code, remains usable as training guidance where it agrees with that rule.
- Q8: the agent found no Utah adoption rule or penalties. It noted that the Food Code's model ordinance leaves fine amounts blank for the adopting jurisdiction.

**One unsupported claim.** Q1's final claim explains the "P" marker on § 2-201.12 as a "provision addressed by a core or noncore item." The cited passage shows the letter "P" but does not define it. The explanation came from the model's own knowledge and is wrong: in the Food Code, "P" marks a Priority item. The other Q1 claims are supported.

**Two smaller issues.** Some years in claims about a source's status (2012, 2020) come from the source-role label the application attaches to each result, not from the passage text. Some section labels (for example ¶ 2-201.11(A)(1)(a)) come from a heading in the preceding chunk.

## Limitations

- Text extraction drops layout. The table structure in Appendix 3 Table 3-A and in the Utah study's rate-ratio table survives only as text, and image-only pages are not indexed (there is no OCR).
- Chunks never cross a page boundary. A table or paragraph that continues onto the next page is split, so the agent may need two searches to recover it.
- Matching citations proves only where a chunk came from. Whether a claim follows from its passage is enforced by the prompt, a numeric check, and manual review, not by an entailment check.
- Semantic search alone handles exact section numbers poorly. The `contains` filter compensates, but it is case-sensitive and matches the exact string.
- Results depend on one chat model at temperature 0. The course service may not be fully deterministic, so repeated runs can differ.
- The check that claims match their citations has gaps. The numeric check catches mismatched numbers but not invented definitions, as Q1's "P" claim shows. Section numbers can also be attributed from a neighboring chunk.
- The chat model was Qwen3.8-Flash-Next rather than the Qwen3.8-27B named in the supplied configuration, because the assigned key returned HTTP 403 for 27B. The service also ignored `tool_choice="required"`, so the controller rejects turns that contain no tool call. The model often sent several tool calls in one turn; only the first is executed.
- Answers are verbose. Several runs add correct but loosely related claims (for example, cleanup procedures in Q2).

## AI-assistance disclosure

- **Claude Code (Anthropic, Claude Opus 5.5)**: Claude used as a tool for creating the agent as well as interpreting the result. 
