"""Command-line interface: `python -m docagent <command>` (or the `docagent` script).

  ingest [--rebuild]          build or resume the persistent Chroma index from the five PDFs
  info                        show the stored index settings and chunk counts
  search QUERY [...]          run the retrieval tool directly (no LLM)
  ask QUESTION [--trace F]    run the agent on one question
  run-queries [--out DIR]     run the nine required queries under one configuration
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PROJECT_ROOT, ConfigError, Settings, load_settings
from .sources import SOURCE_IDS


def _settings(args) -> Settings:
    return load_settings(max_steps=getattr(args, "max_steps", None), top_k=getattr(args, "k", None))


def _embedder(settings: Settings):
    from .embeddings import OpenAICompatibleEmbedder

    return OpenAICompatibleEmbedder(settings)


def _agent(settings: Settings):
    from .agent import Agent
    from .llm import OpenAICompatibleChatModel
    from .retrieval import ChromaRetriever

    retriever = ChromaRetriever(settings, _embedder(settings))
    index = dict(retriever.collection.metadata or {})
    config = {**settings.public(), "index_chunk_count": index.get("chunk_count"),
              "index_built_at": index.get("built_at")}
    return Agent(OpenAICompatibleChatModel(settings), retriever, settings.max_steps, settings.top_k, config)


def cmd_ingest(args) -> int:
    from .ingest import build_index

    settings = _settings(args)
    print(f"Ingesting {settings.docs_dir} into {settings.chroma_dir} (collection {settings.collection}); "
          f"chunk {settings.chunk_size}/{settings.chunk_overlap} chars, model {settings.embedding_model}")
    report = build_index(settings, _embedder(settings), rebuild=args.rebuild, progress=print)
    manifest = report.as_dict()
    out = PROJECT_ROOT / "results" / "index-manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Done: {manifest['pages_total']} pages, {manifest['chunks_total']} chunks "
          f"({manifest['chunks_added']} added, {manifest['chunks_already_present']} already present). "
          f"Manifest: {out.relative_to(PROJECT_ROOT)}")
    return 0


def cmd_info(args) -> int:
    from .ingest import open_client

    settings = _settings(args)
    client = open_client(settings.chroma_dir)
    names = [c.name for c in client.list_collections()]
    if settings.collection not in names:
        print(f"No collection {settings.collection!r} in {settings.chroma_dir}. Run `docagent ingest --rebuild`.")
        return 1
    col = client.get_collection(settings.collection, embedding_function=None)
    print(json.dumps({"collection": settings.collection, "count": col.count(), **(col.metadata or {})}, indent=2))
    return 0


def cmd_search(args) -> int:
    from .retrieval import ChromaRetriever
    from .tools import SearchArgs, passage_record

    settings = _settings(args)
    parsed = SearchArgs(query=args.query, sources=args.source or None, contains=args.contains, k=args.k)
    retriever = ChromaRetriever(settings, _embedder(settings))
    passages = retriever.search(parsed.query, parsed.k or settings.top_k, parsed.sources, parsed.contains)
    for p in passages:
        rec = passage_record(p)
        print(f"[{rec['chunk_id']}] {rec['source_title']}, PDF page {rec['page']} (distance {rec['distance']})")
        print("  " + " ".join(rec["text"].split())[:600] + "\n")
    return 0


def cmd_ask(args) -> int:
    from .queries import render_markdown

    settings = _settings(args)
    result = _agent(settings).run(args.question)
    print(render_markdown(result) if args.verbose else result.rendered_answer())
    print(f"\nterminal_reason={result.terminal_reason} steps={result.steps_used}/{result.max_steps}")
    if args.trace:
        Path(args.trace).write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return 0 if result.terminal_reason in ("answer", "insufficient_evidence") else 1


def cmd_run_queries(args) -> int:
    from .queries import EXPECTED, parse_queries, render_markdown, render_summary

    settings = _settings(args)
    queries = parse_queries(Path(args.queries))
    if args.only:
        queries = [q for q in queries if q.query_id in set(args.only)]
    agent = _agent(settings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(agent.config | {"max_steps": agent.max_steps,
                                                                "retrieval_top_k": agent.top_k},
                                                indent=2) + "\n")
    rows = []
    for q in queries:
        print(f"{q.query_id}: running …", flush=True)
        result = agent.run(q.text)
        data = {"query_id": q.query_id, "title": q.title, "expected": EXPECTED.get(q.query_id), **result.to_dict()}
        (out / f"{q.query_id}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        (out / f"{q.query_id}.md").write_text(render_markdown(result, q))
        rows.append((q, result))
        print(f"{q.query_id}: {result.terminal_reason} in {result.steps_used} steps "
              f"(expected {EXPECTED.get(q.query_id)})", flush=True)
    if not args.only:
        (out / "README.md").write_text(render_summary(rows, json.loads((out / "config.json").read_text())))
    print(f"Results written to {out}")
    return 0 if all(EXPECTED.get(q.query_id) == r.terminal_reason for q, r in rows) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docagent", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="build the vector index")
    p.add_argument("--rebuild", action="store_true", help="drop the collection and re-index everything")
    p.set_defaults(func=cmd_ingest)

    sub.add_parser("info", help="show index metadata").set_defaults(func=cmd_info)

    p = sub.add_parser("search", help="query the index without the LLM")
    p.add_argument("query")
    p.add_argument("--source", action="append", choices=SOURCE_IDS)
    p.add_argument("--contains")
    p.add_argument("-k", type=int)
    p.set_defaults(func=cmd_search)

    for name, func, help_ in (("ask", cmd_ask, "answer one question"),
                              ("run-queries", cmd_run_queries, "run the required queries")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--max-steps", type=int, help="model/tool step limit (default DOCAGENT_MAX_STEPS or 10)")
        p.add_argument("-k", type=int, help="retrieval result count (default DOCAGENT_TOP_K or 5)")
        p.set_defaults(func=func)
        if name == "ask":
            p.add_argument("question")
            p.add_argument("--trace", help="write the full result and trace as JSON")
            p.add_argument("-v", "--verbose", action="store_true", help="print the whole trace")
        else:
            p.add_argument("--queries", default=str(PROJECT_ROOT / "U0-HW-09-materials" /
                                                    "U0-HW-09-required-queries.md"))
            p.add_argument("--out", default=str(PROJECT_ROOT / "results" / "required-queries"))
            p.add_argument("--only", nargs="+", metavar="QID", help="run a subset, e.g. --only Q1 Q8")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # surface service/index problems without a traceback wall
        from .ingest import IngestError
        from .retrieval import IndexNotBuilt

        if isinstance(e, (IngestError, IndexNotBuilt)):
            print(f"error: {e}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
