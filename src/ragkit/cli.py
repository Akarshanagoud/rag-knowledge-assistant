"""Command line interface.

    ragkit ask "how long do I have to request a refund?"
    ragkit search "rate limit" --mode lexical --top-k 5
    ragkit eval                      # score retrieval and answering
    ragkit compare                   # chunkers, embedders, modes, side by side
    ragkit chunks --strategy section # inspect how a corpus was split
    ragkit stats

Exit codes: 0 fine, 1 an evaluation gate failed, 2 usage error.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from .answering import ModelGenerator
from .chunking import CHUNKERS, ChunkConfig
from .evaluation import compare, evaluate_retrieval, load_gold, render_comparison
from .pipeline import PipelineConfig, RAGPipeline, load_corpus
from .retrieval import RetrievalConfig

EXIT_OK, EXIT_GATE, EXIT_USAGE = 0, 1, 2

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data" / "corpus"
DEFAULT_GOLD = ROOT / "data" / "gold_questions.json"


def _config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig()
    config.chunker = getattr(args, "chunker", config.chunker)
    config.embedder = getattr(args, "embedder", config.embedder)
    config.store = getattr(args, "store", config.store)
    config.reranker = getattr(args, "reranker", config.reranker)
    config.chunking = ChunkConfig(
        target_tokens=getattr(args, "chunk_tokens", 220),
        overlap_tokens=getattr(args, "chunk_overlap", 40),
    )
    config.retrieval = RetrievalConfig(
        top_k=getattr(args, "top_k", 5),
        mode=getattr(args, "mode", "hybrid"),
    )
    if config.store == "faiss":
        config.store_options = {"index_type": getattr(args, "faiss_index", "flat")}
    return config


def _pipeline(args: argparse.Namespace) -> RAGPipeline:
    config = _config(args)
    generator = None
    if getattr(args, "generator", "extractive") == "model":
        generator = ModelGenerator(ollama_model=getattr(args, "model", "llama3.2"))
    pipeline = RAGPipeline(config, generator=generator)
    documents = load_corpus(getattr(args, "corpus", None) or DEFAULT_CORPUS)
    if not documents:
        raise FileNotFoundError(f"no documents found in {args.corpus or DEFAULT_CORPUS}")
    pipeline.add_documents(documents)
    return pipeline


def cmd_ask(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    answer = pipeline.ask(args.question, top_k=args.top_k, mode=args.mode)

    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
        return EXIT_OK

    print(answer.text)
    print()
    if answer.abstained:
        print(f"(abstained: {answer.metadata.get('reason')}; "
              f"coverage {answer.metadata.get('coverage')}, "
              f"semantic {answer.metadata.get('semantic')})")
        return EXIT_OK

    print(f"grounded: {answer.grounded}")
    print("sources:")
    for citation in answer.citations:
        print(f"  [{citation.support:.2f}] {citation.citation}")
        print(f"        {citation.text[:100]}")
    return EXIT_OK


def cmd_search(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    where = json.loads(args.where) if args.where else None
    results = pipeline.retrieve(args.query, top_k=args.top_k, mode=args.mode, where=where)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return EXIT_OK

    if not results:
        print("no results")
        return EXIT_OK
    for rank, scored in enumerate(results, start=1):
        print(f"{rank}. [{scored.score:.4f}] {scored.chunk.citation}")
        components = ", ".join(f"{k}={v:.3f}" for k, v in scored.components.items())
        if components:
            print(f"   {components}")
        if scored.matched_terms:
            print(f"   matched: {', '.join(scored.matched_terms[:8])}")
        snippet = " ".join(scored.chunk.text.split())[:160]
        print(f"   {snippet}...")
        print()
    return EXIT_OK


def cmd_eval(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    gold = load_gold(args.gold or DEFAULT_GOLD)
    report = evaluate_retrieval(
        pipeline, gold, k=args.top_k, mode=args.mode,
        label=args.mode, with_answers=not args.retrieval_only,
    )

    if args.json:
        print(json.dumps(report.to_dict(include_questions=args.verbose), indent=2))
    else:
        print(f"questions           : {report.questions}")
        print(f"recall@{args.top_k}            : {report.recall:.3f}")
        print(f"precision@{args.top_k}         : {report.precision:.3f}")
        print(f"MRR                 : {report.mrr:.3f}")
        print(f"nDCG@{args.top_k}              : {report.ndcg:.3f}")
        print(f"hit rate            : {report.hit_rate:.3f}")
        if report.answer_accuracy is not None:
            print(f"answer accuracy     : {report.answer_accuracy:.3f}")
        if report.grounded_rate is not None:
            print(f"grounded rate       : {report.grounded_rate:.3f}")
        if report.abstention_accuracy is not None:
            print(f"abstention accuracy : {report.abstention_accuracy:.3f}")
        if report.false_abstention_rate is not None:
            print(f"false abstentions   : {report.false_abstention_rate:.3f}")
        print(f"latency p50 / p95   : {report.latency_p50:.2f} / {report.latency_p95:.2f} ms")

        failures = report.failures()
        if failures:
            print(f"\nretrieval misses ({len(failures)}):")
            for failure in failures:
                print(f"  - {failure.question}")
                print(f"      returned: {failure.retrieved_docs[:3]}")

    failed = False
    if args.min_recall is not None and report.recall < args.min_recall:
        print(f"\nFAIL: recall {report.recall:.3f} below gate {args.min_recall}", file=sys.stderr)
        failed = True
    if (
        args.min_abstention is not None
        and report.abstention_accuracy is not None
        and report.abstention_accuracy < args.min_abstention
    ):
        print(
            f"\nFAIL: abstention accuracy {report.abstention_accuracy:.3f} "
            f"below gate {args.min_abstention}",
            file=sys.stderr,
        )
        failed = True
    if (
        args.min_grounded is not None
        and report.grounded_rate is not None
        and report.grounded_rate < args.min_grounded
    ):
        print(
            f"\nFAIL: grounded rate {report.grounded_rate:.3f} below gate {args.min_grounded}",
            file=sys.stderr,
        )
        failed = True
    return EXIT_GATE if failed else EXIT_OK


def _variants(args: argparse.Namespace) -> dict[str, PipelineConfig]:
    def build(**overrides) -> PipelineConfig:
        config = PipelineConfig()
        config.chunking = ChunkConfig(target_tokens=overrides.pop("tokens", 220))
        config.retrieval = RetrievalConfig(
            top_k=args.top_k, mode=overrides.pop("mode", "hybrid")
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    variants: dict[str, PipelineConfig] = {
        "hybrid (default)": build(),
        "dense only": build(mode="dense"),
        "lexical only": build(mode="lexical"),
        "hybrid + rerank": build(reranker="default"),
        "hybrid + lexical rerank": build(reranker="lexical"),
    }
    if args.chunkers:
        for name in CHUNKERS:
            variants[f"chunker: {name}"] = build(chunker=name)
    if args.sizes:
        for tokens in (100, 220, 400):
            variants[f"chunk size: {tokens} tok"] = build(tokens=tokens)
    if args.embedders:
        variants["embedder: tfidf"] = build(embedder="tfidf")
    if args.stores:
        for store in ("memory", "faiss", "chroma"):
            try:
                __import__({"faiss": "faiss", "chroma": "chromadb", "memory": "sys"}[store])
            except ImportError:
                print(f"(skipping {store}: not installed)", file=sys.stderr)
                continue
            variants[f"store: {store}"] = build(store=store)
    return variants


def cmd_compare(args: argparse.Namespace) -> int:
    documents = load_corpus(args.corpus or DEFAULT_CORPUS)
    if not documents:
        print("no documents found", file=sys.stderr)
        return EXIT_USAGE
    gold = load_gold(args.gold or DEFAULT_GOLD)

    reports = compare(
        [copy.deepcopy(d) for d in documents],
        gold,
        _variants(args),
        k=args.top_k,
        with_answers=not args.retrieval_only,
    )
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        print(render_comparison(reports))
        print(f"\n{len(gold)} questions, k={args.top_k}, {len(documents)} documents")
    return EXIT_OK


def cmd_chunks(args: argparse.Namespace) -> int:
    documents = load_corpus(args.corpus or DEFAULT_CORPUS)
    if not documents:
        print("no documents found", file=sys.stderr)
        return EXIT_USAGE

    from .chunking import chunk_document

    config = ChunkConfig(
        target_tokens=args.chunk_tokens, overlap_tokens=args.chunk_overlap
    )
    total = 0
    for document in documents:
        chunks = chunk_document(document, args.chunker, config)
        total += len(chunks)
        if args.summary:
            lengths = [len(c.text) for c in chunks]
            print(
                f"{document.doc_id:<20} {len(chunks):>3} chunks  "
                f"mean {sum(lengths) // len(lengths):>4} chars  "
                f"min {min(lengths):>4}  max {max(lengths):>4}"
            )
            continue
        print(f"=== {document.doc_id} ({len(chunks)} chunks) ===")
        for chunk in chunks:
            print(f"  [{chunk.index}] {chunk.citation}  ({len(chunk.text)} chars)")
            print(f"      {' '.join(chunk.text.split())[:140]}...")
        print()
    print(f"{total} chunks from {len(documents)} documents using {args.chunker!r}")
    return EXIT_OK


def cmd_stats(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    print(json.dumps(pipeline.stats(), indent=2))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragkit",
        description="A production-shaped RAG stack with a measurement harness.",
    )
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")

    def add_index_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--corpus", help="directory of .md documents")
        p.add_argument("--chunker", default="section", choices=sorted(CHUNKERS))
        p.add_argument("--chunk-tokens", type=int, default=220)
        p.add_argument("--chunk-overlap", type=int, default=40)
        p.add_argument("--embedder", default="hashing",
                       choices=["hashing", "tfidf", "sentence-transformers"])
        p.add_argument("--store", default="memory", choices=["memory", "faiss", "chroma"])
        p.add_argument("--faiss-index", default="flat", choices=["flat", "ivf", "hnsw"])
        p.add_argument("--reranker", default="none",
                       choices=["none", "lexical", "mmr", "cross-encoder", "default"])
        p.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "lexical"])
        p.add_argument("--top-k", type=int, default=5)
        p.add_argument("--json", action="store_true")

    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    add_index_args(ask)
    ask.add_argument("--generator", default="extractive",
                     choices=["extractive", "template", "model"])
    ask.add_argument("--model", default="llama3.2", help="Ollama model for --generator model")
    ask.set_defaults(func=cmd_ask)

    search = sub.add_parser("search", help="retrieve chunks without answering")
    search.add_argument("query")
    add_index_args(search)
    search.add_argument("--where", help="metadata filter as JSON")
    search.set_defaults(func=cmd_search)

    evaluate = sub.add_parser("eval", help="score retrieval and answering")
    add_index_args(evaluate)
    evaluate.add_argument("--gold", help="gold question set")
    evaluate.add_argument("--retrieval-only", action="store_true")
    evaluate.add_argument("--verbose", action="store_true", help="include per-question rows")
    evaluate.add_argument("--min-recall", type=float)
    evaluate.add_argument("--min-abstention", type=float)
    evaluate.add_argument("--min-grounded", type=float)
    evaluate.set_defaults(func=cmd_eval)

    comparison = sub.add_parser("compare", help="score several configurations side by side")
    comparison.add_argument("--corpus")
    comparison.add_argument("--gold")
    comparison.add_argument("--top-k", type=int, default=5)
    comparison.add_argument("--chunkers", action="store_true", help="include every chunker")
    comparison.add_argument("--sizes", action="store_true", help="include chunk sizes")
    comparison.add_argument("--embedders", action="store_true")
    comparison.add_argument("--stores", action="store_true", help="include faiss and chroma")
    comparison.add_argument("--retrieval-only", action="store_true")
    comparison.add_argument("--json", action="store_true")
    comparison.set_defaults(func=cmd_compare)

    chunks = sub.add_parser("chunks", help="inspect how documents are split")
    chunks.add_argument("--corpus")
    chunks.add_argument("--chunker", default="section", choices=sorted(CHUNKERS))
    chunks.add_argument("--chunk-tokens", type=int, default=220)
    chunks.add_argument("--chunk-overlap", type=int, default=40)
    chunks.add_argument("--summary", action="store_true")
    chunks.set_defaults(func=cmd_chunks)

    stats = sub.add_parser("stats", help="describe the built index")
    add_index_args(stats)
    stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(f"ragkit {__version__}")
        return EXIT_OK

    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE

    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
