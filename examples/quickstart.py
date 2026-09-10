"""Runnable tour of the RAG stack. No keys, no downloads, no network.

    python examples/quickstart.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragkit import (  # noqa: E402
    ChunkConfig,
    PipelineConfig,
    RAGPipeline,
    RetrievalConfig,
    chunk_document,
    compare,
    evaluate_retrieval,
    load_corpus,
    load_gold,
    render_comparison,
)


def rule(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def main() -> None:
    documents = load_corpus(ROOT / "data" / "corpus")
    gold = load_gold(ROOT / "data" / "gold_questions.json")

    rule("1. Chunking keeps the heading and the source offsets")
    for chunk in chunk_document(documents[-1], "section")[:3]:
        print(f"  {chunk.citation}")
        print(f"    section: {chunk.section!r}")
        print(f"    text   : {' '.join(chunk.text.split())[:88]}...")

    pipeline = RAGPipeline()
    pipeline.add_documents(documents)
    stats = pipeline.stats()
    print(f"\n  indexed {stats['documents']} documents into {stats['chunks']} chunks "
          f"(mean {stats['mean_chunk_chars']:.0f} chars)")

    rule("2. Dense and lexical retrieval fail in opposite directions")
    for query in ("What does error E4021 mean?", "how do I get my money back"):
        print(f"\n  query: {query}")
        for mode in ("dense", "lexical", "hybrid"):
            top = pipeline.retrieve(query, top_k=1, mode=mode)
            found = top[0].chunk.doc_id if top else "(nothing)"
            print(f"    {mode:<8} -> {found}")

    rule("3. Fusion reports what each retriever contributed")
    for scored in pipeline.retrieve("error code E4021", top_k=2):
        components = ", ".join(f"{k}={v:.3f}" for k, v in scored.components.items())
        print(f"  [{scored.score:.4f}] {scored.chunk.citation}")
        print(f"           {components}   matched: {', '.join(scored.matched_terms[:5])}")

    rule("4. Answers carry citations that were verified, not requested")
    for question in ("How long are audit records kept?", "What is the default API rate limit?"):
        answer = pipeline.ask(question)
        print(f"\n  Q: {question}")
        print(f"  A: {' '.join(answer.text.split())[:150]}")
        print(f"     grounded={answer.grounded}")
        for citation in answer.citations[:2]:
            print(f"     [support {citation.support:.2f}] {citation.citation}")

    rule("5. The system declines what the corpus cannot answer")
    for question in (
        "What is the company holiday schedule for next year?",
        "How do I configure the Kubernetes ingress controller?",
    ):
        answer = pipeline.ask(question)
        print(f"  Q: {question}")
        print(f"     abstained={answer.abstained}  coverage={answer.metadata.get('coverage')} "
              f"semantic={answer.metadata.get('semantic')}")

    rule("6. Metadata filtering restricts the search")
    filtered = pipeline.retrieve("policy", top_k=3, where={"doc_id": "rate-limits"})
    print(f"  {len(filtered)} results, all from: "
          f"{ {r.chunk.doc_id for r in filtered} }")

    rule("7. Scoring the whole pipeline against a labelled gold set")
    report = evaluate_retrieval(pipeline, gold, k=5, label="default", with_answers=True)
    print(f"  questions           : {report.questions}")
    print(f"  recall@5            : {report.recall:.3f}")
    print(f"  MRR                 : {report.mrr:.3f}")
    print(f"  nDCG@5              : {report.ndcg:.3f}")
    print(f"  answer accuracy     : {report.answer_accuracy:.3f}")
    print(f"  grounded rate       : {report.grounded_rate:.3f}")
    print(f"  abstention accuracy : {report.abstention_accuracy:.3f}")
    print(f"  false abstentions   : {report.false_abstention_rate:.3f}")
    print(f"  latency p50 / p95   : {report.latency_p50:.2f} / {report.latency_p95:.2f} ms")

    rule("8. Which configuration is actually better? Measure it.")

    def variant(**overrides) -> PipelineConfig:
        config = PipelineConfig()
        config.chunking = ChunkConfig(target_tokens=overrides.pop("tokens", 220))
        config.retrieval = RetrievalConfig(top_k=5, mode=overrides.pop("mode", "hybrid"))
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    reports = compare(
        documents,
        gold,
        {
            "hybrid (default)": variant(),
            "dense only": variant(mode="dense"),
            "lexical only": variant(mode="lexical"),
            "hybrid + rerank": variant(reranker="default"),
            "chunker: fixed": variant(chunker="fixed"),
            "chunker: sentence": variant(chunker="sentence"),
            "small chunks (100 tok)": variant(tokens=100),
        },
        k=5,
        with_answers=True,
    )
    print(render_comparison(reports))
    print("\n  Reranking is implemented and measurably hurts on this corpus,")
    print("  so it ships off by default. On a larger corpus, measure again.")


if __name__ == "__main__":
    main()
