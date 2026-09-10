"""Retrieval and answer evaluation.

This exists because every RAG decision — chunk size, embedder, hybrid weights,
which vector store — is an empirical question about *your* corpus, and the
industry answers them with vibes and other people's benchmarks.

Metrics implemented:

* **recall@k** - is the right chunk in the top k at all? The ceiling on
  everything downstream: a reranker cannot recover a chunk retrieval missed.
* **precision@k** - what fraction of what we returned was relevant. Matters
  because context windows are finite and irrelevant chunks displace good ones.
* **MRR** - how far down the list the first correct answer sat. The metric that
  tracks "did the user have to scroll".
* **nDCG@k** - rank-weighted, graded relevance.
* **hit rate** - the blunt one. Did we get *anything* right.
* **abstention accuracy** - for questions with no answer in the corpus, did the
  system correctly decline? Usually unmeasured, and usually the difference
  between a demo and something you can deploy.

A gold record is `{question, relevant_doc_ids, [answer_contains]}`. Labelling
at document level rather than chunk level is deliberate: chunk ids change every
time you change the chunker, which would make the gold set unusable for exactly
the comparison it exists to support.
"""
from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .pipeline import PipelineConfig, RAGPipeline
from .types import Document, ScoredChunk


@dataclass
class GoldQuestion:
    question: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    #: Strings the answer should contain, for end-to-end answer checking.
    answer_contains: list[str] = field(default_factory=list)
    #: True when the corpus genuinely cannot answer this and the system
    #: should abstain.
    unanswerable: bool = False
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldQuestion:
        return cls(
            question=data["question"],
            relevant_doc_ids=data.get("relevant_doc_ids", []),
            answer_contains=data.get("answer_contains", []),
            unanswerable=bool(data.get("unanswerable", False)),
            tags=data.get("tags", []),
        )


def load_gold(path: str | Path) -> list[GoldQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldQuestion.from_dict(item) for item in data]


# --- metrics -----------------------------------------------------------------
def recall_at_k(retrieved_docs: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    found = {d for d in retrieved_docs[:k]} & relevant
    return len(found) / len(relevant)


def precision_at_k(retrieved_docs: list[str], relevant: set[str], k: int) -> float:
    window = retrieved_docs[:k]
    if not window:
        return 0.0
    return sum(1 for d in window if d in relevant) / len(window)


def reciprocal_rank(retrieved_docs: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_docs: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG. Ideal DCG assumes all relevant docs ranked first."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_docs[:k], start=1)
        if doc_id in relevant
    )
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


@dataclass
class QuestionResult:
    question: str
    retrieved_docs: list[str]
    recall: float
    precision: float
    rr: float
    ndcg: float
    hit: bool
    latency_ms: float
    answered: bool = False
    answer_correct: bool | None = None
    grounded: bool = True
    abstained: bool = False
    abstention_correct: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "retrieved_docs": self.retrieved_docs,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "reciprocal_rank": round(self.rr, 4),
            "ndcg": round(self.ndcg, 4),
            "hit": self.hit,
            "latency_ms": round(self.latency_ms, 2),
            "answer_correct": self.answer_correct,
            "grounded": self.grounded,
            "abstained": self.abstained,
            "abstention_correct": self.abstention_correct,
        }


@dataclass
class EvalReport:
    label: str
    k: int
    questions: int
    recall: float
    precision: float
    mrr: float
    ndcg: float
    hit_rate: float
    latency_p50: float
    latency_p95: float
    answer_accuracy: float | None = None
    grounded_rate: float | None = None
    abstention_accuracy: float | None = None
    #: Answerable questions the system declined. The cost of a conservative
    #: abstention gate, reported alongside its benefit so the trade is visible
    #: rather than buried in a single headline number.
    false_abstention_rate: float | None = None
    per_question: list[QuestionResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_questions: bool = False) -> dict[str, Any]:
        payload = {
            "label": self.label,
            "k": self.k,
            "questions": self.questions,
            "recall@k": round(self.recall, 4),
            "precision@k": round(self.precision, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@k": round(self.ndcg, 4),
            "hit_rate": round(self.hit_rate, 4),
            "latency_ms_p50": round(self.latency_p50, 2),
            "latency_ms_p95": round(self.latency_p95, 2),
            "answer_accuracy": round(self.answer_accuracy, 4)
            if self.answer_accuracy is not None
            else None,
            "grounded_rate": round(self.grounded_rate, 4)
            if self.grounded_rate is not None
            else None,
            "abstention_accuracy": round(self.abstention_accuracy, 4)
            if self.abstention_accuracy is not None
            else None,
            "false_abstention_rate": round(self.false_abstention_rate, 4)
            if self.false_abstention_rate is not None
            else None,
            "config": self.config,
        }
        if include_questions:
            payload["per_question"] = [q.to_dict() for q in self.per_question]
        return payload

    def failures(self) -> list[QuestionResult]:
        """Questions where retrieval missed entirely. The list worth reading."""
        return [q for q in self.per_question if not q.hit and not q.abstained]


def evaluate_retrieval(
    pipeline: RAGPipeline,
    gold: list[GoldQuestion],
    k: int = 5,
    mode: str | None = None,
    rerank: bool = True,
    label: str = "run",
    with_answers: bool = False,
) -> EvalReport:
    """Score a pipeline against a gold set."""
    results: list[QuestionResult] = []

    for item in gold:
        started = time.perf_counter()
        retrieved: list[ScoredChunk] = pipeline.retrieve(
            item.question, top_k=k, mode=mode, rerank=rerank
        )
        latency = (time.perf_counter() - started) * 1000

        # Deduplicate to document order: two chunks from the same document are
        # one retrieved source, and counting them twice inflates precision.
        seen: dict[str, None] = {}
        for scored in retrieved:
            seen.setdefault(scored.chunk.doc_id, None)
        retrieved_docs = list(seen)
        relevant = set(item.relevant_doc_ids)

        result = QuestionResult(
            question=item.question,
            retrieved_docs=retrieved_docs,
            recall=recall_at_k(retrieved_docs, relevant, k),
            precision=precision_at_k(retrieved_docs, relevant, k),
            rr=reciprocal_rank(retrieved_docs, relevant),
            ndcg=ndcg_at_k(retrieved_docs, relevant, k),
            hit=bool(set(retrieved_docs[:k]) & relevant),
            latency_ms=latency,
        )

        if with_answers:
            answer = pipeline.ask(item.question, top_k=k, mode=mode)
            result.answered = True
            result.grounded = answer.grounded
            result.abstained = answer.abstained
            if item.unanswerable:
                result.abstention_correct = answer.abstained
            elif item.answer_contains:
                lowered = answer.text.lower()
                result.answer_correct = all(
                    needle.lower() in lowered for needle in item.answer_contains
                )

        results.append(result)

    latencies = sorted(r.latency_ms for r in results) or [0.0]
    answered = [r for r in results if r.answer_correct is not None]
    abstentions = [r for r in results if r.abstention_correct is not None]
    # Unanswerable questions have no relevant documents, so including them in
    # retrieval averages would drag every score toward zero for the right
    # behaviour. They are scored by abstention accuracy instead.
    scored = [r for r, item in zip(results, gold, strict=True) if not item.unanswerable]

    return EvalReport(
        label=label,
        k=k,
        questions=len(results),
        recall=statistics.fmean(r.recall for r in scored) if scored else 0.0,
        precision=statistics.fmean(r.precision for r in scored) if scored else 0.0,
        mrr=statistics.fmean(r.rr for r in scored) if scored else 0.0,
        ndcg=statistics.fmean(r.ndcg for r in scored) if scored else 0.0,
        hit_rate=statistics.fmean(1.0 if r.hit else 0.0 for r in scored) if scored else 0.0,
        latency_p50=latencies[len(latencies) // 2],
        latency_p95=latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)],
        answer_accuracy=statistics.fmean(1.0 if r.answer_correct else 0.0 for r in answered)
        if answered
        else None,
        grounded_rate=statistics.fmean(
            1.0 if r.grounded else 0.0 for r in results if r.answered
        )
        if with_answers
        else None,
        abstention_accuracy=statistics.fmean(
            1.0 if r.abstention_correct else 0.0 for r in abstentions
        )
        if abstentions
        else None,
        false_abstention_rate=statistics.fmean(
            1.0 if r.abstained else 0.0 for r in scored
        )
        if (with_answers and scored)
        else None,
        per_question=results,
        config=pipeline.config.to_dict(),
    )


def compare(
    documents: list[Document],
    gold: list[GoldQuestion],
    variants: dict[str, PipelineConfig],
    k: int = 5,
    with_answers: bool = False,
) -> list[EvalReport]:
    """Build a pipeline per variant and score them all on the same gold set.

    This is the function that answers "which chunker / embedder / store?" with
    a number instead of an opinion.
    """
    reports: list[EvalReport] = []
    for label, config in variants.items():
        pipeline = RAGPipeline(config)
        pipeline.add_documents([_clone(d) for d in documents])
        reports.append(
            evaluate_retrieval(
                pipeline, gold, k=k, label=label, with_answers=with_answers
            )
        )
    reports.sort(key=lambda r: -r.recall)
    return reports


def _clone(document: Document) -> Document:
    """Fresh Document per variant, so chunk objects are never shared."""
    return Document(
        doc_id=document.doc_id,
        text=document.text,
        title=document.title,
        source=document.source,
        metadata=dict(document.metadata),
    )


def render_comparison(reports: list[EvalReport]) -> str:
    """Fixed-width comparison table."""
    header = (
        f"{'VARIANT':<26}{'RECALL':>8}{'PREC':>8}{'MRR':>8}{'NDCG':>8}"
        f"{'HIT':>8}{'P50 ms':>9}"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        lines.append(
            f"{report.label:<26}{report.recall:>8.3f}{report.precision:>8.3f}"
            f"{report.mrr:>8.3f}{report.ndcg:>8.3f}{report.hit_rate:>8.3f}"
            f"{report.latency_p50:>9.2f}"
        )
    if reports and reports[0].answer_accuracy is not None:
        lines += [
            "",
            f"{'VARIANT':<26}{'ANSWER':>8}{'GROUNDED':>10}{'ABSTAIN':>9}{'FALSE-AB':>10}",
        ]
        for report in reports:
            answer = report.answer_accuracy
            grounded = report.grounded_rate
            abstain = report.abstention_accuracy
            false_abstain = report.false_abstention_rate
            lines.append(
                f"{report.label:<26}"
                f"{(f'{answer:.3f}' if answer is not None else '-'):>8}"
                f"{(f'{grounded:.3f}' if grounded is not None else '-'):>10}"
                f"{(f'{abstain:.3f}' if abstain is not None else '-'):>9}"
                f"{(f'{false_abstain:.3f}' if false_abstain is not None else '-'):>10}"
            )
    return "\n".join(lines)
