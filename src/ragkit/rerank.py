"""Reranking.

Retrieval optimises for recall over a wide candidate pool; reranking optimises
for precision over a narrow one. They want different things, which is why doing
both beats doing either harder.

Three rerankers, cheapest first:

* :class:`LexicalReranker` - term coverage and proximity. Free, and it fixes
  the most common hybrid failure: a chunk that scored well on one query term
  while ignoring the other three.
* :class:`MMRReranker` - maximal marginal relevance. Not about relevance at
  all, but about *redundancy*: five near-identical chunks from the same section
  waste the context window and make the answer no better.
* :class:`CrossEncoderReranker` - optional local model, the accurate one.

They compose. The default pipeline is lexical then MMR, because the failure
they each fix is different.
"""
from __future__ import annotations

import abc
from typing import Any

from .embeddings import cosine
from .types import ScoredChunk, tokenize


class Reranker(abc.ABC):
    name: str = "reranker"

    @abc.abstractmethod
    def rerank(
        self, query: str, results: list[ScoredChunk], top_k: int | None = None
    ) -> list[ScoredChunk]:
        ...

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


class LexicalReranker(Reranker):
    """Rescores on query-term coverage and proximity within the chunk.

    Coverage matters more than frequency: a chunk containing all four query
    terms once is a better answer than one containing a single term ten times,
    and pure BM25 does not reliably prefer it.
    """

    name = "lexical"

    def __init__(self, coverage_weight: float = 0.6, proximity_weight: float = 0.2,
                 base_weight: float = 0.2) -> None:
        self.coverage_weight = coverage_weight
        self.proximity_weight = proximity_weight
        self.base_weight = base_weight

    def rerank(
        self, query: str, results: list[ScoredChunk], top_k: int | None = None
    ) -> list[ScoredChunk]:
        terms = set(tokenize(query))
        if not terms or not results:
            return results[: top_k or len(results)]

        # Fused scores are tiny (RRF sums of 1/(60+rank)); normalising keeps
        # the base contribution comparable to coverage and proximity.
        highest = max(r.score for r in results) or 1.0

        for scored in results:
            tokens = tokenize(scored.chunk.text)
            present = terms & set(tokens)
            coverage = len(present) / len(terms)
            proximity = _proximity(tokens, terms)
            scored.components["coverage"] = coverage
            scored.components["proximity"] = proximity
            scored.components["retrieval"] = scored.score
            scored.score = (
                self.coverage_weight * coverage
                + self.proximity_weight * proximity
                + self.base_weight * (scored.score / highest)
            )
            scored.matched_terms = sorted(present)

        results.sort(key=lambda s: -s.score)
        return results[: top_k or len(results)]


def _proximity(tokens: list[str], terms: set[str]) -> float:
    """1.0 when all query terms appear inside a tight window, 0.0 when absent.

    Terms scattered across a long chunk are usually coincidence; terms within a
    sentence or two of each other usually mean the chunk is about the query.
    """
    positions = [i for i, token in enumerate(tokens) if token in terms]
    if len(positions) < 2:
        return 1.0 if positions else 0.0
    distinct = {tokens[i] for i in positions}
    if len(distinct) < 2:
        return 0.3
    span = positions[-1] - positions[0] + 1
    ideal = len(distinct)
    return max(0.0, min(1.0, ideal / span))


class MMRReranker(Reranker):
    """Maximal marginal relevance: trade a little relevance for diversity.

    ``lambda_param`` is the dial. 1.0 is pure relevance (no diversification),
    0.0 is pure novelty. 0.7 keeps the top result intact while stopping the
    next four from being the same paragraph with different overlap.
    """

    name = "mmr"

    def __init__(self, lambda_param: float = 0.7) -> None:
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must be between 0 and 1")
        self.lambda_param = lambda_param

    def rerank(
        self, query: str, results: list[ScoredChunk], top_k: int | None = None
    ) -> list[ScoredChunk]:
        if len(results) <= 1:
            return results

        limit = top_k or len(results)
        remaining = list(results)
        selected: list[ScoredChunk] = [remaining.pop(0)]

        while remaining and len(selected) < limit:
            best_index = 0
            best_value = float("-inf")
            for i, candidate in enumerate(remaining):
                redundancy = max(
                    _similarity(candidate, chosen) for chosen in selected
                )
                value = (
                    self.lambda_param * candidate.score
                    - (1 - self.lambda_param) * redundancy
                )
                if value > best_value:
                    best_value, best_index = value, i
            chosen = remaining.pop(best_index)
            chosen.components["mmr"] = best_value
            selected.append(chosen)

        return selected


def _similarity(a: ScoredChunk, b: ScoredChunk) -> float:
    """Redundancy between two candidates.

    Uses embeddings when both have them, and falls back to Jaccard overlap so
    the reranker still works on chunks that came from a lexical-only run.
    """
    if a.chunk.embedding and b.chunk.embedding:
        return max(0.0, cosine(a.chunk.embedding, b.chunk.embedding))
    left = set(tokenize(a.chunk.text))
    right = set(tokenize(b.chunk.text))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class CrossEncoderReranker(Reranker):
    """Optional local cross-encoder. Slower, and clearly better.

        pip install sentence-transformers
        CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

    A bi-encoder embeds query and document separately, so it can never model
    their interaction. A cross-encoder reads both together, which is why it
    wins on precision and why it cannot be used for first-stage retrieval.
    """

    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CrossEncoderReranker needs: pip install sentence-transformers"
            ) from exc
        self._model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self, query: str, results: list[ScoredChunk], top_k: int | None = None
    ) -> list[ScoredChunk]:
        if not results:
            return results
        scores = self._model.predict([(query, r.chunk.text) for r in results])
        for scored, score in zip(results, scores, strict=True):
            scored.components["retrieval"] = scored.score
            scored.components["cross_encoder"] = float(score)
            scored.score = float(score)
        results.sort(key=lambda s: -s.score)
        return results[: top_k or len(results)]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model_name}


class RerankPipeline(Reranker):
    """Applies several rerankers in order."""

    name = "pipeline"

    def __init__(self, stages: list[Reranker]) -> None:
        self.stages = stages

    def rerank(
        self, query: str, results: list[ScoredChunk], top_k: int | None = None
    ) -> list[ScoredChunk]:
        current = results
        for i, stage in enumerate(self.stages):
            last = i == len(self.stages) - 1
            current = stage.rerank(query, current, top_k if last else None)
        return current

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "stages": [s.describe() for s in self.stages]}


def build_reranker(kind: str, **kwargs: Any) -> Reranker | None:
    """Construct a reranker by name. ``none`` returns ``None``."""
    if kind in {"none", "off", ""}:
        return None
    if kind == "lexical":
        return LexicalReranker(**kwargs)
    if kind == "mmr":
        return MMRReranker(**kwargs)
    if kind == "cross-encoder":
        return CrossEncoderReranker(**kwargs)
    if kind == "default":
        # Precision first, then redundancy removal. The order matters: running
        # MMR first would diversify a list that has not been scored properly.
        return RerankPipeline([LexicalReranker(), MMRReranker()])
    raise ValueError(
        f"unknown reranker {kind!r}; choose from none, lexical, mmr, cross-encoder, default"
    )
