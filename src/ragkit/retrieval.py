"""Retrieval: lexical, dense, and hybrid fusion.

Hybrid is the default because the two retrievers fail in opposite directions,
and the failures are the interesting part:

* **Dense** finds paraphrases but misses exact tokens. A query for error code
  ``E4021`` or drug ``lisinopril`` can rank the right chunk twentieth, because
  a rare literal string contributes almost nothing to a smooth vector.
* **Lexical (BM25)** nails exact tokens and misses every paraphrase. "How do I
  get my money back?" retrieves nothing from a page headed "Refunds".

Fusion is by Reciprocal Rank Fusion rather than score averaging, deliberately.
BM25 scores are unbounded and corpus-dependent; cosine scores sit in [-1, 1].
Averaging them requires a normalisation that is wrong the moment the corpus
changes. RRF only uses *rank*, so it is scale-free and needs no tuning.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .embeddings import Embedder
from .stores.base import VectorStore, matches_filter
from .types import Chunk, ScoredChunk, tokenize


@dataclass
class RetrievalConfig:
    top_k: int = 5
    #: How many candidates each retriever contributes before fusion. Fusion
    #: can only reorder what it is given, so this is deliberately wider than
    #: top_k.
    candidate_k: int = 30
    mode: str = "hybrid"  # hybrid | dense | lexical
    #: RRF smoothing constant. 60 is the value from the original paper and is
    #: not sensitive enough to be worth tuning per corpus.
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    #: Drop results below this fused score. 0 disables.
    min_score: float = 0.0
    #: Maximum chunks per source document, so one verbose document cannot
    #: crowd out every other source.
    max_per_doc: int = 0
    where: dict[str, Any] | None = None


class BM25Index:
    """Okapi BM25 over chunk text.

    Implemented here rather than pulled in as a dependency because the index
    must stay in lockstep with the vector store: the same chunk ids, the same
    metadata filter semantics, the same lifecycle.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._term_frequencies: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0
        #: term -> indices containing it, so scoring touches only candidates
        #: that share a query term instead of the whole corpus.
        self._postings: dict[str, list[int]] = {}

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            index = len(self._chunks)
            tokens = tokenize(f"{chunk.section} {chunk.text}")
            counts = Counter(tokens)
            self._chunks.append(chunk)
            self._term_frequencies.append(counts)
            self._lengths.append(len(tokens) or 1)
            for term in counts:
                self._document_frequency[term] += 1
                self._postings.setdefault(term, []).append(index)
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )

    def _idf(self, term: str) -> float:
        total = len(self._chunks)
        frequency = self._document_frequency.get(term, 0)
        return math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))

    def search(
        self, query: str, top_k: int = 30, where: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        terms = tokenize(query)
        if not terms or not self._chunks:
            return []

        scores: dict[int, float] = {}
        matched: dict[int, list[str]] = {}
        for term in set(terms):
            idf = self._idf(term)
            for index in self._postings.get(term, ()):
                frequency = self._term_frequencies[index][term]
                length = self._lengths[index]
                numerator = frequency * (self.K1 + 1)
                denominator = frequency + self.K1 * (
                    1 - self.B + self.B * length / (self._average_length or 1)
                )
                scores[index] = scores.get(index, 0.0) + idf * numerator / denominator
                matched.setdefault(index, []).append(term)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        results: list[ScoredChunk] = []
        for index, score in ranked:
            chunk = self._chunks[index]
            if not matches_filter(chunk, where):
                continue
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    components={"lexical": score},
                    matched_terms=sorted(matched.get(index, [])),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def clear(self) -> None:
        self._chunks.clear()
        self._term_frequencies.clear()
        self._lengths.clear()
        self._document_frequency.clear()
        self._postings.clear()
        self._average_length = 0.0

    def __len__(self) -> int:
        return len(self._chunks)


@dataclass
class Retriever:
    """Runs the configured retrieval mode over a store and a BM25 index."""

    store: VectorStore
    embedder: Embedder
    bm25: BM25Index = field(default_factory=BM25Index)
    config: RetrievalConfig = field(default_factory=RetrievalConfig)

    def index(self, chunks: list[Chunk]) -> None:
        """Embed and index chunks into both retrievers."""
        pending = [c for c in chunks if c.embedding is None]
        if pending:
            vectors = self.embedder.embed_batch(c.text for c in pending)
            for chunk, vector in zip(pending, vectors, strict=True):
                chunk.embedding = vector
        self.store.add(chunks)
        self.bm25.add(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        config = self.config
        k = top_k or config.top_k
        selected = mode or config.mode
        filters = where if where is not None else config.where

        if selected == "dense":
            results = self._dense(query, config.candidate_k, filters)
        elif selected == "lexical":
            results = self.bm25.search(query, config.candidate_k, filters)
        elif selected == "hybrid":
            results = self._hybrid(query, config.candidate_k, filters)
        else:
            raise ValueError(f"unknown retrieval mode {selected!r}")

        if config.min_score:
            results = [r for r in results if r.score >= config.min_score]
        if config.max_per_doc:
            results = _cap_per_document(results, config.max_per_doc)
        return results[:k]

    def _dense(
        self, query: str, k: int, where: dict[str, Any] | None
    ) -> list[ScoredChunk]:
        return self.store.search(self.embedder.embed(query), top_k=k, where=where)

    def _hybrid(
        self, query: str, k: int, where: dict[str, Any] | None
    ) -> list[ScoredChunk]:
        dense = self._dense(query, k, where)
        lexical = self.bm25.search(query, k, where)
        return reciprocal_rank_fusion(
            {"dense": dense, "lexical": lexical},
            weights={
                "dense": self.config.dense_weight,
                "lexical": self.config.lexical_weight,
            },
            rrf_k=self.config.rrf_k,
        )


def reciprocal_rank_fusion(
    runs: dict[str, list[ScoredChunk]],
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
) -> list[ScoredChunk]:
    """Fuse ranked lists by ``sum(weight / (rrf_k + rank))``.

    Rank-based rather than score-based, so BM25's unbounded scores and cosine's
    bounded ones can be combined without inventing a normalisation that breaks
    when the corpus changes.
    """
    weights = weights or {}
    fused: dict[str, ScoredChunk] = {}

    for run_name, results in runs.items():
        weight = weights.get(run_name, 1.0)
        for rank, scored in enumerate(results, start=1):
            key = scored.chunk.chunk_id
            entry = fused.get(key)
            if entry is None:
                entry = ScoredChunk(
                    chunk=scored.chunk,
                    score=0.0,
                    components={},
                    ranks={},
                    matched_terms=list(scored.matched_terms),
                )
                fused[key] = entry
            entry.score += weight / (rrf_k + rank)
            entry.components.update(scored.components)
            entry.ranks[run_name] = rank
            for term in scored.matched_terms:
                if term not in entry.matched_terms:
                    entry.matched_terms.append(term)

    return sorted(fused.values(), key=lambda s: -s.score)


def _cap_per_document(results: list[ScoredChunk], limit: int) -> list[ScoredChunk]:
    seen: Counter[str] = Counter()
    capped: list[ScoredChunk] = []
    for scored in results:
        if seen[scored.chunk.doc_id] >= limit:
            continue
        seen[scored.chunk.doc_id] += 1
        capped.append(scored)
    return capped
