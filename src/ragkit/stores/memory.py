"""In-memory vector store.

Exhaustive search, which for a corpus of tens of thousands of chunks is both
fast enough and exactly correct — no approximation, no recall loss, no index
build time. That makes it the right *baseline* for the evaluation harness: when
FAISS reports 0.94 recall, this is the 1.00 it is being measured against.

Uses numpy when available for a vectorised dot product, and falls back to pure
Python otherwise. Same results either way; the fallback is just slower.
"""
from __future__ import annotations

import heapq
from typing import Any

from ..embeddings import cosine
from ..types import Chunk, ScoredChunk
from .base import VectorStore, matches_filter

try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


class MemoryStore(VectorStore):
    """Exact nearest-neighbour search over stored chunk vectors."""

    name = "memory"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._by_id: dict[str, int] = {}
        self._matrix: Any | None = None
        self._dirty = True

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"chunk {chunk.chunk_id} has no embedding")
            existing = self._by_id.get(chunk.chunk_id)
            if existing is not None:
                self._chunks[existing] = chunk
            else:
                self._by_id[chunk.chunk_id] = len(self._chunks)
                self._chunks.append(chunk)
        self._dirty = True

    def _rebuild(self) -> None:
        if _np is None or not self._chunks:
            self._matrix = None
        else:
            self._matrix = _np.asarray(
                [c.embedding for c in self._chunks], dtype=_np.float32
            )
        self._dirty = False

    def search(
        self, vector: list[float], top_k: int = 10, where: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        if self._dirty:
            self._rebuild()

        candidates = [
            i for i, chunk in enumerate(self._chunks) if matches_filter(chunk, where)
        ]
        if not candidates:
            return []

        if self._matrix is not None:
            query = _np.asarray(vector, dtype=_np.float32)
            norm = float(_np.linalg.norm(query))
            if norm:
                query = query / norm
            scores = self._matrix[candidates] @ query
            # argpartition is O(n) rather than a full sort; only the top_k
            # slice is then ordered.
            count = min(top_k, len(candidates))
            top = _np.argpartition(-scores, count - 1)[:count]
            top = top[_np.argsort(-scores[top])]
            return [
                ScoredChunk(
                    chunk=self._chunks[candidates[int(i)]],
                    score=float(scores[int(i)]),
                    components={"dense": float(scores[int(i)])},
                )
                for i in top
            ]

        scored = (
            (cosine(vector, self._chunks[i].embedding or []), i) for i in candidates
        )
        best = heapq.nlargest(top_k, scored, key=lambda pair: pair[0])
        return [
            ScoredChunk(chunk=self._chunks[i], score=score, components={"dense": score})
            for score, i in best
        ]

    def get(self, chunk_id: str) -> Chunk | None:
        index = self._by_id.get(chunk_id)
        return self._chunks[index] if index is not None else None

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._by_id.clear()
        self._matrix = None
        self._dirty = True

    def __len__(self) -> int:
        return len(self._chunks)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "exact": True,
            "accelerated": _np is not None,
        }
