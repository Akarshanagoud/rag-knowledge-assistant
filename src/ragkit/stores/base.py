"""Vector store interface.

The point of this abstraction is that swapping FAISS for Chroma should be a
configuration change and a measurement, not a rewrite. The evaluation harness
runs the same questions against each backend so "which vector store?" gets
answered with recall numbers on your corpus rather than with a benchmark from
someone else's.

Every backend must support metadata filtering, because in practice retrieval is
rarely "search everything" — it is "search the current version of the policy
documents this user is allowed to see".
"""
from __future__ import annotations

import abc
from typing import Any

from ..types import Chunk, ScoredChunk


class VectorStore(abc.ABC):
    """Stores chunk vectors and returns nearest neighbours."""

    name: str = "store"

    @abc.abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Index chunks. Each chunk must already carry an embedding."""

    @abc.abstractmethod
    def search(
        self, vector: list[float], top_k: int = 10, where: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        """Return the ``top_k`` nearest chunks, optionally metadata-filtered."""

    @abc.abstractmethod
    def get(self, chunk_id: str) -> Chunk | None:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    def all_chunks(self) -> list[Chunk]:
        """Every stored chunk. Used by lexical search and neighbour expansion."""
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "chunks": len(self)}


def matches_filter(chunk: Chunk, where: dict[str, Any] | None) -> bool:
    """Evaluate a metadata filter against a chunk.

    Supports equality, ``{"$in": [...]}``, ``{"$ne": v}``, ``{"$gt"/"$gte"/
    "$lt"/"$lte": v}``. Fields resolve against the chunk's own attributes first
    and its metadata second, so ``{"doc_id": "policy-1"}`` works without the
    caller knowing how the chunk is laid out.
    """
    if not where:
        return True

    for key, expected in where.items():
        actual = getattr(chunk, key, None)
        if actual is None:
            actual = chunk.metadata.get(key)

        if isinstance(expected, dict):
            for operator, operand in expected.items():
                if not _compare(actual, operator, operand):
                    return False
        elif actual != expected:
            return False
    return True


def _compare(actual: Any, operator: str, operand: Any) -> bool:
    try:
        if operator == "$in":
            return actual in operand
        if operator == "$nin":
            return actual not in operand
        if operator == "$ne":
            return actual != operand
        if operator == "$gt":
            return actual is not None and actual > operand
        if operator == "$gte":
            return actual is not None and actual >= operand
        if operator == "$lt":
            return actual is not None and actual < operand
        if operator == "$lte":
            return actual is not None and actual <= operand
        if operator == "$contains":
            return operand in (actual or "")
    except TypeError:
        # Comparing a string to a number is a filter bug, not a match.
        return False
    raise ValueError(f"unsupported filter operator {operator!r}")
