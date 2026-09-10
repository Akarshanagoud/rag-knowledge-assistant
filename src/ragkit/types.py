"""Core types.

The invariant that runs through this package: **a chunk always knows where it
came from**. Document id, section, and the character offsets in the original
text. Without that, citations are decorative — you can print a filename next to
an answer, but nobody can check it, which is the only thing a citation is for.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A source document before chunking."""

    doc_id: str
    text: str
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class Chunk:
    """A retrievable unit of text, anchored to its source."""

    chunk_id: str
    doc_id: str
    text: str
    #: Character offsets into the *original* document text.
    start: int
    end: int
    #: Position of this chunk within the document, for neighbour expansion.
    index: int = 0
    title: str = ""
    section: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def citation(self) -> str:
        """Human-checkable reference: document, section, character range."""
        base = self.doc_id if not self.section else f"{self.doc_id}#{self.section}"
        return f"{base}:{self.start}-{self.end}"

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.text) // 4)

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "citation": self.citation,
            "title": self.title,
            "section": self.section,
            "start": self.start,
            "end": self.end,
            "index": self.index,
            "metadata": self.metadata,
        }
        if include_text:
            payload["text"] = self.text
        return payload


@dataclass
class ScoredChunk:
    """A chunk with a retrieval score and an explanation of where it came from."""

    chunk: Chunk
    score: float
    #: Per-retriever contributions, e.g. {"dense": 0.71, "lexical": 3.4}.
    components: dict[str, float] = field(default_factory=dict)
    #: Rank in each retriever's own list, used by reciprocal rank fusion.
    ranks: dict[str, int] = field(default_factory=dict)
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        return {
            **self.chunk.to_dict(include_text),
            "score": round(self.score, 5),
            "components": {k: round(v, 5) for k, v in self.components.items()},
            "ranks": self.ranks,
            "matched_terms": self.matched_terms[:8],
        }


@dataclass
class Citation:
    """A claim in an answer, tied to the chunk that supports it."""

    text: str
    chunk_id: str
    citation: str
    doc_id: str
    #: Lexical overlap between the claim and its supporting chunk. Low values
    #: mean the sentence is not really supported by what it cites.
    support: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "citation": self.citation,
            "doc_id": self.doc_id,
            "support": round(self.support, 3),
        }


@dataclass
class Answer:
    """A generated answer plus everything needed to check it."""

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[ScoredChunk] = field(default_factory=list)
    #: False when a claim in the answer is not supported by the retrieved text.
    grounded: bool = True
    #: Set when retrieval found nothing good enough to answer from.
    abstained: bool = False
    elapsed_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sources(self) -> list[str]:
        seen: dict[str, None] = {}
        for citation in self.citations:
            seen.setdefault(citation.citation, None)
        return list(seen)

    def to_dict(self, include_chunks: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": self.question,
            "answer": self.text,
            "grounded": self.grounded,
            "abstained": self.abstained,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "citations": [c.to_dict() for c in self.citations],
            "sources": self.sources,
            "metadata": self.metadata,
        }
        if include_chunks:
            payload["retrieved"] = [c.to_dict(include_text=False) for c in self.retrieved]
        return payload


_WORD = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")

STOPWORDS = frozenset(
    """
    a an and are as at be been but by for from has have how in is it its of on
    or that the this to was were what when where which who will with would
    """.split()  # noqa: SIM905 - readable as prose, not as 37 quoted strings
)


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Lowercase word tokens, stopwords dropped by default."""
    tokens = _WORD.findall(text.lower())
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def sentences(text: str) -> list[tuple[str, int, int]]:
    """Split into ``(sentence, start, end)`` triples, preserving offsets.

    Offsets are kept because chunking, citation and highlighting all need to
    point back at the original document, and a splitter that returns bare
    strings throws that away.
    """
    out: list[tuple[str, int, int]] = []
    # Split on sentence-ending punctuation followed by whitespace and a capital
    # or digit, avoiding the common abbreviation cases.
    pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
    position = 0
    for piece in pattern.split(text):
        if not piece:
            continue
        start = text.find(piece, position)
        if start == -1:
            start = position
        end = start + len(piece)
        stripped = piece.strip()
        if stripped:
            offset = piece.index(stripped[0]) if stripped else 0
            out.append((stripped, start + offset, start + offset + len(stripped)))
        position = end
    return out
