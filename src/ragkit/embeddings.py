"""Embedding backends.

The default embedder is a hashed bag-of-n-grams projection: deterministic, no
download, no network, and honest about what it is. It captures lexical and
sub-word similarity but not synonymy — "car" and "automobile" are unrelated to
it. That limitation is exactly why the retriever defaults to *hybrid* search
rather than dense-only, and why the evaluation harness reports both.

Swap in a real model when you want semantic matching:

    from ragkit.embeddings import SentenceTransformerEmbedder
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")

Both are free and local. The interface is the same, and the evaluation harness
will tell you what the upgrade actually bought you on your corpus, which is the
only way to answer that question.
"""
from __future__ import annotations

import abc
import hashlib
import math
import re
import struct
from collections import Counter
from collections.abc import Iterable
from typing import Any

from .types import tokenize

try:  # optional accelerator; the pure-Python path is always correct
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


class Embedder(abc.ABC):
    """Turns text into a fixed-length vector."""

    name: str = "embedder"
    dimensions: int = 256

    @abc.abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "dimensions": self.dimensions}


def _hash_index(token: str, dimensions: int, salt: str = "") -> tuple[int, int]:
    """Bucket index plus a sign, from a stable hash.

    The sign is the signed-hashing trick: without it, unrelated tokens landing
    in the same bucket only ever add, so collisions inflate similarity. With
    it, collisions cancel on average.
    """
    digest = hashlib.blake2b(f"{salt}{token}".encode(), digest_size=8).digest()
    value = struct.unpack("<Q", digest)[0]
    return value % dimensions, 1 if (value >> 63) & 1 else -1


class HashingEmbedder(Embedder):
    """Hashed character-n-gram and word projection with sub-linear TF.

    Character n-grams give robustness to morphology and typos ("refund",
    "refunds", "refunding" overlap heavily), which is what makes this usable
    for retrieval rather than merely deterministic.
    """

    name = "hashing"

    def __init__(
        self,
        dimensions: int = 256,
        char_ngrams: tuple[int, ...] = (3, 4),
        word_weight: float = 1.0,
        char_weight: float = 0.6,
        salt: str = "ragkit",
    ) -> None:
        self.dimensions = dimensions
        self.char_ngrams = char_ngrams
        self.word_weight = word_weight
        self.char_weight = char_weight
        self.salt = salt

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions

        words = tokenize(text)
        for token, count in Counter(words).items():
            index, sign = _hash_index(token, self.dimensions, self.salt)
            # Sub-linear term frequency: the tenth mention of a word says much
            # less than the second.
            vector[index] += sign * self.word_weight * (1 + math.log(count))

        flattened = re.sub(r"\s+", " ", text.lower())
        for n in self.char_ngrams:
            grams = Counter(flattened[i : i + n] for i in range(max(len(flattened) - n + 1, 0)))
            for gram, count in grams.items():
                if gram.isspace():
                    continue
                index, sign = _hash_index(f"{n}:{gram}", self.dimensions, self.salt)
                vector[index] += sign * self.char_weight * (1 + math.log(count))

        return normalize(vector)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "char_ngrams": list(self.char_ngrams),
            "semantic": False,
            "note": "lexical and sub-word similarity only; no synonymy",
        }


class TfidfEmbedder(Embedder):
    """Corpus-fitted TF-IDF projected onto hashed buckets.

    Unlike :class:`HashingEmbedder` this must see the corpus first, which buys
    proper IDF weighting: a term that appears in every document stops
    contributing. Call :meth:`fit` during indexing.
    """

    name = "tfidf"

    def __init__(self, dimensions: int = 512, salt: str = "ragkit") -> None:
        self.dimensions = dimensions
        self.salt = salt
        self._idf: dict[str, float] = {}
        self._documents = 0

    def fit(self, texts: Iterable[str]) -> TfidfEmbedder:
        document_frequency: Counter[str] = Counter()
        total = 0
        for text in texts:
            total += 1
            for token in set(tokenize(text)):
                document_frequency[token] += 1
        self._documents = total
        self._idf = {
            token: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for token, count in document_frequency.items()
        }
        return self

    @property
    def fitted(self) -> bool:
        return self._documents > 0

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        counts = Counter(tokenize(text))
        if not counts:
            return vector
        max_count = max(counts.values())
        for token, count in counts.items():
            # An unseen token gets the IDF of a term appearing once, so a query
            # word absent from the corpus is treated as highly specific rather
            # than silently dropped.
            idf = self._idf.get(token, math.log(1 + (self._documents + 0.5) / 1.5) if self._documents else 1.0)
            index, sign = _hash_index(token, self.dimensions, self.salt)
            vector[index] += sign * (0.5 + 0.5 * count / max_count) * idf
        return normalize(vector)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "fitted_on_documents": self._documents, "semantic": False}


class SentenceTransformerEmbedder(Embedder):
    """Optional adapter for sentence-transformers models (free, local).

        pip install sentence-transformers
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "SentenceTransformerEmbedder needs: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        return [float(x) for x in self._model.encode(text, normalize_embeddings=True)]

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        batch = list(texts)
        if not batch:
            return []
        encoded = self._model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in row] for row in encoded]

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model": self.model_name, "semantic": True}


# --- vector helpers ----------------------------------------------------------
def normalize(vector: list[float]) -> list[float]:
    """L2-normalise so cosine similarity reduces to a dot product."""
    if _np is not None:
        array = _np.asarray(vector, dtype=_np.float32)
        norm = float(_np.linalg.norm(array))
        return (array / norm).tolist() if norm else array.tolist()
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Assumes normalised inputs but does not require them."""
    if not a or not b:
        return 0.0
    if _np is not None:
        left = _np.asarray(a, dtype=_np.float32)
        right = _np.asarray(b, dtype=_np.float32)
        denominator = float(_np.linalg.norm(left) * _np.linalg.norm(right))
        return float(left @ right / denominator) if denominator else 0.0

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    left = math.sqrt(sum(x * x for x in a))
    right = math.sqrt(sum(y * y for y in b))
    return dot / (left * right) if left and right else 0.0


def build_embedder(kind: str = "hashing", **kwargs: Any) -> Embedder:
    builders = {
        "hashing": HashingEmbedder,
        "tfidf": TfidfEmbedder,
        "sentence-transformers": SentenceTransformerEmbedder,
        "minilm": lambda **kw: SentenceTransformerEmbedder("all-MiniLM-L6-v2", **kw),
    }
    if kind not in builders:
        raise ValueError(f"unknown embedder {kind!r}; choose from {sorted(builders)}")
    return builders[kind](**kwargs)
