"""FAISS and Chroma backends.

Both are optional and free. They are imported lazily so the package installs
and the tests run without either, and so a missing backend produces a clear
message rather than an ImportError at module load.

The interesting difference the evaluation harness exposes: FAISS with an IVF
index is *approximate*, so it can miss a chunk that exact search would find.
That trade is usually correct at scale and usually invisible until someone
measures recall against an exact baseline — which is what `ragkit eval
--compare-stores` does.
"""
from __future__ import annotations

from typing import Any

from ..types import Chunk, ScoredChunk
from .base import VectorStore, matches_filter


class FaissStore(VectorStore):
    """FAISS index with chunk payloads held alongside.

        pip install faiss-cpu

    ``index_type``:
        ``flat``  exact inner-product search (the default; no recall loss)
        ``ivf``   inverted file index, approximate, much faster at scale
        ``hnsw``  graph index, approximate, strong recall/latency trade-off
    """

    name = "faiss"

    def __init__(
        self,
        dimensions: int,
        index_type: str = "flat",
        nlist: int = 64,
        nprobe: int = 8,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "FaissStore requires faiss: pip install faiss-cpu"
            ) from exc
        self._faiss = faiss
        self.dimensions = dimensions
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self._index = self._build_index()
        self._chunks: list[Chunk] = []
        self._by_id: dict[str, int] = {}
        self._trained = index_type == "flat"

    def _build_index(self):
        faiss = self._faiss
        if self.index_type == "flat":
            # Vectors are L2-normalised, so inner product is cosine similarity.
            return faiss.IndexFlatIP(self.dimensions)
        if self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimensions)
            index = faiss.IndexIVFFlat(
                quantizer, self.dimensions, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            index.nprobe = self.nprobe
            return index
        if self.index_type == "hnsw":
            return faiss.IndexHNSWFlat(self.dimensions, 32, faiss.METRIC_INNER_PRODUCT)
        raise ValueError(f"unknown faiss index type {self.index_type!r}")

    def add(self, chunks: list[Chunk]) -> None:
        import numpy as np

        vectors = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"chunk {chunk.chunk_id} has no embedding")
            self._by_id[chunk.chunk_id] = len(self._chunks)
            self._chunks.append(chunk)
            vectors.append(chunk.embedding)

        if not vectors:
            return
        matrix = np.asarray(vectors, dtype=np.float32)

        if not self._trained:
            # IVF needs training data before it can assign vectors to cells.
            # With fewer vectors than centroids, training is meaningless, so
            # the index is shrunk to fit rather than failing.
            if matrix.shape[0] < self.nlist:
                self.nlist = max(1, matrix.shape[0] // 2 or 1)
                self._index = self._build_index()
            self._index.train(matrix)
            self._trained = True

        self._index.add(matrix)

    def search(
        self, vector: list[float], top_k: int = 10, where: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        import numpy as np

        if not self._chunks:
            return []

        query = np.asarray([vector], dtype=np.float32)
        # FAISS has no metadata filtering, so over-fetch and filter after.
        # The multiplier is the honest cost of post-filtering: with a very
        # selective filter this can still under-return, which the caller sees.
        fetch = top_k if not where else min(len(self._chunks), max(top_k * 8, 64))
        scores, indices = self._index.search(query, min(fetch, len(self._chunks)))

        results: list[ScoredChunk] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            chunk = self._chunks[int(index)]
            if not matches_filter(chunk, where):
                continue
            results.append(
                ScoredChunk(chunk=chunk, score=float(score), components={"dense": float(score)})
            )
            if len(results) >= top_k:
                break
        return results

    def get(self, chunk_id: str) -> Chunk | None:
        index = self._by_id.get(chunk_id)
        return self._chunks[index] if index is not None else None

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def clear(self) -> None:
        self._index = self._build_index()
        self._chunks.clear()
        self._by_id.clear()
        self._trained = self.index_type == "flat"

    def __len__(self) -> int:
        return len(self._chunks)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "index_type": self.index_type,
            "exact": self.index_type == "flat",
            "note": "post-filtering: metadata filters are applied after the ANN search",
        }


class ChromaStore(VectorStore):
    """Chroma collection, in-memory or persistent.

        pip install chromadb

    Chroma filters inside the query rather than after it, which is the main
    practical difference from FAISS: a selective filter stays selective instead
    of forcing an over-fetch.
    """

    name = "chroma"

    def __init__(
        self, collection: str = "ragkit", persist_directory: str | None = None
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("ChromaStore requires: pip install chromadb") from exc

        self._client = (
            chromadb.PersistentClient(path=persist_directory)
            if persist_directory
            else chromadb.EphemeralClient()
        )
        self.collection_name = collection
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self._chunks: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids, embeddings, documents, metadatas = [], [], [], []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"chunk {chunk.chunk_id} has no embedding")
            self._chunks[chunk.chunk_id] = chunk
            ids.append(chunk.chunk_id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "start": chunk.start,
                    "end": chunk.end,
                    "index": chunk.index,
                    # Chroma metadata values must be scalars.
                    **{k: v for k, v in chunk.metadata.items()
                       if isinstance(v, (str, int, float, bool))},
                }
            )
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def search(
        self, vector: list[float], top_k: int = 10, where: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        response = self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, len(self._chunks)),
            where=_to_chroma_filter(where),
        )
        ids = response.get("ids", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[ScoredChunk] = []
        for chunk_id, distance in zip(ids, distances, strict=True):
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            # Chroma returns cosine *distance*; convert back to similarity so
            # scores are comparable with the other backends.
            score = 1.0 - float(distance)
            results.append(
                ScoredChunk(chunk=chunk, score=score, components={"dense": score})
            )
        return results

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._chunks.clear()

    def __len__(self) -> int:
        return len(self._chunks)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "collection": self.collection_name,
            "exact": False,
            "note": "pre-filtering: metadata filters are applied inside the query",
        }


def _to_chroma_filter(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate the common filter dialect into Chroma's."""
    if not where:
        return None
    clauses: list[dict[str, Any]] = []
    for key, expected in where.items():
        if isinstance(expected, dict):
            clauses.append({key: expected})
        else:
            clauses.append({key: {"$eq": expected}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def build_store(kind: str, dimensions: int, **kwargs: Any) -> VectorStore:
    """Construct a backend by name. Used by the CLI and the eval harness."""
    from .memory import MemoryStore

    if kind == "memory":
        return MemoryStore()
    if kind == "faiss":
        return FaissStore(dimensions=dimensions, **kwargs)
    if kind == "chroma":
        return ChromaStore(**kwargs)
    raise ValueError(f"unknown store {kind!r}; choose from memory, faiss, chroma")
