"""The assembled RAG pipeline: ingest, chunk, embed, retrieve, rerank, answer."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .answering import AnswerConfig, Generator, answer_question
from .chunking import ChunkConfig, chunk_document
from .embeddings import Embedder, TfidfEmbedder, build_embedder
from .rerank import Reranker, build_reranker
from .retrieval import BM25Index, RetrievalConfig, Retriever
from .stores.base import VectorStore
from .stores.memory import MemoryStore
from .types import Answer, Chunk, Document, ScoredChunk

#: Distinguishes "no reranker" from "not specified". Passing ``reranker=None``
#: must mean *disable reranking*, not *fall back to the configured default* -
#: otherwise there is no way to turn it off through the constructor, and a
#: comparison harness silently measures the default twice.
_UNSET = object()


@dataclass
class PipelineConfig:
    chunker: str = "section"
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    embedder: str = "hashing"
    embedder_options: dict[str, Any] = field(default_factory=dict)
    store: str = "memory"
    store_options: dict[str, Any] = field(default_factory=dict)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    #: Off by default, because on the bundled corpus it measurably hurts:
    #: hybrid retrieval alone scores recall 1.000 / MRR 0.924, and adding the
    #: lexical+MMR pipeline drops those to 0.957 / 0.841 while costing 20x the
    #: latency. Reranking earns its place on larger, more redundant corpora -
    #: `ragkit eval --compare` is how you find out whether yours is one.
    reranker: str = "none"
    answering: AnswerConfig = field(default_factory=AnswerConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunker": self.chunker,
            "chunk_target_tokens": self.chunking.target_tokens,
            "chunk_overlap_tokens": self.chunking.overlap_tokens,
            "embedder": self.embedder,
            "store": self.store,
            "retrieval_mode": self.retrieval.mode,
            "top_k": self.retrieval.top_k,
            "candidate_k": self.retrieval.candidate_k,
            "reranker": self.reranker,
            "min_score_ratio": self.answering.min_score_ratio,
        }


class RAGPipeline:
    """End-to-end question answering over an indexed corpus."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        reranker: Reranker | None | Any = _UNSET,
        generator: Generator | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.embedder = embedder or build_embedder(
            self.config.embedder, **self.config.embedder_options
        )
        self.store = store or self._build_store()
        self.reranker: Reranker | None = (
            build_reranker(self.config.reranker) if reranker is _UNSET else reranker
        )
        self.generator = generator
        self.retriever = Retriever(
            store=self.store,
            embedder=self.embedder,
            bm25=BM25Index(),
            config=self.config.retrieval,
        )
        self.documents: dict[str, Document] = {}

    def _build_store(self) -> VectorStore:
        if self.config.store == "memory":
            return MemoryStore()
        from .stores.external import build_store

        return build_store(
            self.config.store,
            dimensions=self.embedder.dimensions,
            **self.config.store_options,
        )

    # -- indexing -----------------------------------------------------------
    def add_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk, embed and index. Returns the chunks that were created."""
        if not documents:
            return []

        # A corpus-fitted embedder needs the corpus before it can embed
        # anything, so fit here rather than at construction.
        if isinstance(self.embedder, TfidfEmbedder) and not self.embedder.fitted:
            self.embedder.fit(document.text for document in documents)

        chunks: list[Chunk] = []
        for document in documents:
            self.documents[document.doc_id] = document
            chunks.extend(
                chunk_document(document, self.config.chunker, self.config.chunking)
            )

        self.retriever.index(chunks)
        return chunks

    def add_directory(self, path: str | Path, pattern: str = "*.md") -> list[Chunk]:
        """Index every matching file in a directory."""
        directory = Path(path)
        documents = [
            Document(
                doc_id=file.stem,
                text=file.read_text(encoding="utf-8"),
                title=_title_of(file.read_text(encoding="utf-8"), file.stem),
                source=str(file),
            )
            for file in sorted(directory.glob(pattern))
        ]
        return self.add_documents(documents)

    # -- querying -----------------------------------------------------------
    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        mode: str | None = None,
        where: dict[str, Any] | None = None,
        rerank: bool = True,
    ) -> list[ScoredChunk]:
        k = top_k or self.config.retrieval.top_k
        # Retrieve wide, then narrow: a reranker can only reorder what it is
        # given, so handing it exactly top_k makes it useless.
        candidates = self.retriever.retrieve(
            question,
            top_k=self.config.retrieval.candidate_k,
            mode=mode,
            where=where,
        )
        if rerank and self.reranker is not None and candidates:
            return self.reranker.rerank(question, candidates, top_k=k)
        return candidates[:k]

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        mode: str | None = None,
        where: dict[str, Any] | None = None,
    ) -> Answer:
        started = time.perf_counter()
        retrieved = self.retrieve(question, top_k=top_k, mode=mode, where=where)
        answer = answer_question(
            question, retrieved, generator=self.generator, config=self.config.answering
        )
        answer.elapsed_ms = (time.perf_counter() - started) * 1000
        answer.metadata["retrieval_mode"] = mode or self.config.retrieval.mode
        answer.metadata["reranker"] = (
            self.reranker.describe() if self.reranker else None
        )
        return answer

    # -- introspection ------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        chunks = self.store.all_chunks()
        lengths = [len(c.text) for c in chunks]
        return {
            "documents": len(self.documents),
            "chunks": len(chunks),
            "mean_chunk_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "min_chunk_chars": min(lengths) if lengths else 0,
            "max_chunk_chars": max(lengths) if lengths else 0,
            "embedder": self.embedder.describe(),
            "store": self.store.describe(),
            "reranker": self.reranker.describe() if self.reranker else None,
            "config": self.config.to_dict(),
        }

    def save_chunks(self, path: str | Path) -> int:
        """Write the chunk index to JSONL, for inspection or reuse."""
        chunks = self.store.all_chunks()
        with Path(path).open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.to_dict()) + "\n")
        return len(chunks)


def _title_of(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped
    return fallback


def load_corpus(path: str | Path, pattern: str = "*.md") -> list[Document]:
    """Load a directory of text files as documents."""
    directory = Path(path)
    documents: list[Document] = []
    for file in sorted(directory.glob(pattern)):
        text = file.read_text(encoding="utf-8")
        documents.append(
            Document(
                doc_id=file.stem,
                text=text,
                title=_title_of(text, file.stem),
                source=str(file),
                metadata={"filename": file.name},
            )
        )
    return documents
