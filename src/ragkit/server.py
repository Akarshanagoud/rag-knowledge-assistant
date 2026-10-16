"""HTTP service and web UI for the knowledge assistant.

    pip install "ragkit[server]"
    ragkit ui                     # opens http://127.0.0.1:8766/ui/

The service holds one indexed pipeline. Documents can be added at runtime, and
the evaluation endpoints run the same harness as ``ragkit eval`` and
``ragkit compare``, so the numbers in the browser match the command line.

Set ``RAGKIT_CORPUS`` and ``RAGKIT_GOLD`` to serve a different corpus.
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .chunking import CHUNKERS, ChunkConfig
from .cli import DEFAULT_CORPUS, DEFAULT_GOLD
from .evaluation import GoldQuestion, compare, evaluate_retrieval, load_gold
from .pipeline import PipelineConfig, RAGPipeline, load_corpus
from .retrieval import RetrievalConfig
from .types import Document, ScoredChunk

WEB_DIR = Path(__file__).parent / "web"
MODES = ("hybrid", "dense", "lexical")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    mode: str = "hybrid"
    top_k: int = Field(5, ge=1, le=20)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


class DocumentRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=20, max_length=200_000)


class EvalRequest(BaseModel):
    k: int = Field(5, ge=1, le=20)
    mode: str = "hybrid"


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {', '.join(MODES)}")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "document"


def _chunk(scored: ScoredChunk, rank: int) -> dict[str, Any]:
    payload = scored.to_dict(include_text=True)
    payload["rank"] = rank
    return payload


def _gold_dict(question: GoldQuestion) -> dict[str, Any]:
    return {
        "question": question.question,
        "relevant_doc_ids": question.relevant_doc_ids,
        "answer_contains": question.answer_contains,
        "unanswerable": question.unanswerable,
        "tags": question.tags,
    }


def _variants(k: int) -> dict[str, PipelineConfig]:
    """The configurations the comparison view scores, mirroring ``ragkit compare``."""

    def build(tokens: int = 220, mode: str = "hybrid", **overrides: Any) -> PipelineConfig:
        config = PipelineConfig()
        config.chunking = ChunkConfig(target_tokens=tokens)
        config.retrieval = RetrievalConfig(top_k=k, mode=mode)
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    variants = {
        "hybrid (default)": build(),
        "dense only": build(mode="dense"),
        "lexical only": build(mode="lexical"),
        "hybrid + rerank": build(reranker="default"),
    }
    for name in CHUNKERS:
        if name != "section":
            variants[f"chunker: {name}"] = build(chunker=name)
    variants["small chunks (100 tok)"] = build(tokens=100)
    variants["large chunks (400 tok)"] = build(tokens=400)
    return variants


class Assistant:
    """One indexed pipeline plus the corpus and gold set it was built from."""

    def __init__(self, corpus: Path | None, gold: Path | None) -> None:
        self.corpus_path = corpus
        self.base_documents = load_corpus(corpus) if corpus and corpus.is_dir() else []
        self.added: list[Document] = []
        self.gold = load_gold(gold) if gold and gold.is_file() else []
        self.pipeline = self._build()

    @property
    def documents(self) -> list[Document]:
        return self.base_documents + self.added

    def _build(self) -> RAGPipeline:
        pipeline = RAGPipeline(PipelineConfig())
        if self.documents:
            pipeline.add_documents([copy.deepcopy(d) for d in self.documents])
        return pipeline

    # -- views ----------------------------------------------------------------
    def overview(self) -> dict[str, Any]:
        stats = self.pipeline.stats()
        answerable = [q for q in self.gold if not q.unanswerable]
        unanswerable = [q for q in self.gold if q.unanswerable]
        return {
            "version": __version__,
            "corpus": str(self.corpus_path) if self.corpus_path else None,
            "documents": stats["documents"],
            "chunks": stats["chunks"],
            "mean_chunk_chars": stats["mean_chunk_chars"],
            "embedder": stats["embedder"],
            "store": stats["store"],
            "reranker": stats["reranker"],
            "chunker": self.pipeline.config.chunker,
            "gold_questions": len(self.gold),
            "suggested": [q.question for q in answerable[:6]] + [q.question for q in unanswerable[:2]],
        }

    def ask(self, body: AskRequest) -> dict[str, Any]:
        _check_mode(body.mode)
        answer = self.pipeline.ask(body.question, top_k=body.top_k, mode=body.mode)
        payload = answer.to_dict(include_chunks=False)
        payload["retrieved"] = [_chunk(c, i + 1) for i, c in enumerate(answer.retrieved)]
        payload["mode"] = body.mode
        payload["top_k"] = body.top_k
        return payload

    def retrieve(self, body: RetrieveRequest) -> dict[str, Any]:
        runs = {
            mode: [_chunk(c, i + 1) for i, c in enumerate(self.pipeline.retrieve(body.query, top_k=body.top_k, mode=mode))]
            for mode in MODES
        }
        gold = next((q for q in self.gold if q.question.strip().lower() == body.query.strip().lower()), None)
        return {"query": body.query, "runs": runs, "gold": _gold_dict(gold) if gold else None}

    def list_documents(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for chunk in self.pipeline.store.all_chunks():
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        added = {d.doc_id for d in self.added}
        return [
            {
                "doc_id": d.doc_id,
                "title": d.title or d.doc_id,
                "chars": len(d.text),
                "words": len(d.text.split()),
                "chunks": counts.get(d.doc_id, 0),
                "added": d.doc_id in added,
            }
            for d in self.documents
        ]

    def document(self, doc_id: str) -> dict[str, Any]:
        doc = next((d for d in self.documents if d.doc_id == doc_id), None)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"no document {doc_id!r}")
        chunks = sorted(
            (c for c in self.pipeline.store.all_chunks() if c.doc_id == doc_id), key=lambda c: c.index
        )
        return {
            "doc_id": doc.doc_id,
            "title": doc.title or doc.doc_id,
            "text": doc.text,
            "added": doc in self.added,
            "chunks": [
                {"chunk_id": c.chunk_id, "index": c.index, "section": c.section, "start": c.start, "end": c.end, "chars": len(c.text), "citation": c.citation}
                for c in chunks
            ],
        }

    def add_document(self, body: DocumentRequest) -> dict[str, Any]:
        doc_id = _slug(body.title)
        if any(d.doc_id == doc_id for d in self.documents):
            raise HTTPException(status_code=409, detail=f"a document called {doc_id!r} already exists")
        self.added.append(Document(doc_id=doc_id, text=body.text, title=body.title.strip(), source="uploaded"))
        self.pipeline = self._build()
        return next(d for d in self.list_documents() if d["doc_id"] == doc_id)

    def remove_document(self, doc_id: str) -> dict[str, Any]:
        doc = next((d for d in self.added if d.doc_id == doc_id), None)
        if doc is None:
            raise HTTPException(status_code=404, detail="only documents added in this session can be removed")
        self.added.remove(doc)
        self.pipeline = self._build()
        return {"removed": doc_id}

    def evaluate(self, body: EvalRequest) -> dict[str, Any]:
        _check_mode(body.mode)
        if not self.gold:
            raise HTTPException(status_code=409, detail="no gold question set is loaded")
        report = evaluate_retrieval(self.pipeline, self.gold, k=body.k, mode=body.mode, label=body.mode, with_answers=True)
        payload = report.to_dict(include_questions=True)
        for row, gold in zip(payload["per_question"], self.gold, strict=True):
            row.update(_gold_dict(gold))
        return payload

    def compare(self, k: int) -> list[dict[str, Any]]:
        if not self.gold:
            raise HTTPException(status_code=409, detail="no gold question set is loaded")
        documents = [copy.deepcopy(d) for d in self.documents]
        return [r.to_dict() for r in compare(documents, self.gold, _variants(k), k=k, with_answers=True)]


def create_app(corpus: Path | None = DEFAULT_CORPUS, gold: Path | None = DEFAULT_GOLD) -> FastAPI:
    assistant = Assistant(corpus, gold)
    app = FastAPI(title="ragkit", version=__version__, description="Knowledge assistant with a measurement harness.")
    app.state.assistant = assistant

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "documents": len(assistant.documents)}

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        return assistant.overview()

    @app.post("/api/ask")
    def ask(body: AskRequest) -> dict[str, Any]:
        return assistant.ask(body)

    @app.post("/api/retrieve")
    def retrieve(body: RetrieveRequest) -> dict[str, Any]:
        return assistant.retrieve(body)

    @app.get("/api/documents")
    def documents() -> list[dict[str, Any]]:
        return assistant.list_documents()

    @app.get("/api/documents/{doc_id}")
    def document(doc_id: str) -> dict[str, Any]:
        return assistant.document(doc_id)

    @app.post("/api/documents")
    def add_document(body: DocumentRequest) -> dict[str, Any]:
        return assistant.add_document(body)

    @app.delete("/api/documents/{doc_id}")
    def remove_document(doc_id: str) -> dict[str, Any]:
        return assistant.remove_document(doc_id)

    @app.get("/api/gold")
    def gold() -> list[dict[str, Any]]:
        return [_gold_dict(q) for q in assistant.gold]

    @app.post("/api/evaluate")
    def evaluate(body: EvalRequest) -> dict[str, Any]:
        return assistant.evaluate(body)

    @app.post("/api/compare")
    def run_compare(body: EvalRequest) -> list[dict[str, Any]]:
        return assistant.compare(body.k)

    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    return app


def _env_path(name: str, default: Path) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else default


app = create_app(_env_path("RAGKIT_CORPUS", DEFAULT_CORPUS), _env_path("RAGKIT_GOLD", DEFAULT_GOLD))
