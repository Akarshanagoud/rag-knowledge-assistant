"""ragkit - a production-shaped RAG stack with a measurement harness."""
from .answering import (
    REFUSAL,
    AnswerConfig,
    ExtractiveGenerator,
    ModelGenerator,
    TemplateGenerator,
    answer_question,
    verify_citations,
)
from .chunking import CHUNKERS, ChunkConfig, chunk_document, chunk_documents, get_chunker
from .embeddings import (
    Embedder,
    HashingEmbedder,
    TfidfEmbedder,
    build_embedder,
    cosine,
)
from .evaluation import (
    EvalReport,
    GoldQuestion,
    compare,
    evaluate_retrieval,
    load_gold,
    render_comparison,
)
from .pipeline import PipelineConfig, RAGPipeline, load_corpus
from .rerank import (
    CrossEncoderReranker,
    LexicalReranker,
    MMRReranker,
    RerankPipeline,
    build_reranker,
)
from .retrieval import BM25Index, RetrievalConfig, Retriever, reciprocal_rank_fusion
from .stores import MemoryStore, VectorStore
from .types import Answer, Chunk, Citation, Document, ScoredChunk

__version__ = "0.1.0"

__all__ = [
    "CHUNKERS",
    "REFUSAL",
    "Answer",
    "AnswerConfig",
    "BM25Index",
    "Chunk",
    "ChunkConfig",
    "Citation",
    "CrossEncoderReranker",
    "Document",
    "Embedder",
    "EvalReport",
    "ExtractiveGenerator",
    "GoldQuestion",
    "HashingEmbedder",
    "LexicalReranker",
    "MMRReranker",
    "MemoryStore",
    "ModelGenerator",
    "PipelineConfig",
    "RAGPipeline",
    "RerankPipeline",
    "RetrievalConfig",
    "Retriever",
    "ScoredChunk",
    "TemplateGenerator",
    "TfidfEmbedder",
    "VectorStore",
    "answer_question",
    "build_embedder",
    "build_reranker",
    "chunk_document",
    "chunk_documents",
    "compare",
    "cosine",
    "evaluate_retrieval",
    "get_chunker",
    "load_corpus",
    "load_gold",
    "reciprocal_rank_fusion",
    "render_comparison",
    "verify_citations",
    "__version__",
]
