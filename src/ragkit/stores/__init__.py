"""Vector store backends: in-memory (exact), FAISS and Chroma."""
from .base import VectorStore, matches_filter
from .memory import MemoryStore

__all__ = ["MemoryStore", "VectorStore", "matches_filter"]
