"""Embeddings, BM25, fusion, filtering and reranking."""
import pytest

from ragkit import (
    BM25Index,
    HashingEmbedder,
    LexicalReranker,
    MemoryStore,
    MMRReranker,
    RetrievalConfig,
    Retriever,
    TfidfEmbedder,
    build_embedder,
    cosine,
    reciprocal_rank_fusion,
)
from ragkit.rerank import RerankPipeline, build_reranker
from ragkit.stores.base import matches_filter
from ragkit.types import Chunk, ScoredChunk


def make_chunk(chunk_id: str, text: str, doc_id: str = "d", **metadata) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=doc_id, text=text, start=0, end=len(text),
        metadata=metadata,
    )


# --- embeddings --------------------------------------------------------------
def test_embeddings_are_deterministic():
    embedder = HashingEmbedder()
    assert embedder.embed("refund policy") == embedder.embed("refund policy")


def test_embeddings_are_normalised():
    vector = HashingEmbedder().embed("refund policy applies within 30 days")
    assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-5)


def test_related_text_scores_above_unrelated():
    embedder = HashingEmbedder()
    query = embedder.embed("refund window")
    related = embedder.embed("Customers may request a refund within 30 days")
    unrelated = embedder.embed("Kubernetes ingress controller configuration")
    assert cosine(query, related) > cosine(query, unrelated)


def test_character_ngrams_absorb_inflection():
    embedder = HashingEmbedder()
    assert cosine(embedder.embed("refund"), embedder.embed("refunds")) > 0.4


def test_tfidf_downweights_ubiquitous_terms():
    embedder = TfidfEmbedder(dimensions=128)
    embedder.fit(["the policy applies", "the policy is here", "the rare aardvark"])
    assert embedder.fitted
    assert embedder._idf["aardvark"] > embedder._idf["policy"]


def test_unfitted_tfidf_still_embeds():
    assert any(TfidfEmbedder().embed("anything at all"))


def test_empty_text_embeds_without_error():
    assert len(HashingEmbedder().embed("")) == 256


def test_unknown_embedder_is_rejected():
    with pytest.raises(ValueError, match="unknown embedder"):
        build_embedder("nope")


# --- BM25 --------------------------------------------------------------------
def _bm25() -> BM25Index:
    index = BM25Index()
    index.add(
        [
            make_chunk("a", "Refunds are issued within 30 days of purchase."),
            make_chunk("b", "Error E4021 indicates an expired session token."),
            make_chunk("c", "The default rate limit is 1000 requests per minute."),
        ]
    )
    return index


def test_bm25_finds_the_exact_token():
    results = _bm25().search("E4021")
    assert results[0].chunk.chunk_id == "b"


def test_bm25_returns_matched_terms():
    results = _bm25().search("refunds purchase")
    assert set(results[0].matched_terms) >= {"refunds", "purchase"}


def test_bm25_ignores_stopword_only_queries():
    assert _bm25().search("the and of") == []


def test_bm25_returns_nothing_for_an_absent_term():
    assert _bm25().search("kubernetes") == []


def test_bm25_respects_a_metadata_filter():
    index = BM25Index()
    index.add([make_chunk("a", "refund policy", doc_id="one"),
               make_chunk("b", "refund policy", doc_id="two")])
    results = index.search("refund", where={"doc_id": "two"})
    assert [r.chunk.chunk_id for r in results] == ["b"]


# --- fusion ------------------------------------------------------------------
def test_rrf_rewards_agreement_between_retrievers():
    shared = make_chunk("shared", "text")
    dense_only = make_chunk("dense-only", "text")
    lexical_only = make_chunk("lexical-only", "text")

    fused = reciprocal_rank_fusion(
        {
            "dense": [ScoredChunk(dense_only, 0.9), ScoredChunk(shared, 0.5)],
            "lexical": [ScoredChunk(lexical_only, 9.0), ScoredChunk(shared, 4.0)],
        }
    )
    # Ranked second by both beats ranked first by one: agreement is the signal.
    assert fused[0].chunk.chunk_id == "shared"


def test_rrf_is_insensitive_to_score_scale():
    """The reason fusion uses rank: BM25 and cosine live on different scales."""
    chunk_a, chunk_b = make_chunk("a", "x"), make_chunk("b", "y")
    small = reciprocal_rank_fusion(
        {"dense": [ScoredChunk(chunk_a, 0.4), ScoredChunk(chunk_b, 0.3)]}
    )
    huge = reciprocal_rank_fusion(
        {"dense": [ScoredChunk(chunk_a, 4000.0), ScoredChunk(chunk_b, 3000.0)]}
    )
    assert [c.chunk.chunk_id for c in small] == [c.chunk.chunk_id for c in huge]


def test_rrf_records_per_retriever_ranks():
    chunk = make_chunk("a", "x")
    fused = reciprocal_rank_fusion(
        {"dense": [ScoredChunk(chunk, 0.5)], "lexical": [ScoredChunk(chunk, 3.0)]}
    )
    assert fused[0].ranks == {"dense": 1, "lexical": 1}


def test_weights_shift_the_fusion():
    chunk_a, chunk_b = make_chunk("a", "x"), make_chunk("b", "y")
    runs = {
        "dense": [ScoredChunk(chunk_a, 1.0), ScoredChunk(chunk_b, 0.9)],
        "lexical": [ScoredChunk(chunk_b, 5.0), ScoredChunk(chunk_a, 1.0)],
    }
    dense_heavy = reciprocal_rank_fusion(runs, weights={"dense": 3.0, "lexical": 1.0})
    lexical_heavy = reciprocal_rank_fusion(runs, weights={"dense": 1.0, "lexical": 3.0})
    assert dense_heavy[0].chunk.chunk_id == "a"
    assert lexical_heavy[0].chunk.chunk_id == "b"


# --- store and filters -------------------------------------------------------
def test_memory_store_round_trip():
    embedder = HashingEmbedder()
    store = MemoryStore()
    chunk = make_chunk("a", "refund policy")
    chunk.embedding = embedder.embed(chunk.text)
    store.add([chunk])
    assert len(store) == 1
    assert store.get("a") is chunk
    assert store.search(embedder.embed("refund"), top_k=1)[0].chunk.chunk_id == "a"


def test_store_rejects_chunks_without_embeddings():
    with pytest.raises(ValueError, match="no embedding"):
        MemoryStore().add([make_chunk("a", "text")])


def test_store_upserts_rather_than_duplicating():
    embedder = HashingEmbedder()
    store = MemoryStore()
    for text in ("first", "second"):
        chunk = make_chunk("a", text)
        chunk.embedding = embedder.embed(text)
        store.add([chunk])
    assert len(store) == 1
    assert store.get("a").text == "second"


@pytest.mark.parametrize(
    "where,expected",
    [
        ({"doc_id": "one"}, True),
        ({"doc_id": "two"}, False),
        ({"doc_id": {"$in": ["one", "three"]}}, True),
        ({"doc_id": {"$ne": "two"}}, True),
        ({"year": {"$gte": 2024}}, True),
        ({"year": {"$lt": 2024}}, False),
        ({"missing": "x"}, False),
        (None, True),
    ],
)
def test_metadata_filters(where, expected):
    chunk = make_chunk("a", "text", doc_id="one", year=2025)
    assert matches_filter(chunk, where) is expected


def test_filter_type_mismatch_does_not_raise():
    chunk = make_chunk("a", "text", doc_id="one", label="x")
    assert matches_filter(chunk, {"label": {"$gt": 5}}) is False


def test_unsupported_operator_is_rejected():
    with pytest.raises(ValueError, match="unsupported filter operator"):
        matches_filter(make_chunk("a", "t"), {"doc_id": {"$regex": "x"}})


# --- retriever ---------------------------------------------------------------
def _retriever(mode: str = "hybrid") -> Retriever:
    embedder = HashingEmbedder()
    retriever = Retriever(
        store=MemoryStore(), embedder=embedder,
        config=RetrievalConfig(mode=mode, top_k=3, candidate_k=10),
    )
    retriever.index(
        [
            make_chunk("a", "Refunds are issued within 30 days of purchase.", doc_id="refund"),
            make_chunk("b", "Error E4021 indicates an expired session token.", doc_id="auth"),
            make_chunk("c", "The default rate limit is 1000 requests per minute.", doc_id="rate"),
            make_chunk("d", "Backups are retained for 35 days on a rolling basis.", doc_id="data"),
        ]
    )
    return retriever


def test_indexing_populates_both_retrievers():
    retriever = _retriever()
    assert len(retriever.store) == 4
    assert len(retriever.bm25) == 4


def test_hybrid_finds_an_exact_token():
    assert _retriever().retrieve("E4021")[0].chunk.chunk_id == "b"


def test_hybrid_results_carry_both_components():
    results = _retriever().retrieve("refunds within 30 days")
    assert "dense" in results[0].components
    assert "lexical" in results[0].components


def test_dense_mode_uses_only_the_vector_store():
    results = _retriever("dense").retrieve("refund")
    assert all("lexical" not in r.components for r in results)


def test_retriever_respects_top_k():
    assert len(_retriever().retrieve("days", top_k=2)) == 2


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        _retriever().retrieve("x", mode="telepathy")


def test_max_per_doc_prevents_one_document_dominating():
    embedder = HashingEmbedder()
    retriever = Retriever(
        store=MemoryStore(), embedder=embedder,
        config=RetrievalConfig(mode="dense", top_k=5, candidate_k=10, max_per_doc=1),
    )
    retriever.index(
        [make_chunk(f"a{i}", f"refund policy part {i}", doc_id="same") for i in range(4)]
        + [make_chunk("b", "refund policy elsewhere", doc_id="other")]
    )
    doc_ids = [r.chunk.doc_id for r in retriever.retrieve("refund policy")]
    assert doc_ids.count("same") == 1


# --- rerankers ---------------------------------------------------------------
def test_lexical_reranker_prefers_full_term_coverage():
    covers_all = ScoredChunk(
        make_chunk("all", "The refund window is 30 days from purchase."), 0.02
    )
    covers_one = ScoredChunk(
        make_chunk("one", "refund refund refund refund refund refund"), 0.03
    )
    ranked = LexicalReranker().rerank("refund window days", [covers_one, covers_all])
    assert ranked[0].chunk.chunk_id == "all"


def test_lexical_reranker_records_its_components():
    result = LexicalReranker().rerank(
        "refund window", [ScoredChunk(make_chunk("a", "refund window here"), 0.5)]
    )[0]
    assert {"coverage", "proximity", "retrieval"} <= set(result.components)


def test_mmr_removes_near_duplicates():
    duplicate = "The refund window is 30 days from the date of purchase."
    results = [
        ScoredChunk(make_chunk("a", duplicate), 0.9),
        ScoredChunk(make_chunk("b", duplicate), 0.89),
        ScoredChunk(make_chunk("c", "Rate limits are 1000 requests per minute."), 0.5),
    ]
    ranked = MMRReranker(lambda_param=0.5).rerank("refund", results, top_k=2)
    assert [r.chunk.chunk_id for r in ranked] == ["a", "c"]


def test_mmr_with_lambda_one_preserves_relevance_order():
    results = [
        ScoredChunk(make_chunk("a", "same text here"), 0.9),
        ScoredChunk(make_chunk("b", "same text here"), 0.8),
    ]
    ranked = MMRReranker(lambda_param=1.0).rerank("text", results)
    assert [r.chunk.chunk_id for r in ranked] == ["a", "b"]


def test_mmr_rejects_an_out_of_range_lambda():
    with pytest.raises(ValueError, match="between 0 and 1"):
        MMRReranker(lambda_param=1.5)


def test_pipeline_applies_stages_in_order():
    pipeline = RerankPipeline([LexicalReranker(), MMRReranker(0.9)])
    results = [
        ScoredChunk(make_chunk("a", "refund window is 30 days"), 0.5),
        ScoredChunk(make_chunk("b", "unrelated content entirely"), 0.4),
    ]
    ranked = pipeline.rerank("refund window", results, top_k=2)
    assert ranked[0].chunk.chunk_id == "a"
    assert pipeline.describe()["stages"][0]["name"] == "lexical"


def test_build_reranker_none_disables_reranking():
    assert build_reranker("none") is None


def test_unknown_reranker_is_rejected():
    with pytest.raises(ValueError, match="unknown reranker"):
        build_reranker("magic")
