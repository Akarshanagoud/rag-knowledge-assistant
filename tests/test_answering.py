"""Answer generation, citation verification, abstention and evaluation."""
import json
from pathlib import Path

import pytest

from ragkit import (
    REFUSAL,
    AnswerConfig,
    ExtractiveGenerator,
    PipelineConfig,
    RAGPipeline,
    TemplateGenerator,
    answer_question,
    load_corpus,
    load_gold,
    verify_citations,
)
from ragkit.answering import coverage, support_score
from ragkit.evaluation import (
    evaluate_retrieval,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragkit.types import Chunk, ScoredChunk

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
GOLD = ROOT / "data" / "gold_questions.json"


def scored(
    chunk_id: str, text: str, score: float = 0.5, doc_id: str = "d",
    dense: float = 0.02,
) -> ScoredChunk:
    """A retrieved chunk.

    ``dense`` defaults low so the semantic escape hatch stays closed unless a
    test deliberately opens it; otherwise every fixture would silently bypass
    the abstention gate under test.
    """
    chunk = Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text, start=0, end=len(text))
    return ScoredChunk(chunk=chunk, score=score, components={"dense": dense})


# --- metrics -----------------------------------------------------------------
def test_recall_and_precision():
    retrieved = ["a", "b", "c"]
    assert recall_at_k(retrieved, {"a", "z"}, 3) == 0.5
    assert precision_at_k(retrieved, {"a"}, 3) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg_is_one_when_perfectly_ranked():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)


def test_ndcg_penalises_a_lower_rank():
    top = ndcg_at_k(["a", "x"], {"a"}, 2)
    lower = ndcg_at_k(["x", "a"], {"a"}, 2)
    assert top > lower


def test_metrics_with_no_relevant_documents():
    assert recall_at_k(["a"], set(), 3) == 0.0
    assert ndcg_at_k(["a"], set(), 3) == 0.0


# --- support and coverage ----------------------------------------------------
def test_support_score_measures_overlap():
    assert support_score("refund within 30 days", "The refund is within 30 days.") == 1.0
    assert support_score("kubernetes ingress", "The refund is within 30 days.") == 0.0


def test_coverage_allows_inflection():
    chunks = [scored("a", "Refunds are indicated in the policy document.")]
    assert coverage("refund indicate", chunks) == pytest.approx(1.0)


def test_coverage_does_not_match_unrelated_prefixes():
    # "plan" must not match "planet": a prefix is only a stem when the suffix
    # is short enough to be an inflection.
    chunks = [scored("a", "The planetarium opens at noon.")]
    assert coverage("plan", chunks) == 0.0


# --- generation --------------------------------------------------------------
def test_extractive_answer_is_copied_from_the_source():
    chunks = [scored("a", "Customers may request a refund within 30 days of purchase.")]
    answer = answer_question("How long to request a refund?", chunks)
    assert "30 days" in answer.text
    assert not answer.abstained
    # Every word came from the chunk, so grounding is structural.
    assert answer.grounded


def test_extractive_answer_carries_citation_markers():
    chunks = [scored("a", "Refunds are processed within 5 to 7 business days.")]
    answer = answer_question("How long do refunds take to process?", chunks)
    assert "[1]" in answer.text
    assert answer.citations
    assert answer.citations[0].chunk_id == "a"


def test_template_generator_returns_the_top_passage():
    chunks = [scored("a", "Setup fees are non-refundable.")]
    answer = answer_question(
        "Are setup fees refundable?", chunks, generator=TemplateGenerator()
    )
    assert answer.text.startswith("Setup fees are non-refundable.")


def test_irrelevant_sentences_are_not_stitched_in():
    chunks = [
        scored("a", "Customers may request a refund within 30 days of purchase.", 0.9),
        scored("b", "API tokens do not expire automatically but can be revoked.", 0.8),
    ]
    answer = answer_question("How long to request a refund?", chunks)
    assert "API tokens" not in answer.text


# --- abstention --------------------------------------------------------------
def test_abstains_when_nothing_was_retrieved():
    answer = answer_question("anything", [])
    assert answer.abstained
    assert answer.text == REFUSAL


def test_abstains_when_coverage_is_too_low():
    chunks = [scored("a", "Refunds are issued within 30 days of purchase.", 0.02)]
    answer = answer_question("How do I configure the Kubernetes ingress?", chunks)
    assert answer.abstained
    assert "cover enough" in answer.metadata["reason"]


def test_a_strong_semantic_match_overrides_weak_lexical_coverage():
    # Coverage is 0.25 (only "money" appears), below the 0.34 gate, but the
    # dense score is decisive - the paraphrase case the escape hatch exists for.
    chunk = scored(
        "a",
        "Customers get money returned to the original payment method.",
        score=0.9,
        dense=0.9,
    )
    question = "How do I get my money back after buying the wrong plan?"

    gated = answer_question(question, [chunk], config=AnswerConfig(semantic_floor=0.99))
    assert gated.abstained
    assert "cover enough" in gated.metadata["reason"]

    allowed = answer_question(question, [chunk], config=AnswerConfig(semantic_floor=0.5))
    assert not allowed.abstained


def test_abstention_reports_the_numbers_behind_the_decision():
    answer = answer_question(
        "kubernetes ingress controller", [scored("a", "refunds within 30 days")]
    )
    assert answer.abstained
    assert "coverage" in answer.metadata
    assert "semantic" in answer.metadata
    assert "min_coverage" in answer.metadata


def test_score_floor_is_relative_not_absolute():
    """A regression guard: an absolute floor once abstained on 100% of queries.

    Raw RRF scores are around 0.03 while reranked scores are near 1.0, so an
    absolute threshold tuned for one configuration rejects everything under
    the other.
    """
    tiny = scored("a", "Customers may request a refund within 30 days.", score=0.0164)
    answer = answer_question("refund request window", [tiny])
    assert not answer.abstained
    assert "30 days" in answer.text


# --- citation verification ---------------------------------------------------
def test_uncited_sentence_marks_the_answer_ungrounded():
    chunks = [scored("a", "Refunds take 30 days.")]
    citations, grounded = verify_citations(
        "Refunds take 30 days. And something invented.", chunks, AnswerConfig()
    )
    assert not grounded


def test_citation_to_a_nonexistent_source_is_ungrounded():
    chunks = [scored("a", "Refunds take 30 days.")]
    _citations, grounded = verify_citations(
        "Refunds take 30 days. [7]", chunks, AnswerConfig()
    )
    assert not grounded


def test_unsupported_claim_is_ungrounded():
    chunks = [scored("a", "Refunds take 30 days.")]
    _citations, grounded = verify_citations(
        "The service runs on Kubernetes with an ingress controller. [1]",
        chunks,
        AnswerConfig(),
    )
    assert not grounded


def test_supported_claim_is_grounded():
    chunks = [scored("a", "Refunds are processed within 30 days of the request.")]
    citations, grounded = verify_citations(
        "Refunds are processed within 30 days. [1]", chunks, AnswerConfig()
    )
    assert grounded
    assert citations[0].support > 0.8


def test_refusal_counts_as_grounded():
    _citations, grounded = verify_citations(REFUSAL, [], AnswerConfig())
    assert grounded


# --- end to end --------------------------------------------------------------
@pytest.fixture(scope="module")
def pipeline() -> RAGPipeline:
    p = RAGPipeline()
    p.add_documents(load_corpus(CORPUS))
    return p


def test_corpus_indexes(pipeline):
    stats = pipeline.stats()
    assert stats["documents"] == 6
    assert stats["chunks"] > 10


def test_answers_a_direct_question(pipeline):
    answer = pipeline.ask("How long do I have to request a refund?")
    assert "30 days" in answer.text
    assert answer.grounded
    assert any("refund-policy" in s for s in answer.sources)


def test_answers_an_exact_token_question(pipeline):
    answer = pipeline.ask("What does error E4021 mean?")
    assert "E4021" in answer.text
    assert any("authentication" in s for s in answer.sources)


def test_declines_a_question_the_corpus_cannot_answer(pipeline):
    answer = pipeline.ask("What is the company holiday schedule for next year?")
    assert answer.abstained
    assert answer.text == REFUSAL


def test_every_citation_points_at_real_source_text(pipeline):
    answer = pipeline.ask("How long are audit records kept?")
    for citation in answer.citations:
        chunk = pipeline.store.get(citation.chunk_id)
        assert chunk is not None
        document = pipeline.documents[chunk.doc_id]
        assert document.text[chunk.start : chunk.end].strip()


def test_metadata_filter_restricts_retrieval(pipeline):
    results = pipeline.retrieve("policy", top_k=5, where={"doc_id": "rate-limits"})
    assert results
    assert all(r.chunk.doc_id == "rate-limits" for r in results)


def test_gold_set_is_well_formed():
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    doc_ids = {p.stem for p in CORPUS.glob("*.md")}
    for item in gold:
        for doc_id in item.get("relevant_doc_ids", []):
            assert doc_id in doc_ids, f"gold references unknown document {doc_id}"
        if item.get("unanswerable"):
            assert not item.get("relevant_doc_ids")


def test_published_evaluation_numbers_hold(pipeline):
    """The README's headline metrics are enforced, not just claimed."""
    report = evaluate_retrieval(
        pipeline, load_gold(GOLD), k=5, label="default", with_answers=True
    )
    assert report.recall >= 0.95
    assert report.mrr >= 0.85
    assert report.abstention_accuracy == 1.0
    assert report.grounded_rate == 1.0
    assert report.answer_accuracy >= 0.9
    # The measured cost of a conservative gate, held to a documented ceiling.
    assert report.false_abstention_rate <= 0.25


def test_reranking_can_be_disabled_explicitly():
    """`reranker=None` must mean off, not 'fall back to the default'."""
    config = PipelineConfig()
    config.reranker = "default"
    assert RAGPipeline(config).reranker is not None
    assert RAGPipeline(config, reranker=None).reranker is None


def test_extractive_generator_refuses_an_empty_question():
    assert ExtractiveGenerator().generate("", [scored("a", "text")], AnswerConfig()) == REFUSAL
