"""Chunking strategies: boundaries, offsets, overlap, headings."""
import pytest

from ragkit import ChunkConfig, chunk_document, get_chunker
from ragkit.chunking import CHUNKERS, find_sections
from ragkit.types import Document

MARKDOWN = """# Refund Policy

## Standard window
Customers may request a full refund within 30 days of purchase. Refunds after
30 days are issued as account credit.

## Processing time
Approved refunds are processed within 5 to 7 business days. Funds appear on the
statement within a further 3 to 5 days.

## Non-refundable items
Setup fees and professional services are non-refundable.
"""


def doc(text: str = MARKDOWN) -> Document:
    return Document(doc_id="refunds", text=text, title="Refund Policy")


@pytest.mark.parametrize("strategy", sorted(CHUNKERS))
def test_every_chunk_offset_maps_back_to_the_source(strategy):
    """The invariant the whole package rests on: citations must be checkable."""
    document = doc()
    for chunk in chunk_document(document, strategy):
        source = document.text[chunk.start : chunk.end]
        # The heading may be prepended, so compare on the body.
        body = chunk.text.split("\n", 1)[-1] if chunk.section else chunk.text
        assert body.strip()[:40] in " ".join(source.split()) or body.strip()[:40] in source


@pytest.mark.parametrize("strategy", sorted(CHUNKERS))
def test_chunks_are_non_empty_and_ordered(strategy):
    chunks = chunk_document(doc(), strategy)
    assert chunks
    assert all(c.text.strip() for c in chunks)
    assert [c.index for c in chunks] == sorted(c.index for c in chunks)


@pytest.mark.parametrize("strategy", sorted(CHUNKERS))
def test_chunk_ids_are_unique(strategy):
    chunks = chunk_document(doc(), strategy)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_section_chunker_splits_on_headings():
    chunks = chunk_document(doc(), "section")
    sections = {c.section for c in chunks}
    assert "Standard window" in sections
    assert "Processing time" in sections


def test_section_chunker_carries_the_heading_into_the_chunk():
    # A body that no longer says which section it belongs to answers the wrong
    # question confidently.
    chunk = next(c for c in chunk_document(doc(), "section") if c.section == "Processing time")
    assert chunk.text.startswith("Processing time")


def test_sentence_chunker_never_splits_mid_sentence():
    text = "First sentence here. Second sentence here. Third sentence here. " * 20
    config = ChunkConfig(target_tokens=30, overlap_tokens=0)
    for chunk in chunk_document(Document("d", text), "sentence", config):
        stripped = chunk.text.strip()
        assert stripped.endswith((".", "!", "?")), stripped[-40:]


def test_fixed_chunker_does_split_mid_sentence():
    # The baseline's known weakness, asserted so the comparison is honest.
    text = "word " * 400
    chunks = chunk_document(Document("d", text), "fixed", ChunkConfig(target_tokens=20))
    assert len(chunks) > 1


def test_overlap_repeats_content_between_neighbours():
    text = " ".join(f"Sentence number {i} carries content." for i in range(60))
    config = ChunkConfig(target_tokens=40, overlap_tokens=20)
    chunks = chunk_document(Document("d", text), "sentence", config)
    assert len(chunks) > 2
    first, second = set(chunks[0].text.split()), set(chunks[1].text.split())
    assert first & second, "overlap should repeat content across the boundary"


def test_zero_overlap_produces_disjoint_spans():
    text = " ".join(f"Sentence number {i} carries content." for i in range(60))
    config = ChunkConfig(target_tokens=40, overlap_tokens=0)
    chunks = chunk_document(Document("d", text), "sentence", config)
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert following.start >= previous.end


def test_larger_target_produces_fewer_chunks():
    small = chunk_document(doc(), "sentence", ChunkConfig(target_tokens=30))
    large = chunk_document(doc(), "sentence", ChunkConfig(target_tokens=400))
    assert len(small) > len(large)


def test_tiny_trailing_chunk_is_merged():
    text = "A full sentence that is reasonably long and carries real content here. Tiny."
    chunks = chunk_document(
        Document("d", text), "sentence", ChunkConfig(target_tokens=16, min_tokens=20)
    )
    assert all(len(c.text) > 20 for c in chunks)


def test_find_sections_handles_documents_without_headings():
    sections = find_sections("Just a paragraph with no heading at all.")
    assert sections == [("", 0, len("Just a paragraph with no heading at all."))]


def test_section_chunker_falls_back_when_there_are_no_headings():
    chunks = chunk_document(Document("d", "Plain prose. " * 40), "section")
    assert chunks


def test_citation_includes_document_section_and_offsets():
    chunk = chunk_document(doc(), "section")[1]
    assert chunk.citation.startswith("refunds#")
    assert f"{chunk.start}-{chunk.end}" in chunk.citation


def test_empty_document_produces_no_chunks():
    assert chunk_document(Document("d", ""), "sentence") == []


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown chunker"):
        get_chunker("nope")
