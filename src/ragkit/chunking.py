"""Chunking strategies.

Chunking is where most RAG quality is won or lost, and it is usually the least
examined part of the stack. Three failure modes matter:

* **Chunks that split a fact in half.** "The refund window is" | "30 days."
  Neither chunk answers the question, so retrieval cannot either.
* **Chunks that bury a fact in noise.** A 2000-token chunk retrieves on its
  dominant topic; a fact mentioned once inside it is invisible to the scorer.
* **Chunks that lose their heading.** A section body that no longer says which
  policy it belongs to answers the wrong question confidently.

So every strategy here respects natural boundaries where it can, carries the
heading into the chunk, and always preserves character offsets so a citation
can be checked against the source.

The strategies are directly comparable through the evaluation harness, which
is the point: "which chunker?" should be a measurement, not a preference.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .types import Chunk, Document, sentences

#: Rough characters-per-token for English prose. Used for size targets only.
CHARS_PER_TOKEN = 4


@dataclass
class ChunkConfig:
    #: Target chunk size in tokens.
    target_tokens: int = 220
    #: Overlap between adjacent chunks, in tokens. Overlap exists so a fact
    #: sitting on a boundary appears whole in at least one chunk.
    overlap_tokens: int = 40
    #: Chunks shorter than this are merged into their neighbour rather than
    #: emitted: a 12-token fragment retrieves noisily and answers nothing.
    min_tokens: int = 40
    #: Prefix each chunk with its section heading.
    include_heading: bool = True

    @property
    def target_chars(self) -> int:
        return self.target_tokens * CHARS_PER_TOKEN

    @property
    def overlap_chars(self) -> int:
        return self.overlap_tokens * CHARS_PER_TOKEN

    @property
    def min_chars(self) -> int:
        return self.min_tokens * CHARS_PER_TOKEN


class Chunker(Protocol):
    name: str

    def split(self, document: Document, config: ChunkConfig) -> list[Chunk]:
        ...


# Markdown-style headings and underlined/uppercase section titles.
_HEADING = re.compile(
    r"(?m)^(?:(#{1,6})\s+(?P<hash>.+)|(?P<upper>[A-Z][A-Z0-9 ,'/&()-]{3,60})\s*$)"
)


def find_sections(text: str) -> list[tuple[str, int, int]]:
    """Return ``(heading, body_start, body_end)`` for each section."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", 0, len(text))]

    sections: list[tuple[str, int, int]] = []
    if matches[0].start() > 0:
        sections.append(("", 0, matches[0].start()))

    for i, match in enumerate(matches):
        heading = (match.group("hash") or match.group("upper") or "").strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((heading, body_start, body_end))
    return sections


def _make_chunk(
    document: Document, text: str, start: int, end: int, index: int, section: str,
    config: ChunkConfig,
) -> Chunk:
    body = text
    if config.include_heading and section and not body.lstrip().startswith(section):
        # The heading travels with the body so the chunk still says what it is
        # about when it is retrieved in isolation.
        body = f"{section}\n{body}"
    return Chunk(
        chunk_id=f"{document.doc_id}::{index}",
        doc_id=document.doc_id,
        text=body.strip(),
        start=start,
        end=end,
        index=index,
        title=document.title,
        section=section,
        source=document.source,
        metadata=dict(document.metadata),
    )


class FixedChunker:
    """Fixed-size character windows with overlap.

    The baseline. Fast, predictable, and it will happily cut a sentence in
    half — which is exactly why it is worth having in the comparison.
    """

    name = "fixed"

    def split(self, document: Document, config: ChunkConfig) -> list[Chunk]:
        text = document.text
        chunks: list[Chunk] = []
        step = max(config.target_chars - config.overlap_chars, 1)
        index = 0
        for start in range(0, max(len(text), 1), step):
            end = min(start + config.target_chars, len(text))
            piece = text[start:end]
            if not piece.strip():
                continue
            chunks.append(_make_chunk(document, piece, start, end, index, "", config))
            index += 1
            if end >= len(text):
                break
        return chunks


class SentenceChunker:
    """Packs whole sentences up to the target size.

    Never splits mid-sentence, so a fact is never cut in half. Overlap is
    applied by repeating trailing sentences into the next chunk.
    """

    name = "sentence"

    def split(self, document: Document, config: ChunkConfig) -> list[Chunk]:
        parts = sentences(document.text)
        if not parts:
            return []

        chunks: list[Chunk] = []
        current: list[tuple[str, int, int]] = []
        current_chars = 0
        index = 0

        for sentence in parts:
            sentence_chars = len(sentence[0])
            if current and current_chars + sentence_chars > config.target_chars:
                chunks.append(self._emit(document, current, index, config))
                index += 1
                current = self._overlap_tail(current, config)
                current_chars = sum(len(s[0]) for s in current)
            current.append(sentence)
            current_chars += sentence_chars

        if current:
            chunks.append(self._emit(document, current, index, config))
        return self._merge_small(chunks, config)

    @staticmethod
    def _overlap_tail(
        current: list[tuple[str, int, int]], config: ChunkConfig
    ) -> list[tuple[str, int, int]]:
        tail: list[tuple[str, int, int]] = []
        total = 0
        for sentence in reversed(current):
            if total >= config.overlap_chars:
                break
            tail.insert(0, sentence)
            total += len(sentence[0])
        return tail

    def _emit(
        self, document: Document, group: list[tuple[str, int, int]], index: int,
        config: ChunkConfig,
    ) -> Chunk:
        start, end = group[0][1], group[-1][2]
        return _make_chunk(document, document.text[start:end], start, end, index, "", config)

    @staticmethod
    def _merge_small(chunks: list[Chunk], config: ChunkConfig) -> list[Chunk]:
        """Fold a runt trailing chunk back into its predecessor."""
        if len(chunks) < 2:
            return chunks
        last = chunks[-1]
        if len(last.text) >= config.min_chars:
            return chunks
        previous = chunks[-2]
        merged = Chunk(
            chunk_id=previous.chunk_id,
            doc_id=previous.doc_id,
            text=f"{previous.text} {last.text}".strip(),
            start=previous.start,
            end=last.end,
            index=previous.index,
            title=previous.title,
            section=previous.section,
            source=previous.source,
            metadata=previous.metadata,
        )
        return chunks[:-2] + [merged]


class SectionChunker:
    """Splits on headings first, then packs sentences within each section.

    This is the default. Documents that have structure already tell you where
    the semantic boundaries are; ignoring that in favour of a character count
    is throwing away free signal.
    """

    name = "section"

    def split(self, document: Document, config: ChunkConfig) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0

        for heading, body_start, body_end in find_sections(document.text):
            body = document.text[body_start:body_end]
            if not body.strip():
                continue

            for piece_start, piece_end in self._pack(body, config):
                absolute_start = body_start + piece_start
                absolute_end = body_start + piece_end
                text = document.text[absolute_start:absolute_end]
                if not text.strip():
                    continue
                chunks.append(
                    _make_chunk(
                        document, text, absolute_start, absolute_end, index, heading, config
                    )
                )
                index += 1

        return chunks or FixedChunker().split(document, config)

    @staticmethod
    def _pack(body: str, config: ChunkConfig) -> list[tuple[int, int]]:
        parts = sentences(body)
        if not parts:
            return [(0, len(body))]

        spans: list[tuple[int, int]] = []
        group: list[tuple[str, int, int]] = []
        chars = 0
        for sentence in parts:
            if group and chars + len(sentence[0]) > config.target_chars:
                spans.append((group[0][1], group[-1][2]))
                tail: list[tuple[str, int, int]] = []
                total = 0
                for previous in reversed(group):
                    if total >= config.overlap_chars:
                        break
                    tail.insert(0, previous)
                    total += len(previous[0])
                group = tail
                chars = total
            group.append(sentence)
            chars += len(sentence[0])
        if group:
            spans.append((group[0][1], group[-1][2]))
        return spans


class ParagraphChunker:
    """Blank-line separated paragraphs, merged up to the target size.

    Good for prose without headings, where a paragraph is the natural unit and
    a sentence is too small to stand alone.
    """

    name = "paragraph"

    def split(self, document: Document, config: ChunkConfig) -> list[Chunk]:
        text = document.text
        chunks: list[Chunk] = []
        index = 0
        buffer_start: int | None = None
        buffer_end = 0

        for match in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
            start, end = match.span()
            if buffer_start is None:
                buffer_start = start
            buffer_end = end

            if buffer_end - buffer_start >= config.target_chars:
                chunks.append(
                    _make_chunk(
                        document, text[buffer_start:buffer_end], buffer_start, buffer_end,
                        index, "", config,
                    )
                )
                index += 1
                buffer_start = None

        if buffer_start is not None and text[buffer_start:buffer_end].strip():
            chunks.append(
                _make_chunk(
                    document, text[buffer_start:buffer_end], buffer_start, buffer_end,
                    index, "", config,
                )
            )
        return chunks


CHUNKERS: dict[str, Chunker] = {
    "fixed": FixedChunker(),
    "sentence": SentenceChunker(),
    "section": SectionChunker(),
    "paragraph": ParagraphChunker(),
}


def get_chunker(name: str) -> Chunker:
    if name not in CHUNKERS:
        raise ValueError(f"unknown chunker {name!r}; choose from {sorted(CHUNKERS)}")
    return CHUNKERS[name]


def chunk_document(
    document: Document, strategy: str = "section", config: ChunkConfig | None = None
) -> list[Chunk]:
    return get_chunker(strategy).split(document, config or ChunkConfig())


def chunk_documents(
    documents: list[Document], strategy: str = "section", config: ChunkConfig | None = None
) -> list[Chunk]:
    settings = config or ChunkConfig()
    return [c for document in documents for c in chunk_document(document, strategy, settings)]
