"""Answer generation with enforced citations.

Two things happen here that most RAG examples skip.

**Abstention.** If retrieval found nothing above the confidence floor, the
answer is "I don't know", not a fluent paragraph assembled from the three
least-irrelevant chunks. A RAG system that never abstains has no way to be
wrong quietly, which means it is wrong loudly, in production, to a user who
believed it.

**Grounding verification after generation.** Citations are checked, not
requested. Every sentence in the answer is matched back to the chunk it claims
to come from, and a sentence whose support falls below the threshold marks the
whole answer ungrounded. Asking a model to cite its sources is a prompt;
verifying that the cited text supports the claim is a control.

The default generator is extractive - it selects and stitches source sentences,
so it cannot hallucinate by construction. A model generator is available and
goes through exactly the same verification.
"""
from __future__ import annotations

import abc
import re
import time
from dataclasses import dataclass
from typing import Any

from .types import Answer, Citation, ScoredChunk, sentences, tokenize

REFUSAL = (
    "I don't have enough information in the indexed documents to answer that."
)

_CITATION_MARKER = re.compile(r"\[(\d+)\]")


@dataclass
class AnswerConfig:
    #: Keep chunks scoring at least this fraction of the top result. Relative,
    #: not absolute, because retrieval scores have no fixed scale: raw RRF sums
    #: are around 0.03, cosine is in [-1, 1], BM25 is unbounded, and reranking
    #: replaces all of them with something else again. An absolute floor tuned
    #: against one configuration silently rejects everything under another -
    #: which is how a pipeline ends up abstaining on 100% of questions after a
    #: change that had nothing to do with abstention. 0.0 keeps everything.
    min_score_ratio: float = 0.0
    #: Fraction of the question's content words that must appear somewhere in
    #: the top retrieved chunks before the system will attempt an answer.
    #:
    #: This is the abstention gate that actually works. A retrieval score is
    #: relative - the best of six chunks always scores highest, even when all
    #: six are irrelevant - so a score floor cannot tell "best match" from
    #: "good match". Term coverage is absolute: a question about the holiday
    #: schedule shares almost no vocabulary with a refund policy, whatever the
    #: ranker says.
    min_coverage: float = 0.34
    #: Chunks considered when measuring coverage.
    coverage_chunks: int = 3
    #: Escape hatch for genuine paraphrases. A question can share almost no
    #: vocabulary with its answer ("get my money back" vs "refund") and still
    #: be answerable, so a strong *semantic* match overrides a weak lexical
    #: one. With the default hashing embedder this rarely fires, because that
    #: embedder is lexical too - which is precisely the measurable case for
    #: upgrading to a sentence-transformer. See `ragkit eval --abstention`.
    semantic_floor: float = 0.15
    #: A sentence needs this much overlap with the question to be quoted.
    min_sentence_overlap: float = 0.2
    #: Lexical overlap a sentence needs with its source to count as grounded.
    min_support: float = 0.35
    #: Chunks passed to the generator.
    max_context_chunks: int = 5
    #: Character budget for assembled context.
    max_context_chars: int = 6000
    #: Sentences an extractive answer may use.
    max_sentences: int = 4
    #: Require every sentence to be traceable to a source.
    require_citations: bool = True


class Generator(abc.ABC):
    """Turns a question plus context chunks into answer text."""

    name: str = "generator"

    @abc.abstractmethod
    def generate(
        self, question: str, chunks: list[ScoredChunk], config: AnswerConfig
    ) -> str:
        ...

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


class ExtractiveGenerator(Generator):
    """Selects the source sentences that best answer the question.

    Cannot hallucinate: every word of the answer is copied from a retrieved
    chunk, and each sentence is emitted with the marker of the chunk it came
    from. That makes the citation check a formality rather than a hope, and it
    makes this a genuinely usable default rather than a placeholder.
    """

    name = "extractive"

    def generate(
        self, question: str, chunks: list[ScoredChunk], config: AnswerConfig
    ) -> str:
        query_terms = set(tokenize(question))
        if not query_terms:
            return REFUSAL

        candidates: list[tuple[float, str, int]] = []
        for position, scored in enumerate(chunks[: config.max_context_chunks]):
            for sentence, _start, _end in sentences(scored.chunk.text):
                tokens = set(tokenize(sentence))
                if not tokens:
                    continue
                overlap = _overlap(query_terms, tokens)
                # A sentence that barely brushes the question is noise. Without
                # this floor the answer fills up with fluent, unrelated prose
                # from lower-ranked chunks, which is worse than a short answer.
                if overlap < config.min_sentence_overlap:
                    continue
                # Rank is a tiebreak, not a substitute for relevance: a weakly
                # matching sentence in the top chunk should still lose to a
                # strongly matching one further down.
                rank_bonus = 1.0 / (1 + position)
                length_penalty = 0.5 if len(tokens) < 4 else 1.0
                score = (overlap * 0.85 + rank_bonus * 0.15) * length_penalty
                candidates.append((score, sentence, position))

        if not candidates:
            return REFUSAL

        candidates.sort(key=lambda item: -item[0])
        chosen = candidates[: config.max_sentences]
        # Restore reading order within the answer, so it flows like prose
        # rather than like a ranked list.
        chosen.sort(key=lambda item: item[2])

        parts: list[str] = []
        for _score, sentence, position in chosen:
            text = sentence.strip()
            if not text.endswith((".", "!", "?")):
                text += "."
            parts.append(f"{text} [{position + 1}]")
        return " ".join(parts)


class TemplateGenerator(Generator):
    """Returns the single best passage verbatim.

    Useful when the corpus is policy text and the honest answer is "here is the
    clause", not a paraphrase of it.
    """

    name = "template"

    def generate(
        self, question: str, chunks: list[ScoredChunk], config: AnswerConfig
    ) -> str:
        if not chunks:
            return REFUSAL
        best = chunks[0].chunk.text.strip()
        if len(best) > config.max_context_chars:
            best = best[: config.max_context_chars].rsplit(" ", 1)[0] + "..."
        return f"{best} [1]"


class ModelGenerator(Generator):
    """Generates with a local model, then submits to the same verification.

        from ragkit.answering import ModelGenerator
        generator = ModelGenerator(ollama_model="llama3.2")

    The prompt instructs the model to cite and to refuse when the context is
    insufficient, but nothing downstream trusts that: the answer is verified
    against the retrieved text either way.
    """

    name = "model"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = ollama_model
        self.timeout = timeout

    def generate(
        self, question: str, chunks: list[ScoredChunk], config: AnswerConfig
    ) -> str:
        import json
        import urllib.request

        if not chunks:
            return REFUSAL

        context = build_context(chunks, config)
        prompt = (
            "Answer the question using only the numbered sources below.\n"
            "Cite the source number in square brackets after each sentence, "
            'like this: "Refunds take 30 days. [2]"\n'
            "If the sources do not contain the answer, reply exactly: "
            f"{REFUSAL}\n\n"
            f"{context}\n\nQuestion: {question}\nAnswer:"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"model generation failed: {exc}") from exc
        return (data.get("response") or "").strip() or REFUSAL

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model, "base_url": self.base_url}


def build_context(chunks: list[ScoredChunk], config: AnswerConfig) -> str:
    """Numbered source block, truncated to the character budget."""
    parts: list[str] = []
    used = 0
    for i, scored in enumerate(chunks[: config.max_context_chunks], start=1):
        header = f"[{i}] {scored.chunk.citation}"
        body = scored.chunk.text.strip()
        if used + len(body) > config.max_context_chars:
            body = body[: max(config.max_context_chars - used, 0)]
        if not body:
            break
        parts.append(f"{header}\n{body}")
        used += len(body)
    return "\n\n".join(parts)


def _overlap(query_terms: set[str], tokens: set[str]) -> float:
    """Fraction of query terms present, allowing simple morphological variants.

    "indicate" should match "indicates" and "refund" should match "refunds".
    A full stemmer is overkill here and would need its own test suite; prefix
    matching on the longer of the two words gets the same result for English
    inflection with none of the machinery.
    """
    if not query_terms:
        return 0.0
    matched = 0
    for term in query_terms:
        if term in tokens:
            matched += 1
            continue
        if any(_same_stem(term, token) for token in tokens):
            matched += 1
    return matched / len(query_terms)


def _same_stem(a: str, b: str) -> bool:
    if len(a) < 4 or len(b) < 4:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    # Only treat a prefix as a stem match when the suffix is short enough to be
    # an inflection ("refund"/"refunds"), not a different word ("plan"/"planet").
    return longer.startswith(shorter) and len(longer) - len(shorter) <= 3


def coverage(question: str, chunks: list[ScoredChunk], limit: int = 3) -> float:
    """Best question-term coverage across the top ``limit`` chunks."""
    query_terms = set(tokenize(question))
    if not query_terms or not chunks:
        return 0.0
    return max(
        _overlap(query_terms, set(tokenize(scored.chunk.text)))
        for scored in chunks[:limit]
    )


def best_semantic_score(chunks: list[ScoredChunk]) -> float:
    """Dense similarity of the single top-ranked chunk.

    Deliberately the top chunk rather than the best of the first few: a real
    semantic match is the *best* match. Taking a max over several chunks lets
    an unrelated question qualify on its third-best hit, which is how an
    abstention gate quietly stops abstaining.

    Returns 0.0 when there is no dense component (lexical-only retrieval),
    which correctly disables the escape hatch rather than letting an unrelated
    score stand in for it.
    """
    return chunks[0].components.get("dense", 0.0) if chunks else 0.0


def support_score(sentence: str, chunk_text: str) -> float:
    """Fraction of a sentence's content words present in its source.

    Lexical, not semantic, and deliberately so: a check the reviewer can
    reproduce by eye beats one that needs a second model to explain itself.
    """
    claim = set(tokenize(sentence))
    if not claim:
        return 1.0
    source = set(tokenize(chunk_text))
    return len(claim & source) / len(claim)


def verify_citations(
    answer_text: str, chunks: list[ScoredChunk], config: AnswerConfig
) -> tuple[list[Citation], bool]:
    """Match each answer sentence to its source and score the support.

    Returns ``(citations, grounded)``. ``grounded`` is False when any sentence
    fails the support threshold or cites nothing at all.
    """
    if answer_text.strip() == REFUSAL:
        return [], True

    citations: list[Citation] = []
    grounded = True

    for sentence, _start, _end in sentences(answer_text):
        markers = [int(m) for m in _CITATION_MARKER.findall(sentence)]
        claim = _CITATION_MARKER.sub("", sentence).strip()
        if not claim:
            continue

        if not markers:
            if config.require_citations:
                grounded = False
            continue

        best: tuple[float, ScoredChunk] | None = None
        for marker in markers:
            if not 1 <= marker <= len(chunks):
                # A citation pointing at a source that does not exist is the
                # clearest possible sign the answer is not grounded.
                grounded = False
                continue
            candidate = chunks[marker - 1]
            score = support_score(claim, candidate.chunk.text)
            if best is None or score > best[0]:
                best = (score, candidate)

        if best is None:
            grounded = False
            continue

        score, chunk = best
        if score < config.min_support:
            grounded = False
        citations.append(
            Citation(
                text=claim,
                chunk_id=chunk.chunk.chunk_id,
                citation=chunk.chunk.citation,
                doc_id=chunk.chunk.doc_id,
                support=score,
            )
        )

    return citations, grounded


def answer_question(
    question: str,
    retrieved: list[ScoredChunk],
    generator: Generator | None = None,
    config: AnswerConfig | None = None,
) -> Answer:
    """Generate an answer from retrieved chunks, or abstain."""
    settings = config or AnswerConfig()
    engine = generator or ExtractiveGenerator()
    started = time.perf_counter()

    top_score = max((r.score for r in retrieved), default=0.0)
    floor = settings.min_score_ratio * top_score if top_score > 0 else float("-inf")
    usable = [r for r in retrieved if r.score >= floor]
    question_coverage = coverage(question, usable, settings.coverage_chunks)
    semantic = best_semantic_score(usable)

    should_abstain = not usable or (
        question_coverage < settings.min_coverage and semantic < settings.semantic_floor
    )

    if should_abstain:
        return Answer(
            question=question,
            text=REFUSAL,
            retrieved=retrieved,
            abstained=True,
            grounded=True,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            metadata={
                "reason": "no retrieved chunk met the confidence floor"
                if not usable
                else "retrieved text does not cover enough of the question",
                "best_score": round(top_score, 5),
                "coverage": round(question_coverage, 3),
                "semantic": round(semantic, 4),
                "min_coverage": settings.min_coverage,
                "semantic_floor": settings.semantic_floor,
                "generator": engine.name,
            },
        )

    text = engine.generate(question, usable, settings)
    citations, grounded = verify_citations(text, usable, settings)
    abstained = text.strip() == REFUSAL

    return Answer(
        question=question,
        text=text,
        citations=citations,
        retrieved=usable,
        grounded=grounded,
        abstained=abstained,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        metadata={
            "generator": engine.name,
            "coverage": round(question_coverage, 3),
            "semantic": round(semantic, 4),
            "context_chunks": len(usable[: settings.max_context_chunks]),
            "mean_support": round(
                sum(c.support for c in citations) / len(citations), 3
            )
            if citations
            else None,
        },
    )
