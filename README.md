# RAG Knowledge Assistant

A production-shaped retrieval-augmented generation stack — ingestion, chunking,
embeddings, hybrid search, reranking, grounded answers — with the evaluation
harness that tells you whether any of it is working.

[![CI](https://github.com/akarshanamachanpally/rag-knowledge-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/akarshanamachanpally/rag-knowledge-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/core%20dependencies-none-brightgreen)

```bash
pip install -e ".[dev]"
ragkit ask "how long do I have to request a refund?"
ragkit compare --chunkers --sizes     # which configuration is actually better
```

No API key, no model download, no network. The core has zero dependencies;
numpy, FAISS, Chroma and sentence-transformers are optional accelerators, and
every one of them is free.

---

## The part most RAG repos skip

Every RAG decision — chunk size, chunker, embedder, hybrid weights, whether to
rerank, which vector store — is an empirical question about *your* corpus. The
usual answer is a preference. This repo answers with a table:

```
$ ragkit compare --chunkers --sizes

VARIANT                     RECALL    PREC     MRR    NDCG     HIT   P50 ms
---------------------------------------------------------------------------
hybrid (default)             1.000   0.322   0.924   0.940   1.000     0.29
dense only                   1.000   0.306   0.935   0.948   1.000     6.52
chunker: section             1.000   0.322   0.924   0.940   1.000     0.31
chunker: sentence            1.000   0.240   0.833   0.875   1.000     2.47
chunker: fixed               1.000   0.233   0.802   0.851   1.000     2.62
lexical only                 0.957   0.549   0.862   0.887   0.957     0.20
hybrid + rerank              0.913   0.288   0.819   0.840   0.913     5.94

VARIANT                     ANSWER  GROUNDED  ABSTAIN  FALSE-AB
hybrid (default)             1.000     1.000    1.000     0.174
lexical only                 0.909     1.000    1.000     0.261
hybrid + rerank              1.000     1.000    1.000     0.217
```

**That table changed the defaults.** Reranking is implemented (lexical, MMR,
cross-encoder) and it *hurts* on this corpus — recall 1.000 → 0.913, MRR 0.924
→ 0.819, at 20x the latency. So reranking ships off by default, and the README
says why instead of claiming a best practice the repo's own numbers contradict.
On a larger, more redundant corpus the answer will differ; `ragkit compare` is
how you find out rather than assume.

Headline numbers on the bundled corpus (26 questions, 6 documents, k=5),
enforced by CI:

| Metric | Value |
|---|---|
| recall@5 | 1.000 |
| MRR | 0.924 |
| nDCG@5 | 0.940 |
| answer accuracy | 1.000 |
| grounded rate | 1.000 |
| abstention accuracy | 1.000 |
| false abstentions | 0.174 |
| latency p50 / p95 | 0.28 / 0.56 ms |

---

## Design decisions

### Hybrid search, fused by rank rather than score

Dense and lexical retrieval fail in opposite directions, and the failures are
the interesting part:

- **Dense** finds paraphrases and misses exact tokens. A query for `E4021` or
  `lisinopril` can rank the right chunk twentieth, because a rare literal
  string contributes almost nothing to a smooth vector.
- **Lexical (BM25)** nails exact tokens and misses every paraphrase. "How do I
  get my money back?" retrieves nothing from a page headed "Refunds".

Fusion uses Reciprocal Rank Fusion, not score averaging. BM25 scores are
unbounded and corpus-dependent; cosine sits in [-1, 1]. Averaging them needs a
normalisation that breaks the moment the corpus changes. RRF uses only *rank*,
so it is scale-free and needs no tuning.

```
$ ragkit search "error code E4021" --top-k 1
1. [0.0328] authentication#Error codes:725-958
   dense=0.303, lexical=7.759
   matched: e4021, error
```

Both retrievers' contributions are reported on every result, so a bad ranking
is diagnosable rather than mysterious.

### Abstention, measured honestly

A RAG system that never says "I don't know" has no way to be wrong quietly —
so it is wrong loudly, in production, to a user who believed it. Three of the
26 gold questions are unanswerable from the corpus, and the system declines all
three.

The gate is **term coverage**, not a score threshold, because a retrieval score
is *relative*: the best of six chunks always scores highest, even when all six
are irrelevant. Coverage is absolute — a question about the holiday schedule
shares almost no vocabulary with a refund policy, whatever the ranker says.

A strong dense score overrides weak coverage, for genuine paraphrases. With the
default hashing embedder that escape hatch rarely fires, because that embedder
is lexical too — which is exactly the measurable case for upgrading to a
sentence-transformer.

The cost is reported, not hidden: **false abstention rate 0.174** — four of 23
answerable questions declined. That is the deliberate operating point (never
fabricate, occasionally decline a paraphrase), and it is a tracked metric so a
change that improves abstention by wrecking availability shows up immediately.

### Citations are verified, not requested

Asking a model to cite its sources is a prompt. Checking that the cited text
supports the claim is a control. Every answer sentence is matched back to the
chunk it cites and scored for lexical support; a sentence below threshold, or
citing a source that does not exist, marks the whole answer ungrounded.

The default generator is **extractive** — it selects and stitches source
sentences, so it cannot hallucinate by construction. A local model generator
(Ollama) is available and goes through exactly the same verification.

```
$ ragkit ask "how long are audit records kept?"
Audit records are retained for 7 years to satisfy financial reporting obligations. [1]

grounded: True
sources:
  [1.00] data-retention#Retention periods:24-286
```

### A chunk always knows where it came from

`Chunk` cannot exist without `(doc_id, section, start, end)` into the original
document, and a test asserts that every chunk from every strategy maps back to
its source text. Without that, citations are decorative: you can print a
filename next to an answer, but nobody can check it, which is the only thing a
citation is for.

Chunking is where most RAG quality is won or lost, and the three failure modes
each have a countermeasure:

| Failure | Countermeasure |
|---|---|
| A fact split in half across a boundary | Sentence-aware packing, plus configurable overlap |
| A fact buried in a 2000-token chunk | Size targets tuned by measurement, not by default |
| A section body that loses its heading | The heading travels into the chunk text |

### Vector stores are swappable *and* comparable

`memory` (exact, the baseline), `faiss` (flat/IVF/HNSW), `chroma`
(ephemeral/persistent). The abstraction earns its keep by making the difference
measurable: FAISS with an IVF index is approximate, so it can miss a chunk that
exact search finds. That trade is usually right at scale and usually invisible
until someone measures it against an exact baseline.

The other real difference the code documents: FAISS has no metadata filtering,
so filters are applied *after* the ANN search (requiring over-fetch), while
Chroma filters inside the query.

---

## Usage

### CLI

```bash
ragkit ask "what should a client do when it receives HTTP 429?"
ragkit search "rate limit" --mode lexical --top-k 5
ragkit search "policy" --where '{"doc_id": "rate-limits"}'
ragkit eval --min-recall 0.95 --min-grounded 1.0     # CI-friendly gates
ragkit compare --chunkers --sizes --stores
ragkit chunks --chunker section --summary
ragkit stats
```

Exit codes: `0` fine, `1` an evaluation gate failed, `2` usage.

### Library

```python
from ragkit import RAGPipeline, load_corpus

pipeline = RAGPipeline()
pipeline.add_documents(load_corpus("data/corpus"))

answer = pipeline.ask("how long are backups retained?")
answer.text            # "...35 days on a rolling basis. [1]"
answer.grounded        # every sentence traced to a source
answer.abstained       # True when the corpus cannot answer
answer.citations[0].support   # lexical support for that specific claim
answer.sources         # ['data-retention#Backups:288-520']
```

Tuning is configuration:

```python
from ragkit import PipelineConfig, ChunkConfig, RetrievalConfig

config = PipelineConfig(
    chunker="section",
    chunking=ChunkConfig(target_tokens=220, overlap_tokens=40),
    embedder="hashing",          # or tfidf, sentence-transformers
    store="faiss",               # or memory, chroma
    retrieval=RetrievalConfig(mode="hybrid", top_k=5, candidate_k=30),
    reranker="none",             # or lexical, mmr, cross-encoder, default
)
```

### With real models (still free, still local)

```bash
pip install -e ".[models]"      # sentence-transformers
ollama pull llama3.2            # optional, for generated rather than extracted answers
ragkit ask "..." --embedder sentence-transformers --generator model
```

Then re-run `ragkit compare` to see what the upgrade actually bought on your
corpus. That is the whole point of the harness.

---

## Evaluation harness

Gold records label **documents**, not chunks — deliberately. Chunk ids change
every time you change the chunker, which would make the gold set useless for
exactly the comparison it exists to support.

```json
{
  "question": "What does error E4021 mean?",
  "relevant_doc_ids": ["authentication"],
  "answer_contains": ["E4021"],
  "tags": ["exact-token", "lexical-favours"]
}
```

Metrics: recall@k, precision@k, MRR, nDCG@k, hit rate, answer accuracy,
grounded rate, abstention accuracy, false abstention rate, and latency
percentiles. Retrieval averages exclude unanswerable questions — including them
would drag every score toward zero for the *correct* behaviour — and those are
scored by abstention accuracy instead.

```bash
ragkit eval --verbose        # per-question rows, and the retrieval misses
```

---

## Project layout

```
src/ragkit/
  types.py         Document, Chunk, ScoredChunk, Citation, Answer
  chunking.py      Fixed, sentence, section and paragraph strategies
  embeddings.py    Hashing, TF-IDF, sentence-transformers adapter
  retrieval.py     BM25, dense, and reciprocal rank fusion
  rerank.py        Lexical, MMR, cross-encoder, composable pipeline
  answering.py     Extractive/template/model generation + citation verification
  evaluation.py    Metrics, gold sets, side-by-side comparison
  pipeline.py      The assembled stack
  cli.py           ask / search / eval / compare / chunks / stats
  stores/          base + memory (exact), FAISS, Chroma
data/corpus/       Six synthetic policy documents
data/gold_questions.json   26 labelled questions, 3 unanswerable
tests/             119 tests
```

## Testing

```bash
pytest                       # 119 tests, under a second
ragkit eval                  # quality gates
ruff check src tests
```

CI runs the suite on 3.10–3.12, again with numpy uninstalled (it is an
accelerator, not a requirement, and the pure-Python path must agree), enforces
the published quality numbers, and — when FAISS and Chroma are installed — runs
the same evaluation against those backends.

## Limitations

- The default embedder is lexical. It captures sub-word and morphological
  similarity, not synonymy: "car" and "automobile" are unrelated to it. That is
  why hybrid is the default and why paraphrase questions are the ones that get
  declined. Install `[models]` and re-run `compare` to quantify the fix on your
  corpus.
- Support scoring is lexical overlap, so it detects unsupported claims but not
  subtly wrong ones. A claim can quote the right chunk and still misread it.
- The bundled corpus is six documents. The numbers above are real but small;
  they demonstrate that the harness works, not that the configuration
  generalises. Point it at your corpus.
- No incremental reindexing, deletion, or multi-tenancy. Adding documents
  re-embeds only the new ones, but a changed document must be re-added.

## License

MIT
