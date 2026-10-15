# Verified results

Output captured by actually running this repository. Nothing here is hand-written.

| | |
|---|---|
| Date | 2026-10-15 |
| Commit | `d548d2d` |
| Python | Python 3.11.9 |
| OS | Windows 11 |

Timing figures (ms, rps) vary by machine and run. Counts and accuracy figures are deterministic.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]" pytest
pytest -q
python examples\quickstart.py
```

## Test suite

```text
........................................................................ [ 60%]
...............................................                          [100%]
119 passed in 0.89s
```

## `examples/quickstart.py`

```text

============================================================================
1. Chunking keeps the heading and the source offsets
============================================================================
  refund-policy#Standard refund window:60-318
    section: 'Standard refund window'
    text   : Standard refund window Customers may request a full refund within 30 days of the origina...
  refund-policy#Processing time:339-621
    section: 'Processing time'
    text   : Processing time Approved refunds are processed within 5 to 7 business days. The funds ma...
  refund-policy#Annual subscriptions:647-893
    section: 'Annual subscriptions'
    text   : Annual subscriptions may be cancelled at any time. Cancellation stops the next renewal b...

  indexed 6 documents into 20 chunks (mean 231 chars)

============================================================================
2. Dense and lexical retrieval fail in opposite directions
============================================================================

  query: What does error E4021 mean?
    dense    -> authentication
    lexical  -> authentication
    hybrid   -> authentication

  query: how do I get my money back
    dense    -> onboarding
    lexical  -> authentication
    hybrid   -> authentication

============================================================================
3. Fusion reports what each retriever contributed
============================================================================
  [0.0328] authentication#Error codes:725-958
           dense=0.303, lexical=7.759   matched: e4021, error
  [0.0161] refund-policy#Annual subscriptions:647-893
           dense=0.103   matched: 

============================================================================
4. Answers carry citations that were verified, not requested
============================================================================

  Q: How long are audit records kept?
  A: Audit records are retained for 7 years to satisfy financial reporting obligations. [1] Backups Encrypted backups are kept for 35 days on a rolling bas
     grounded=True
     [support 0.57] data-retention#Backups:323-537

  Q: What is the default API rate limit?
  A: Increasing limits Rate limit increases are reviewed on request for accounts on the Business plan and above. [1] Default limits The default limit is 10
     grounded=True
     [support 0.46] rate-limits#Default limits:37-213

============================================================================
5. The system declines what the corpus cannot answer
============================================================================
  Q: What is the company holiday schedule for next year?
     abstained=True  coverage=0.2 semantic=0.1198
  Q: How do I configure the Kubernetes ingress controller?
     abstained=True  coverage=0.2 semantic=-0.0161

============================================================================
6. Metadata filtering restricts the search
============================================================================
  3 results, all from: {'rate-limits'}

============================================================================
7. Scoring the whole pipeline against a labelled gold set
============================================================================
  questions           : 26
  recall@5            : 1.000
  MRR                 : 0.924
  nDCG@5              : 0.940
  answer accuracy     : 1.000
  grounded rate       : 1.000
  abstention accuracy : 1.000
  false abstentions   : 0.174
  latency p50 / p95   : 0.17 / 0.24 ms

============================================================================
8. Which configuration is actually better? Measure it.
============================================================================
VARIANT                     RECALL    PREC     MRR    NDCG     HIT   P50 ms
---------------------------------------------------------------------------
hybrid (default)             1.000   0.322   0.924   0.940   1.000     0.19
dense only                   1.000   0.294   0.938   0.950   1.000     0.15
chunker: fixed               1.000   0.252   0.880   0.914   1.000     0.19
chunker: sentence            1.000   0.260   0.880   0.911   1.000     0.18
small chunks (100 tok)       1.000   0.322   0.924   0.940   1.000     0.19
lexical only                 0.957   0.543   0.873   0.894   0.957     0.01
hybrid + rerank              0.913   0.288   0.819   0.840   0.913     2.73

VARIANT                     ANSWER  GROUNDED  ABSTAIN  FALSE-AB
hybrid (default)             1.000     1.000    1.000     0.174
dense only                   1.000     1.000    0.667     0.174
chunker: fixed               1.000     1.000    1.000     0.130
chunker: sentence            1.000     1.000    1.000     0.130
small chunks (100 tok)       1.000     1.000    1.000     0.174
lexical only                 0.909     1.000    1.000     0.261
hybrid + rerank              1.000     1.000    1.000     0.217

  Reranking is implemented and measurably hurts on this corpus,
  so it ships off by default. On a larger corpus, measure again.
```
