"""CLI behaviour and exit codes."""
import json

from ragkit.cli import main


def test_ask_prints_an_answer_with_sources(capsys):
    assert main(["ask", "How long do I have to request a refund?"]) == 0
    out = capsys.readouterr().out
    assert "30 days" in out
    assert "sources:" in out
    assert "refund-policy" in out


def test_ask_declines_when_the_corpus_cannot_answer(capsys):
    assert main(["ask", "What is the company holiday schedule for next year?"]) == 0
    out = capsys.readouterr().out
    assert "abstained" in out


def test_ask_json_is_machine_readable(capsys):
    main(["ask", "What does error E4021 mean?", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["grounded"] is True
    assert payload["citations"]
    assert "E4021" in payload["answer"]


def test_search_shows_per_retriever_components(capsys):
    assert main(["search", "error code E4021", "--top-k", "3"]) == 0
    out = capsys.readouterr().out
    assert "dense=" in out and "lexical=" in out


def test_search_respects_a_metadata_filter(capsys):
    main(["search", "policy", "--where", '{"doc_id": "rate-limits"}', "--json"])
    results = json.loads(capsys.readouterr().out)
    assert results
    assert all(r["doc_id"] == "rate-limits" for r in results)


def test_lexical_mode_wins_on_exact_tokens(capsys):
    main(["search", "E4033", "--mode", "lexical", "--top-k", "1", "--json"])
    results = json.loads(capsys.readouterr().out)
    assert results[0]["doc_id"] == "authentication"


def test_eval_reports_the_headline_metrics(capsys):
    assert main(["eval"]) == 0
    out = capsys.readouterr().out
    for label in ("recall@5", "MRR", "abstention accuracy", "grounded rate"):
        assert label in out


def test_eval_gates_pass_at_the_published_thresholds(capsys):
    code = main(["eval", "--min-recall", "0.95", "--min-abstention", "1.0",
                 "--min-grounded", "1.0"])
    capsys.readouterr()
    assert code == 0


def test_eval_gate_failure_exits_nonzero(capsys):
    code = main(["eval", "--min-recall", "1.01"])
    capsys.readouterr()
    assert code == 1


def test_eval_json_output(capsys):
    main(["eval", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["questions"] > 20
    assert payload["recall@k"] >= 0.95


def test_compare_ranks_configurations(capsys):
    assert main(["compare"]) == 0
    out = capsys.readouterr().out
    assert "VARIANT" in out and "RECALL" in out
    assert "dense only" in out and "lexical only" in out


def test_compare_across_chunkers(capsys):
    main(["compare", "--chunkers", "--retrieval-only", "--json"])
    reports = json.loads(capsys.readouterr().out)
    labels = {r["label"] for r in reports}
    assert {"chunker: fixed", "chunker: section"} <= labels


def test_chunks_summary(capsys):
    assert main(["chunks", "--summary"]) == 0
    out = capsys.readouterr().out
    assert "refund-policy" in out
    assert "chunks" in out


def test_stats_describes_the_index(capsys):
    assert main(["stats"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["documents"] == 6
    assert payload["embedder"]["semantic"] is False


def test_version(capsys):
    assert main(["--version"]) == 0
    assert "ragkit" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_missing_corpus_is_a_usage_error(capsys, tmp_path):
    assert main(["ask", "anything", "--corpus", str(tmp_path)]) == 2
    assert "no documents" in capsys.readouterr().err
