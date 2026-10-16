"""The HTTP service behind the web UI. Same pipeline, same numbers as the CLI."""
import re

import pytest
from fastapi.testclient import TestClient

from ragkit.server import create_app


@pytest.fixture()
def client():
    return TestClient(create_app())


def test_ui_is_served_and_root_redirects(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"
    assert "Knowledge Assistant" in client.get("/ui/").text


def test_answers_cite_evidence_by_rank(client):
    body = client.post("/api/ask", json={"question": "How long are audit records kept?"}).json()
    assert body["grounded"] is True
    assert body["abstained"] is False
    assert "7 years" in body["answer"]
    markers = {int(n) for n in re.findall(r"\[(\d+)\]", body["answer"])}
    ranks = [c["rank"] for c in body["retrieved"]]
    assert markers and markers <= set(ranks)
    assert ranks == sorted(ranks)
    first = body["retrieved"][0]
    assert {"dense", "lexical"} <= set(first["components"])
    assert first["text"]


def test_questions_outside_the_corpus_are_declined(client):
    body = client.post("/api/ask", json={"question": "How do I configure the Kubernetes ingress controller?"}).json()
    assert body["abstained"] is True
    assert body["citations"] == []
    assert body["metadata"]["coverage"] < body["metadata"]["min_coverage"]


def test_retrieval_runs_every_mode_and_recognises_gold_questions(client):
    body = client.post("/api/retrieve", json={"query": "Can I ask for a higher throughput allowance?"}).json()
    assert set(body["runs"]) == {"hybrid", "dense", "lexical"}
    assert body["gold"]["relevant_doc_ids"] == ["rate-limits"]
    assert body["runs"]["dense"][0]["doc_id"] == "rate-limits"
    assert all(r["doc_id"] != "rate-limits" for r in body["runs"]["lexical"])


def test_an_added_document_is_searchable_then_removable(client):
    text = "# Travel Policy\n\n## Meals\nMeals are reimbursed up to 60 dollars per day with itemised receipts."
    added = client.post("/api/documents", json={"title": "Travel Policy", "text": text}).json()
    assert added["doc_id"] == "travel-policy" and added["added"] is True and added["chunks"] >= 1
    hits = client.post("/api/retrieve", json={"query": "meal reimbursement per day"}).json()["runs"]["hybrid"]
    assert hits[0]["doc_id"] == "travel-policy"
    assert client.post("/api/documents", json={"title": "Travel Policy", "text": text}).status_code == 409
    assert client.delete("/api/documents/travel-policy").json() == {"removed": "travel-policy"}
    assert client.delete("/api/documents/refund-policy").status_code == 404


def test_document_view_exposes_chunk_boundaries(client):
    doc = client.get("/api/documents/refund-policy").json()
    assert doc["chunks"]
    for chunk in doc["chunks"]:
        assert 0 <= chunk["start"] < chunk["end"] <= len(doc["text"])


def test_evaluation_meets_the_same_gates_as_ci(client):
    report = client.post("/api/evaluate", json={"k": 5, "mode": "hybrid"}).json()
    assert report["recall@k"] >= 0.95
    assert report["abstention_accuracy"] == 1.0
    assert report["grounded_rate"] == 1.0
    assert len(report["per_question"]) == report["questions"] == len(client.get("/api/gold").json())


def test_comparison_scores_each_configuration(client):
    labels = {r["label"] for r in client.post("/api/compare", json={"k": 5}).json()}
    assert {"hybrid (default)", "dense only", "lexical only", "hybrid + rerank"} <= labels


def test_bad_modes_are_rejected(client):
    assert client.post("/api/ask", json={"question": "refunds", "mode": "fuzzy"}).status_code == 422
    assert client.post("/api/evaluate", json={"mode": "fuzzy"}).status_code == 422
