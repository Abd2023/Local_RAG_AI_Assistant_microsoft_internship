import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src import api


def test_health_endpoint_reports_local_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "count_chunks", lambda: 4)
    monkeypatch.setattr(api, "count_vectors", lambda: 4)
    response = TestClient(api.app).get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["chunks"] == 4


def test_query_endpoint_returns_rag_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api,
        "answer_query",
        lambda question: {"answer": question, "trace_id": "trace-1"},
    )
    response = TestClient(api.app).post("/api/query", json={"question": "hello"})

    assert response.status_code == 200
    assert response.json()["trace_id"] == "trace-1"


def test_query_endpoint_rejects_blank_question() -> None:
    response = TestClient(api.app).post("/api/query", json={"question": " "})

    assert response.status_code == 422
