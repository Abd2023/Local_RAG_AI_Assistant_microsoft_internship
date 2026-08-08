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


def test_upload_endpoint_stores_and_indexes_document(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(api.config, "SAMPLE_DOCS_PATH", tmp_path)
    monkeypatch.setattr(
        api,
        "ingest_documents",
        lambda docs_path: {
            "files": 1,
            "indexed_files": 1,
            "skipped_files": 0,
            "chunks": 2,
            "final_rows": 2,
            "final_vectors": 2,
        },
    )

    response = TestClient(api.app).post(
        "/api/upload",
        files={"file": ("new-research.pdf", b"%PDF test", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["document_name"] == "new-research.pdf"
    assert response.json()["index"]["indexed_files"] == 1
    assert (tmp_path / "new-research.pdf").read_bytes() == b"%PDF test"


def test_upload_endpoint_rejects_unsupported_documents() -> None:
    response = TestClient(api.app).post(
        "/api/upload",
        files={"file": ("malware.exe", b"not allowed", "application/octet-stream")},
    )

    assert response.status_code == 400
