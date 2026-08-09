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
    monkeypatch.setattr(api.config, "UPLOADS_PATH", tmp_path)
    monkeypatch.setattr(
        api,
        "ingest_documents",
        lambda: {
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


def test_upload_endpoint_accepts_multiple_documents_and_indexes_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(api.config, "UPLOADS_PATH", tmp_path)
    ingest_calls: list[object] = []

    def fake_ingest() -> dict[str, object]:
        ingest_calls.append(True)
        return {
            "files": 2,
            "indexed_files": 2,
            "skipped_files": 0,
            "chunks": 3,
            "final_rows": 3,
            "final_vectors": 3,
            "documents": [
                {"document_name": "one.md", "status": "indexed", "chunks": 1, "vectors": 1},
                {"document_name": "two.txt", "status": "indexed", "chunks": 2, "vectors": 2},
            ],
        }

    monkeypatch.setattr(api, "ingest_documents", fake_ingest)
    response = TestClient(api.app).post(
        "/api/upload",
        files=[
            ("files", ("one.md", b"one", "text/markdown")),
            ("files", ("two.txt", b"two", "text/plain")),
        ],
    )

    assert response.status_code == 200
    assert ingest_calls == [True]
    assert [item["document_name"] for item in response.json()["files"]] == [
        "one.md",
        "two.txt",
    ]
    assert [item["chunks"] for item in response.json()["files"]] == [1, 2]
    assert (tmp_path / "one.md").read_bytes() == b"one"
    assert (tmp_path / "two.txt").read_bytes() == b"two"


def test_ingest_endpoint_without_path_uses_combined_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def fake_ingest(path=None, *, rebuild=False):
        calls.append((path, rebuild))
        return {"indexed_files": 0}

    monkeypatch.setattr(api, "ingest_documents", fake_ingest)
    response = TestClient(api.app).post("/api/ingest")

    assert response.status_code == 200
    assert calls == [(None, False)]


def test_upload_endpoint_rejects_unsupported_documents() -> None:
    response = TestClient(api.app).post(
        "/api/upload",
        files={"file": ("malware.exe", b"not allowed", "application/octet-stream")},
    )

    assert response.status_code == 400
