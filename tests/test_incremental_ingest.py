from pathlib import Path

from src.ingest import add_document_file, add_document_files, ingest_documents
from src.storage import count_chunks, fetch_all_chunks
from src.vector_store import count_vectors


def test_incremental_indexing_skips_updates_and_handles_changes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    source = docs_path / "handbook.md"
    source.write_text("# Handbook\n\nThe daily standup starts at 10:00 AM.", encoding="utf-8")
    db_path = tmp_path / "rag.db"
    vector_db_path = tmp_path / "lancedb"
    embedding_calls: list[list[str]] = []

    def fake_embeddings(texts: list[str]) -> list[list[float]]:
        embedding_calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("src.ingest.generate_embeddings", fake_embeddings)

    first = ingest_documents(docs_path, db_path=db_path, vector_db_path=vector_db_path)
    second = ingest_documents(docs_path, db_path=db_path, vector_db_path=vector_db_path)
    source.write_text("# Handbook\n\nThe daily standup starts at 11:00 AM.", encoding="utf-8")
    third = ingest_documents(docs_path, db_path=db_path, vector_db_path=vector_db_path)
    rows_after_change = fetch_all_chunks(db_path)
    source.unlink()
    fourth = ingest_documents(docs_path, db_path=db_path, vector_db_path=vector_db_path)

    assert first["indexed_files"] == 1
    assert first["stored_rows"] == 1
    assert second["skipped_files"] == 1
    assert second["chunks"] == 0
    assert third["indexed_files"] == 1
    assert count_chunks(db_path) == 0
    assert count_vectors(vector_db_path) == 0
    assert fourth["removed_files"] == 1
    assert len(embedding_calls) == 2
    assert "11:00 AM" in rows_after_change[0].content


def test_failed_changed_document_is_retried_and_does_not_keep_stale_chunks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    source = docs_path / "broken.md"
    source.write_text("Working content", encoding="utf-8")
    db_path = tmp_path / "rag.db"
    vector_db_path = tmp_path / "lancedb"

    monkeypatch.setattr("src.ingest.generate_embeddings", lambda texts: [[1.0, 0.0] for _ in texts])
    first = ingest_documents(docs_path, db_path=db_path, vector_db_path=vector_db_path)
    assert first["indexed_files"] == 1
    source.write_bytes(b"\xff\xfe not valid utf8")

    failed = ingest_documents(docs_path, db_path=db_path, vector_db_path=vector_db_path)
    retried = ingest_documents(docs_path, db_path=db_path, vector_db_path=vector_db_path)

    assert failed["error_files"] == 1
    assert retried["error_files"] == 1
    assert count_chunks(db_path) == 0
    assert count_vectors(vector_db_path) == 0


def test_add_document_file_copies_external_document_and_indexes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "incoming" / "research.pdf"
    source.parent.mkdir()
    source.write_bytes(b"pdf bytes")
    docs_path = tmp_path / "knowledge"
    calls: list[Path] = []

    def fake_ingest(path: Path) -> dict[str, int]:
        calls.append(path)
        return {"files": 1, "indexed_files": 1}

    monkeypatch.setattr("src.ingest.ingest_documents", fake_ingest)

    result = add_document_file(source, docs_path)

    assert result["document_name"] == "research.pdf"
    assert (docs_path / "research.pdf").read_bytes() == b"pdf bytes"
    assert calls == [docs_path.resolve()]


def test_add_document_files_recursively_stages_supported_files_in_one_pass(
    monkeypatch,
    tmp_path: Path,
) -> None:
    incoming = tmp_path / "incoming"
    (incoming / "nested").mkdir(parents=True)
    (incoming / "one.md").write_text("one", encoding="utf-8")
    (incoming / "nested" / "two.txt").write_text("two", encoding="utf-8")
    (incoming / "ignore.py").write_text("ignore", encoding="utf-8")
    destination = tmp_path / "knowledge"
    calls: list[Path] = []

    def fake_ingest(path: Path) -> dict[str, object]:
        calls.append(path)
        return {"files": 2, "indexed_files": 2}

    monkeypatch.setattr("src.ingest.ingest_documents", fake_ingest)

    result = add_document_files([incoming], destination)

    assert {item["document_name"] for item in result["staged"]} == {"one.md", "two.txt"}
    assert (destination / "one.md").read_text(encoding="utf-8") == "one"
    assert (destination / "two.txt").read_text(encoding="utf-8") == "two"
    assert calls == [destination.resolve()]
