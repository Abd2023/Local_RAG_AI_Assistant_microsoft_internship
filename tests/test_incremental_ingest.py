from pathlib import Path

from src.ingest import ingest_documents
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
