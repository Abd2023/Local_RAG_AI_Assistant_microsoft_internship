from pathlib import Path

import pytest

from src.storage import StoredChunk
from src.vector_store import (
    count_vectors,
    rebuild_vector_table,
    replace_document_vectors,
    search_vectors,
)


def make_chunk(
    chunk_id: int,
    source_name: str,
    document_id: str,
) -> StoredChunk:
    return StoredChunk(
        id=chunk_id,
        source_name=source_name,
        chunk_index=chunk_id - 1,
        content=f"Content from {source_name}",
        embedding=[],
        created_at="2026-07-01T00:00:00+00:00",
        chunk_uid=f"chunk-{chunk_id}",
        document_id=document_id,
        source_path=f"C:/docs/{source_name}",
        extraction_method="markdown",
    )


def test_lancedb_rebuild_and_search_uses_deterministic_vectors(tmp_path: Path) -> None:
    vector_db_path = tmp_path / "lancedb"
    best = make_chunk(1, "best.md", "doc-1")
    other = make_chunk(2, "other.md", "doc-2")

    stored = rebuild_vector_table(
        [(best, [1.0, 0.0]), (other, [0.0, 1.0])],
        vector_db_path=vector_db_path,
    )
    results = search_vectors([1.0, 0.0], top_k=2, vector_db_path=vector_db_path)

    assert stored == 2
    assert count_vectors(vector_db_path) == 2
    assert [result["source_name"] for result in results] == ["best.md", "other.md"]
    assert results[0]["similarity"] == pytest.approx(1.0)
    assert results[1]["similarity"] == pytest.approx(0.0)


def test_replace_document_vectors_removes_old_rows(tmp_path: Path) -> None:
    vector_db_path = tmp_path / "lancedb"
    old = make_chunk(1, "old.md", "doc-1")
    new = make_chunk(2, "new.md", "doc-1")

    rebuild_vector_table([(old, [1.0, 0.0])], vector_db_path=vector_db_path)
    replace_document_vectors("doc-1", [(new, [0.0, 1.0])], vector_db_path=vector_db_path)
    results = search_vectors([0.0, 1.0], top_k=1, vector_db_path=vector_db_path)

    assert count_vectors(vector_db_path) == 1
    assert results[0]["source_name"] == "new.md"
