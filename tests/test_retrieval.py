from pathlib import Path

import pytest

from src import retrieval
from src.storage import StoredChunk


def make_stored_chunk(
    chunk_id: int,
    source_name: str,
    embedding: list[float],
) -> StoredChunk:
    return StoredChunk(
        id=chunk_id,
        source_name=source_name,
        chunk_index=0,
        content=f"Content from {source_name}",
        embedding=embedding,
        created_at="2026-07-01T00:00:00+00:00",
    )


@pytest.mark.parametrize(
    ("vector_a", "vector_b", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([0.0, 0.0], [1.0, 0.0], 0.0),
    ],
)
def test_cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
    expected: float,
) -> None:
    assert retrieval.cosine_similarity(vector_a, vector_b) == pytest.approx(expected)


def test_cosine_similarity_rejects_invalid_vectors() -> None:
    with pytest.raises(ValueError, match="empty"):
        retrieval.cosine_similarity([], [])

    with pytest.raises(ValueError, match="shapes must match"):
        retrieval.cosine_similarity([1.0], [1.0, 2.0])


def test_rank_chunks_by_similarity_uses_fake_vectors_deterministically() -> None:
    chunks = [
        make_stored_chunk(1, "orthogonal.md", [0.0, 1.0]),
        make_stored_chunk(2, "best.md", [1.0, 0.0]),
        make_stored_chunk(3, "opposite.md", [-1.0, 0.0]),
    ]

    results = retrieval.rank_chunks_by_similarity(
        query_embedding=[1.0, 0.0],
        chunks=chunks,
        top_k=2,
    )

    assert [result.source_name for result in results] == [
        "best.md",
        "orthogonal.md",
    ]
    assert [result.similarity for result in results] == pytest.approx([1.0, 0.0])


def test_rank_chunks_by_similarity_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        retrieval.rank_chunks_by_similarity([1.0], [], top_k=0)


def test_retrieve_top_chunks_uses_injected_embedding_and_database_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chunks = [
        make_stored_chunk(1, "first.md", [0.0, 1.0]),
        make_stored_chunk(2, "second.md", [1.0, 0.0]),
    ]
    calls: list[str] = []

    monkeypatch.setattr(retrieval, "fetch_all_chunks", lambda _db_path: chunks)

    def fake_generate_embedding(query: str) -> list[float]:
        calls.append(query)
        return [1.0, 0.0]

    monkeypatch.setattr(retrieval, "generate_embedding", fake_generate_embedding)

    results = retrieval.retrieve_top_chunks(
        "When is the demo?",
        top_k=1,
        db_path=tmp_path / "unused.db",
    )

    assert calls == ["When is the demo?"]
    assert [result.source_name for result in results] == ["second.md"]


def test_retrieve_top_chunks_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        retrieval.retrieve_top_chunks("   ")


def test_retrieve_top_chunks_requires_ingested_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(retrieval, "fetch_all_chunks", lambda _db_path: [])

    with pytest.raises(retrieval.RetrievalError, match="src.ingest"):
        retrieval.retrieve_top_chunks(
            "When is the demo?",
            db_path=tmp_path / "empty.db",
        )
