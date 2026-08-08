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


def test_retrieve_top_chunks_uses_injected_embedding_and_vector_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fake_generate_embedding(query: str) -> list[float]:
        calls.append(query)
        return [1.0, 0.0]

    def fake_search_vectors(
        query_embedding: list[float],
        top_k: int,
        vector_db_path: Path,
    ) -> list[dict[str, object]]:
        assert query_embedding == [1.0, 0.0]
        assert top_k == 1
        assert vector_db_path == tmp_path / "lancedb"
        return [
            {
                "sqlite_id": 2,
                "source_name": "second.md",
                "chunk_index": 0,
                "content": "Content from second.md",
                "similarity": 0.99,
                "source_path": "C:/docs/second.md",
                "page_start": None,
                "page_end": None,
                "extraction_method": "markdown",
                "heading": "Second",
                "chunk_uid": "chunk-2",
            }
        ]

    monkeypatch.setattr(retrieval, "generate_embedding", fake_generate_embedding)
    monkeypatch.setattr(retrieval, "search_vectors", fake_search_vectors)

    results = retrieval.retrieve_top_chunks(
        "When is the demo?",
        top_k=1,
        db_path=tmp_path / "lancedb",
    )

    assert calls == ["When is the demo?"]
    assert [result.source_name for result in results] == ["second.md"]
    assert results[0].retrieval_score == pytest.approx(0.99)
    assert results[0].source_path == "C:/docs/second.md"


def test_retrieve_top_chunks_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        retrieval.retrieve_top_chunks("   ")


def test_retrieve_top_chunks_requires_ingested_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(retrieval, "generate_embedding", lambda _query: [1.0, 0.0])

    def fake_search_vectors(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        raise retrieval.VectorStoreError("Run `python -m src.ingest` before retrieval.")

    monkeypatch.setattr(retrieval, "search_vectors", fake_search_vectors)

    with pytest.raises(retrieval.RetrievalError, match="src.ingest"):
        retrieval.retrieve_top_chunks(
            "When is the demo?",
            db_path=tmp_path / "empty-lancedb",
        )
