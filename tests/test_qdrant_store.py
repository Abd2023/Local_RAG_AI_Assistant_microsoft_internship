from pathlib import Path

import pytest

from src import config, vector_store
from src.storage import StoredChunk


def test_qdrant_backend_round_trip_with_local_in_memory_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qdrant = pytest.importorskip("qdrant_client")
    client = qdrant.QdrantClient(":memory:")
    monkeypatch.setattr(config, "VECTOR_BACKEND", "qdrant")
    monkeypatch.setattr(vector_store, "_qdrant_client", lambda: client)

    chunk = StoredChunk(
        id=1,
        source_name="notes.md",
        chunk_index=0,
        content="The standup starts at 10 AM.",
        embedding=[1.0, 0.0],
        created_at="2026-08-08T00:00:00+00:00",
        document_id="doc-1",
        source_path=str(Path("notes.md")),
    )

    assert vector_store.add_chunk_vectors([(chunk, [1.0, 0.0])]) == 1
    rows = vector_store.search_vectors([1.0, 0.0], top_k=1)

    assert rows[0]["source_name"] == "notes.md"
    assert rows[0]["similarity"] == pytest.approx(1.0)
