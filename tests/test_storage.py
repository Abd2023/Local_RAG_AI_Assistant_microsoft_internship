from pathlib import Path

from src.ingest import DocumentChunk
from src.storage import (
    connect,
    count_chunks,
    create_chunks_table,
    fetch_all_chunks,
    insert_chunk,
    rebuild_chunks,
)


def test_insert_and_fetch_chunk_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.db"
    chunk = DocumentChunk(
        source_name="handbook.md",
        chunk_index=2,
        content="The demo begins at 2 PM.",
    )

    connection = connect(db_path)
    try:
        create_chunks_table(connection)
        row_id = insert_chunk(connection, chunk, [0.25, -0.5, 1.0])
        connection.commit()
    finally:
        connection.close()

    rows = fetch_all_chunks(db_path)

    assert row_id > 0
    assert count_chunks(db_path) == 1
    assert len(rows) == 1
    assert rows[0].id == row_id
    assert rows[0].source_name == "handbook.md"
    assert rows[0].chunk_index == 2
    assert rows[0].content == "The demo begins at 2 PM."
    assert rows[0].embedding == [0.25, -0.5, 1.0]
    assert rows[0].created_at


def test_rebuild_chunks_replaces_existing_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.db"
    original = DocumentChunk("old.md", 0, "Old content")
    replacement = DocumentChunk("new.md", 0, "New content")

    rebuild_chunks([(original, [1.0, 0.0])], db_path=db_path)
    stored_rows = rebuild_chunks(
        [(replacement, [0.0, 1.0])],
        db_path=db_path,
    )

    rows = fetch_all_chunks(db_path)

    assert stored_rows == 1
    assert count_chunks(db_path) == 1
    assert [row.source_name for row in rows] == ["new.md"]
    assert rows[0].embedding == [0.0, 1.0]
