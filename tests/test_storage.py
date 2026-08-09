from pathlib import Path

from src.ingest import DocumentChunk
from src.storage import (
    connect,
    count_chunks,
    create_chunks_table,
    fetch_all_chunks,
    insert_chunk,
    rebuild_lexical_index,
    rebuild_chunks,
    search_lexical_chunks,
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


def test_lexical_search_finds_exact_terms_after_rebuilding_index(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.db"
    connection = connect(db_path)
    try:
        create_chunks_table(connection)
        insert_chunk(
            connection,
            DocumentChunk("setup.md", 0, "Install Tesseract before indexing scanned PDFs."),
            [1.0, 0.0],
        )
        insert_chunk(
            connection,
            DocumentChunk("other.md", 0, "The final demo lasts five minutes."),
            [0.0, 1.0],
        )
        connection.commit()
    finally:
        connection.close()

    assert rebuild_lexical_index(db_path) is True
    results = search_lexical_chunks("Tesseract scanned PDFs", db_path=db_path)

    assert results
    assert results[0]["source_name"] == "setup.md"
    assert results[0]["lexical_score"] == 1.0
