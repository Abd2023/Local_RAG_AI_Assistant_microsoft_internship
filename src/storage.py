"""SQLite storage helpers for document metadata, chunks, and legacy embeddings."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from src import config

LEXICAL_TABLE = "chunks_fts"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_uid TEXT,
    document_id TEXT,
    source_path TEXT,
    source_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    extraction_method TEXT,
    heading TEXT,
    chunk_hash TEXT,
    updated_at TEXT,
    created_at TEXT NOT NULL
);
"""

DOCUMENTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    source_name TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_at TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT
);
"""

CHUNK_OPTIONAL_COLUMNS = {
    "chunk_uid": "TEXT",
    "document_id": "TEXT",
    "source_path": "TEXT",
    "page_start": "INTEGER",
    "page_end": "INTEGER",
    "extraction_method": "TEXT",
    "heading": "TEXT",
    "chunk_hash": "TEXT",
    "updated_at": "TEXT",
}


@dataclass(frozen=True)
class StoredChunk:
    """A chunk row read back from SQLite."""

    id: int
    source_name: str
    chunk_index: int
    content: str
    embedding: list[float]
    created_at: str
    chunk_uid: str | None = None
    document_id: str | None = None
    source_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    extraction_method: str | None = None
    heading: str | None = None
    chunk_hash: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class StoredDocument:
    """A document manifest row read from SQLite."""

    document_id: str
    source_path: str
    source_name: str
    file_hash: str
    size_bytes: int
    modified_at: str
    indexed_at: str
    status: str
    error: str | None = None


class ChunkLike(Protocol):
    """Minimal chunk shape accepted by storage helpers."""

    source_name: str
    chunk_index: int
    content: str


def connect(db_path: Path = config.DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection, creating the data directory if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def create_chunks_table(connection: sqlite3.Connection) -> None:
    """Create/migrate metadata tables if they do not exist."""
    connection.execute(DOCUMENTS_SCHEMA_SQL)
    connection.execute(SCHEMA_SQL)
    existing_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(chunks)").fetchall()
    }
    for column_name, column_type in CHUNK_OPTIONAL_COLUMNS.items():
        if column_name not in existing_columns:
            connection.execute(f"ALTER TABLE chunks ADD COLUMN {column_name} {column_type}")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_uid ON chunks(chunk_uid)")
    try:
        connection.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS {LEXICAL_TABLE}
            USING fts5(content, source_name, heading, content='chunks', content_rowid='id')"""
        )
    except sqlite3.OperationalError:
        # Some minimal SQLite builds omit FTS5. Vector retrieval remains available.
        pass
    connection.commit()


def rebuild_lexical_index(db_path: Path = config.DATABASE_PATH) -> bool:
    """Rebuild the SQLite FTS5 index from the canonical chunk table."""
    connection = connect(db_path)
    try:
        create_chunks_table(connection)
        try:
            connection.execute(
                f"INSERT INTO {LEXICAL_TABLE}({LEXICAL_TABLE}) VALUES ('rebuild')"
            )
        except sqlite3.OperationalError:
            return False
        connection.commit()
        return True
    finally:
        connection.close()


def _fts_query(query: str) -> str:
    """Build a forgiving OR query for exact terms and identifiers."""
    tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def search_lexical_chunks(
    query: str,
    top_k: int = config.LEXICAL_CANDIDATE_K,
    db_path: Path = config.DATABASE_PATH,
) -> list[dict[str, object]]:
    """Return FTS5-ranked chunk rows for exact-term retrieval."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    match_query = _fts_query(query)
    if not match_query:
        return []

    connection = connect(db_path)
    try:
        create_chunks_table(connection)
        try:
            rows = connection.execute(
                f"""SELECT c.id, c.chunk_uid, c.document_id, c.source_path,
                    c.source_name, c.chunk_index, c.content, c.page_start, c.page_end,
                    c.extraction_method, c.heading,
                    bm25({LEXICAL_TABLE}) AS bm25_score
                    FROM {LEXICAL_TABLE}
                    JOIN chunks AS c ON c.id = {LEXICAL_TABLE}.rowid
                    WHERE {LEXICAL_TABLE} MATCH ?
                    ORDER BY bm25_score ASC
                    LIMIT ?""",
                (match_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        results: list[dict[str, object]] = []
        for rank, row in enumerate(rows):
            results.append(
                {
                    "sqlite_id": int(row["id"]),
                    "chunk_uid": row["chunk_uid"],
                    "document_id": row["document_id"],
                    "source_path": row["source_path"],
                    "source_name": row["source_name"],
                    "chunk_index": int(row["chunk_index"]),
                    "content": row["content"],
                    "page_start": row["page_start"],
                    "page_end": row["page_end"],
                    "extraction_method": row["extraction_method"],
                    "heading": row["heading"],
                    "lexical_score": 1.0 / (rank + 1),
                    "bm25_score": float(row["bm25_score"]),
                }
            )
        return results
    finally:
        connection.close()


def count_chunks_for_document(
    document_id: str,
    db_path: Path = config.DATABASE_PATH,
) -> int:
    """Return the number of indexed chunks belonging to one document."""
    connection = connect(db_path)
    try:
        create_chunks_table(connection)
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM chunks WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    finally:
        connection.close()
    return int(row["count"])


def clear_chunks(connection: sqlite3.Connection) -> None:
    """Remove all existing chunk rows."""
    connection.execute("DELETE FROM chunks")
    connection.commit()


def _optional_attr(item: object, name: str) -> object | None:
    return getattr(item, name, None)


def insert_chunk(
    connection: sqlite3.Connection,
    chunk: ChunkLike,
    embedding: Sequence[float],
) -> int:
    """Insert one chunk and its embedding vector."""
    embedding_json = json.dumps([float(value) for value in embedding])
    created_at = datetime.now(UTC).isoformat()
    cursor = connection.execute(
        """
        INSERT INTO chunks (
            chunk_uid, document_id, source_path, source_name, chunk_index, content,
            embedding_json, page_start, page_end, extraction_method, heading,
            chunk_hash, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _optional_attr(chunk, "chunk_uid"),
            _optional_attr(chunk, "document_id"),
            _optional_attr(chunk, "source_path"),
            chunk.source_name,
            chunk.chunk_index,
            chunk.content,
            embedding_json,
            _optional_attr(chunk, "page_start"),
            _optional_attr(chunk, "page_end"),
            _optional_attr(chunk, "extraction_method"),
            _optional_attr(chunk, "heading"),
            _optional_attr(chunk, "chunk_hash"),
            created_at,
            created_at,
        ),
    )
    return int(cursor.lastrowid)


def rebuild_chunks(
    chunk_embeddings: Iterable[tuple[ChunkLike, Sequence[float]]],
    db_path: Path = config.DATABASE_PATH,
) -> int:
    """Create the table, clear old rows, and insert chunk/embedding pairs."""
    connection = connect(db_path)
    try:
        create_chunks_table(connection)
        connection.execute("DELETE FROM chunks")
        row_count = 0
        for chunk, embedding in chunk_embeddings:
            insert_chunk(connection, chunk, embedding)
            row_count += 1
        connection.commit()
        return row_count
    finally:
        connection.close()


def _stored_chunk_from_row(row: sqlite3.Row) -> StoredChunk:
    return StoredChunk(
        id=int(row["id"]),
        source_name=str(row["source_name"]),
        chunk_index=int(row["chunk_index"]),
        content=str(row["content"]),
        embedding=[float(value) for value in json.loads(row["embedding_json"])],
        created_at=str(row["created_at"]),
        chunk_uid=row["chunk_uid"],
        document_id=row["document_id"],
        source_path=row["source_path"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        extraction_method=row["extraction_method"],
        heading=row["heading"],
        chunk_hash=row["chunk_hash"],
        updated_at=row["updated_at"],
    )


def fetch_all_chunks(db_path: Path = config.DATABASE_PATH) -> list[StoredChunk]:
    """Read all stored chunks back with embeddings decoded from JSON."""
    connection = connect(db_path)
    try:
        create_chunks_table(connection)
        rows = connection.execute(
            """
            SELECT
                id, chunk_uid, document_id, source_path, source_name, chunk_index,
                content, embedding_json, page_start, page_end, extraction_method,
                heading, chunk_hash, created_at, updated_at
            FROM chunks
            ORDER BY source_name, chunk_index, id
            """
        ).fetchall()
    finally:
        connection.close()

    return [_stored_chunk_from_row(row) for row in rows]


def count_chunks(db_path: Path = config.DATABASE_PATH) -> int:
    """Return the number of chunk rows in the database."""
    connection = connect(db_path)
    try:
        create_chunks_table(connection)
        row = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
    finally:
        connection.close()
    return int(row["count"])


def fetch_documents(db_path: Path = config.DATABASE_PATH) -> dict[str, StoredDocument]:
    """Return document manifest rows keyed by source path."""
    connection = connect(db_path)
    try:
        create_chunks_table(connection)
        rows = connection.execute(
            """
            SELECT document_id, source_path, source_name, file_hash, size_bytes,
                   modified_at, indexed_at, status, error
            FROM documents
            ORDER BY source_path
            """
        ).fetchall()
    finally:
        connection.close()

    return {
        str(row["source_path"]): StoredDocument(
            document_id=str(row["document_id"]),
            source_path=str(row["source_path"]),
            source_name=str(row["source_name"]),
            file_hash=str(row["file_hash"]),
            size_bytes=int(row["size_bytes"]),
            modified_at=str(row["modified_at"]),
            indexed_at=str(row["indexed_at"]),
            status=str(row["status"]),
            error=row["error"],
        )
        for row in rows
    }


def upsert_document(
    connection: sqlite3.Connection,
    *,
    document_id: str,
    source_path: str,
    source_name: str,
    file_hash: str,
    size_bytes: int,
    modified_at: str,
    status: str,
    error: str | None = None,
) -> None:
    """Insert or update one document manifest row."""
    indexed_at = datetime.now(UTC).isoformat()
    connection.execute(
        """
        INSERT INTO documents (
            document_id, source_path, source_name, file_hash, size_bytes,
            modified_at, indexed_at, status, error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id) DO UPDATE SET
            source_path = excluded.source_path,
            source_name = excluded.source_name,
            file_hash = excluded.file_hash,
            size_bytes = excluded.size_bytes,
            modified_at = excluded.modified_at,
            indexed_at = excluded.indexed_at,
            status = excluded.status,
            error = excluded.error
        """,
        (
            document_id,
            source_path,
            source_name,
            file_hash,
            size_bytes,
            modified_at,
            indexed_at,
            status,
            error,
        ),
    )


def delete_document(connection: sqlite3.Connection, document_id: str) -> None:
    """Delete one document and its chunk metadata."""
    connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))


def delete_document_chunks(connection: sqlite3.Connection, document_id: str) -> None:
    """Delete chunk metadata for one document."""
    connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))


def clear_all_metadata(connection: sqlite3.Connection) -> None:
    """Remove all document and chunk metadata rows."""
    connection.execute("DELETE FROM chunks")
    connection.execute("DELETE FROM documents")


def replace_document_chunks(
    connection: sqlite3.Connection,
    *,
    document_id: str,
    chunk_embeddings: Iterable[tuple[ChunkLike, Sequence[float]]],
) -> list[StoredChunk]:
    """Replace chunk metadata for one document and return stored rows."""
    delete_document_chunks(connection, document_id)
    row_ids: list[int] = []
    for chunk, embedding in chunk_embeddings:
        row_ids.append(insert_chunk(connection, chunk, embedding))

    if not row_ids:
        return []

    placeholders = ",".join("?" for _ in row_ids)
    rows = connection.execute(
        f"""
        SELECT
            id, chunk_uid, document_id, source_path, source_name, chunk_index,
            content, embedding_json, page_start, page_end, extraction_method,
            heading, chunk_hash, created_at, updated_at
        FROM chunks
        WHERE id IN ({placeholders})
        ORDER BY chunk_index, id
        """,
        row_ids,
    ).fetchall()

    return [_stored_chunk_from_row(row) for row in rows]
