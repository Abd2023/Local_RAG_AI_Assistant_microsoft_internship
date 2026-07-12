"""SQLite storage helpers for document chunks and embeddings."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from src import config

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class StoredChunk:
    """A chunk row read back from SQLite."""

    id: int
    source_name: str
    chunk_index: int
    content: str
    embedding: list[float]
    created_at: str


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
    """Create the chunks table if it does not exist."""
    connection.execute(SCHEMA_SQL)
    connection.commit()


def clear_chunks(connection: sqlite3.Connection) -> None:
    """Remove all existing chunk rows."""
    connection.execute("DELETE FROM chunks")
    connection.commit()


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
        INSERT INTO chunks (source_name, chunk_index, content, embedding_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            chunk.source_name,
            chunk.chunk_index,
            chunk.content,
            embedding_json,
            created_at,
        ),
    )
    return int(cursor.lastrowid)


def rebuild_chunks(
    chunk_embeddings: Iterable[tuple[ChunkLike, Sequence[float]]],
    db_path: Path = config.DATABASE_PATH,
) -> int:
    """Create the table, clear old rows, and insert chunk/embedding pairs."""
    with connect(db_path) as connection:
        create_chunks_table(connection)
        connection.execute("DELETE FROM chunks")
        row_count = 0
        for chunk, embedding in chunk_embeddings:
            insert_chunk(connection, chunk, embedding)
            row_count += 1
        connection.commit()
        return row_count


def fetch_all_chunks(db_path: Path = config.DATABASE_PATH) -> list[StoredChunk]:
    """Read all stored chunks back with embeddings decoded from JSON."""
    with connect(db_path) as connection:
        create_chunks_table(connection)
        rows = connection.execute(
            """
            SELECT id, source_name, chunk_index, content, embedding_json, created_at
            FROM chunks
            ORDER BY source_name, chunk_index, id
            """
        ).fetchall()

    return [
        StoredChunk(
            id=int(row["id"]),
            source_name=str(row["source_name"]),
            chunk_index=int(row["chunk_index"]),
            content=str(row["content"]),
            embedding=[float(value) for value in json.loads(row["embedding_json"])],
            created_at=str(row["created_at"]),
        )
        for row in rows
    ]


def count_chunks(db_path: Path = config.DATABASE_PATH) -> int:
    """Return the number of chunk rows in the database."""
    with connect(db_path) as connection:
        create_chunks_table(connection)
        row = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
    return int(row["count"])
