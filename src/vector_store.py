"""LanceDB vector storage for document chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src import config
from src.storage import StoredChunk

TABLE_NAME = "chunks"


class VectorStoreError(RuntimeError):
    """Raised when vector search/storage cannot be performed."""


def _import_lancedb():
    try:
        import lancedb
    except ImportError as exc:  # pragma: no cover - dependency is installed in normal setup
        raise VectorStoreError("LanceDB is required for vector search. Run '.\\run.ps1 setup'.") from exc
    return lancedb


def _import_qdrant():
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:  # pragma: no cover - dependency is installed in Docker setup
        raise VectorStoreError("qdrant-client is required when VECTOR_BACKEND=qdrant.") from exc
    return QdrantClient, models


def _qdrant_client():
    qdrant_client, _ = _import_qdrant()
    return qdrant_client(url=config.QDRANT_URL)


def _qdrant_collection_exists(client) -> bool:
    if hasattr(client, "collection_exists"):
        return bool(client.collection_exists(config.QDRANT_COLLECTION))
    try:
        client.get_collection(config.QDRANT_COLLECTION)
    except Exception:
        return False
    return True


def _qdrant_payload(record: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in record.items() if key != "vector"}


def _qdrant_points(chunk_embeddings: Sequence[tuple[StoredChunk, Sequence[float]]]):
    _, models = _import_qdrant()
    points = []
    for chunk, embedding in chunk_embeddings:
        record = chunk_vector_record(chunk, embedding)
        points.append(
            models.PointStruct(
                id=int(chunk.id),
                vector=record["vector"],
                payload=_qdrant_payload(record),
            )
        )
    return points


def _qdrant_ensure_collection(client, dimension: int) -> None:
    if _qdrant_collection_exists(client):
        return
    _, models = _import_qdrant()
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """Return a unit-length vector for cosine-equivalent LanceDB search."""
    array = np.asarray(vector, dtype=np.float32)
    if array.size == 0:
        raise ValueError("Cannot normalize an empty vector.")
    norm = np.linalg.norm(array)
    if norm == 0:
        return [0.0 for _ in array.tolist()]
    return (array / norm).astype(float).tolist()


def squared_l2_distance_to_cosine_similarity(distance: float) -> float:
    """Convert LanceDB squared L2 distance on unit vectors to cosine similarity."""
    return 1.0 - (float(distance) / 2.0)


def connect(vector_db_path: Path = config.VECTOR_DB_PATH):
    """Connect to the local LanceDB database."""
    vector_db_path.mkdir(parents=True, exist_ok=True)
    return _import_lancedb().connect(str(vector_db_path))


def _open_table(db):
    try:
        return db.open_table(TABLE_NAME)
    except Exception as exc:
        raise VectorStoreError(
            f"No LanceDB table found in {config.VECTOR_DB_PATH}. Run `python -m src.ingest` before retrieval."
        ) from exc


def _list_tables(db) -> list[str]:
    if hasattr(db, "list_tables"):
        response = db.list_tables()
        if hasattr(response, "tables"):
            return list(response.tables)
        return list(response)
    return list(db.table_names())


def _drop_table_if_exists(db) -> None:
    if TABLE_NAME in _list_tables(db):
        db.drop_table(TABLE_NAME, ignore_missing=True)


def drop_vector_table(vector_db_path: Path = config.VECTOR_DB_PATH) -> None:
    """Drop the vector table if it exists."""
    if config.VECTOR_BACKEND == "qdrant":
        client = _qdrant_client()
        if _qdrant_collection_exists(client):
            client.delete_collection(config.QDRANT_COLLECTION)
        return

    db = connect(vector_db_path)
    _drop_table_if_exists(db)


def _safe_str(value: object | None) -> str:
    return "" if value is None else str(value)


def _page_value(value: int | None) -> int:
    return int(value) if value is not None else -1


def chunk_vector_record(chunk: StoredChunk, embedding: Sequence[float]) -> dict[str, object]:
    """Build a LanceDB row from stored chunk metadata and an embedding."""
    return {
        "chunk_uid": _safe_str(chunk.chunk_uid or chunk.id),
        "sqlite_id": int(chunk.id),
        "document_id": _safe_str(chunk.document_id),
        "source_path": _safe_str(chunk.source_path),
        "source_name": chunk.source_name,
        "chunk_index": int(chunk.chunk_index),
        "content": chunk.content,
        "vector": normalize_vector(embedding),
        "page_start": _page_value(chunk.page_start),
        "page_end": _page_value(chunk.page_end),
        "extraction_method": _safe_str(chunk.extraction_method),
        "heading": _safe_str(chunk.heading),
        "chunk_hash": _safe_str(chunk.chunk_hash),
    }


def rebuild_vector_table(
    chunk_embeddings: Sequence[tuple[StoredChunk, Sequence[float]]],
    vector_db_path: Path = config.VECTOR_DB_PATH,
) -> int:
    """Replace the LanceDB table with the provided chunk vectors."""
    if config.VECTOR_BACKEND == "qdrant":
        client = _qdrant_client()
        if _qdrant_collection_exists(client):
            client.delete_collection(config.QDRANT_COLLECTION)
        if not chunk_embeddings:
            return 0
        first_embedding = normalize_vector(chunk_embeddings[0][1])
        _qdrant_ensure_collection(client, len(first_embedding))
        client.upsert(
            collection_name=config.QDRANT_COLLECTION,
            points=_qdrant_points(chunk_embeddings),
            wait=True,
        )
        return len(chunk_embeddings)

    db = connect(vector_db_path)
    _drop_table_if_exists(db)
    records = [chunk_vector_record(chunk, embedding) for chunk, embedding in chunk_embeddings]
    if not records:
        return 0
    db.create_table(TABLE_NAME, data=records, mode="overwrite")
    return len(records)


def delete_document_vectors(document_id: str, vector_db_path: Path = config.VECTOR_DB_PATH) -> None:
    """Delete vector rows for one document if the table exists."""
    if config.VECTOR_BACKEND == "qdrant":
        client = _qdrant_client()
        if not _qdrant_collection_exists(client):
            return
        _, models = _import_qdrant()
        client.delete(
            collection_name=config.QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
        return

    db = connect(vector_db_path)
    if TABLE_NAME not in _list_tables(db):
        return
    table = db.open_table(TABLE_NAME)
    escaped = document_id.replace("'", "''")
    table.delete(f"document_id = '{escaped}'")


def add_chunk_vectors(
    chunk_embeddings: Sequence[tuple[StoredChunk, Sequence[float]]],
    vector_db_path: Path = config.VECTOR_DB_PATH,
) -> int:
    """Append chunk vectors to the LanceDB table, creating it if needed."""
    if not chunk_embeddings:
        return 0
    if config.VECTOR_BACKEND == "qdrant":
        client = _qdrant_client()
        first_embedding = normalize_vector(chunk_embeddings[0][1])
        _qdrant_ensure_collection(client, len(first_embedding))
        client.upsert(
            collection_name=config.QDRANT_COLLECTION,
            points=_qdrant_points(chunk_embeddings),
            wait=True,
        )
        return len(chunk_embeddings)

    db = connect(vector_db_path)
    records = [chunk_vector_record(chunk, embedding) for chunk, embedding in chunk_embeddings]
    if TABLE_NAME not in _list_tables(db):
        db.create_table(TABLE_NAME, data=records, mode="overwrite")
    else:
        db.open_table(TABLE_NAME).add(records)
    return len(records)


def replace_document_vectors(
    document_id: str,
    chunk_embeddings: Sequence[tuple[StoredChunk, Sequence[float]]],
    vector_db_path: Path = config.VECTOR_DB_PATH,
) -> int:
    """Replace vectors for one document."""
    delete_document_vectors(document_id, vector_db_path)
    return add_chunk_vectors(chunk_embeddings, vector_db_path)


def count_vectors(vector_db_path: Path = config.VECTOR_DB_PATH) -> int:
    """Return the number of vector rows."""
    if config.VECTOR_BACKEND == "qdrant":
        client = _qdrant_client()
        if not _qdrant_collection_exists(client):
            return 0
        return int(client.count(collection_name=config.QDRANT_COLLECTION, exact=True).count)

    db = connect(vector_db_path)
    if TABLE_NAME not in _list_tables(db):
        return 0
    return int(db.open_table(TABLE_NAME).count_rows())


def _optional_page(value: object) -> int | None:
    number = int(value)
    return number if number > 0 else None


def search_vectors(
    query_embedding: Sequence[float],
    top_k: int = config.TOP_K,
    vector_db_path: Path = config.VECTOR_DB_PATH,
) -> list[dict[str, object]]:
    """Search the configured vector backend and return ranked chunk dictionaries."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if config.VECTOR_BACKEND == "qdrant":
        client = _qdrant_client()
        if not _qdrant_collection_exists(client):
            raise VectorStoreError(
                f"No Qdrant collection found at {config.QDRANT_URL}. Run ingestion first."
            )
        query_vector = normalize_vector(query_embedding)
        try:
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=config.QDRANT_COLLECTION,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True,
                )
                points = response.points
            else:  # pragma: no cover - compatibility with older qdrant-client releases
                points = client.search(
                    collection_name=config.QDRANT_COLLECTION,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True,
                )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc

        rows: list[dict[str, object]] = []
        for point in points:
            payload = dict(point.payload or {})
            payload["sqlite_id"] = int(payload.get("sqlite_id") or point.id)
            payload["similarity"] = float(point.score)
            payload["page_start"] = _optional_page(payload.get("page_start", -1))
            payload["page_end"] = _optional_page(payload.get("page_end", -1))
            rows.append(payload)
        return rows

    db = connect(vector_db_path)
    if TABLE_NAME not in _list_tables(db):
        raise VectorStoreError(
            f"No LanceDB table found in {vector_db_path}. Run `python -m src.ingest` before retrieval."
        )

    table = _open_table(db)
    if table.count_rows() == 0:
        raise VectorStoreError(
            f"No vectors found in {vector_db_path}. Run `python -m src.ingest` before retrieval."
        )

    query_vector = normalize_vector(query_embedding)
    rows = table.search(query_vector).limit(top_k).to_list()
    for row in rows:
        row["similarity"] = squared_l2_distance_to_cosine_similarity(float(row.get("_distance", 0.0)))
        row["page_start"] = _optional_page(row.get("page_start", -1))
        row["page_end"] = _optional_page(row.get("page_end", -1))
    return rows
