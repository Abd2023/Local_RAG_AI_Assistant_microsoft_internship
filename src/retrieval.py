"""Vector similarity retrieval over stored document chunks."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from src import config
from src.foundry_client import generate_embedding
from src.storage import StoredChunk
from src.storage import search_lexical_chunks
from src.vector_store import VectorStoreError, search_vectors


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot be performed."""


@dataclass(frozen=True)
class RetrievalResult:
    """A ranked chunk returned by vector search."""

    chunk_id: int
    source_name: str
    chunk_index: int
    content: str
    similarity: float
    source_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    retrieval_score: float | None = None
    extraction_method: str | None = None
    heading: str | None = None
    chunk_uid: str | None = None
    rerank_score: float | None = None
    lexical_score: float | None = None
    hybrid_score: float | None = None
    ranking_method: str = "vector"


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Return cosine similarity for two vectors."""
    a = np.asarray(vector_a, dtype=np.float32)
    b = np.asarray(vector_b, dtype=np.float32)

    if a.size == 0 or b.size == 0:
        raise ValueError("Cannot compare empty vectors.")
    if a.shape != b.shape:
        raise ValueError(f"Vector shapes must match. Got {a.shape} and {b.shape}.")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def rank_chunks_by_similarity(
    query_embedding: Sequence[float],
    chunks: Sequence[StoredChunk],
    top_k: int = config.TOP_K,
) -> list[RetrievalResult]:
    """Rank stored chunks by similarity to a query embedding."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    ranked = [
        RetrievalResult(
            chunk_id=chunk.id,
            source_name=chunk.source_name,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            similarity=cosine_similarity(query_embedding, chunk.embedding),
            source_path=chunk.source_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            retrieval_score=cosine_similarity(query_embedding, chunk.embedding),
            extraction_method=chunk.extraction_method,
            heading=chunk.heading,
            chunk_uid=chunk.chunk_uid,
        )
        for chunk in chunks
    ]
    ranked.sort(key=lambda result: result.similarity, reverse=True)
    return ranked[:top_k]


def _result_from_vector_row(row: dict[str, object]) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=int(row.get("sqlite_id") or 0),
        source_name=str(row.get("source_name") or ""),
        chunk_index=int(row.get("chunk_index") or 0),
        content=str(row.get("content") or ""),
        similarity=float(row.get("similarity") or 0.0),
        source_path=str(row.get("source_path") or "") or None,
        page_start=row.get("page_start") if row.get("page_start") is not None else None,
        page_end=row.get("page_end") if row.get("page_end") is not None else None,
        retrieval_score=float(row.get("similarity") or 0.0),
        extraction_method=str(row.get("extraction_method") or "") or None,
        heading=str(row.get("heading") or "") or None,
        chunk_uid=str(row.get("chunk_uid") or "") or None,
        rerank_score=(
            float(row.get("rerank_score"))
            if row.get("rerank_score") is not None
            else None
        ),
        lexical_score=(
            float(row.get("lexical_score"))
            if row.get("lexical_score") is not None
            else None
        ),
        hybrid_score=(
            float(row.get("hybrid_score"))
            if row.get("hybrid_score") is not None
            else None
        ),
        ranking_method=str(row.get("ranking_method") or "vector"),
    )


def _merge_hybrid_rows(
    vector_rows: list[dict[str, object]],
    lexical_rows: list[dict[str, object]],
    top_k: int,
) -> list[RetrievalResult]:
    """Merge vector and exact-term candidates before cross-encoder reranking."""
    merged: dict[str, dict[str, object]] = {}

    def key_for(row: dict[str, object]) -> str:
        return str(row.get("chunk_uid") or row.get("sqlite_id") or (
            row.get("source_name"), row.get("chunk_index")
        ))

    for row in vector_rows:
        merged[key_for(row)] = dict(row)

    for row in lexical_rows:
        key = key_for(row)
        if key not in merged:
            merged[key] = dict(row)
        else:
            merged[key]["lexical_score"] = row.get("lexical_score")

    results: list[RetrievalResult] = []
    for row in merged.values():
        vector_score = float(row.get("similarity") or 0.0)
        lexical_score = float(row.get("lexical_score") or 0.0)
        vector_component = max(0.0, min(1.0, (vector_score + 1.0) / 2.0))
        hybrid_score = (0.65 * vector_component) + (0.35 * lexical_score)
        enriched = dict(row)
        enriched.update(
            {
                "lexical_score": lexical_score,
                "hybrid_score": hybrid_score,
                "ranking_method": "hybrid",
            }
        )
        results.append(_result_from_vector_row(enriched))

    results.sort(
        key=lambda result: (
            result.hybrid_score if result.hybrid_score is not None else float("-inf"),
            result.similarity,
            result.lexical_score if result.lexical_score is not None else 0.0,
        ),
        reverse=True,
    )
    return results[:top_k]


def _trace_span(trace: object | None, name: str):
    if trace is not None and hasattr(trace, "span"):
        return trace.span(name)

    class _NoOpSpan:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    return _NoOpSpan()


def retrieve_top_chunks(
    query: str,
    top_k: int = config.TOP_K,
    db_path: Path = config.VECTOR_DB_PATH,
    metadata_db_path: Path | None = None,
    trace: object | None = None,
) -> list[RetrievalResult]:
    """Embed a query and merge semantic and exact-term retrieval candidates."""
    if not query.strip():
        raise ValueError("Query must not be empty.")

    with _trace_span(trace, "query_embedding"):
        query_embedding = generate_embedding(query)

    try:
        with _trace_span(trace, "vector_search"):
            rows = search_vectors(query_embedding, top_k=top_k, vector_db_path=db_path)
    except VectorStoreError as exc:
        raise RetrievalError(f"{exc}") from exc

    lexical_rows: list[dict[str, object]] = []
    if config.HYBRID_RETRIEVAL_ENABLED:
        lexical_path = metadata_db_path
        if lexical_path is None:
            lexical_path = (
                config.DATABASE_PATH
                if db_path == config.VECTOR_DB_PATH
                else db_path.parent / "rag.db"
            )
        lexical_rows = search_lexical_chunks(
            query,
            top_k=config.LEXICAL_CANDIDATE_K,
            db_path=lexical_path,
        )

    results = _merge_hybrid_rows(rows, lexical_rows, top_k)
    if trace is not None and hasattr(trace, "set"):
        trace.set("retrieval_backend", "hybrid" if config.HYBRID_RETRIEVAL_ENABLED else "vector")
        trace.set("vector_retrieval_count", len(rows))
        trace.set("lexical_retrieval_count", len(lexical_rows))
        trace.set("retrieval_count", len(results))
    return results


def main() -> None:
    """Run a simple retrieval query from the command line."""
    query = " ".join(sys.argv[1:]).strip() or "What time does the daily standup start?"
    results = retrieve_top_chunks(query)

    print(f"Query: {query}")
    for index, result in enumerate(results, start=1):
        preview = result.content.replace("\n", " ")[:120]
        print(
            f"{index}. {result.source_name}#{result.chunk_index} "
            f"score={result.similarity:.4f}: {preview}"
        )


if __name__ == "__main__":
    main()
