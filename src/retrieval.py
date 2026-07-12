"""Vector similarity retrieval over stored document chunks."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src import config
from src.foundry_client import generate_embedding
from src.storage import StoredChunk, fetch_all_chunks


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
        )
        for chunk in chunks
    ]
    ranked.sort(key=lambda result: result.similarity, reverse=True)
    return ranked[:top_k]


def retrieve_top_chunks(
    query: str,
    top_k: int = config.TOP_K,
    db_path: Path = config.DATABASE_PATH,
) -> list[RetrievalResult]:
    """Embed a query and return the most similar stored chunks."""
    if not query.strip():
        raise ValueError("Query must not be empty.")

    chunks = fetch_all_chunks(db_path)
    if not chunks:
        raise RetrievalError(
            f"No chunks found in {db_path}. Run `python -m src.ingest` before retrieval."
        )

    query_embedding = generate_embedding(query)
    return rank_chunks_by_similarity(query_embedding, chunks, top_k=top_k)


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
