"""Text loading, chunking, and indexing for the local RAG knowledge base."""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src import config
from src.document_loaders import (
    LoadedTextBlock,
    iter_supported_files,
    load_document,
)
from src.foundry_client import generate_embeddings
from src.storage import (
    clear_all_metadata,
    connect,
    count_chunks,
    create_chunks_table,
    delete_document,
    fetch_documents,
    replace_document_chunks,
    upsert_document,
)
from src.vector_store import (
    count_vectors,
    delete_document_vectors,
    drop_vector_table,
    replace_document_vectors,
)

TEXT_EXTENSIONS = {".md", ".txt"}
MAX_CHUNK_WORDS = int(config.CHUNK_TARGET_WORDS * 1.4)


@dataclass(frozen=True)
class DocumentChunk:
    """A chunk of source text with metadata needed for retrieval."""

    source_name: str
    chunk_index: int
    content: str
    source_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    extraction_method: str | None = None
    heading: str | None = None
    document_id: str | None = None
    chunk_hash: str | None = None
    chunk_uid: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a plain dictionary representation for storage/debugging."""
        return {
            "source_name": self.source_name,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "source_path": self.source_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "extraction_method": self.extraction_method,
            "heading": self.heading,
            "document_id": self.document_id,
            "chunk_hash": self.chunk_hash,
            "chunk_uid": self.chunk_uid,
        }


@dataclass(frozen=True)
class FileFingerprint:
    """Stable file metadata used for incremental indexing decisions."""

    document_id: str
    source_path: str
    source_name: str
    file_hash: str
    size_bytes: int
    modified_at: str


def count_words(text: str) -> int:
    """Count non-whitespace word-like tokens in text."""
    return len(re.findall(r"\S+", text))


def iter_text_files(docs_path: Path = config.SAMPLE_DOCS_PATH) -> list[Path]:
    """Return supported text files from a directory in deterministic order."""
    return sorted(
        path
        for path in docs_path.iterdir()
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS
    )


def split_text_blocks(text: str) -> list[str]:
    """Split text into paragraph and Markdown-heading blocks."""
    blocks: list[str] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            block = " ".join(paragraph_lines).strip()
            if block:
                blocks.append(block)
            paragraph_lines.clear()

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue

        if line.startswith("#"):
            flush_paragraph()
            blocks.append(line)
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    return blocks


def split_oversized_block(block: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Split one oversized block into word windows."""
    words = block.split()
    if len(words) <= max_words:
        return [block]

    return [
        " ".join(words[index : index + max_words]).strip()
        for index in range(0, len(words), max_words)
    ]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _heading_from_content(content: str, fallback: str | None = None) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def _make_chunk(
    *,
    content: str,
    source_name: str,
    chunk_index: int,
    source_path: str | None,
    page_start: int | None,
    page_end: int | None,
    extraction_method: str | None,
    heading: str | None,
    document_id: str | None,
) -> DocumentChunk:
    chunk_hash = _content_hash(content)
    chunk_uid = f"{document_id}:{chunk_index}:{chunk_hash[:16]}" if document_id else None
    return DocumentChunk(
        source_name=source_name,
        chunk_index=chunk_index,
        content=content,
        source_path=source_path,
        page_start=page_start,
        page_end=page_end,
        extraction_method=extraction_method,
        heading=_heading_from_content(content, heading),
        document_id=document_id,
        chunk_hash=chunk_hash,
        chunk_uid=chunk_uid,
    )


def chunk_text(
    text: str,
    source_name: str,
    target_words: int = config.CHUNK_TARGET_WORDS,
    max_words: int = MAX_CHUNK_WORDS,
    *,
    source_path: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    extraction_method: str | None = None,
    heading: str | None = None,
    document_id: str | None = None,
    start_index: int = 0,
) -> list[DocumentChunk]:
    """Create document chunks from one source text block."""
    blocks = split_text_blocks(text)
    chunks: list[DocumentChunk] = []
    current_blocks: list[str] = []
    current_word_count = 0

    def next_chunk_index() -> int:
        return start_index + len(chunks)

    def flush_chunk() -> None:
        nonlocal current_word_count
        if not current_blocks:
            return

        content = "\n\n".join(current_blocks).strip()
        if content:
            chunks.append(
                _make_chunk(
                    content=content,
                    source_name=source_name,
                    chunk_index=next_chunk_index(),
                    source_path=source_path,
                    page_start=page_start,
                    page_end=page_end,
                    extraction_method=extraction_method,
                    heading=heading,
                    document_id=document_id,
                )
            )
        current_blocks.clear()
        current_word_count = 0

    for block in blocks:
        block_word_count = count_words(block)
        if block_word_count == 0:
            continue

        if block_word_count > max_words:
            flush_chunk()
            for split_block in split_oversized_block(block, max_words=max_words):
                chunks.append(
                    _make_chunk(
                        content=split_block,
                        source_name=source_name,
                        chunk_index=next_chunk_index(),
                        source_path=source_path,
                        page_start=page_start,
                        page_end=page_end,
                        extraction_method=extraction_method,
                        heading=heading,
                        document_id=document_id,
                    )
                )
            continue

        if current_blocks and (
            current_word_count >= target_words
            or current_word_count + block_word_count > max_words
        ):
            flush_chunk()

        current_blocks.append(block)
        current_word_count += block_word_count

    flush_chunk()
    return chunks


def document_id_for_path(path: Path) -> str:
    """Return a stable document id for a path."""
    normalized_path = str(path.resolve()).lower()
    return hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()


def fingerprint_file(path: Path) -> FileFingerprint:
    """Return hash, size, mtime, and identity metadata for one source file."""
    data = path.read_bytes()
    stat = path.stat()
    return FileFingerprint(
        document_id=document_id_for_path(path),
        source_path=str(path.resolve()),
        source_name=path.name,
        file_hash=hashlib.sha256(data).hexdigest(),
        size_bytes=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    )


def _chunks_from_blocks(
    blocks: list[LoadedTextBlock],
    *,
    document_id: str,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for block in blocks:
        chunks.extend(
            chunk_text(
                block.content,
                source_name=block.source_name,
                source_path=block.source_path,
                page_start=block.page_start,
                page_end=block.page_end,
                extraction_method=block.extraction_method,
                heading=block.heading,
                document_id=document_id,
                start_index=len(chunks),
            )
        )
    return chunks


def load_document_chunks(docs_path: Path = config.SAMPLE_DOCS_PATH) -> list[DocumentChunk]:
    """Load all supported sample documents and return their chunks."""
    chunks: list[DocumentChunk] = []
    for path in iter_supported_files(docs_path):
        blocks = load_document(path)
        if not blocks:
            continue
        chunks.extend(_chunks_from_blocks(blocks, document_id=document_id_for_path(path)))
    return chunks


def build_chunk_embeddings(chunks: list[DocumentChunk]) -> list[tuple[DocumentChunk, list[float]]]:
    """Generate embeddings for chunks and pair each vector with its source chunk."""
    texts = [chunk.content for chunk in chunks]
    embeddings = generate_embeddings(texts)
    return list(zip(chunks, embeddings, strict=True))


def _is_unchanged(stored_file_hash: str, fingerprint: FileFingerprint) -> bool:
    return stored_file_hash == fingerprint.file_hash


def ingest_documents(
    docs_path: Path = config.SAMPLE_DOCS_PATH,
    *,
    db_path: Path = config.DATABASE_PATH,
    vector_db_path: Path = config.VECTOR_DB_PATH,
    rebuild: bool = False,
) -> dict[str, int]:
    """Incrementally index supported documents into SQLite metadata and LanceDB."""
    files = iter_supported_files(docs_path)
    connection = connect(db_path)
    summary = {
        "files": len(files),
        "indexed_files": 0,
        "skipped_files": 0,
        "removed_files": 0,
        "error_files": 0,
        "chunks": 0,
        "stored_rows": 0,
        "stored_vectors": 0,
        "final_rows": 0,
        "final_vectors": 0,
    }

    try:
        create_chunks_table(connection)
        existing_documents = fetch_documents(db_path)
        if rebuild or (not existing_documents and count_chunks(db_path) > 0):
            clear_all_metadata(connection)
            connection.commit()
            drop_vector_table(vector_db_path)
            existing_documents = {}

        current_paths = {str(path.resolve()): path for path in files}
        for source_path, document in existing_documents.items():
            if source_path not in current_paths:
                delete_document_vectors(document.document_id, vector_db_path)
                delete_document(connection, document.document_id)
                summary["removed_files"] += 1

        connection.commit()

        for path in files:
            fingerprint = fingerprint_file(path)
            stored_document = existing_documents.get(fingerprint.source_path)
            if (
                not rebuild
                and stored_document is not None
                and _is_unchanged(stored_document.file_hash, fingerprint)
            ):
                summary["skipped_files"] += 1
                continue

            try:
                blocks = load_document(path)
                chunks = _chunks_from_blocks(blocks, document_id=fingerprint.document_id)
                chunk_embeddings = build_chunk_embeddings(chunks) if chunks else []

                upsert_document(
                    connection,
                    document_id=fingerprint.document_id,
                    source_path=fingerprint.source_path,
                    source_name=fingerprint.source_name,
                    file_hash=fingerprint.file_hash,
                    size_bytes=fingerprint.size_bytes,
                    modified_at=fingerprint.modified_at,
                    status="indexed" if chunks else "empty",
                    error=None,
                )
                stored_chunks = replace_document_chunks(
                    connection,
                    document_id=fingerprint.document_id,
                    chunk_embeddings=chunk_embeddings,
                )
                connection.commit()

                vector_count = replace_document_vectors(
                    fingerprint.document_id,
                    list(zip(stored_chunks, [embedding for _, embedding in chunk_embeddings], strict=True)),
                    vector_db_path,
                )
                summary["indexed_files"] += 1
                summary["chunks"] += len(chunks)
                summary["stored_rows"] += len(stored_chunks)
                summary["stored_vectors"] += vector_count
            except Exception as exc:
                upsert_document(
                    connection,
                    document_id=fingerprint.document_id,
                    source_path=fingerprint.source_path,
                    source_name=fingerprint.source_name,
                    file_hash=fingerprint.file_hash,
                    size_bytes=fingerprint.size_bytes,
                    modified_at=fingerprint.modified_at,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                connection.commit()
                summary["error_files"] += 1
                print(f"Error indexing {path.name}: {type(exc).__name__}: {exc}")
    finally:
        connection.close()

    summary["final_rows"] = count_chunks(db_path)
    summary["final_vectors"] = count_vectors(vector_db_path)
    return summary


def ingest_sample_documents() -> dict[str, int]:
    """Load sample docs, embed chunks, and incrementally update the local index."""
    return ingest_documents(config.SAMPLE_DOCS_PATH)


def main() -> None:
    """Build or update the local knowledge base from sample documents."""
    parser = argparse.ArgumentParser(description="Index local documents for RAG retrieval.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild all metadata and vectors.")
    args = parser.parse_args()

    summary = ingest_documents(rebuild=args.rebuild)
    print(f"Found {summary['files']} supported files in {config.SAMPLE_DOCS_PATH}")
    print(f"Indexed files: {summary['indexed_files']}")
    print(f"Skipped unchanged files: {summary['skipped_files']}")
    print(f"Removed files: {summary['removed_files']}")
    print(f"Files with errors: {summary['error_files']}")
    print(f"New or changed chunks embedded: {summary['chunks']}")
    print(f"Stored rows written this run: {summary['stored_rows']}")
    print(f"Stored vectors written this run: {summary['stored_vectors']}")
    print(f"Final SQLite chunk rows: {summary['final_rows']}")
    print(f"Final {config.VECTOR_BACKEND} vector rows: {summary['final_vectors']}")


if __name__ == "__main__":
    main()
