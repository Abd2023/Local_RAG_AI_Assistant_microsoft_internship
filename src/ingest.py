"""Text loading and chunking for the local RAG knowledge base."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.foundry_client import generate_embeddings
from src.storage import count_chunks, rebuild_chunks

SUPPORTED_EXTENSIONS = {".md", ".txt"}
MAX_CHUNK_WORDS = int(config.CHUNK_TARGET_WORDS * 1.4)


@dataclass(frozen=True)
class DocumentChunk:
    """A chunk of source text with metadata needed for retrieval."""

    source_name: str
    chunk_index: int
    content: str

    def as_dict(self) -> dict[str, object]:
        """Return a plain dictionary representation for storage/debugging."""
        return {
            "source_name": self.source_name,
            "chunk_index": self.chunk_index,
            "content": self.content,
        }


def count_words(text: str) -> int:
    """Count non-whitespace word-like tokens in text."""
    return len(re.findall(r"\S+", text))


def iter_text_files(docs_path: Path = config.SAMPLE_DOCS_PATH) -> list[Path]:
    """Return supported text files from a directory in deterministic order."""
    return sorted(
        path
        for path in docs_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
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


def chunk_text(
    text: str,
    source_name: str,
    target_words: int = config.CHUNK_TARGET_WORDS,
    max_words: int = MAX_CHUNK_WORDS,
) -> list[DocumentChunk]:
    """Create document chunks from one source document."""
    blocks = split_text_blocks(text)
    chunks: list[DocumentChunk] = []
    current_blocks: list[str] = []
    current_word_count = 0

    def flush_chunk() -> None:
        nonlocal current_word_count
        if not current_blocks:
            return

        content = "\n\n".join(current_blocks).strip()
        if content:
            chunks.append(
                DocumentChunk(
                    source_name=source_name,
                    chunk_index=len(chunks),
                    content=content,
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
                    DocumentChunk(
                        source_name=source_name,
                        chunk_index=len(chunks),
                        content=split_block,
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


def load_document_chunks(docs_path: Path = config.SAMPLE_DOCS_PATH) -> list[DocumentChunk]:
    """Load all supported sample documents and return their chunks."""
    chunks: list[DocumentChunk] = []
    for path in iter_text_files(docs_path):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        chunks.extend(chunk_text(text, source_name=path.name))
    return chunks


def build_chunk_embeddings(chunks: list[DocumentChunk]) -> list[tuple[DocumentChunk, list[float]]]:
    """Generate embeddings for chunks and pair each vector with its source chunk."""
    texts = [chunk.content for chunk in chunks]
    embeddings = generate_embeddings(texts)
    return list(zip(chunks, embeddings, strict=True))


def ingest_sample_documents() -> dict[str, int]:
    """Load sample docs, embed chunks, and rebuild the local SQLite database."""
    files = iter_text_files(config.SAMPLE_DOCS_PATH)
    chunks = load_document_chunks(config.SAMPLE_DOCS_PATH)
    if not chunks:
        return {
            "files": len(files),
            "chunks": 0,
            "stored_rows": 0,
        }

    chunk_embeddings = build_chunk_embeddings(chunks)
    stored_rows = rebuild_chunks(chunk_embeddings)
    return {
        "files": len(files),
        "chunks": len(chunks),
        "stored_rows": stored_rows,
    }


def main() -> None:
    """Build the local SQLite knowledge base from sample documents."""
    files = iter_text_files(config.SAMPLE_DOCS_PATH)
    chunks = load_document_chunks()
    if not files:
        print(f"No supported .md or .txt files found in {config.SAMPLE_DOCS_PATH}")
        return
    if not chunks:
        print(f"No non-empty chunks found in {config.SAMPLE_DOCS_PATH}")
        return

    print(f"Found {len(files)} supported files in {config.SAMPLE_DOCS_PATH}")
    print(f"Prepared {len(chunks)} chunks for embedding")
    for chunk in chunks:
        preview = chunk.content.replace("\n", " ")[:90]
        print(
            f"{chunk.source_name}#{chunk.chunk_index} "
            f"({count_words(chunk.content)} words): {preview}"
        )

    chunk_embeddings = build_chunk_embeddings(chunks)
    stored_rows = rebuild_chunks(chunk_embeddings)
    final_row_count = count_chunks()
    print(f"Stored {stored_rows} chunks in {config.DATABASE_PATH}")
    print(f"Database row count after rebuild: {final_row_count}")


if __name__ == "__main__":
    main()
