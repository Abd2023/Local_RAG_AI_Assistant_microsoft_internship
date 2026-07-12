"""Text loading and chunking for the local RAG knowledge base."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src import config

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


def main() -> None:
    """Print a chunking summary for the configured sample documents."""
    chunks = load_document_chunks()
    if not chunks:
        print(f"No chunks found in {config.SAMPLE_DOCS_PATH}")
        return

    print(f"Loaded {len(chunks)} chunks from {config.SAMPLE_DOCS_PATH}")
    for chunk in chunks:
        preview = chunk.content.replace("\n", " ")[:90]
        print(
            f"{chunk.source_name}#{chunk.chunk_index} "
            f"({count_words(chunk.content)} words): {preview}"
        )


if __name__ == "__main__":
    main()
