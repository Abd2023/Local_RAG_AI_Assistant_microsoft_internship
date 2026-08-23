from pathlib import Path

from src.ingest import (
    chunk_text,
    count_words,
    iter_text_files,
    load_document_chunks,
)


def test_chunk_text_preserves_source_metadata_and_sequential_indexes() -> None:
    text = """# Schedule

Daily activities begin promptly.

Attendance is recorded every morning.
"""

    chunks = chunk_text(
        text,
        source_name="schedule.md",
        target_words=4,
        max_words=8,
    )

    assert [chunk.source_name for chunk in chunks] == [
        "schedule.md",
        "schedule.md",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert chunks[0].content == "# Schedule\n\nDaily activities begin promptly."
    assert chunks[1].content == "Attendance is recorded every morning."


def test_chunk_text_splits_oversized_blocks_at_max_words() -> None:
    text = " ".join(f"word{index}" for index in range(10))

    chunks = chunk_text(
        text,
        source_name="large.txt",
        target_words=3,
        max_words=4,
    )

    assert [count_words(chunk.content) for chunk in chunks] == [4, 4, 2]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert " ".join(chunk.content for chunk in chunks) == text


def test_chunk_text_overlaps_normal_markdown_chunk_boundaries() -> None:
    text = "\n\n".join(
        [
            "one two three four five six",
            "seven eight nine ten eleven twelve",
            "thirteen fourteen fifteen sixteen seventeen eighteen",
        ]
    )

    chunks = chunk_text(text, source_name="guide.md", target_words=10, max_words=32)

    assert len(chunks) == 2
    assert "six" in chunks[0].content
    assert "six" in chunks[1].content
    assert "seven" in chunks[1].content


def test_load_document_chunks_reads_supported_files_in_name_order(
    tmp_path: Path,
) -> None:
    (tmp_path / "b.txt").write_text("Second document.", encoding="utf-8")
    (tmp_path / "a.md").write_text("# First\n\nFirst document.", encoding="utf-8")
    (tmp_path / "ignored.json").write_text('{"ignored": true}', encoding="utf-8")
    (tmp_path / "empty.md").write_text("  \n", encoding="utf-8")

    files = iter_text_files(tmp_path)
    chunks = load_document_chunks(tmp_path)

    assert [path.name for path in files] == ["a.md", "b.txt", "empty.md"]
    assert [chunk.source_name for chunk in chunks] == ["a.md", "b.txt"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 0]


def test_chunk_text_returns_no_chunks_for_whitespace() -> None:
    assert chunk_text(" \n\t ", source_name="empty.txt") == []
