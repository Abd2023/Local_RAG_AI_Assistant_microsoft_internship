"""Interactive command-line interface for the Local RAG AI Assistant."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shlex

from src.ingest import add_document_files, replace_document_files, reset_knowledge_base
from src.rag import answer_query

EXIT_COMMANDS = {"exit", "quit"}
UPLOAD_COMMAND = "/upload"
CLEAN_COMMAND = "/clean"


def _parse_path_command(
    command: str,
    *,
    command_name: str,
    allow_empty: bool,
    usage: str,
) -> list[Path] | None:
    """Parse a slash command with quoted Windows paths."""
    stripped = command.strip()
    lowered = stripped.lower()
    if lowered != command_name and not lowered.startswith(f"{command_name} "):
        return None
    remainder = stripped[len(command_name) :].strip()
    if not remainder:
        if allow_empty:
            return []
        raise ValueError(usage)

    try:
        tokens = shlex.split(remainder, posix=False)
    except ValueError as exc:
        raise ValueError(f"Could not parse paths: {exc}") from exc

    paths = [Path(token.strip("\"'")) for token in tokens if token.strip("\"'")]
    if not paths:
        if allow_empty:
            return []
        raise ValueError("At least one path is required.")
    return paths


def parse_upload_command(command: str) -> list[Path] | None:
    """Parse a quoted Windows `/upload` command, returning None for normal questions."""
    return _parse_path_command(
        command,
        command_name=UPLOAD_COMMAND,
        allow_empty=False,
        usage="Usage: /upload 'C:\\path\\file.pdf' ['C:\\path\\other.pdf']",
    )


def parse_clean_command(command: str) -> list[Path] | None:
    """Parse `/clean` with optional replacement document paths."""
    return _parse_path_command(
        command,
        command_name=CLEAN_COMMAND,
        allow_empty=True,
        usage="Usage: /clean ['C:\\path\\file.pdf' | 'C:\\path\\documents-folder']",
    )


def print_clean_summary(summary: dict[str, object]) -> None:
    """Print a reset summary."""
    print("\nKnowledge base cleared.")
    print(f"Removed uploaded documents: {summary.get('removed_uploads', 0)}")
    print(f"Stored rows: {summary.get('final_rows', 0)}")
    print(f"Stored vectors: {summary.get('final_vectors', 0)}")


def print_upload_summary(summary: dict[str, object]) -> None:
    """Print staged files and their incremental indexing outcomes."""
    reset_summary = summary.get("reset")
    if isinstance(reset_summary, dict):
        print_clean_summary(reset_summary)
    print(f"\nUpload complete: {len(summary.get('staged', []))} document(s) staged.")
    print(
        "Indexed: {indexed_files} | Skipped: {skipped_files} | Errors: {error_files}".format(
            indexed_files=summary.get("indexed_files", 0),
            skipped_files=summary.get("skipped_files", 0),
            error_files=summary.get("error_files", 0),
        )
    )
    outcomes = {
        str(item.get("document_name")): item
        for item in summary.get("documents", [])
        if isinstance(item, dict)
    }
    for staged in summary.get("staged", []):
        if not isinstance(staged, dict):
            continue
        name = str(staged.get("document_name", "document"))
        outcome = outcomes.get(name, {})
        print(
            f"- {name}: {outcome.get('status', 'staged')} "
            f"chunks={outcome.get('chunks', 0)} vectors={outcome.get('vectors', 0)}"
        )


def print_answer(result: dict[str, object]) -> None:
    """Print an answer result in a readable terminal format."""
    answer = str(result["answer"]).split("\n\nSources:", maxsplit=1)[0].strip()
    print("\nAnswer:")
    print(answer)

    trace_id = result.get("trace_id")
    if trace_id:
        print(f"\nTrace: {trace_id}")

    retrieved_chunks = result.get("retrieved_chunks", [])
    if retrieved_chunks:
        print("\nSources:")
        for chunk in retrieved_chunks:
            source_name = chunk["source_name"]
            chunk_number = chunk["chunk_number"]
            similarity = chunk["similarity"]
            similarity_percent = chunk["similarity_percent"]
            print(
                f"- {source_name} "
                f"(chunk {chunk_number}, cosine similarity {similarity:.3f} / {similarity_percent:.1f}%)"
            )


def run_cli(input_func: Callable[[str], str] = input) -> None:
    """Run the interactive CLI loop."""
    print("Local RAG AI Assistant")
    print(
        "Ask a question about the indexed documents. "
        "Use /upload to add files or /clean to start fresh. "
        "Type 'exit' or 'quit' to close."
    )

    while True:
        try:
            question = input_func("\nQuestion> ").strip()
        except EOFError:
            print("\nGoodbye.")
            return

        if not question:
            print("Enter a question, or type 'exit' to quit.")
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Goodbye.")
            return

        try:
            clean_paths = parse_clean_command(question)
        except ValueError as exc:
            print(f"\nClean error: {exc}")
            continue

        if clean_paths is not None:
            try:
                if clean_paths:
                    print_upload_summary(replace_document_files(clean_paths))
                else:
                    print_clean_summary(reset_knowledge_base())
            except Exception as exc:
                print(f"\nClean error: {exc}")
            continue

        try:
            upload_paths = parse_upload_command(question)
        except ValueError as exc:
            print(f"\nUpload error: {exc}")
            continue

        if upload_paths is not None:
            try:
                print_upload_summary(add_document_files(upload_paths))
            except Exception as exc:
                print(f"\nUpload error: {exc}")
            continue

        try:
            result = answer_query(question)
        except Exception as exc:
            print(f"\nError: {exc}")
            continue

        print_answer(result)


def main() -> None:
    """Run the CLI."""
    run_cli()


if __name__ == "__main__":
    main()
