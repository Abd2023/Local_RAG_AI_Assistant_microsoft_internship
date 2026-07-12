"""Interactive command-line interface for the Local RAG AI Assistant."""

from __future__ import annotations

from collections.abc import Callable

from src.rag import answer_query

EXIT_COMMANDS = {"exit", "quit"}


def print_answer(result: dict[str, object]) -> None:
    """Print an answer result in a readable terminal format."""
    answer = str(result["answer"]).split("\n\nSources:", maxsplit=1)[0].strip()
    print("\nAnswer:")
    print(answer)

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
    print("Ask a question about the sample documents. Type 'exit' or 'quit' to close.")

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
