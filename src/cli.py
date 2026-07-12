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

    sources = result.get("sources", [])
    if sources:
        print("\nSources:")
        for source in sources:
            print(f"- {source}")


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
