"""RAG prompt construction and answer generation."""

from __future__ import annotations

import sys

from src import config
from src.foundry_client import complete_chat_messages
from src.retrieval import RetrievalResult, retrieve_top_chunks

SYSTEM_INSTRUCTION = """You are a local document Q&A assistant. Answer only using the provided context.
If the answer is not in the context, say you do not know based on the available documents.
Be concise and include the source names you used."""


def build_context_block(results: list[RetrievalResult]) -> str:
    """Build a labeled context block from retrieved chunks."""
    context_parts = []
    for result in results:
        label = f"{result.source_name}#{result.chunk_index}"
        context_parts.append(f"[Source: {label}]\n{result.content}")
    return "\n\n---\n\n".join(context_parts)


def build_user_prompt(question: str, results: list[RetrievalResult]) -> str:
    """Build the user message containing context and the actual question."""
    context = build_context_block(results)
    return f"""Use the context below to answer the question.
If the answer is missing from the context, begin with: I do not know based on the available documents.
End with a separate line in this exact format: Sources: source_name#chunk_index, source_name#chunk_index

Context:
{context}

Question:
{question}

Answer:"""


def _chunk_debug_info(results: list[RetrievalResult]) -> list[dict[str, object]]:
    """Return compact retrieved chunk details for debugging."""
    return [
        {
            "chunk_id": result.chunk_id,
            "source_name": result.source_name,
            "chunk_index": result.chunk_index,
            "similarity": result.similarity,
            "preview": result.content.replace("\n", " ")[:240],
        }
        for result in results
    ]


def _ensure_answer_has_sources(answer: str, sources: list[str]) -> str:
    """Append source labels when the model omits them."""
    if not sources:
        return answer.strip()

    normalized_answer = answer.lower()
    if all(source.lower() in normalized_answer for source in sources):
        return answer.strip()

    return f"{answer.strip()}\n\nSources: {', '.join(sources)}"


def answer_query(question: str) -> dict[str, object]:
    """Answer a question using retrieved local document context."""
    if not question.strip():
        raise ValueError("Question must not be empty.")

    retrieved_chunks = retrieve_top_chunks(question, top_k=config.TOP_K)
    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": build_user_prompt(question, retrieved_chunks),
        },
    ]
    answer = complete_chat_messages(messages)
    sources = [f"{result.source_name}#{result.chunk_index}" for result in retrieved_chunks]
    final_answer = _ensure_answer_has_sources(answer, sources)

    return {
        "answer": final_answer,
        "sources": sources,
        "retrieved_chunks": _chunk_debug_info(retrieved_chunks),
    }


def main() -> None:
    """Run one RAG question from the command line."""
    question = " ".join(sys.argv[1:]).strip() or "What time does the daily standup start?"
    result = answer_query(question)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print("Sources:")
    for source in result["sources"]:
        print(f"- {source}")
    print("Retrieved chunks:")
    for chunk in result["retrieved_chunks"]:
        print(f"- {chunk['source_name']}#{chunk['chunk_index']} score={chunk['similarity']:.4f}")


if __name__ == "__main__":
    main()
