"""RAG prompt construction and answer generation."""

from __future__ import annotations

import re
import sys

from src import config
from src.foundry_client import complete_chat_messages
from src.retrieval import RetrievalResult, retrieve_top_chunks

STOP_WORDS = {
    "about",
    "after",
    "before",
    "does",
    "each",
    "from",
    "have",
    "into",
    "that",
    "the",
    "their",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
}

CALENDAR_DATE_PATTERN = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{1,2}(?:,\s*\d{4})?\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    flags=re.IGNORECASE,
)
STREET_ADDRESS_PATTERN = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9 .'-]+"
    r"(?:street|st\.|avenue|ave\.|road|rd\.|boulevard|blvd\.|lane|ln\.|drive|dr\.|court|ct\.)\b",
    flags=re.IGNORECASE,
)

SYSTEM_INSTRUCTION = """You are a local document Q&A assistant. Answer only using the provided context.
If the answer is not in the context, say you do not know based on the available documents.
Be concise and include the source names you used."""


def _keywords(text: str) -> set[str]:
    """Return simple lowercase keywords for context snippet selection."""
    words = {
        word
        for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(word) > 2 and word not in STOP_WORDS
    }
    if "v1" in words:
        words.update({"first", "version"})
    return words


def _relevant_snippet(question: str, content: str, max_sentences: int = 3) -> str:
    """Extract the most question-relevant sentences from a retrieved chunk."""
    question_keywords = _keywords(question)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", content)
        if sentence.strip()
    ]
    if not sentences or not question_keywords:
        return content

    scored = []
    for index, sentence in enumerate(sentences):
        score = len(question_keywords & _keywords(sentence))
        if score > 0:
            scored.append((score, index, sentence))

    if not scored:
        return " ".join(sentences[:max_sentences])

    selected = sorted(
        sorted(scored, key=lambda item: item[0], reverse=True)[:max_sentences],
        key=lambda item: item[1],
    )
    return " ".join(sentence for _, _, sentence in selected)


def build_context_block(results: list[RetrievalResult], question: str = "") -> str:
    """Build a labeled context block from retrieved chunks."""
    context_parts = []
    for result in results:
        label = _context_source_label(result)
        content = _relevant_snippet(question, result.content) if question else result.content
        context_parts.append(f"[Source: {label}]\n{content}")
    return "\n\n---\n\n".join(context_parts)


def build_user_prompt(question: str, results: list[RetrievalResult]) -> str:
    """Build the user message containing context and the actual question."""
    context = build_context_block(results, question=question)
    return f"""Use the context below to answer the question.
If the answer is missing from the context, begin with: I do not know based on the available documents.
Do not invent facts or source names.
Return exactly two parts:
Answer: one or two concise sentences that answer the question using only the context.
Sources: source_name#chunk_index, source_name#chunk_index

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
            "chunk_number": result.chunk_index + 1,
            "similarity": result.similarity,
            "similarity_percent": result.similarity * 100,
            "source_label": _display_source_label(result),
            "preview": result.content.replace("\n", " ")[:240],
        }
        for result in results
    ]


def _context_source_label(result: RetrievalResult) -> str:
    """Return the compact source label used inside model context."""
    return f"{result.source_name}#{result.chunk_index}"


def _display_source_label(result: RetrievalResult) -> str:
    """Return a human-readable source label for terminal/API output."""
    return f"{result.source_name} (chunk {result.chunk_index + 1})"


def _ensure_answer_has_sources(answer: str, sources: list[str]) -> str:
    """Append canonical source labels and remove model-invented source lines."""
    answer_without_sources = re.sub(
        r"\n*\s*Sources:\s*.*$",
        "",
        answer.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if not sources:
        return answer_without_sources

    return f"{answer_without_sources}\n\nSources: {', '.join(sources)}"


def _low_confidence_answer(results: list[RetrievalResult]) -> str | None:
    """Return a deterministic no-answer response when retrieval is weak."""
    if not results or results[0].similarity < config.RETRIEVAL_MIN_TOP_SCORE:
        return "I do not know based on the available documents."
    return None


def _context_text(results: list[RetrievalResult]) -> str:
    """Return all retrieved content as one text block for deterministic guards."""
    return "\n\n".join(result.content for result in results)


def _contains_role_identity(context: str, role: str) -> bool:
    """Return whether context explicitly names a person for a role."""
    proper_name = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+"
    role_pattern = re.escape(role)
    return any(
        re.search(pattern, context)
        for pattern in (
            rf"\b{role_pattern}\s+(?:is|was|:)\s+{proper_name}\b",
            rf"\b{proper_name}\s+(?:is|was|serves as|acts as)\s+(?:the\s+)?{role_pattern}\b",
        )
    )


def _missing_specific_detail_answer(
    question: str,
    results: list[RetrievalResult],
) -> str | None:
    """Refuse exact-detail questions when the retrieved context lacks that evidence."""
    normalized_question = question.lower()
    context = _context_text(results)

    asks_for_calendar_date = (
        "calendar date" in normalized_question
        or ("exact" in normalized_question and "date" in normalized_question)
    )
    if asks_for_calendar_date and not CALENDAR_DATE_PATTERN.search(context):
        return "I do not know based on the available documents."

    asks_for_address = "address" in normalized_question
    if asks_for_address and not STREET_ADDRESS_PATTERN.search(context):
        return "I do not know based on the available documents."

    asks_for_instructor_identity = (
        normalized_question.strip().startswith("who ")
        and "instructor" in normalized_question
    )
    if asks_for_instructor_identity and not _contains_role_identity(context, "instructor"):
        return "I do not know based on the available documents."

    return None


def answer_query(question: str) -> dict[str, object]:
    """Answer a question using retrieved local document context."""
    if not question.strip():
        raise ValueError("Question must not be empty.")

    retrieved_chunks = retrieve_top_chunks(question, top_k=config.TOP_K)
    sources = [_display_source_label(result) for result in retrieved_chunks]
    no_answer = _low_confidence_answer(retrieved_chunks) or _missing_specific_detail_answer(
        question,
        retrieved_chunks,
    )
    if no_answer is not None:
        return {
            "answer": _ensure_answer_has_sources(no_answer, sources),
            "sources": sources,
            "retrieved_chunks": _chunk_debug_info(retrieved_chunks),
        }

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
        print(
            f"- {chunk['source_name']} "
            f"(chunk {chunk['chunk_number']}, "
            f"cosine similarity {chunk['similarity']:.3f} / {chunk['similarity_percent']:.1f}%)"
        )


if __name__ == "__main__":
    main()
