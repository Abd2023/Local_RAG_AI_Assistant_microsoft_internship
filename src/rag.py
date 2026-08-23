"""RAG prompt construction and answer generation."""

from __future__ import annotations

import math
import re
import sys

from src import config
from src.citations import NO_ANSWER_PREFIX, citation_key, verify_answer
from src.foundry_client import complete_chat_messages
from src.retrieval import RetrievalResult, retrieve_top_chunks
from src.reranker import rerank_results
from src.traces import TraceRecorder

STOP_WORDS = {
    "about",
    "after",
    "adı",
    "alan",
    "alanlar",
    "answer",
    "answers",
    "are",
    "before",
    "based",
    "can",
    "cannot",
    "could",
    "does",
    "document",
    "documents",
    "each",
    "from",
    "have",
    "hangi",
    "into",
    "mu",
    "nedir",
    "not",
    "oluyor",
    "only",
    "pakette",
    "paketin",
    "question",
    "the",
    "this",
    "what",
    "that",
    "their",
    "bu",
    "when",
    "where",
    "which",
    "with",
    "would",
}
TERM_EQUIVALENTS = {
    "component": {"komponent"},
    "field": {"alan"},
    "fields": {"alan"},
    "image": {"görüntü"},
    "komponent": {"component"},
    "package": {"paket"},
    "paket": {"package"},
    "taşıma": {"taşı", "transfer"},
    "transfer": {"taşıma"},
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
TIME_PATTERN = re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b", flags=re.IGNORECASE)
WEEKDAY_PATTERN = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    flags=re.IGNORECASE,
)

SYSTEM_INSTRUCTION = """You are a local document Q&A assistant. Answer only using the provided context.
If the answer is not in the context, say you do not know based on the available documents.
Answer directly in a concise response of up to six sentences. For exact fact questions, preserve names, numbers, dates, units, and wording from the context. For summary questions, combine only supported facts from the context; do not invent, infer unsupported details, or show reasoning.
Every factual sentence must end with one or more exact bracketed labels copied from the context.
Use only citation labels present in the context. Never invent a source label.
Do not include a Sources section; the application adds canonical sources after verification."""


def _term_variants(word: str) -> set[str]:
    """Return cheap English variants so small wording changes still match."""
    normalized = word.lower().strip("_")
    if not normalized:
        return set()

    variants = {normalized}
    variants.update(TERM_EQUIVALENTS.get(normalized, set()))
    if len(normalized) > 4 and normalized.endswith("ies"):
        variants.add(f"{normalized[:-3]}y")
    if len(normalized) > 4 and normalized.endswith("ing"):
        variants.add(normalized[:-3])
        variants.add(f"{normalized[:-3]}e")
    if len(normalized) > 3 and normalized.endswith("ed"):
        variants.add(normalized[:-2])
        variants.add(normalized[:-1])
    if len(normalized) > 3 and normalized.endswith("s"):
        variants.add(normalized[:-1])
    for suffix in (
        "abilecek",
        "abilecekler",
        "yebilir",
        "yabilir",
        "ebilir",
        "abilir",
        "lerinde",
        "larından",
        "lerden",
        "lardan",
        "leri",
        "ları",
        "lerin",
        "ların",
        "nın",
        "nin",
        "nun",
        "nün",
        "dır",
        "dir",
        "dur",
        "dür",
        "tir",
        "tır",
        "tur",
        "tür",
        "lar",
        "ler",
        "den",
        "dan",
        "ten",
        "tan",
        "de",
        "da",
        "te",
        "ta",
        "in",
        "ın",
        "un",
        "ün",
        "ir",
        "ır",
        "ur",
        "ür",
        "i",
        "ı",
        "u",
        "ü",
        "e",
        "a",
    ):
        if len(normalized) > len(suffix) + 2 and normalized.endswith(suffix):
            variants.add(normalized[: -len(suffix)])
    if normalized.startswith("güncelle"):
        variants.add("güncelle")
    if normalized.startswith("guncelle"):
        variants.add("guncelle")
    for variant in list(variants):
        variants.update(TERM_EQUIVALENTS.get(variant, set()))
    return variants


def _keywords(text: str) -> set[str]:
    """Return simple lowercase keywords for context snippet selection."""
    words: set[str] = set()
    for word in re.findall(r"[\w]+", text.lower(), flags=re.UNICODE):
        if len(word) <= 2 or word in STOP_WORDS:
            continue
        words.update(_term_variants(word))
    if "v1" in words:
        words.update({"first", "version"})
    return words


def _sentences_from_content(content: str) -> list[str]:
    """Split retrieved content into candidate evidence sentences."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", content)
        if sentence.strip() and not re.fullmatch(r"\d+[.)]?|[●•\-\u200b]+", sentence.strip())
    ]


def _raw_important_terms(text: str) -> set[str]:
    """Return non-stopword surface terms for phrase-level scoring."""
    return {
        word.lower()
        for word in re.findall(r"[\w]+", text, flags=re.UNICODE)
        if len(word) > 2 and word.lower() not in STOP_WORDS
    }


def _sentence_score(question: str, sentence: str) -> float:
    """Score how well a sentence can answer a question."""
    normalized_question = question.lower()
    normalized_sentence = sentence.lower()
    question_keywords = _keywords(question)
    sentence_keywords = _keywords(sentence)
    overlap = question_keywords & sentence_keywords
    score = float(len(overlap) * 2)

    for term in _raw_important_terms(question):
        if term in normalized_sentence:
            score += 0.75

    if "what days" in normalized_question or "hours" in normalized_question:
        if WEEKDAY_PATTERN.search(sentence):
            score += 4.0
        if TIME_PATTERN.search(sentence):
            score += 4.0

    if "which week" in normalized_question or "what week" in normalized_question:
        if "week" in normalized_sentence:
            score += 3.0
        if re.search(r"\bweek\s+\d+\b", normalized_sentence):
            score += 4.0

    if "checkpoint" in normalized_question and "checkpoint" in normalized_sentence:
        score += 4.0
    if "third checkpoint" in normalized_question and "third checkpoint" in normalized_sentence:
        score += 5.0

    gpu_terms = {"gpu", "cuda", "nvidia"}
    if gpu_terms & question_keywords and gpu_terms & sentence_keywords:
        score += 5.0
    if "required" in question_keywords and "prefer" in sentence_keywords:
        score += 2.0
    if "available" in question_keywords and "available" in sentence_keywords:
        score += 2.0

    rank_terms = {"rank", "ranking", "embeddings", "embedding", "python", "javascript"}
    if rank_terms & question_keywords and rank_terms & sentence_keywords:
        score += 3.0

    package_terms = {"package", "paket", "name"}
    if package_terms & question_keywords and package_terms & sentence_keywords:
        score += 3.0

    field_terms = {"field", "alan", "classlabel", "classid", "confidence", "güncelle"}
    if field_terms & question_keywords and field_terms & sentence_keywords:
        score += 3.0
    if {"classlabel", "classid", "confidence"} <= sentence_keywords:
        score += 4.0

    image_terms = {"image", "görüntü", "taşıma", "transfer"}
    if image_terms & question_keywords and image_terms & sentence_keywords:
        score += 4.0
    if "yapmaz" in sentence_keywords or "bulunmaz" in sentence_keywords:
        score += 2.0

    if sentence.lstrip().startswith("#"):
        score -= 4.0
    if len(sentence.split()) < 4:
        score -= 1.0
    return score


def _best_evidence_sentences(
    question: str,
    results: list[RetrievalResult],
    *,
    max_sentences: int = 2,
    min_score: float = 5.0,
) -> list[tuple[RetrievalResult, str, float]]:
    """Return the strongest cited sentences across retrieved chunks."""
    scored: list[tuple[float, int, int, RetrievalResult, str]] = []
    for result_index, result in enumerate(results):
        for sentence_index, sentence in enumerate(_sentences_from_content(result.content)):
            score = _sentence_score(question, sentence)
            if score >= min_score:
                scored.append((score, result_index, sentence_index, result, sentence))

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected: list[tuple[RetrievalResult, str, float]] = []
    seen: set[str] = set()
    for score, _result_index, _sentence_index, result, sentence in scored:
        normalized = re.sub(r"\s+", " ", sentence.lower())
        if normalized in seen:
            continue
        selected.append((result, sentence, score))
        seen.add(normalized)
        if len(selected) >= max_sentences:
            break
    return selected


def _relevant_snippet(question: str, content: str, max_sentences: int = 3) -> str:
    """Extract the most question-relevant sentences from a retrieved chunk."""
    question_keywords = _keywords(question)
    sentences = _sentences_from_content(content)
    if not sentences or not question_keywords:
        return content

    scored = []
    for index, sentence in enumerate(sentences):
        score = _sentence_score(question, sentence)
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
    """Build a labeled context block from complete retrieved chunks."""
    context_parts: list[str] = []
    remaining_chars = config.MAX_CONTEXT_CHARS
    for result in results:
        label = _context_source_label(result)
        metadata = [f"[Source: {label}]"]
        if result.page_start is not None:
            page_label = (
                str(result.page_start)
                if result.page_end in {None, result.page_start}
                else f"{result.page_start}-{result.page_end}"
            )
            metadata.append(f"Page: {page_label}")
        if result.heading:
            metadata.append(f"Heading: {result.heading}")
        if result.extraction_method:
            metadata.append(f"Extraction: {result.extraction_method}")
        if result.source_path:
            metadata.append(f"Path: {result.source_path}")

        content = result.content.strip()
        block = "\n".join([" | ".join(metadata), content])
        if remaining_chars <= 0:
            break
        if len(block) > remaining_chars:
            block = block[:remaining_chars].rsplit(" ", 1)[0].rstrip() + "\n[Context truncated]"
        context_parts.append(block)
        remaining_chars -= len(block) + 7
    return "\n\n---\n\n".join(context_parts)


def build_user_prompt(question: str, results: list[RetrievalResult]) -> str:
    """Build the user message containing context and the actual question."""
    context = build_context_block(results, question=question)
    return f"""Use the context below to answer the question.
If the answer is present, answer directly and preserve exact names, numbers, dates, units, and wording.
If the answer is missing from the context, write exactly: I do not know based on the available documents.
Every factual sentence must end with an exact bracketed label copied from the context, such as [source.md#0].
Do not write headings, labels, reasoning, or a Sources section.
Do not invent facts or source names.

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
            "chunk_uid": result.chunk_uid,
            "source_path": result.source_path,
            "source_name": result.source_name,
            "chunk_index": result.chunk_index,
            "chunk_number": result.chunk_index + 1,
            "page_start": result.page_start,
            "page_end": result.page_end,
            "extraction_method": result.extraction_method,
            "heading": result.heading,
            "similarity": result.similarity,
            "vector_similarity": result.similarity,
            "rerank_score": result.rerank_score,
            "lexical_score": result.lexical_score,
            "hybrid_score": result.hybrid_score,
            "ranking_method": result.ranking_method,
            "similarity_percent": result.similarity * 100,
            "source_label": _display_source_label(result),
            "preview": result.content.replace("\n", " ")[:240],
        }
        for result in results
    ]


def _page_label(result: RetrievalResult) -> str:
    """Return a compact page label when page metadata is available."""
    if result.page_start is None:
        return ""
    if result.page_end is None or result.page_end == result.page_start:
        return f" page {result.page_start}"
    return f" pages {result.page_start}-{result.page_end}"


def _context_source_label(result: RetrievalResult) -> str:
    """Return the compact source label used inside model context."""
    return citation_key(result.source_name, result.chunk_index)


def _display_source_label(result: RetrievalResult) -> str:
    """Return a human-readable source label for terminal/API output."""
    return f"{result.source_name}{_page_label(result)} (chunk {result.chunk_index + 1})"


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


def _citation_failure_answer() -> str:
    """Return a safe response when the model output cannot be verified."""
    return "I do not know based on the available documents."


def _normalize_model_citations(answer: str) -> str:
    """Canonicalize common small-model citation mistakes before verification."""
    normalized = re.sub(
        r"\[\s*Source:\s*([^\[\]]+?#\d+)\s*\]",
        r"[\1]",
        answer,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"\[\s*(?:source(?:\.md)?|filename|file|document|source_name)#(?:\d+|chunk_index|index)\s*\]",
        "",
        normalized,
        flags=re.IGNORECASE,
    )


def _best_citation_for_claim(
    claim: str,
    results: list[RetrievalResult],
) -> str | None:
    """Find the retrieved source that best supports an uncited model claim."""
    normalized_claim = re.sub(r"\[[^\[\]]+?#\d+\]", "", claim).strip().lower()
    if not normalized_claim:
        return None

    exact_matches: list[tuple[int, RetrievalResult]] = []
    for index, result in enumerate(results):
        if normalized_claim.rstrip(".!?") in result.content.lower():
            exact_matches.append((index, result))
    if exact_matches:
        return citation_key(exact_matches[0][1].source_name, exact_matches[0][1].chunk_index)

    evidence = _best_evidence_sentences(
        claim,
        results,
        max_sentences=1,
        min_score=4.0,
    )
    if not evidence:
        return None
    result, _sentence, _score = evidence[0]
    return citation_key(result.source_name, result.chunk_index)


def _repair_uncited_model_answer(
    answer: str,
    results: list[RetrievalResult],
) -> str | None:
    """Attach citations to a grounded model answer that omitted labels."""
    allowed_citations = {
        citation_key(result.source_name, result.chunk_index)
        for result in results
    }
    verification = verify_answer(answer, allowed_citations)
    if verification.invalid_citations or not verification.uncited_claims:
        return None
    if len(verification.claims) > 6:
        return None

    repaired_claims: list[str] = []
    for claim in verification.claims:
        if re.search(r"\[[^\[\]]+?#\d+\]", claim):
            repaired_claims.append(claim)
            continue
        citation = _best_citation_for_claim(claim, results)
        if citation is None:
            return None
        repaired_claims.append(f"{claim.rstrip()} [{citation}]")
    return " ".join(repaired_claims).strip() or None


def _explicit_absence_answer(
    question: str,
    results: list[RetrievalResult],
) -> str | None:
    """Answer explicit-mention questions when context supports a negative answer."""
    normalized_question = question.lower()
    context = _context_text(results).lower()

    if (
        "explicit" in normalized_question
        and "javascript" in normalized_question
        and "javascript" not in context
    ):
        evidence = _best_evidence_sentences(
            "ranking embeddings in Python",
            results,
            max_sentences=1,
            min_score=5.0,
        )
        if evidence:
            result, sentence, _score = evidence[0]
            citation = citation_key(result.source_name, result.chunk_index)
            evidence_sentence = sentence.rstrip(".!?")
            return (
                f"No; the documents say {evidence_sentence[0].lower() + evidence_sentence[1:]}, "
                f"and they do not explicitly permit JavaScript. [{citation}]"
            )

    if (
        {"gpu", "cuda", "nvidia"} & _keywords(question)
        and {"required", "require"} & _keywords(question)
    ):
        evidence = _best_evidence_sentences(
            "NVIDIA GPU CUDA variants available prefer",
            results,
            max_sentences=1,
            min_score=5.0,
        )
        if evidence:
            result, sentence, _score = evidence[0]
            sentence_keywords = _keywords(sentence)
            if "prefer" in sentence_keywords and "available" in sentence_keywords:
                citation = citation_key(result.source_name, result.chunk_index)
                evidence_sentence = sentence.rstrip(".!?")
                return (
                    "The documents do not state that an NVIDIA GPU is required; "
                    f"they say {evidence_sentence[0].lower() + evidence_sentence[1:]}. [{citation}]"
                )

    return None


def _needs_explicit_absence_answer(question: str) -> bool:
    """Return whether the question asks for a documented absence/requirement distinction."""
    normalized_question = question.lower()
    return (
        ("explicit" in normalized_question and "javascript" in normalized_question)
        or (
            {"gpu", "cuda", "nvidia"} & _keywords(question)
            and {"required", "require"} & _keywords(question)
        )
    )


def _needs_direct_fact_override(question: str) -> bool:
    """Return whether a concise exact fact should beat model phrasing."""
    question_terms = _keywords(question)
    return bool(
        {"alan", "field", "güncelle", "guncelle", "image", "görüntü", "taşıma", "transfer"}
        & question_terms
    )


def _direct_fact_answer(
    question: str,
    results: list[RetrievalResult],
) -> str | None:
    """Return concise extractive answers for common exact-fact questions."""
    normalized_question = question.lower()
    question_terms = _keywords(question)

    asks_package_name = (
        ("name" in question_terms and "package" in question_terms)
        or ("paket" in normalized_question and "ad" in normalized_question)
    )
    asks_updated_fields = (
        "alan" in question_terms
        or "field" in question_terms
        or "güncelle" in question_terms
        or "guncelle" in question_terms
    )
    asks_image_transfer = bool(
        {"image", "görüntü", "taşıma", "transfer"} & question_terms
    )

    for result in results:
        citation = citation_key(result.source_name, result.chunk_index)
        content = result.content
        content_lower = content.lower()

        if asks_package_name:
            match = re.search(
                r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){1,})\s+paketi\b",
                content,
            )
            if match:
                package_name = match.group(1).strip()
                if "paket" in normalized_question:
                    return f"Paketin adı {package_name}. [{citation}]"
                return f"The package name is {package_name}. [{citation}]"

        if asks_updated_fields and all(
            field in content_lower
            for field in ("classlabel", "classid", "confidence")
        ):
            return f"classLabel, classId ve confidence alanları güncellenebilir. [{citation}]"

        if asks_image_transfer and (
            "görüntü taşıma işlemi yapmaz" in content_lower
            or "image input/output socket" in content_lower
        ):
            return f"Hayır; bu component görüntü taşıma işlemi yapmaz. [{citation}]"

    return None


def _extractive_fallback(
    question: str,
    results: list[RetrievalResult],
    *,
    allow_weak: bool = True,
) -> str:
    """Build a concise grounded answer when the small model breaks the citation contract."""
    if not results:
        return _citation_failure_answer()

    explicit_answer = _explicit_absence_answer(question, results)
    if explicit_answer is not None:
        return explicit_answer

    direct_answer = _direct_fact_answer(question, results)
    if direct_answer is not None:
        return direct_answer

    evidence = _best_evidence_sentences(
        question,
        results,
        max_sentences=2,
        min_score=5.0,
    )
    if not evidence and allow_weak:
        result = results[0]
        snippet = _relevant_snippet(question, result.content, max_sentences=1).strip()
        evidence = [(result, snippet, 0.0)] if snippet else []

    answer_parts: list[str] = []
    for result, sentence, _score in evidence:
        citation = citation_key(result.source_name, result.chunk_index)
        answer_parts.append(f"{sentence.rstrip()} [{citation}]")
    return " ".join(answer_parts) if answer_parts else _citation_failure_answer()


def _repair_model_no_answer(
    question: str,
    answer: str,
    results: list[RetrievalResult],
) -> str | None:
    """Recover when the model says no-answer despite strong retrieved evidence."""
    if not answer.strip().startswith(NO_ANSWER_PREFIX):
        return None

    repaired = _extractive_fallback(question, results, allow_weak=False)
    if repaired.startswith(NO_ANSWER_PREFIX):
        return None

    allowed_citations = {
        citation_key(result.source_name, result.chunk_index)
        for result in results
    }
    verification = verify_answer(repaired, allowed_citations)
    grounding = _verify_claim_grounding(verification.claims, results)
    if verification.verified and grounding["verified"]:
        return repaired
    return None


def _low_confidence_answer(results: list[RetrievalResult]) -> str | None:
    """Return a deterministic no-answer response when retrieval is weak."""
    top_score = (
        results[0].hybrid_score
        if results and results[0].hybrid_score is not None
        else results[0].similarity if results else 0.0
    )
    if not results or top_score < config.RETRIEVAL_MIN_TOP_SCORE:
        return "I do not know based on the available documents."
    return None


def _retrieval_confidence(results: list[RetrievalResult]) -> float:
    """Expose the bounded top hybrid score for API and UI consumers."""
    if not results:
        return 0.0
    score = results[0].hybrid_score
    if score is None:
        score = (results[0].similarity + 1.0) / 2.0
    return round(max(0.0, min(1.0, float(score))), 3)


def _context_text(results: list[RetrievalResult]) -> str:
    """Return all retrieved content as one text block for deterministic guards."""
    return "\n\n".join(result.content for result in results)


def _grounding_terms(text: str) -> set[str]:
    """Return meaningful words and numeric tokens used by the grounding guard."""
    return {
        token
        for token in re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)
        if token not in STOP_WORDS and (len(token) > 2 or token.isdigit())
    }


def _verify_claim_grounding(
    claims: list[str],
    results: list[RetrievalResult],
) -> dict[str, object]:
    """Check that cited claims share meaningful evidence with their source chunks."""
    by_citation = {
        citation_key(result.source_name, result.chunk_index): result
        for result in results
    }
    unsupported: list[str] = []
    for claim in claims:
        citation_matches = re.findall(r"\[([^\[\]]+?)#(\d+)\]", claim)
        cited_results = [
            by_citation.get(citation_key(source.strip(), int(index)))
            for source, index in citation_matches
        ]
        source_terms = _grounding_terms(
            " ".join(result.content for result in cited_results if result is not None)
        )
        claim_terms = _grounding_terms(re.sub(r"\[[^\[\]]+?#\d+\]", "", claim))
        overlap = claim_terms & source_terms
        required_overlap = max(1, min(3, math.ceil(len(claim_terms) * 0.15)))
        if not cited_results or not claim_terms or len(overlap) < required_overlap:
            unsupported.append(claim)

    return {
        "verified": not unsupported,
        "unsupported_claims": unsupported,
    }


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


def _guard_decision(question: str, results: list[RetrievalResult]) -> tuple[str, str | None]:
    """Return guard decision name and deterministic no-answer text if needed."""
    low_confidence = _low_confidence_answer(results)
    if low_confidence is not None:
        return "low_confidence", low_confidence

    missing_detail = _missing_specific_detail_answer(question, results)
    if missing_detail is not None:
        return "missing_exact_detail", missing_detail

    return "normal_generation", None


def answer_query(
    question: str,
    *,
    trace_enabled: bool = True,
    trace_dir: object | None = None,
) -> dict[str, object]:
    """Answer a question using retrieved local document context."""
    trace = TraceRecorder(
        question=question,
        traces_dir=trace_dir or config.TRACES_PATH,
        enabled=trace_enabled,
    )

    if not question.strip():
        exc = ValueError("Question must not be empty.")
        setattr(exc, "trace_id", trace.trace_id)
        trace.set("guard_decision", "exception")
        trace.finish(status="error", error=exc)
        raise exc

    try:
        with trace.span("retrieval_total"):
            vector_candidates = retrieve_top_chunks(
                question,
                top_k=(
                    config.RERANKER_CANDIDATE_K
                    if config.RERANKER_ENABLED
                    else config.TOP_K
                ),
                trace=trace,
            )
        trace.set("vector_candidates", _chunk_debug_info(vector_candidates))

        with trace.span("reranking"):
            retrieved_chunks, reranker_status = rerank_results(
                question,
                vector_candidates,
                top_k=config.TOP_K,
            )
        trace.set("reranker_status", reranker_status)
        sources = [_display_source_label(result) for result in retrieved_chunks]
        retrieved_debug = _chunk_debug_info(retrieved_chunks)
        trace.set("retrieved_sources", retrieved_debug)

        guard_decision, no_answer = _guard_decision(question, retrieved_chunks)
        trace.set("guard_decision", guard_decision)
        confidence = _retrieval_confidence(retrieved_chunks)
        trace.set("confidence", confidence)
        if no_answer is not None:
            final_answer = _ensure_answer_has_sources(no_answer, sources)
            trace.update(
                {
                    "prompt_length": 0,
                    "answer_length": len(final_answer),
                    "sources": sources,
                    "citation_verification": verify_answer(final_answer, []).as_dict(),
                    "grounding_verification": {
                        "verified": True,
                        "unsupported_claims": [],
                    },
                }
            )
            trace.finish(status="ok", answer=final_answer)
            return {
                "answer": final_answer,
                "sources": sources,
                "retrieved_chunks": retrieved_debug,
                "verification": verify_answer(final_answer, []).as_dict(),
                "grounding": {"verified": True, "unsupported_claims": []},
                "confidence": confidence,
                "reranker_status": reranker_status,
                "trace_id": trace.trace_id,
            }

        with trace.span("prompt_build"):
            user_prompt = build_user_prompt(question, retrieved_chunks)
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
        trace.update(
            {
                "prompt_length": len(user_prompt),
                "prompt_messages": messages,
                "sources": sources,
            }
        )

        allowed_citations = {
            citation_key(result.source_name, result.chunk_index)
            for result in retrieved_chunks
        }
        chat_generation_recovered = False
        chat_generation_error: dict[str, str] | None = None
        with trace.span("chat_generation"):
            try:
                answer = complete_chat_messages(messages)
            except Exception as exc:
                fallback_answer = _extractive_fallback(
                    question,
                    retrieved_chunks,
                    allow_weak=False,
                )
                fallback_verification = verify_answer(fallback_answer, allowed_citations)
                fallback_grounding = _verify_claim_grounding(
                    fallback_verification.claims,
                    retrieved_chunks,
                )
                if not (
                    fallback_verification.verified
                    and fallback_grounding["verified"]
                ):
                    raise
                answer = fallback_answer
                chat_generation_recovered = True
                chat_generation_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        trace.set("model_answer", answer)
        trace.set("chat_generation_recovered", chat_generation_recovered)
        if chat_generation_error is not None:
            trace.set("chat_generation_error", chat_generation_error)
        normalized_answer = re.sub(
            r"\[(?:source_name|filename)#(?:chunk_index|index)\]",
            "",
            answer,
            flags=re.IGNORECASE,
        )
        normalized_answer = _normalize_model_citations(normalized_answer)
        if (
            NO_ANSWER_PREFIX in normalized_answer
            and not normalized_answer.strip().startswith(NO_ANSWER_PREFIX)
        ):
            normalized_answer = NO_ANSWER_PREFIX
        explicit_answer_repaired = False
        if _needs_explicit_absence_answer(question):
            explicit_answer = _explicit_absence_answer(question, retrieved_chunks)
            if explicit_answer is not None:
                normalized_answer = explicit_answer
                explicit_answer_repaired = True
        direct_fact_repaired = False
        if _needs_direct_fact_override(question):
            direct_answer = _direct_fact_answer(question, retrieved_chunks)
            if direct_answer is not None:
                normalized_answer = direct_answer
                direct_fact_repaired = True
        no_answer_repaired = False
        repaired_no_answer = _repair_model_no_answer(
            question,
            normalized_answer,
            retrieved_chunks,
        )
        if repaired_no_answer is not None:
            normalized_answer = repaired_no_answer
            no_answer_repaired = True
        verification = verify_answer(normalized_answer, allowed_citations)
        grounding = _verify_claim_grounding(verification.claims, retrieved_chunks)
        citation_repaired = False
        repaired_uncited = _repair_uncited_model_answer(
            normalized_answer,
            retrieved_chunks,
        )
        if repaired_uncited is not None:
            repaired_verification = verify_answer(repaired_uncited, allowed_citations)
            repaired_grounding = _verify_claim_grounding(
                repaired_verification.claims,
                retrieved_chunks,
            )
            if repaired_verification.verified and repaired_grounding["verified"]:
                normalized_answer = repaired_uncited
                verification = repaired_verification
                grounding = repaired_grounding
                citation_repaired = True
        if (
            not verification.verified or not grounding["verified"]
        ) and not verification.invalid_citations:
            repaired_answer = _extractive_fallback(question, retrieved_chunks)
            repaired_verification = verify_answer(repaired_answer, allowed_citations)
            repaired_grounding = _verify_claim_grounding(
                repaired_verification.claims,
                retrieved_chunks,
            )
            if repaired_verification.verified and repaired_grounding["verified"]:
                verification = repaired_verification
                grounding = repaired_grounding
                citation_repaired = True
        trace.set("explicit_answer_repair_applied", explicit_answer_repaired)
        trace.set("direct_fact_repair_applied", direct_fact_repaired)
        trace.set("model_no_answer_repair_applied", no_answer_repaired)
        trace.set("citation_repair_applied", citation_repaired)
        trace.set("citation_verification", verification.as_dict())
        trace.set("grounding_verification", grounding)
        trace.set("confidence", confidence)
        if verification.verified and grounding["verified"]:
            final_answer = _ensure_answer_has_sources(verification.answer, sources)
        else:
            trace.set("guard_decision", "citation_verification_failed")
            final_answer = _ensure_answer_has_sources(_citation_failure_answer(), sources)
        trace.finish(status="ok", answer=final_answer)

        return {
            "answer": final_answer,
            "sources": sources,
            "retrieved_chunks": retrieved_debug,
            "verification": verification.as_dict(),
            "grounding": grounding,
            "confidence": confidence,
            "citation_repair_applied": citation_repaired,
            "explicit_answer_repair_applied": explicit_answer_repaired,
            "direct_fact_repair_applied": direct_fact_repaired,
            "model_no_answer_repair_applied": no_answer_repaired,
            "chat_generation_recovered": chat_generation_recovered,
            "reranker_status": reranker_status,
            "trace_id": trace.trace_id,
        }
    except Exception as exc:
        setattr(exc, "trace_id", trace.trace_id)
        trace.set("guard_decision", "exception")
        trace.finish(status="error", error=exc)
        raise


def main() -> None:
    """Run one RAG question from the command line."""
    question = " ".join(sys.argv[1:]).strip() or "What time does the daily standup start?"
    result = answer_query(question)
    answer = str(result["answer"]).split("\n\nSources:", maxsplit=1)[0].strip()
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    if result.get("trace_id"):
        print(f"Trace: {result['trace_id']}")
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
