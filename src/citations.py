"""Citation parsing and provenance verification for generated answers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

CITATION_PATTERN = re.compile(r"\[([^\[\]]+?)#(\d+)\]")
SOURCES_LINE_PATTERN = re.compile(r"\n*\s*Sources:\s*.*$", re.IGNORECASE | re.DOTALL)
NO_ANSWER_PREFIX = "I do not know based on the available documents."


@dataclass(frozen=True)
class CitationVerification:
    """Result of checking generated citations against retrieved chunks."""

    verified: bool
    claims: list[str]
    citations: list[str]
    invalid_citations: list[str]
    uncited_claims: list[str]
    answer: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def citation_key(source_name: str, chunk_index: int) -> str:
    """Return the stable citation key used in prompts and answers."""
    return f"{source_name}#{chunk_index}"


def _answer_body(answer: str) -> str:
    body = SOURCES_LINE_PATTERN.sub("", answer.strip()).strip()
    if body.lower().startswith("answer:"):
        body = body[len("answer:") :].strip()
    return body


def _claims_from_answer(answer: str) -> list[str]:
    body = _answer_body(answer)
    if not body or body.startswith(NO_ANSWER_PREFIX):
        return []
    claims: list[str] = []
    for line in body.splitlines():
        line = line.strip(" -*\t")
        if not line:
            continue
        protected_tokens: list[str] = []

        def protect(match: re.Match[str]) -> str:
            protected_tokens.append(match.group(0).replace(".", "<citation-dot>"))
            return f"<citation-{len(protected_tokens) - 1}>"

        protected_line = CITATION_PATTERN.sub(protect, line)
        protected_line = re.sub(r"(?<=\w)\.(?=\w)", "<text-dot>", protected_line)
        sentences = re.findall(
            r"[^.!?]+?[.!?](?=\s|$)(?:\s*<citation-\d+>)*|[^.!?]+(?:\s*<citation-\d+>)*",
            protected_line,
        )
        for sentence in sentences:
            restored = sentence.strip()
            for index, token in enumerate(protected_tokens):
                restored = restored.replace(
                    f"<citation-{index}>",
                    token.replace("<citation-dot>", "."),
                )
            restored = restored.replace("<text-dot>", ".")
            if restored:
                claims.append(restored)
    return claims


def verify_answer(answer: str, allowed_citations: Iterable[str]) -> CitationVerification:
    """Verify that each factual claim has only citations from retrieved chunks."""
    allowed = set(allowed_citations)
    body = _answer_body(answer)
    claims = _claims_from_answer(answer)
    citations: list[str] = []
    invalid: list[str] = []
    uncited: list[str] = []

    for claim in claims:
        matches = CITATION_PATTERN.findall(claim)
        if not matches:
            uncited.append(claim)
            continue
        for source_name, chunk_index in matches:
            key = citation_key(source_name.strip(), int(chunk_index))
            citations.append(key)
            if key not in allowed:
                invalid.append(key)

    verified = not invalid and not uncited
    if not claims and body.startswith(NO_ANSWER_PREFIX):
        verified = True

    return CitationVerification(
        verified=verified,
        claims=claims,
        citations=citations,
        invalid_citations=invalid,
        uncited_claims=uncited,
        answer=body,
    )


def repair_answer_with_citations(
    answer: str,
    claims: Iterable[str],
    claim_citations: Iterable[str],
) -> str:
    """Attach backend-selected citations when the model omitted all citations."""
    body = _answer_body(answer)
    repaired_claims = [
        f"{claim.strip()} [{citation}]"
        for claim, citation in zip(claims, claim_citations, strict=True)
        if claim.strip() and not re.fullmatch(r"\d+[.)]", claim.strip())
    ]
    return " ".join(repaired_claims) or body
