import pytest

from src import rag
from src.retrieval import RetrievalResult


def make_result(
    source_name: str,
    chunk_index: int,
    content: str,
    similarity: float = 0.9,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_index + 1,
        source_name=source_name,
        chunk_index=chunk_index,
        content=content,
        similarity=similarity,
    )


def test_build_user_prompt_includes_context_question_and_source_names() -> None:
    results = [
        make_result(
            "schedule.md",
            0,
            "The daily standup starts at 10 AM. Lunch starts at noon.",
        ),
        make_result(
            "policies.txt",
            2,
            "Attendance is recorded during standup.",
        ),
    ]

    prompt = rag.build_user_prompt(
        "When does the daily standup start?",
        results,
    )

    assert "[Source: schedule.md#0]" in prompt
    assert "[Source: policies.txt#2]" in prompt
    assert "The daily standup starts at 10 AM." in prompt
    assert "Attendance is recorded during standup." in prompt
    assert "Question:\nWhen does the daily standup start?" in prompt
    assert "I do not know based on the available documents." in prompt
    assert "Do not write headings, labels, reasoning, or a Sources section." in prompt


def test_build_context_block_separates_retrieved_chunks() -> None:
    results = [
        make_result("one.md", 0, "First context."),
        make_result("two.md", 1, "Second context."),
    ]

    context = rag.build_context_block(results)

    assert context == (
        "[Source: one.md#0]\nFirst context."
        "\n\n---\n\n"
        "[Source: two.md#1]\nSecond context."
    )


def test_answer_query_skips_chat_model_when_retrieval_is_low_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        make_result(
            "unrelated.md",
            0,
            "Unrelated context.",
            similarity=0.1,
        )
    ]
    monkeypatch.setattr(rag, "retrieve_top_chunks", lambda *_args, **_kwargs: results)

    def unexpected_model_call(_messages: list[dict[str, str]]) -> str:
        raise AssertionError("The chat model must not run for low-confidence retrieval.")

    monkeypatch.setattr(rag, "complete_chat_messages", unexpected_model_call)

    response = rag.answer_query("What is the instructor's phone number?")

    assert str(response["answer"]).startswith(
        "I do not know based on the available documents."
    )
    assert response["sources"] == ["unrelated.md (chunk 1)"]


@pytest.mark.parametrize(
    ("question", "content"),
    [
        (
            "Who is the instructor for Northstar AI Summer School?",
            "The instructor uses standup to identify teams that need extra help.",
        ),
        (
            "What exact calendar date is the final demo?",
            "Final demos happen on the last Friday of Week 4.",
        ),
        (
            "What is the street address of the classroom?",
            "Office hours are held every Tuesday and Thursday.",
        ),
    ],
)
def test_answer_query_refuses_missing_exact_details(
    question: str,
    content: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        make_result(
            "handbook.md",
            0,
            content,
            similarity=0.9,
        )
    ]
    monkeypatch.setattr(rag, "retrieve_top_chunks", lambda *_args, **_kwargs: results)

    def unexpected_model_call(_messages: list[dict[str, str]]) -> str:
        raise AssertionError("The chat model must not run for missing exact details.")

    monkeypatch.setattr(rag, "complete_chat_messages", unexpected_model_call)

    response = rag.answer_query(question)

    assert str(response["answer"]).startswith(
        "I do not know based on the available documents."
    )
    assert response["sources"] == ["handbook.md (chunk 1)"]


def test_answer_query_allows_explicit_exact_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        make_result("handbook.md", 0, "The final demo date is July 31, 2026.")
    ]
    monkeypatch.setattr(rag, "retrieve_top_chunks", lambda *_args, **_kwargs: results)
    monkeypatch.setattr(
        rag,
        "complete_chat_messages",
        lambda _messages: "The final demo date is July 31, 2026. [handbook.md#0]",
    )

    response = rag.answer_query("What exact calendar date is the final demo?")

    assert str(response["answer"]).startswith("The final demo date is July 31, 2026.")


def test_build_context_block_preserves_pdf_metadata_and_full_content() -> None:
    result = RetrievalResult(
        chunk_id=1,
        source_name="guide.pdf",
        chunk_index=0,
        content="The Turkish installation instruction is complete and exact.",
        similarity=0.9,
        source_path=r"C:\docs\guide.pdf",
        page_start=7,
        page_end=7,
        extraction_method="pymupdf",
        heading="Installation",
    )

    context = rag.build_context_block([result])

    assert "Page: 7" in context
    assert "Heading: Installation" in context
    assert "Extraction: pymupdf" in context
    assert r"Path: C:\docs\guide.pdf" in context
    assert result.content in context


def test_answer_query_builds_messages_and_canonicalizes_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        make_result(
            "schedule.md",
            1,
            "The final demo begins at 2 PM.",
        )
    ]
    captured_messages: list[dict[str, str]] = []
    monkeypatch.setattr(rag, "retrieve_top_chunks", lambda *_args, **_kwargs: results)

    def fake_complete_chat_messages(messages: list[dict[str, str]]) -> str:
        captured_messages.extend(messages)
        return "The final demo begins at 2 PM. [schedule.md#1]\n\nSources: invented.md#99"

    monkeypatch.setattr(rag, "complete_chat_messages", fake_complete_chat_messages)

    response = rag.answer_query("When does the final demo begin?")

    assert captured_messages[0] == {
        "role": "system",
        "content": rag.SYSTEM_INSTRUCTION,
    }
    assert "[Source: schedule.md#1]" in captured_messages[1]["content"]
    assert response["answer"] == (
        "The final demo begins at 2 PM. [schedule.md#1]"
        "\n\nSources: schedule.md (chunk 2)"
    )
    assert response["sources"] == ["schedule.md (chunk 2)"]


def test_answer_query_uses_grounded_fallback_when_model_omits_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        make_result(
            "schedule.md",
            0,
            "The daily standup starts at 10 AM. Lunch starts at noon.",
        )
    ]
    monkeypatch.setattr(rag, "retrieve_top_chunks", lambda *_args, **_kwargs: results)
    monkeypatch.setattr(rag, "complete_chat_messages", lambda _messages: "The standup is at 10 AM.")

    response = rag.answer_query("When does the daily standup start?")

    assert response["citation_repair_applied"] is True
    assert response["verification"]["verified"] is True
    assert "The daily standup starts at 10 AM. [schedule.md#0]" in response["answer"]


def test_answer_query_repairs_uncited_answer_without_replacing_it_with_first_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        make_result(
            "course_overview.md",
            0,
            (
                "Northstar AI Summer School is a fictional four-week program. "
                "The program meets Monday through Friday from 10:00 AM to 4:00 PM."
            ),
        ),
        make_result(
            "schedule_and_attendance.md",
            0,
            "The standard daily schedule starts with a 15-minute standup at 10:00 AM.",
        ),
    ]
    monkeypatch.setattr(rag, "retrieve_top_chunks", lambda *_args, **_kwargs: results)
    monkeypatch.setattr(
        rag,
        "rerank_results",
        lambda _query, candidates, *, top_k: (list(candidates)[:top_k], "test"),
    )
    monkeypatch.setattr(
        rag,
        "complete_chat_messages",
        lambda _messages: "The program meets Monday through Friday from 10:00 AM to 4:00 PM.",
    )

    response = rag.answer_query("What days and hours does the program meet?")

    assert response["citation_repair_applied"] is True
    assert response["model_no_answer_repair_applied"] is False
    assert response["answer"].startswith(
        "The program meets Monday through Friday from 10:00 AM to 4:00 PM. [course_overview.md#0]"
    )
    assert "fictional four-week program" not in response["answer"]


def test_answer_query_repairs_wrong_no_answer_when_strong_evidence_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        make_result(
            "schedule_and_attendance.md",
            0,
            (
                "The first project checkpoint is at the end of Week 1. "
                "The second checkpoint is at the end of Week 2. "
                "The third checkpoint is at the end of Week 3, when teams must show document chunks stored in SQLite."
            ),
        )
    ]
    monkeypatch.setattr(rag, "retrieve_top_chunks", lambda *_args, **_kwargs: results)
    monkeypatch.setattr(
        rag,
        "complete_chat_messages",
        lambda _messages: "I do not know based on the available documents.",
    )

    response = rag.answer_query("At the end of which week is the third checkpoint?")

    assert response["direct_fact_repair_applied"] is False
    assert response["model_no_answer_repair_applied"] is True
    assert "The third checkpoint is at the end of Week 3" in response["answer"]
    assert "[schedule_and_attendance.md#0]" in response["answer"]


def test_answer_query_answers_explicit_javascript_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        make_result(
            "project_requirements.md",
            0,
            (
                "Retrieval must use cosine similarity for the first version. "
                "For the small sample dataset, it is acceptable to load all stored embeddings into memory and rank them in Python."
            ),
        )
    ]
    monkeypatch.setattr(rag, "retrieve_top_chunks", lambda *_args, **_kwargs: results)
    monkeypatch.setattr(
        rag,
        "complete_chat_messages",
        lambda _messages: "I do not know based on the available documents.",
    )

    response = rag.answer_query("Does the document explicitly permit ranking embeddings in JavaScript?")

    assert response["explicit_answer_repair_applied"] is True
    assert response["model_no_answer_repair_applied"] is False
    assert response["answer"].startswith("No;")
    assert "Python" in response["answer"]
    assert "JavaScript" in response["answer"]


def test_answer_query_answers_gpu_required_vs_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        make_result(
            "tools_and_setup.md",
            0,
            (
                "Microsoft Foundry Local is used to run models on the student device. "
                "On machines with a supported NVIDIA GPU, teams should prefer CUDA GPU variants when available."
            ),
        )
    ]
    monkeypatch.setattr(rag, "retrieve_top_chunks", lambda *_args, **_kwargs: results)
    monkeypatch.setattr(
        rag,
        "complete_chat_messages",
        lambda _messages: "I do not know based on the available documents.",
    )

    response = rag.answer_query(
        "Is an NVIDIA GPU required, or are CUDA variants only recommended when available?"
    )

    assert response["explicit_answer_repair_applied"] is True
    assert response["model_no_answer_repair_applied"] is False
    assert "do not state that an NVIDIA GPU is required" in response["answer"]
    assert "prefer CUDA GPU variants when available" in response["answer"]


def test_answer_query_accepts_model_source_prefix_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        make_result(
            "package.pdf",
            0,
            "Detections Classes Replacement paketi, NovaVision workflow içinde geliştirilmiş bir component paketidir.",
        )
    ]
    monkeypatch.setattr(rag, "retrieve_top_chunks", lambda *_args, **_kwargs: results)
    monkeypatch.setattr(
        rag,
        "complete_chat_messages",
        lambda _messages: "The name of the package is Detections Classes Replacement. [Source: package.pdf#0]",
    )

    response = rag.answer_query("what is the name of package")

    assert response["verification"]["verified"] is True
    assert response["answer"].startswith(
        "The name of the package is Detections Classes Replacement. [package.pdf#0]"
    )


def test_answer_query_repairs_placeholder_citation_for_turkish_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        make_result(
            "package.pdf",
            0,
            (
                "Temel özellikler ● Detection bounding box koordinatlarını korur "
                "● classLabel, classId ve confidence alanlarını güncelleyebilir."
            ),
        )
    ]
    monkeypatch.setattr(rag, "retrieve_top_chunks", lambda *_args, **_kwargs: results)
    monkeypatch.setattr(
        rag,
        "complete_chat_messages",
        lambda _messages: "Bu pakette classLabel, classId ve confidence alanları güncellenir [source.md#0].",
    )

    response = rag.answer_query("bu pakette hangi alanlar güncellenir")

    assert response["verification"]["verified"] is True
    assert response["direct_fact_repair_applied"] is True
    assert "classLabel, classId ve confidence alanları güncellenebilir" in response["answer"]
    assert "[package.pdf#0]" in response["answer"]


def test_answer_query_repairs_turkish_no_answer_for_image_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        make_result(
            "package.pdf",
            0,
            (
                "1. Genel Bakış Paketin amacı ve ne yaptığı. "
                "Bu component görüntü taşıma işlemi yapmaz. "
                "Girişte ve çıkışta yalnızca detection listesi kullanır. "
                "Bu nedenle Suite canvas üzerinde image input/output socket'i bulunmaz."
            ),
        )
    ]
    monkeypatch.setattr(rag, "retrieve_top_chunks", lambda *_args, **_kwargs: results)
    monkeypatch.setattr(
        rag,
        "complete_chat_messages",
        lambda _messages: "Bu komponentte görüntü taşıma yapıyor mu? I do not know based on the available documents.",
    )

    response = rag.answer_query("bu komponentte görüntü taşıma oluyor mu")

    assert response["direct_fact_repair_applied"] is True
    assert response["model_no_answer_repair_applied"] is False
    assert "Hayır; bu component görüntü taşıma işlemi yapmaz. [package.pdf#0]" in response["answer"]
    assert not response["answer"].startswith("1.")


def test_answer_query_recovers_with_verified_evidence_when_chat_generation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        make_result(
            "package.pdf",
            0,
            "Detections Classes Replacement paketi, NovaVision workflow içinde geliştirilmiş bir component paketidir.",
        )
    ]
    monkeypatch.setattr(rag, "retrieve_top_chunks", lambda *_args, **_kwargs: results)

    def fail_chat(_messages: list[dict[str, str]]) -> str:
        raise RuntimeError("onnx allocation failed")

    monkeypatch.setattr(rag, "complete_chat_messages", fail_chat)

    response = rag.answer_query("what is the name of package")

    assert response["chat_generation_recovered"] is True
    assert response["answer"].startswith("The package name is Detections Classes Replacement.")
    assert "[package.pdf#0]" in response["answer"]


def test_answer_query_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        rag.answer_query(" \n ")
