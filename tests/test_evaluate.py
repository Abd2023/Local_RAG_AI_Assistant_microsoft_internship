from pathlib import Path

from src.evaluate import (
    EXPANDED_EVALUATION_QUESTIONS,
    TURKISH_EVALUATION_QUESTIONS,
    run_evaluation,
)


def test_run_evaluation_saves_all_question_results(tmp_path: Path) -> None:
    questions = [
        {"id": "A1", "category": "answerable", "question": "Known?", "expected": "Known."},
        {"id": "E1", "category": "edge", "question": "", "expected": "Error."},
    ]

    def fake_answer(question: str) -> dict[str, object]:
        if not question:
            raise ValueError("Question must not be empty.")
        return {
            "answer": "Known.",
            "trace_id": "trace-1",
            "sources": ["doc.md (chunk 1)"],
            "retrieved_chunks": [
                {
                    "source_name": "doc.md",
                    "chunk_index": 0,
                    "chunk_number": 1,
                    "similarity": 0.9,
                    "source_label": "doc.md (chunk 1)",
                    "preview": "Known.",
                }
            ],
        }

    report = run_evaluation(
        output_dir=tmp_path,
        questions=questions,
        answer_func=fake_answer,
    )

    assert report["summary"] == {
        "total": 2,
        "ok": 1,
        "errors": 1,
        "by_category": {
            "answerable": {"total": 1, "errors": 0},
            "edge": {"total": 1, "errors": 1},
        },
    }
    assert Path(report["output_path"]).exists()
    assert [record["id"] for record in report["records"]] == ["A1", "E1"]
    assert report["records"][0]["trace_id"] == "trace-1"
    assert report["records"][1]["error"]["type"] == "ValueError"


def test_expanded_evaluation_profiles_cover_requested_question_types() -> None:
    expanded_ids = {question["id"] for question in EXPANDED_EVALUATION_QUESTIONS}

    assert {"A6", "A7", "A8", "A9"}.issubset(expanded_ids)
    assert TURKISH_EVALUATION_QUESTIONS[0]["category"] == "turkish"
