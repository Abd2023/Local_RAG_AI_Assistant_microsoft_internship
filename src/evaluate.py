"""Manual evaluation runner for the local RAG assistant."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src import config
from src.rag import answer_query

EvaluationAnswerFunc = Callable[[str], dict[str, Any]]

EVALUATION_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "A1",
        "category": "answerable",
        "question": "What time does the daily standup start?",
        "expected": "The daily standup starts at 10:00 AM.",
    },
    {
        "id": "A2",
        "category": "answerable",
        "question": "How much of the final project grade is RAG pipeline correctness worth?",
        "expected": "RAG pipeline correctness is worth 40 percent.",
    },
    {
        "id": "A3",
        "category": "answerable",
        "question": "What storage layer is required for the first version?",
        "expected": "SQLite is required for the first version.",
    },
    {
        "id": "A4",
        "category": "answerable",
        "question": "What are the default chat and embedding models for the course?",
        "expected": "The chat model is qwen2.5-0.5b and the embedding model is qwen3-embedding-0.6b.",
    },
    {
        "id": "A5",
        "category": "answerable",
        "question": "What should students do if they are blocked for more than 30 minutes?",
        "expected": "They should ask for help, starting with a teammate, then notes, then the instructor during lab time.",
    },
    {
        "id": "U1",
        "category": "unanswerable",
        "question": "Who is the instructor for Northstar AI Summer School?",
        "expected": "The documents do not specify the instructor's identity.",
    },
    {
        "id": "U2",
        "category": "unanswerable",
        "question": "What is the street address of the classroom?",
        "expected": "The documents do not provide a street address.",
    },
    {
        "id": "U3",
        "category": "unanswerable",
        "question": "What exact calendar date is the final demo?",
        "expected": "The documents do not provide an exact calendar date.",
    },
    {
        "id": "E1",
        "category": "edge",
        "question": "Tell me about setup.",
        "expected": "A cautious setup summary or clarification request.",
    },
    {
        "id": "E2",
        "category": "edge",
        "question": "   ",
        "expected": "A clear empty-question validation error.",
        "expected_error": "ValueError",
    },
]


def _compact_chunks(chunks: object) -> list[dict[str, Any]]:
    if not isinstance(chunks, list):
        return []
    compact: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        compact.append(
            {
                "source_name": chunk.get("source_name"),
                "chunk_index": chunk.get("chunk_index"),
                "chunk_number": chunk.get("chunk_number"),
                "similarity": chunk.get("similarity"),
                "source_label": chunk.get("source_label"),
                "preview": chunk.get("preview"),
            }
        )
    return compact


def run_evaluation(
    *,
    output_dir: Path = config.EVALUATIONS_PATH,
    questions: list[dict[str, str]] = EVALUATION_QUESTIONS,
    answer_func: EvaluationAnswerFunc = answer_query,
) -> dict[str, Any]:
    """Run evaluation questions and save a structured JSON report."""
    started_at = datetime.now(UTC).isoformat()
    records: list[dict[str, Any]] = []

    for question in questions:
        record: dict[str, Any] = {
            "id": question["id"],
            "category": question["category"],
            "question": question["question"],
            "expected": question["expected"],
            "status": "running",
        }
        try:
            result = answer_func(question["question"])
            if question.get("expected_error"):
                record.update(
                    {
                        "status": "error",
                        "error": {
                            "type": "ExpectedErrorNotRaised",
                            "message": f"Expected {question['expected_error']}.",
                        },
                        "answer": result.get("answer"),
                        "trace_id": result.get("trace_id"),
                        "sources": result.get("sources", []),
                        "retrieved_chunks": _compact_chunks(result.get("retrieved_chunks")),
                    }
                )
                records.append(record)
                continue
            record.update(
                {
                    "status": "ok",
                    "answer": result.get("answer"),
                    "trace_id": result.get("trace_id"),
                    "sources": result.get("sources", []),
                    "retrieved_chunks": _compact_chunks(result.get("retrieved_chunks")),
                }
            )
        except Exception as exc:
            if question.get("expected_error") == type(exc).__name__:
                record.update(
                    {
                        "status": "ok",
                        "expected_error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                        "answer": None,
                        "trace_id": getattr(exc, "trace_id", None),
                        "sources": [],
                        "retrieved_chunks": [],
                    }
                )
            else:
                record.update(
                    {
                        "status": "error",
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                        "answer": None,
                        "trace_id": getattr(exc, "trace_id", None),
                        "sources": [],
                        "retrieved_chunks": [],
                    }
                )
        records.append(record)

    summary = {
        "total": len(records),
        "ok": sum(1 for record in records if record["status"] == "ok"),
        "errors": sum(1 for record in records if record["status"] == "error"),
        "by_category": {},
    }
    for record in records:
        category = record["category"]
        bucket = summary["by_category"].setdefault(category, {"total": 0, "errors": 0})
        bucket["total"] += 1
        if record["status"] == "error":
            bucket["errors"] += 1

    report = {
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "summary": summary,
        "records": records,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"evaluation-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    report["output_path"] = str(output_path)
    return report


def main() -> None:
    """Run the manual evaluation suite."""
    parser = argparse.ArgumentParser(description="Run the local RAG evaluation questions.")
    parser.add_argument("--output-dir", type=Path, default=config.EVALUATIONS_PATH)
    args = parser.parse_args()

    report = run_evaluation(output_dir=args.output_dir)
    summary = report["summary"]
    print(f"Evaluation saved to {report['output_path']}")
    print(f"Total: {summary['total']} | OK: {summary['ok']} | Errors: {summary['errors']}")
    for record in report["records"]:
        trace_id = record.get("trace_id") or "none"
        print(f"- {record['id']} {record['status']} trace={trace_id}")


if __name__ == "__main__":
    main()
