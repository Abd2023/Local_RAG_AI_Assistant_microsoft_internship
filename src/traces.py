"""Local JSONL tracing for RAG runs."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from src import config


def utc_now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(UTC).isoformat()


def _today_trace_path(traces_dir: Path = config.TRACES_PATH) -> Path:
    return traces_dir / f"{datetime.now(UTC).date().isoformat()}.jsonl"


def save_trace_record(record: dict[str, Any], traces_dir: Path = config.TRACES_PATH) -> Path:
    """Append one trace record to the local trace JSONL file."""
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = _today_trace_path(traces_dir)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
        handle.write("\n")
    return path


@dataclass
class TraceRecorder:
    """Collect timings and metadata for one answer_query call."""

    question: str
    traces_dir: Path = config.TRACES_PATH
    enabled: bool = True
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(default_factory=utc_now_iso)
    _start_time: float = field(default_factory=time.perf_counter)
    record: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.record.update(
            {
                "trace_id": self.trace_id,
                "started_at": self.started_at,
                "question": self.question,
                "models": {
                    "chat": (
                        config.OLLAMA_CHAT_MODEL
                        if config.RAG_PROVIDER == "ollama"
                        else config.CHAT_MODEL_ALIAS
                    ),
                    "embedding": (
                        config.OLLAMA_EMBEDDING_MODEL
                        if config.RAG_PROVIDER == "ollama"
                        else config.EMBEDDING_MODEL_ALIAS
                    ),
                    "provider": config.RAG_PROVIDER,
                    "reranker": config.RERANKER_MODEL if config.RERANKER_ENABLED else None,
                },
                "settings": {
                    "top_k": config.TOP_K,
                    "reranker_candidate_k": config.RERANKER_CANDIDATE_K,
                    "lexical_candidate_k": config.LEXICAL_CANDIDATE_K,
                    "hybrid_retrieval": config.HYBRID_RETRIEVAL_ENABLED,
                    "reranker_device": config.RERANKER_DEVICE,
                    "chat_max_tokens": config.CHAT_MAX_TOKENS,
                    "max_context_chars": config.MAX_CONTEXT_CHARS,
                    "retrieval_min_top_score": config.RETRIEVAL_MIN_TOP_SCORE,
                },
                "timings_ms": {},
                "retrieved_sources": [],
                "prompt_messages": [],
                "citation_verification": None,
                "guard_decision": "not_evaluated",
                "status": "running",
            }
        )

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Measure a named step in milliseconds."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record.setdefault("timings_ms", {})[name] = round(
                (time.perf_counter() - start) * 1000,
                3,
            )

    def set(self, key: str, value: Any) -> None:
        """Set one trace field."""
        self.record[key] = value

    def update(self, values: dict[str, Any]) -> None:
        """Update multiple trace fields."""
        self.record.update(values)

    def finish(
        self,
        *,
        status: str,
        answer: str | None = None,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        """Finalize and optionally save the trace record."""
        self.record["finished_at"] = utc_now_iso()
        self.record["status"] = status
        self.record.setdefault("timings_ms", {})["total"] = round(
            (time.perf_counter() - self._start_time) * 1000,
            3,
        )

        if answer is not None:
            self.record["answer"] = answer
            self.record["answer_length"] = len(answer)

        if error is not None:
            self.record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }

        if self.enabled:
            path = save_trace_record(self.record, self.traces_dir)
            self.record["trace_path"] = str(path)

        return self.record


def _load_trace_file(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def load_trace_records(
    traces_dir: Path = config.TRACES_PATH,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load recent trace records from local JSONL trace files."""
    if not traces_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    for path in sorted(traces_dir.glob("*.jsonl"), reverse=True):
        records.extend(_load_trace_file(path))

    records.sort(key=lambda record: str(record.get("started_at", "")), reverse=True)
    if limit is not None:
        return records[:limit]
    return records


def is_notable_trace(record: dict[str, Any]) -> bool:
    """Return whether a trace is a failure or conservative no-answer."""
    return record.get("status") != "ok" or record.get("guard_decision") in {
        "low_confidence",
        "missing_exact_detail",
    }


def format_trace_summary(record: dict[str, Any]) -> str:
    """Return a compact single-line trace summary."""
    timings = record.get("timings_ms", {})
    top_source = "none"
    sources = record.get("retrieved_sources") or []
    if sources:
        first = sources[0]
        top_source = f"{first.get('source_name')}#{first.get('chunk_index')} score={first.get('similarity')}"

    return (
        f"{record.get('started_at')} {record.get('trace_id')} "
        f"status={record.get('status')} guard={record.get('guard_decision')} "
        f"total_ms={timings.get('total')} top={top_source} "
        f"question={record.get('question')!r}"
    )


def main() -> None:
    """Print recent trace summaries."""
    parser = argparse.ArgumentParser(description="Inspect local RAG trace records.")
    parser.add_argument("command", choices={"latest", "failures"}, nargs="?", default="latest")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    records = load_trace_records(limit=None)
    if args.command == "failures":
        records = [record for record in records if is_notable_trace(record)]
    records = records[: args.limit]

    if not records:
        print(f"No trace records found in {config.TRACES_PATH}")
        return

    for record in records:
        print(format_trace_summary(record))


if __name__ == "__main__":
    main()
