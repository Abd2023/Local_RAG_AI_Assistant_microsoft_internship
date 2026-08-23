from pathlib import Path

from src.traces import TraceRecorder, is_notable_trace, load_trace_records


def test_trace_recorder_writes_timing_and_answer(tmp_path: Path) -> None:
    trace = TraceRecorder("What time is standup?", traces_dir=tmp_path)
    with trace.span("retrieval"):
        pass
    trace.set("guard_decision", "normal_generation")
    record = trace.finish(status="ok", answer="Standup starts at 10:00 AM.")

    records = load_trace_records(tmp_path)

    assert record["trace_id"] == records[0]["trace_id"]
    assert records[0]["question"] == "What time is standup?"
    assert records[0]["answer_length"] == len("Standup starts at 10:00 AM.")
    assert "retrieval" in records[0]["timings_ms"]
    assert records[0]["timings_ms"]["total"] >= 0
    assert not is_notable_trace(records[0])


def test_trace_recorder_records_exceptions(tmp_path: Path) -> None:
    trace = TraceRecorder("Broken question", traces_dir=tmp_path)
    error = ValueError("bad input")

    record = trace.finish(status="error", error=error)

    assert record["error"] == {"type": "ValueError", "message": "bad input"}
    assert is_notable_trace(record)
