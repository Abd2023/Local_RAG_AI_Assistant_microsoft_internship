# Advanced Phase 1-2 Notes

This document records what was implemented for the first two advanced phases.

## Phase 1: Observability

Every `answer_query` call now creates a local trace record under `data/traces/*.jsonl`.

Trace records include:

- `trace_id`, timestamps, question, answer, status, model aliases, top-k, and retrieval threshold.
- Timings for query embedding, vector search, total retrieval, prompt build, chat generation, and total runtime.
- Retrieved sources with rank order, chunk metadata, score, and preview.
- Guard decision: `normal_generation`, `low_confidence`, `missing_exact_detail`, or `exception`.
- Prompt and answer lengths when available.
- Error type/message for failures.

Commands:

```powershell
.\\run.ps1 eval
.\\run.ps1 traces
python -m src.evaluate
python -m src.traces latest
python -m src.traces failures
```

## Phase 2: Data Pipeline

The live retrieval backend is now LanceDB at `data/lancedb`. SQLite remains the metadata store at `data/rag.db`.

Supported source formats:

- `.md`
- `.txt`
- `.pdf`
- `.docx`

Incremental indexing behavior:

- Scans supported files in `data/sample_docs`.
- Computes stable document IDs and SHA-256 file hashes.
- Skips unchanged files.
- Re-indexes changed files.
- Deletes metadata and vectors for removed files.
- Embeds only new or changed chunks.

Commands:

```powershell
.\\run.ps1 ingest
.\\run.ps1 rebuild
.\\run.ps1 status
```

## Design Choices

- LanceDB stores normalized vectors. LanceDB reports squared L2 distance, so the adapter converts distance back to cosine-like similarity with `1 - distance / 2`.
- SQLite keeps the document manifest and chunk metadata so future phases can add filtering, permissions, and evaluation summaries without coupling everything to vector rows.
- OCR is a fallback only. Text PDFs use direct PyMuPDF extraction. Sparse image pages use Tesseract OCR when available.

## References

- LanceDB Python docs: https://lancedb.github.io/lancedb/python/python/
- LanceDB search docs: https://lancedb.github.io/lancedb/search/
- PyMuPDF text extraction and OCR docs: https://pymupdf.readthedocs.io/
- python-docx docs: https://python-docx.readthedocs.io/
- OpenTelemetry Python instrumentation docs: https://opentelemetry.io/docs/languages/python/instrumentation/
- Phoenix tracing docs: https://docs.arize.com/phoenix/tracing
