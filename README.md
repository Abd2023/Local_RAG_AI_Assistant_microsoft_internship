# Local RAG AI Assistant

Offline document Q&A assistant built with Microsoft Foundry Local, Python, SQLite metadata, LanceDB vector search, and a direct RAG pipeline.

The project answers questions from local documents only. It retrieves relevant chunks, builds a grounded prompt, runs a local chat model, returns sources, and records local traces for debugging.

## Current Capabilities

- CLI question answering with local Foundry Local models.
- Markdown, text, PDF, and DOCX ingestion.
- Incremental indexing that skips unchanged files and removes deleted documents.
- SQLite metadata storage at `data/rag.db`.
- LanceDB vector storage at `data/lancedb`.
- Local JSONL traces at `data/traces`.
- Structured evaluation reports at `data/evaluations`.
- Optional cross-encoder reranking with `sentence-transformers`.
- Strict citation provenance verification for generated claims.
- FastAPI service and React chat interface.
- Docker Compose packaging with Ollama and Qdrant.
- Unit tests for chunking, storage, retrieval, tracing, evaluation, document loading, vector search, and incremental indexing.

## Quick Start

From the project root:

```powershell
cd C:\fun_project\microsoft_internship\project
.\run.ps1 setup
.\run.ps1 rebuild
.\run.ps1 cli
```

Ask one question without starting the interactive CLI:

```powershell
.\run.ps1 ask "What time does the daily standup start?"
```

Run tests:

```powershell
.\run.ps1 test
```

Run the manual evaluation set and inspect traces:

```powershell
.\run.ps1 eval
.\run.ps1 traces
```

If PowerShell blocks local scripts:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 cli
```

## Runner Commands

- `.\\run.ps1 setup`: create `.venv` if needed and install dependencies.
- `.\\run.ps1 ingest`: incrementally update changed documents.
- `.\\run.ps1 rebuild`: fully rebuild SQLite metadata and LanceDB vectors.
- `.\\run.ps1 cli`: start the interactive assistant.
- `.\\run.ps1 ask "question"`: ask one question and exit.
- `.\\run.ps1 eval`: run the 10-question evaluation suite.
- `.\\run.ps1 traces`: print recent local trace summaries.
- `.\\run.ps1 status`: show database/vector counts.
- `.\\run.ps1 test`: run unit tests.

## Architecture

```text
data/sample_docs
  -> document loaders (.md, .txt, .pdf, .docx)
  -> chunking with source metadata
  -> local embeddings through Foundry Local or Ollama
  -> SQLite document/chunk metadata
  -> LanceDB or Qdrant vector rows
  -> query embedding
  -> vector search (top 10 candidates)
  -> cross-encoder reranking (top 3)
  -> guarded RAG prompt
  -> local chat model
  -> citation verification
  -> answer + sources + verification + trace_id
```

## Reranking and Evidence

The default reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`. It is loaded lazily and can be configured with `RERANKER_MODEL`, `RERANKER_DEVICE`, `RERANKER_CANDIDATE_K`, `RAG_TOP_K`, and `RERANKER_ENABLED`. Unit tests inject a fake scorer, so normal test runs do not download a model.

Every factual sentence must use a retrieved citation such as `[tools_and_setup.md#0]`. The backend rejects uncited or invented citations and returns verification details in the response and local trace.

## HTTP API

Start the native service with `run.ps1 api`, then use:

- `GET http://localhost:8000/api/health`
- `POST http://localhost:8000/api/query` with `{ "question": "..." }`
- `POST http://localhost:8000/api/ingest`
- `GET http://localhost:8000/api/traces/{trace_id}`

## Docker Compose

The Compose stack runs the backend with Ollama and Qdrant and serves the React interface at `http://localhost:3000`:

```powershell
docker compose up --build
docker compose exec ollama ollama pull qwen2.5:0.5b
docker compose exec ollama ollama pull nomic-embed-text
```

The first model pulls require internet access. After Docker images and model volumes are cached, normal querying runs locally without cloud inference. The backend rebuilds an empty Qdrant collection automatically on its first start.

## OCR Notes

PDF text extraction uses PyMuPDF. OCR is only attempted for sparse PDF pages that appear scanned and contain images.

OCR requires a separate Tesseract installation on Windows. If scanned PDFs need OCR, install Tesseract and set `TESSDATA_PREFIX` so PyMuPDF can find tessdata. Normal text PDFs do not need OCR.

## Generated Local Data

These paths are generated and intentionally ignored by Git:

- `data/rag.db`
- `data/lancedb/`
- `data/traces/`
- `data/evaluations/`
- `.venv/`

## Useful Test Questions

Answerable:

- What time does the daily standup start?
- How much of the final project grade is RAG pipeline correctness worth?
- What storage layer is required for the first version?
- What are the default chat and embedding models for the course?

Unanswerable:

- Who is the instructor for Northstar AI Summer School?
- What is the street address of the classroom?
- What exact calendar date is the final demo?

Known weak case before reranking:

- Tell me about setup.

The trace output shows this vague question currently retrieves `grading_and_demo.md` before `tools_and_setup.md`, which is exactly the kind of failure Phase 3 reranking should address.
