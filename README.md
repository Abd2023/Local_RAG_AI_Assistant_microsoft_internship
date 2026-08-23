# Local RAG AI Assistant

Offline document Q&A assistant built with Microsoft Foundry Local, Python, SQLite metadata, LanceDB vector search, and a direct RAG pipeline.

The project answers questions from local documents only. It retrieves relevant chunks, builds a grounded prompt, runs a local chat model, returns sources, and records local traces for debugging.

## In Simple Language

This project is a private question-answering assistant for your own files.

You give it Markdown, text, PDF, or Word files. It breaks those documents into small pieces, finds the pieces related to a question, and asks a local AI model to answer using only those pieces. During normal use, document text does not need to be sent to a cloud AI service.

The important idea is RAG: **retrieve relevant information first, then generate an answer from that information**. If the documents do not contain an answer, the assistant is instructed to say that it does not know instead of inventing a fact.

## What You Can Demonstrate

- Upload local `.md`, `.txt`, `.pdf`, or `.docx` documents.
- Ask questions through the CLI or the browser interface.
- See retrieved source documents and chunk metadata.
- Get grounded answers with validated source citations.
- Test answerable, unanswerable, and edge-case questions.
- Run the pipeline locally after the required packages and models have been downloaded.

## Current Capabilities

- CLI question answering with Microsoft Foundry Local and the configured Phi chat model.
- Markdown, text, PDF, and DOCX ingestion.
- Incremental indexing that skips unchanged files and removes deleted documents.
- Hybrid semantic plus SQLite FTS5 lexical retrieval for exact names, numbers, dates, and terminology.
- Full-chunk prompts with PDF page, heading, extraction method, and source-path metadata.
- Phi-4-mini chat default: `phi-4-mini` through Foundry Local.
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

`setup` creates the Python environment and installs the dependencies. `rebuild` indexes the six sample documents in `data/sample_docs`. `cli` starts the interactive assistant.

For a fresh clone, the first model download may require internet access. After the models are cached, normal questions can run locally. If the machine does not have a supported GPU, try the CPU configuration before rebuilding:

```powershell
$env:FOUNDRY_REQUIRE_CHAT_GPU = "false"
$env:FOUNDRY_REQUIRE_EMBEDDING_GPU = "false"
.\run.ps1 rebuild
.\run.ps1 cli
```

Ask one question without starting the interactive CLI:

```powershell
.\run.ps1 ask "What time does the daily standup start?"
```

Add a new local document and index it immediately:

```powershell
.\run.ps1 upload "C:\path\to\your-document.pdf"
.\run.ps1 ask "What is this document about?"
```

Uploads support `.md`, `.txt`, `.pdf`, and `.docx`. The file is copied into the ignored `data\uploads` directory, then the normal incremental pipeline extracts text, runs OCR when required, chunks the content, creates embeddings, updates both search indexes, and records metadata. Re-uploading an unchanged file skips embedding; a changed file replaces its chunks and vectors.

To start over with a fresh document set, use `clean`:

```powershell
.\run.ps1 clean
.\run.ps1 clean "C:\path\to\documents-folder"
```

The interactive CLI has the same behavior:

```text
Question> /clean
Question> /clean 'C:\path\documents-folder'
```

`/clean` removes staged uploaded documents, clears SQLite metadata, clears vector rows, and rebuilds the empty lexical index. `/clean <path>` does that first, then recursively stages and indexes the new `.pdf`, `.docx`, `.md`, and `.txt` documents from the path.

The interactive CLI accepts one path, multiple quoted paths, or a recursive directory:

```text
Question> /upload 'C:\path\Package Geliştirici Kılavuzu.pdf'
Question> /upload 'C:\path\one.pdf' 'C:\path\two.pdf'
Question> /upload 'C:\path\documents-folder'
Question> What are the main requirements in this document?
```

The directory form recursively includes `.pdf`, `.docx`, `.md`, and `.txt` files and performs one combined indexing pass. Failed files are reported without blocking valid files. Upload commands never call the chat model.

## Run the Browser Interface

Start the backend in one terminal from the project root:

```powershell
.\run.ps1 api
```

Install and start the React frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The page lets you upload documents, ask questions, and inspect retrieved evidence, confidence, citations, and trace IDs.

Run tests:

```powershell
.\run.ps1 test
```

## Short Video Demo

For a 3–5 minute screen recording in Turkish, use [docs/video_demo_tr.md](docs/video_demo_tr.md). It includes the speaking script, architecture explanation, upload flow, answerable and unanswerable questions, and the “what I learned” section. The small files in `demo_data` are safe, deterministic documents for the live upload demonstration.

## Screenshots

Screenshots can be added here later. Suggested files are `docs/images/web-ui.png`, `docs/images/uploaded-documents.png`, and `docs/images/grounded-answer.png`.

<!-- Example:
![Local RAG browser interface](docs/images/web-ui.png)
-->

Run the manual evaluation set and inspect traces:

```powershell
.\run.ps1 eval
.\run.ps1 eval --expanded
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
- `.\\run.ps1 upload "path"`: copy one supported document into `data\\uploads` and index it immediately.
- `.\\run.ps1 clean ["path"]`: clear uploaded documents, metadata, and vectors; with a path, index a fresh corpus.
- `.\\run.ps1 cli`: start the interactive assistant.
- `.\\run.ps1 ask "question"`: ask one question and exit.
- `.\\run.ps1 eval`: run the 10-question acceptance suite (5 answerable, 3 unanswerable, 2 edge cases).
- `.\\run.ps1 eval --expanded`: add exact-fact, multi-chunk, and full-summary cases.
- `.\\run.ps1 eval --expanded --turkish`: also test a Turkish-language document after uploading it.
- `.\\run.ps1 traces`: print recent local trace summaries.
- `.\\run.ps1 status`: show database/vector counts.
- `.\\run.ps1 test`: run unit tests.

## Architecture

```text
data/sample_docs + data/uploads
  -> document loaders (.md, .txt, .pdf, .docx)
  -> chunking with source metadata
  -> local embeddings through Foundry Local
  -> SQLite document/chunk metadata
  -> LanceDB or Qdrant vector rows
  -> query embedding
  -> hybrid vector + SQLite FTS5 search (up to 20 candidates)
  -> CPU cross-encoder reranking (best 5 chunks)
  -> guarded RAG prompt
  -> local chat model
  -> citation verification
  -> answer + sources + verification + trace_id
```

## Reranking and Evidence

The default reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`, configured for CPU by default so Phi-4-mini can use the GPU. It is loaded lazily and can be configured with `RERANKER_MODEL`, `RERANKER_DEVICE`, `RERANKER_BATCH_SIZE`, `RERANKER_CANDIDATE_K`, `RAG_TOP_K`, and `RERANKER_ENABLED`. Unit tests inject a fake scorer, so normal test runs do not download a model.

Every factual sentence must use a retrieved citation such as `[tools_and_setup.md#0]`. The backend rejects uncited or invented citations and returns verification details in the response and local trace.

## HTTP API

Start the native service with `run.ps1 api`, then use:

- `GET http://localhost:8000/api/health`
- `POST http://localhost:8000/api/query` with `{ "question": "..." }`
- `POST http://localhost:8000/api/ingest`
- `POST http://localhost:8000/api/upload` as multipart form data with one or more repeated `files` fields; the legacy `file` field is also accepted.
- `GET http://localhost:8000/api/traces/{trace_id}`

## Docker Compose

The Compose stack runs the backend with Ollama and Qdrant and serves the React interface at `http://localhost:3000`:

```powershell
docker compose up -d qdrant ollama
docker compose exec ollama ollama pull phi4-mini
docker compose exec ollama ollama pull nomic-embed-text
docker compose up --build -d backend frontend
```

The first model pulls require internet access. After Docker images and model volumes are cached, normal querying runs locally without cloud inference. The backend rebuilds an empty Qdrant collection automatically on its first start.

## Foundry Local GPU Notes

Native Foundry runs use Phi-4-mini as the default chat model. Chat requires the CUDA GPU variant by default, while embeddings use the CPU variant by default so query embedding does not occupy VRAM before Phi-4-mini loads. The app does not automatically fall back to a weaker chat model.

Useful overrides:

- `FOUNDRY_REQUIRE_CHAT_GPU=false`: explicitly use the Foundry CPU chat variant of the same `phi-4-mini` alias.
- `FOUNDRY_REQUIRE_EMBEDDING_GPU=true`: force embedding on GPU.
- `FOUNDRY_ALLOW_CPU_FALLBACK=true`: if GPU loading fails, retry the same `phi-4-mini` alias on CPU.
- `FOUNDRY_UNLOAD_CHAT_AFTER_USE=false`: keep Phi-4-mini loaded between answers for speed when memory allows.
- `FOUNDRY_UNLOAD_EMBEDDING_AFTER_USE=true`: unload the embedding model after each embedding call when chat memory is more important than embedding reload speed.
- `RAG_CHAT_MAX_TOKENS=220`: default concise answer budget; increase only if longer summaries are needed and memory allows.
- `FOUNDRY_REQUIRE_GPU=false`: legacy switch that disables GPU requirements for both chat and embeddings unless the more specific variables are set.

## OCR Notes

PDF text extraction uses PyMuPDF. OCR is only attempted for sparse PDF pages that appear scanned and contain images.

OCR requires a separate Tesseract installation on Windows. If scanned PDFs need OCR, install Tesseract and set `TESSDATA_PREFIX` so PyMuPDF can find tessdata. The Docker backend image includes English Tesseract OCR. Normal text PDFs do not need OCR.

## Generated Local Data

These paths are generated and intentionally ignored by Git:

- `data/rag.db`
- `data/lancedb/`
- `data/traces/`
- `data/evaluations/`
- `data/uploads/`
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
