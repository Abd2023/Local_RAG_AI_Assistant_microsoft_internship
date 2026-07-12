# Local RAG AI Assistant With Microsoft Foundry Local

## Purpose

Build a local offline Q&A assistant that answers questions from a small document collection using Retrieval-Augmented Generation (RAG), Microsoft Foundry Local, local embeddings, SQLite, and a command-line interface.

The first version should prioritize a working, understandable prototype over frameworks or advanced UI. The goal is to learn and prove every part of the RAG pipeline before adding abstractions.

## Framework Decision

For v1, do not use LangChain, LangGraph, or an agentic framework.

Use a direct Python pipeline:

```text
documents -> chunks -> embeddings -> SQLite -> retrieval -> prompt -> Foundry Local LLM -> CLI answer
```

Reason:

- The internship project is about understanding how RAG works.
- Frameworks can hide important parts like chunking, embedding, vector similarity, prompt construction, and grounding.
- A direct implementation is easier to debug step by step.
- LangChain or LangGraph can be added later as stretch goals after the core project works.

Possible future use:

- LangChain: document loaders, prompt templates, retrievers, and standard RAG abstractions.
- LangGraph: agentic workflows, tool routing, memory, human-in-the-loop, or multi-step decision flows.

## Project Defaults

- Project folder: `C:\fun_project\microsoft_internship\project`
- Language: Python
- Python version: 3.11 or newer
- First interface: CLI
- First document type: `.txt` and `.md`
- First dataset: small sample documents
- Database: SQLite file at `data/rag.db`
- Embedding model: `qwen3-embedding-0.6b`
- Chat model: `qwen2.5-0.5b`
- Retrieval count: top 3 chunks
- First priority: working offline prototype

## Target Project Structure

```text
project/
  data/
    sample_docs/
    rag.db
  docs/
    implementation_plan.md
    architecture.md
    evaluation.md
  src/
    __init__.py
    config.py
    foundry_client.py
    ingest.py
    storage.py
    retrieval.py
    rag.py
    cli.py
  tests/
    test_chunking.py
    test_storage.py
    test_retrieval.py
    test_rag_prompt.py
  README.md
  requirements.txt
```

## Step 1: Prepare The Python Environment

### Implementation

1. Open a terminal in:

   ```powershell
   C:\fun_project\microsoft_internship\project
   ```

2. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

3. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Create `requirements.txt` with:

   ```text
   foundry-local-sdk-winml
   openai
   numpy
   pytest
   rich
   ```

5. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

### Control Check

This step is complete only if:

- `.venv` exists.
- `pip list` shows `foundry-local-sdk-winml`, `openai`, `numpy`, `pytest`, and `rich`.
- `python --version` reports Python 3.11 or newer.
- `python -c "import foundry_local_sdk"` runs without an import error.

## Step 2: Create Project Skeleton

### Implementation

Create the folders and empty source files:

- `data/sample_docs/`
- `docs/`
- `src/`
- `tests/`
- `src/__init__.py`
- `src/config.py`
- `src/foundry_client.py`
- `src/ingest.py`
- `src/storage.py`
- `src/retrieval.py`
- `src/rag.py`
- `src/cli.py`
- `README.md`

### Control Check

This step is complete only if:

- All folders exist.
- All listed files exist.
- `python -m src.cli` can be run and exits cleanly with a placeholder message.

## Step 3: Add Configuration

### Implementation

In `src/config.py`, define constants for:

- Project root path
- Sample docs path
- SQLite database path
- Embedding model alias
- Chat model alias
- Top-k retrieval count
- Chunk size target

Use these default values:

```python
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
CHAT_MODEL_ALIAS = "qwen2.5-0.5b"
TOP_K = 3
CHUNK_TARGET_WORDS = 500
```

### Control Check

This step is complete only if:

- Other modules can import `src.config`.
- Paths resolve relative to the project root.
- Running `python -c "from src import config; print(config.TOP_K)"` prints `3`.

## Step 4: Verify Foundry Local Chat

### Implementation

In `src/foundry_client.py`, implement:

- Foundry Local SDK initialization.
- Windows execution provider download/registration.
- Chat model download/load.
- A small function to send one prompt to the local model.

Use the chat model:

```text
qwen2.5-0.5b
```

### Control Check

This step is complete only if:

- A local model downloads or is already cached.
- The chat model loads successfully.
- A test prompt such as `Say hello in one sentence.` returns a generated answer.
- The test works without calling a cloud API.

## Step 5: Verify Foundry Local Embeddings

### Implementation

Extend `src/foundry_client.py` with:

- Embedding model download/load.
- A function that accepts one text string and returns one embedding vector.
- A function that accepts multiple strings and returns multiple embedding vectors.

Use the embedding model:

```text
qwen3-embedding-0.6b
```

### Control Check

This step is complete only if:

- The embedding model downloads or is already cached.
- A sentence produces a list of numeric values.
- The embedding vector length is greater than zero.
- Two calls with similar sentences produce vectors that can be compared numerically.

## Step 6: Create Sample Documents

### Implementation

Create 5 to 8 small `.md` or `.txt` files in `data/sample_docs/`.

Use one controlled topic so retrieval can be tested easily. Good options:

- A fictional university course handbook
- A small product FAQ
- A technical onboarding manual
- A Microsoft internship FAQ if the content is non-confidential

Each document should have:

- A clear title
- 3 to 8 short paragraphs
- Information that can answer obvious test questions
- Some information that is intentionally absent, so unanswerable questions can be tested

### Control Check

This step is complete only if:

- `data/sample_docs/` contains at least 5 readable files.
- Each file has enough text to split into chunks.
- We can write at least 5 answerable questions from the documents.
- We can write at least 3 unanswerable questions not covered by the documents.

## Step 7: Implement Text Loading And Chunking

### Implementation

In `src/ingest.py`, implement:

- Read `.txt` and `.md` files from `data/sample_docs/`.
- Split text by paragraphs/headings.
- Combine small paragraphs into chunks around 300 to 700 words.
- Preserve metadata:
  - `source_name`
  - `chunk_index`
  - `content`

For v1, do not support PDF or DOCX ingestion.

### Control Check

This step is complete only if:

- Running the chunking code prints or returns chunks from every sample document.
- Empty files are skipped or handled cleanly.
- Every chunk includes `source_name`, `chunk_index`, and `content`.
- No chunk is empty.

## Step 8: Implement SQLite Storage

### Implementation

In `src/storage.py`, implement:

- Connect to `data/rag.db`.
- Create a `chunks` table.
- Clear/rebuild the table during ingestion.
- Insert chunks with embeddings.
- Read all chunks back for retrieval.

Use this v1 schema:

```sql
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

Store embeddings as JSON text for simplicity.

### Control Check

This step is complete only if:

- `data/rag.db` is created.
- The `chunks` table exists.
- Inserted chunks can be read back.
- The row count matches the number of chunks produced by ingestion.

## Step 9: Implement The Ingestion Pipeline

### Implementation

In `src/ingest.py`, connect the earlier pieces:

1. Load sample documents.
2. Chunk the documents.
3. Generate embeddings for chunks.
4. Rebuild the SQLite table.
5. Store chunk content and embedding vectors.

Run it with:

```powershell
python -m src.ingest
```

### Control Check

This step is complete only if:

- `python -m src.ingest` completes without errors.
- The database contains chunks from all sample documents.
- A rerun rebuilds cleanly without duplicate rows.
- The script prints a useful summary such as number of files, chunks, and stored rows.

## Step 10: Implement Vector Similarity Retrieval

### Implementation

In `src/retrieval.py`, implement:

- Cosine similarity between two vectors.
- Query embedding generation.
- Load all stored chunk embeddings from SQLite.
- Rank chunks by similarity.
- Return top `TOP_K` chunks.

For v1, compute similarity in Python with `numpy`. Do not add a vector database yet.

### Control Check

This step is complete only if:

- Known test vectors return expected similarity order.
- A query related to one sample document retrieves a chunk from that document.
- Retrieval returns source names and chunk content.
- Retrieval handles an empty database with a clear error message.

## Step 11: Build The RAG Prompt And Answer Function

### Implementation

In `src/rag.py`, implement:

- `answer_query(question: str) -> dict`
- Retrieve top chunks.
- Build a context block with source labels.
- Send messages to the local chat model.
- Return:
  - final answer
  - retrieved sources
  - retrieved chunk previews or IDs

System instruction:

```text
You are a local document Q&A assistant. Answer only using the provided context.
If the answer is not in the context, say you do not know based on the available documents.
Be concise and include the source names you used.
```

### Control Check

This step is complete only if:

- Answerable questions produce answers grounded in the sample documents.
- Unanswerable questions produce an "I do not know" style answer.
- Answers include source names.
- Retrieved chunks can be inspected for debugging.

## Step 12: Build The CLI

### Implementation

In `src/cli.py`, implement a loop:

- Print a startup message.
- Accept user questions.
- Ignore empty input.
- Exit on `exit` or `quit`.
- Call `answer_query`.
- Print the answer and sources.

Run it with:

```powershell
python -m src.cli
```

### Control Check

This step is complete only if:

- CLI starts successfully.
- A user can ask multiple questions in one session.
- `exit` and `quit` close the program cleanly.
- Empty input does not call the model.
- Answers and sources are readable in the terminal.

## Step 13: Add Unit Tests

### Implementation

Add tests for:

- Chunking behavior
- SQLite insert/read behavior
- Cosine similarity
- Retrieval ranking with fake embeddings
- Prompt construction rules

Tests that require Foundry Local model downloads should be marked as manual or integration tests. Unit tests should be able to run without model downloads.

Run tests with:

```powershell
pytest
```

### Control Check

This step is complete only if:

- `pytest` runs.
- Unit tests pass without requiring a model download.
- Retrieval logic is tested with deterministic fake vectors.
- Prompt construction test confirms context and source names are included.

## Step 14: Manual Evaluation

### Implementation

Create `docs/evaluation.md` with:

- 5 answerable questions
- 3 unanswerable questions
- 2 vague or edge-case questions
- Expected behavior
- Actual answer
- Retrieved sources
- Pass/fail judgment
- Notes for improvement

### Control Check

This step is complete only if:

- Every evaluation question has a recorded result.
- Answerable questions mostly retrieve the right source.
- Unanswerable questions do not hallucinate.
- Any failures are documented with a likely cause.

## Step 15: Write Project Documentation

### Implementation

Update `README.md` with:

- Project overview
- Architecture explanation
- Setup instructions
- Ingestion command
- CLI run command
- Example questions
- Limitations
- Future improvements

Create `docs/architecture.md` explaining:

```text
sample docs -> chunking -> embeddings -> SQLite -> query embedding -> similarity search -> context prompt -> local LLM -> answer
```

### Control Check

This step is complete only if:

- A new person can follow the README and run the project.
- Architecture documentation matches the actual code.
- Known limitations are clearly stated.

## Step 16: Prepare Demo

### Implementation

Prepare a short demo script:

1. Show the sample documents.
2. Run ingestion.
3. Start CLI.
4. Ask 2 answerable questions.
5. Ask 1 unanswerable question.
6. Explain retrieved sources.
7. Summarize what was learned.

### Control Check

This step is complete only if:

- Demo can be completed in 3 to 5 minutes.
- The assistant works offline after models are downloaded.
- The unanswerable question demonstrates responsible behavior.
- The project explanation is clear enough for internship review.

## Stretch Goals After V1

Only start these after the CLI prototype is working and tested:

- Add Streamlit or Gradio UI.
- Add PDF and DOCX ingestion.
- Add source citations with chunk IDs.
- Add conversation memory.
- Add LangChain version of the same RAG pipeline for comparison.
- Add LangGraph agentic retrieval where the model decides whether retrieval is needed.
- Add a real vector database or SQLite vector extension.
- Add better evaluation metrics.
- Add packaging or one-command setup.

## Definition Of Done For V1

The v1 project is done when:

- `python -m src.ingest` builds the database from sample documents.
- `python -m src.cli` starts an offline Q&A assistant.
- Answerable questions retrieve relevant context and produce grounded answers.
- Unanswerable questions do not produce fabricated answers.
- Sources are shown with answers.
- Unit tests pass.
- Manual evaluation is documented.
- README explains setup, usage, architecture, and limitations.

## Main References

- Microsoft Foundry Local SDK reference: https://learn.microsoft.com/en-us/azure/foundry-local/reference/reference-sdk-current
- Foundry Local embeddings: https://learn.microsoft.com/en-us/azure/foundry-local/how-to/how-to-generate-embeddings
- Foundry Local native chat completions: https://learn.microsoft.com/en-us/azure/foundry-local/how-to/how-to-use-native-chat-completions
- Foundry Local with LangChain: https://learn.microsoft.com/en-us/azure/foundry-local/how-to/how-to-use-langchain-with-foundry-local
- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
