# Local RAG Assistant Architecture

This file is a compact, screen-recording-friendly view of the actual project architecture.

```text
User
  |
  | question / document upload
  v
React UI or CLI
  |
  v
FastAPI API / application layer
  |
  +--> Document loaders: MD, TXT, PDF, DOCX
  |       |
  |       v
  |    Chunking + source/page/heading metadata
  |       |
  |       v
  |    Foundry Local embedding model
  |       |
  |       +--> SQLite: documents, chunks, metadata, JSON embeddings
  |       +--> LanceDB: normalized vector rows for fast vector search
  |
  +--> Query embedding
          |
          v
      Hybrid retrieval: vector similarity + SQLite FTS5
          |
          v
      CPU cross-encoder reranking
          |
          v
      Grounded prompt containing only retrieved context
          |
          v
      Microsoft Foundry Local chat model
          |
          v
      Citation verification + answer + sources + trace ID
```

## What happens during a document upload?

1. The UI sends the selected local files to the upload endpoint.
2. The backend extracts text and preserves source metadata.
3. Text is divided into readable chunks.
4. Foundry Local creates one embedding vector per chunk.
5. SQLite stores the canonical document/chunk records and JSON vectors.
6. LanceDB stores normalized vector rows used by the live vector search.

## What happens during a question?

1. The question is embedded locally.
2. The retriever combines semantic vector matches with SQLite FTS5 lexical matches.
3. The best candidates are reranked.
4. The prompt tells the local chat model to answer only from the retrieved context.
5. The answer is checked for valid citations. If the documents do not contain the answer, the assistant returns an explicit “I do not know” response.
6. The request is recorded in a local trace for debugging and evaluation.

The project can run without cloud inference after the required packages and local models have been downloaded and cached.
