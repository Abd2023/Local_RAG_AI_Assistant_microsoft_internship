"""FastAPI service for the local RAG assistant."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src import config
from src.ingest import document_path_for_name, ingest_documents
from src.rag import answer_query
from src.storage import count_chunks
from src.traces import load_trace_records
from src.vector_store import count_vectors


class QueryRequest(BaseModel):
    """Question submitted to the assistant."""

    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be empty.")
        return value.strip()


class IngestRequest(BaseModel):
    """Optional ingestion controls for the local documents directory."""

    documents_path: str | None = None
    rebuild: bool = False


app = FastAPI(title="Local RAG AI Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _safe_documents_path(documents_path: str | None) -> Path | None:
    """Validate an optional data directory, preserving the combined default."""
    if not documents_path:
        return None

    path = Path(documents_path)
    resolved = path.resolve()
    data_root = config.DATA_DIR.resolve()
    if resolved != data_root and data_root not in resolved.parents:
        raise HTTPException(
            status_code=400,
            detail="documents_path must be inside the project's data directory.",
        )
    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(status_code=400, detail="documents_path must be an existing directory.")
    return resolved


@app.get("/api/health")
def health() -> dict[str, object]:
    """Return service configuration and local index counts."""
    try:
        chunks = count_chunks()
        vectors = count_vectors()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Local vector store unavailable: {exc}") from exc

    return {
        "status": "ok",
        "provider": config.RAG_PROVIDER,
        "vector_backend": config.VECTOR_BACKEND,
        "chunks": chunks,
        "vectors": vectors,
        "reranker_enabled": config.RERANKER_ENABLED,
    }


@app.post("/api/query")
def query(request: QueryRequest) -> dict[str, object]:
    """Answer one question with sources and verification metadata."""
    try:
        return answer_query(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        trace_id = getattr(exc, "trace_id", None)
        detail: dict[str, object] = {"message": str(exc)}
        if trace_id:
            detail["trace_id"] = trace_id
        raise HTTPException(status_code=500, detail=detail) from exc


@app.post("/api/ingest")
def ingest(request: IngestRequest | None = None) -> dict[str, object]:
    """Incrementally index an allowed directory, or the sample and upload corpus."""
    request = request or IngestRequest()
    documents_path = _safe_documents_path(request.documents_path)
    try:
        return ingest_documents(documents_path, rebuild=request.rebuild)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/upload")
async def upload(
    files: list[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
) -> dict[str, object]:
    """Store one or more documents and immediately run one indexing pass."""
    uploads = list(files or [])
    if file is not None:
        uploads.append(file)
    if not uploads:
        raise HTTPException(status_code=400, detail="At least one document is required.")

    staged_paths: list[Path] = []
    response_files: list[dict[str, object]] = []
    reserved_destinations: set[Path] = set()

    for upload_file in uploads:
        temporary_path: Path | None = None
        try:
            destination = document_path_for_name(upload_file.filename or "")
            if destination in reserved_destinations:
                destination = destination.with_name(
                    f"{destination.stem}_{len(reserved_destinations)}{destination.suffix}"
                )
            reserved_destinations.add(destination)

            size_bytes = 0
            with NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".uploading",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                while chunk := await upload_file.read(1024 * 1024):
                    size_bytes += len(chunk)
                    if size_bytes > config.MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"Document exceeds the "
                                f"{config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
                            ),
                        )
                    temporary.write(chunk)

            if size_bytes == 0:
                raise HTTPException(status_code=400, detail="Uploaded document is empty.")

            temporary_path.replace(destination)
            temporary_path = None
            staged_paths.append(destination)
            response_files.append(
                {
                    "document_name": destination.name,
                    "document_path": str(destination),
                    "status": "staged",
                }
            )
        except HTTPException as exc:
            response_files.append(
                {
                    "document_name": upload_file.filename or "unknown",
                    "status": "error",
                    "error": str(exc.detail),
                }
            )
        except Exception as exc:
            response_files.append(
                {
                    "document_name": upload_file.filename or "unknown",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            await upload_file.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    if not staged_paths:
        raise HTTPException(status_code=400, detail={"files": response_files})

    try:
        summary = ingest_documents()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    outcomes = {
        str(item.get("document_name")): item
        for item in summary.get("documents", [])
        if isinstance(item, dict)
    }
    for item in response_files:
        outcome = outcomes.get(str(item.get("document_name")))
        if outcome:
            item.update(outcome)

    return {
        "document_name": response_files[0].get("document_name"),
        "document_path": response_files[0].get("document_path"),
        "files": response_files,
        "index": summary,
    }


@app.get("/api/traces/{trace_id}")
def trace(trace_id: str) -> dict[str, object]:
    """Return one locally persisted trace record."""
    for record in load_trace_records(limit=None):
        if record.get("trace_id") == trace_id:
            return record
    raise HTTPException(status_code=404, detail="Trace not found.")
