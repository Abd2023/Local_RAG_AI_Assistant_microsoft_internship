"""Configuration constants for the Local RAG AI Assistant."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DOCS_PATH = DATA_DIR / "sample_docs"
DATABASE_PATH = DATA_DIR / "rag.db"
VECTOR_DB_PATH = DATA_DIR / "lancedb"
TRACES_PATH = DATA_DIR / "traces"
EVALUATIONS_PATH = DATA_DIR / "evaluations"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


RAG_PROVIDER = os.getenv("RAG_PROVIDER", "foundry").strip().lower()
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "lancedb").strip().lower()

EMBEDDING_MODEL_ALIAS = os.getenv("FOUNDRY_EMBEDDING_MODEL", "qwen3-embedding-0.6b")
CHAT_MODEL_ALIAS = os.getenv("FOUNDRY_CHAT_MODEL", "qwen2.5-0.5b")
PREFERRED_EXECUTION_PROVIDER = os.getenv("FOUNDRY_EXECUTION_PROVIDER", "CUDAExecutionProvider")
REQUIRE_GPU_MODELS = _env_bool("FOUNDRY_REQUIRE_GPU", True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:0.5b")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "local_rag_chunks")

TOP_K = _env_int("RAG_TOP_K", 3)
RERANKER_ENABLED = _env_bool("RERANKER_ENABLED", True)
RERANKER_CANDIDATE_K = _env_int("RERANKER_CANDIDATE_K", max(10, TOP_K))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto")
RERANKER_BATCH_SIZE = _env_int("RERANKER_BATCH_SIZE", 16)
RERANKER_STRICT = _env_bool("RERANKER_STRICT", False)

CHUNK_TARGET_WORDS = _env_int("RAG_CHUNK_TARGET_WORDS", 500)
RETRIEVAL_MIN_TOP_SCORE = _env_float("RETRIEVAL_MIN_TOP_SCORE", 0.35)
OCR_MIN_TEXT_CHARS = _env_int("OCR_MIN_TEXT_CHARS", 40)
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
