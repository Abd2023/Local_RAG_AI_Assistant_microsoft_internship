"""Configuration constants for the Local RAG AI Assistant."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DOCS_PATH = DATA_DIR / "sample_docs"
DATABASE_PATH = DATA_DIR / "rag.db"

EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
CHAT_MODEL_ALIAS = "qwen2.5-0.5b"
PREFERRED_EXECUTION_PROVIDER = "CUDAExecutionProvider"
REQUIRE_GPU_MODELS = True
TOP_K = 3
CHUNK_TARGET_WORDS = 500
