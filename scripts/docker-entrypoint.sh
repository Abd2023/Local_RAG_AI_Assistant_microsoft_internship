#!/bin/sh
set -eu

if [ "$(python -c 'from src.vector_store import count_vectors; print(count_vectors())')" = "0" ]; then
  python -m src.ingest --rebuild
else
  python -m src.ingest
fi
exec uvicorn src.api:app --host 0.0.0.0 --port 8000
