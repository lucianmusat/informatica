#!/usr/bin/env bash
# Run the app against backends you already have listening on localhost.
# For a full local stack (Ollama + Weaviate + app) use docker-compose.yml instead.
set -euo pipefail

export WEAVIATE_URL=${WEAVIATE_URL:-localhost}
export WEAVIATE_HTTP_PORT=${WEAVIATE_HTTP_PORT:-8080}
export WEAVIATE_GRPC_PORT=${WEAVIATE_GRPC_PORT:-50051}

export OLLAMA_URL=${OLLAMA_URL:-http://localhost:11434}
export OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-30m}
export OLLAMA_NUM_CTX=${OLLAMA_NUM_CTX:-4096}

export LLM_MODEL=${LLM_MODEL:-qwen3:4b-instruct}

streamlit run app.py
