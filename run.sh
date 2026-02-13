#!/usr/local/env bash

# For local run
# kubectl -n default port-forward svc/ollama 11434:11434
# kubectl -n default port-forward svc/weaviate 8080:8080 50051:50051

export WEAVIATE_URL=localhost
export WEAVIATE_HTTP_PORT=8080
export WEAVIATE_GRPC_PORT=50051
export OLLAMA_URL=http://localhost:11434
export OLLAMA_KEEP_ALIVE=30m

streamlit run app.py
