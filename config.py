"""Runtime configuration, all overridable by environment variable."""

import os

WEAVIATE_CLASS_NAME = "DocumentConversationAlUsers"

# Chat model. Defaults to Qwen3 4B Instruct (the non-thinking 2507 build, Q4_K_M,
# ~2.5GB of weights) which fits a 4GB card alongside the embedder provided Ollama
# runs with OLLAMA_FLASH_ATTENTION=1 and OLLAMA_KV_CACHE_TYPE=q8_0.
# Fall back to "qwen2.5:3b-instruct" if the GPU turns out to be too tight.
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b-instruct")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "nomic-embed-text")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama.default.svc.cluster.local:11434")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.6"))

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "weaviate.default.svc.cluster.local")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_HTTP_SECURE = os.getenv("WEAVIATE_HTTP_SECURE", "false").lower() == "true"
WEAVIATE_GRPC_SECURE = os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true"

# Retrieval. RETRIEVAL_MAX_DISTANCE is a *cosine distance* (0 = identical,
# 2 = opposite), so lower is more similar. In "Auto" mode a chunk is only shown
# to the model when it scores at or below this. Calibrate with tools/calibrate_retrieval.py.
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
RETRIEVAL_MAX_DISTANCE = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0.50"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# How many prior turns to replay to the model each request.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
