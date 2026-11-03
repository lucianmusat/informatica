"""Runtime configuration, all overridable by environment variable."""

import os

WEAVIATE_CLASS_NAME = os.getenv("WEAVIATE_CLASS_NAME", "DocumentConversationAlUsers")

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

# Optional "big" backend: an Ollama on a machine that is only sometimes available
# (the gaming desktop). When it answers and has LLM_MODEL_BIG pulled, chat goes
# there; otherwise it falls back to OLLAMA_URL + LLM_MODEL. Empty URL = disabled.
# Embeddings never move: the Weaviate vectors and RETRIEVAL_MAX_DISTANCE are tied
# to EMBEDDER_MODEL on OLLAMA_URL.
OLLAMA_URL_BIG = os.getenv("OLLAMA_URL_BIG", "").rstrip("/")
OLLAMA_BIG_NAME = os.getenv("OLLAMA_BIG_NAME", "Briareos")
OLLAMA_SMALL_NAME = os.getenv("OLLAMA_SMALL_NAME", "Kottos")
# Qwen 3.8 27B (default tag is the Q4_K_M build): ~17GB of weights, leaving room on a 24GB 4090
# for a 16k q8_0 KV cache. The tag must match `ollama list` exactly.
LLM_MODEL_BIG = os.getenv("LLM_MODEL_BIG", "qwen3.8:27b")
OLLAMA_NUM_CTX_BIG = int(os.getenv("OLLAMA_NUM_CTX_BIG", "16384"))
# Seconds to trust a probe result, and to skip the big backend after it fails.
OLLAMA_BIG_PROBE_TTL = int(os.getenv("OLLAMA_BIG_PROBE_TTL", "20"))
OLLAMA_BIG_COOLDOWN = int(os.getenv("OLLAMA_BIG_COOLDOWN", "60"))

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

# --------------------------------------------------------------------------
# Access control
# --------------------------------------------------------------------------
# Google sign-in is configured through .streamlit/secrets.toml ([auth]), which the
# container entrypoint writes from GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET /
# COOKIE_SECRET / APP_URL. See write_secrets.py.

# Always allowed in, and the only one who sees the access-requests page.
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "amibadesign@gmail.com").strip().lower()
APP_URL = os.getenv("APP_URL", "https://informatica.lucianmusat.nl").rstrip("/")

CASSANDRA_HOSTS = [h.strip() for h in os.getenv(
    "CASSANDRA_HOSTS", "cassandra.default.svc.cluster.local").split(",") if h.strip()]
# Not CASSANDRA_PORT: Kubernetes injects that as "tcp://<ip>:9042" for the
# `cassandra` Service.
CASSANDRA_PORT = int(os.getenv("CASSANDRA_CQL_PORT", "9042"))
CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "informatica")

# Access-request alerts. Disabled when SMTP_PASSWORD is empty.
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "amibadesign@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_EMAIL = os.getenv("ALERT_EMAIL", ADMIN_EMAIL)
