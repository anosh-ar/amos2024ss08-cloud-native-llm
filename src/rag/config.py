"""
Central configuration for the DeepCNCF RAG pipeline.

All values can be overridden with environment variables or a `src/rag/.env` file
(see `.env.example`). The LLM is reached through any OpenAI-compatible endpoint,
so the same code works with Ollama, LocalAI (Docker) or a llama.cpp server.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

RAG_DIR = Path(__file__).resolve().parent
load_dotenv(RAG_DIR / ".env")

# --- Data -------------------------------------------------------------------
DATASET_ID = os.getenv("RAG_DATASET_ID", "Kubermatic/cncf-raw-data-for-llm-training")
BENCHMARK_DATASET_ID = os.getenv("RAG_BENCHMARK_DATASET_ID", "Kubermatic/Benchmark-Questions")

# --- Embeddings & vector store ---------------------------------------------
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge-*-v1.5 models retrieve better when the *query* (not the passages) gets this prefix.
EMBED_QUERY_PREFIX = os.getenv(
    "RAG_EMBED_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
)
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", RAG_DIR / "index"))
COLLECTION = os.getenv("RAG_COLLECTION", "cncf")
CHUNK_TOKENS = int(os.getenv("RAG_CHUNK_TOKENS", "400"))  # bge-small reads at most 512 tokens
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# --- LLM (OpenAI-compatible server) ------------------------------------------
# Ollama default: http://localhost:11434/v1   LocalAI default: http://localhost:8080/v1
LLM_BASE_URL = os.getenv("RAG_LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "deepcncf")
LLM_API_KEY = os.getenv("RAG_LLM_API_KEY", "not-needed")  # local servers ignore it
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("RAG_LLM_MAX_TOKENS", "512"))

OUTPUT_DIR = RAG_DIR / "output"
