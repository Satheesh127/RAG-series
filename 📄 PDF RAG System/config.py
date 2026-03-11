"""
Configuration Settings for PDF RAG Q&A System
==============================================

All constants are defined here. Environment variables (from .env) take
precedence over the defaults below so the project is portable across
different machines without touching this file.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # load .env before reading os.getenv()

# ── Directories ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
LOG_DIR      = os.path.join(BASE_DIR, "logs")

# ── API Keys ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Models ────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GROQ_MODEL_NAME      = os.getenv("GROQ_MODEL",      "llama-3.1-8b-instant")

# ── Text processing ───────────────────────────────────────────────────────────
CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE",    "800"))   # characters
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP", "150"))   # characters
MAX_CHUNKS_PER_QUERY = int(os.getenv("MAX_CHUNKS",   "10"))     # chunks returned

# ── HTTP / retry ───────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 30
MAX_RETRIES     = 3

# ── Logging ─────────────────────────────────────────────────────────────────────────
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ── Groq token limits ───────────────────────────────────────────────────────────────
MAX_TOKENS_PER_REQUEST = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))
MAX_RESPONSE_TOKENS    = int(os.getenv("MAX_RESPONSE_TOKENS", "1000"))

# ── ChromaDB ────────────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "knowledge_base"
SIMILARITY_THRESHOLD   = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
