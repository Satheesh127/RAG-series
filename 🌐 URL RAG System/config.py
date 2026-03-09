"""
Configuration Settings for RAG Knowledge Assistant
=================================================

This module contains all configurable constants and settings for the
RAG (Retrieval Augmented Generation) system.
"""

import os

# Directory Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Model Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# Text Processing Configuration
CHUNK_SIZE = 600  # Target chunk size in characters
CHUNK_OVERLAP = 100  # Overlap between chunks in characters
MAX_CHUNKS_PER_QUERY = 5  # Maximum chunks to retrieve for each query

# API Configuration
REQUEST_TIMEOUT = 30  # HTTP request timeout in seconds
MAX_RETRIES = 3  # Maximum retries for failed requests

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Groq API Configuration
MAX_TOKENS_PER_REQUEST = 4000  # Free tier limit for context
MAX_RESPONSE_TOKENS = 1000  # Maximum tokens for response

# ChromaDB Configuration
CHROMA_COLLECTION_NAME = "knowledge_base"
SIMILARITY_THRESHOLD = 0.6  # Minimum similarity score for retrieval

# Web Scraping Configuration
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"