"""
Configuration for the AI Resume Screener.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Hybrid scoring: retrieval similarity of resume chunks vs the job description,
# plus the LLM evaluation of those retrieved chunks.
RETRIEVAL_WEIGHT = 0.4
LLM_WEIGHT = 0.6

# Vector database
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
COLLECTION_NAME = "resume_chunks"

# Embeddings (downloaded on first run)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Gemini
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
TEMPERATURE = 0.2

# Resume chunking for retrieval
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K_CHUNKS = 5
