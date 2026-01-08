"""
Configuration file for AI Resume Screener
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Scoring Weights
SEMANTIC_SEARCH_WEIGHT = 0.6
LLM_WEIGHT = 0.4

# Vector Database Configuration
VECTOR_DB_PATH = "./vector_db"
COLLECTION_NAME = "resumes"

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight and efficient

# LLM Configuration
LLM_MODEL = "gemini-2.0-flash"  # Available model (or use "gemini-pro-latest" for latest stable)
TEMPERATURE = 0.3  # Lower temperature for more consistent matching
