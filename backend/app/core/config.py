"""
Application Configuration

This module centralizes all configurable settings for the
AI Research Copilot backend.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Directories 
BASE_DIR = Path(__file__).resolve().parent[2]
DATA_DIR = BASE_DIR/"data"
DATABASE_DIR = DATA_DIR/"database"
VECTOR_STORE_DIR = DATA_DIR/"vector_store"
CACHE_DIR = DATA_DIR/"cache"
EXPORT_DIR = DATA_DIR/"export"


class Settings(BaseSettings):
    
    # Application
    app_name: str = "AI Research Copilot"
    app_version: str = "0.1.0"
    debug: bool = True

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    #Database
    database_path: Path = DATABASE_DIR/"papers.db"
    vector_store_path: Path = VECTOR_STORE_DIR

    #embeddings
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_size: int = 800
    chunk_overlap: int = 200

    #Retrieval Chunks
    top_k: int = 5

    #LLM 
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    max_tokens: int = 1024

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding= "utf-8",
        case_sensitive= False,
        extra= 'ignore',
    )

settings = Settings()