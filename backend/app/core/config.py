"""
Application Configuration

This module centralizes all configurable settings for the
AI Research Copilot backend.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.core.constants import DATABASE_DIR, VECTOR_STORE_DIR
class Settings(BaseSettings): 

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