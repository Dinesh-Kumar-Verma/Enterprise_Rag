from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    # LLM
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # Embeddings
    hf_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    
    # Vector Store
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "enterprise_rag"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False