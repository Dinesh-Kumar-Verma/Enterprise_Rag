from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    # LLM
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Embeddings
    hf_api_key: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector Store
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "enterprise_rag"

    # Retrieval
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    relevance_threshold: float = 0.35
    hybrid_alpha: float = 0.5

    # Generation
    max_context_tokens: int = 8000
    temperature: float = 0.1

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = "enterprise-rag"
    langchain_tracing_v2: str = "true"

    # App
    app_name: str = "EnterpriseRAG"
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
