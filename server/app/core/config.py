from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/ecommerce_rag"

    milvus_uri: str = "http://localhost:19530"
    milvus_token: str | None = None
    milvus_collection: str = "product_chunks"
    text_milvus_collection: str = "product_chunks"
    image_milvus_collection: str = "product_image_chunks"
    organizer_dataset_dir: str | None = "./ecommerce_agent_dataset"

    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str = "Doubao-Seed-2.0-lite"
    llm_timeout_seconds: float = 60
    llm_max_retries: int = 2
    llm_retry_backoff_seconds: float = 0.8
    llm_input_price_per_1k: float | None = None
    llm_output_price_per_1k: float | None = None
    llm_stream_include_usage: bool = True
    llm_thinking_type: str | None = "disabled"

    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_dim: int = 384
    embedding_timeout_seconds: float = 30

    dashscope_api_key: str | None = None
    image_embedding_backend: str = "dashscope"
    image_embedding_api_key: str | None = None
    image_embedding_model: str = "tongyi-embedding-vision-flash-2026-03-06"
    clip_model_name: str = "ViT-B-16"
    clip_device: str = "auto"
    image_embedding_dim: int = 768
    image_relevance_threshold: float = 0.20

    vlm_api_key: str | None = None
    vlm_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    vlm_model: str = "ep-20260514111645-lmgt2"
    vlm_timeout_seconds: float = 10
    vlm_max_retries: int = 0

    evidence_cache_ttl_seconds: int = 3600
    evidence_cache_recent_turns: int = 20
    evidence_cache_max_candidates_per_turn: int = 20

    rerank_api_key: str | None = None
    rerank_base_url: str | None = None
    rerank_model: str | None = None
    rerank_backend: str = "hybrid"
    rerank_timeout_seconds: float = 30

    enable_trajectory_log: bool = True
    trajectory_log_path: str = "server/logs/multi_need_trajectories.jsonl"

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None

    fallback_dataset_dir: str = Field(default="../data/raw/ecommerce_agent_dataset")


@lru_cache
def get_settings() -> Settings:
    return Settings()
