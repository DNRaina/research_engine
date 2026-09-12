"""
config.py
---------
Centralized, typed application settings and environment configuration.
Serves as the single source of truth for paths, model parameters,
networking limits, and credentials.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── Application Info ─────────────────────────────────────────────────────
    app_name: str = "JANE Research Agent"
    app_version: str = "2.0.0"
    user_agent: str = "JANEResearchAgent/2.0 (contact: jane-research@agent.local)"

    # ─── LLM & Embeddings ─────────────────────────────────────────────────────
    google_api_key: Optional[str] = Field(
        default=None,
        validation_alias="GOOGLE_API_KEY",
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
    )
    default_model: str = "gemini-2.5-flash-lite"
    available_models: List[str] = [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    embedding_model: str = "models/text-embedding-004"
    embedding_batch_size: int = 50

    # ─── Storage Directories ──────────────────────────────────────────────────
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)
    books_dir: str = "books"
    scans_dir: str = "scans"
    past_chats_dir: str = "past_chats"
    past_sessions_dir: str = "past_sessions"
    kb_index_subdir: str = "faiss_index"
    memory_index_subdir: str = "memory_index"
    manifest_filename: str = "manifest.json"
    memory_manifest_filename: str = "memory_manifest.json"

    # ─── Document Chunking ────────────────────────────────────────────────────
    kb_chunk_size: int = 800
    kb_chunk_overlap: int = 150
    memory_chunk_size: int = 600
    memory_chunk_overlap: int = 100

    # ─── Tool Execution & Networking ──────────────────────────────────────────
    http_timeout_s: float = 12.0
    arxiv_timeout_s: float = 15.0
    wikipedia_timeout_s: float = 10.0
    pubmed_timeout_s: float = 12.0
    web_timeout_s: float = 12.0
    kb_search_timeout_s: float = 20.0
    memory_search_timeout_s: float = 15.0

    max_tool_retries: int = 3
    base_retry_delay_s: float = 1.5
    rate_limit_backoff_s: float = 10.0

    # ─── Helpers ──────────────────────────────────────────────────────────────
    @property
    def api_key(self) -> str:
        """Return the active Gemini/Google API key or empty string."""
        return self.google_api_key or self.gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key.strip())

    def ensure_directories(self) -> None:
        """Idempotently ensure all runtime directories exist."""
        for folder in [self.books_dir, self.scans_dir, self.past_chats_dir, self.past_sessions_dir]:
            os.makedirs(folder, exist_ok=True)


settings = Settings()
