"""Embedding model configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings


@dataclass(frozen=True)
class EmbeddingsConfig:
    """Embedding model settings."""

    model: str = "text-embedding-3-small"
    provider: str = "openai"  # openai or deepseek


def build_embeddings(config: EmbeddingsConfig | None = None) -> OpenAIEmbeddings:
    """Create embeddings client for OpenAI or DeepSeek compatible APIs."""

    config = config or EmbeddingsConfig()
    provider = config.provider.lower()
    if provider not in {"openai", "deepseek"}:
        raise ValueError("provider must be 'openai' or 'deepseek'")

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if provider == "deepseek":
        base_url = base_url or "https://api.deepseek.com"
    return OpenAIEmbeddings(
        model=config.model,
        api_key=api_key,
        base_url=base_url,
    )
