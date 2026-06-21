"""Embedding model configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


@dataclass(frozen=True)
class EmbeddingsConfig:
    """Embedding model settings."""

    model: str = "text-embedding-3-small"
    provider: str = "openai"  # openai or deepseek


class CompatibleEmbeddings(Embeddings):
    """OpenAI-compatible embeddings client that sends raw text inputs."""

    def __init__(self, model: str, api_key: str, base_url: str | None) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def build_embeddings(config: EmbeddingsConfig | None = None) -> Embeddings:
    """Create embeddings client for OpenAI or DeepSeek compatible APIs."""

    config = config or EmbeddingsConfig()
    provider = config.provider.lower()
    if provider not in {"openai", "deepseek"}:
        raise ValueError("provider must be 'openai' or 'deepseek'")

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("EMBEDDING_MODEL", config.model)
    if provider == "deepseek":
        base_url = base_url or "https://api.deepseek.com"
    use_dashscope = bool(base_url and "dashscope.aliyuncs.com" in base_url)
    if use_dashscope and not os.getenv("EMBEDDING_MODEL"):
        model = "text-embedding-v2"
    if use_dashscope:
        return CompatibleEmbeddings(model=model, api_key=api_key, base_url=base_url)
    return OpenAIEmbeddings(model=model, api_key=api_key, base_url=base_url)
