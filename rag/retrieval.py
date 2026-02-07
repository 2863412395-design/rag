"""Retrieval utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval configuration."""

    top_k: int = 4


def retrieve_chunks(vectorstore, query: str, config: RetrievalConfig | None = None) -> List[Document]:
    """Retrieve top-k chunks for a query."""

    config = config or RetrievalConfig()
    return vectorstore.similarity_search(query, k=config.top_k)
