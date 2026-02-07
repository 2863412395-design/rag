"""Chunking utilities for telecom documents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for document chunking."""

    chunk_size: int = 1024
    chunk_overlap: int = 128


def split_documents(
    documents: Iterable[Document], config: ChunkingConfig | None = None
) -> List[Document]:
    """Split documents into chunks with telecom-friendly separators.

    Args:
        documents: Source documents.
        config: Chunking configuration (size + overlap).

    Returns:
        List of chunked documents.
    """

    config = config or ChunkingConfig()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))
