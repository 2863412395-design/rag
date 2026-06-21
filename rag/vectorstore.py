"""Vector store adapters for Chroma and FAISS."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS

from rag.embeddings import EmbeddingsConfig, build_embeddings


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for vector store selection."""

    backend: str = "chroma"  # chroma or faiss
    persist_dir: str = "data/chroma"
    collection_name: str = "telecom-rag"


def build_vectorstore(
    documents: Iterable[Document],
    config: VectorStoreConfig | None = None,
    embeddings_config: EmbeddingsConfig | None = None,
):
    """Build and persist a vector store."""

    config = config or VectorStoreConfig()
    embeddings = build_embeddings(embeddings_config)
    persist_path = Path(config.persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    filtered_docs: list[Document] = []
    for doc in documents:
        if isinstance(doc.page_content, bytes):
            doc.page_content = doc.page_content.decode("utf-8", errors="ignore")
        if not isinstance(doc.page_content, str):
            continue
        if not doc.page_content.strip():
            continue
        filtered_docs.append(doc)
    if not filtered_docs:
        raise ValueError("No valid documents available for embedding.")
    texts = [doc.page_content for doc in filtered_docs]
    metadatas = [doc.metadata for doc in filtered_docs]

    if config.backend == "chroma":
        return Chroma.from_texts(
            texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=str(persist_path),
            collection_name=config.collection_name,
        )
    if config.backend == "faiss":
        store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
        store.save_local(str(persist_path))
        return store
    raise ValueError("backend must be 'chroma' or 'faiss'")


def load_vectorstore(
    config: VectorStoreConfig | None = None,
    embeddings_config: EmbeddingsConfig | None = None,
):
    """Load a persisted vector store."""

    config = config or VectorStoreConfig()
    embeddings = build_embeddings(embeddings_config)
    persist_path = Path(config.persist_dir)

    if config.backend == "chroma":
        return Chroma(
            embedding_function=embeddings,
            persist_directory=str(persist_path),
            collection_name=config.collection_name,
        )
    if config.backend == "faiss":
        return FAISS.load_local(
            str(persist_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    raise ValueError("backend must be 'chroma' or 'faiss'")
