"""Document ingestion pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader

from rag.chunking import ChunkingConfig, split_documents
from rag.vectorstore import VectorStoreConfig, build_vectorstore


def clean_text(text: str) -> str:
    """Basic cleanup for telecom documents."""

    lines = [line.strip() for line in text.splitlines()]
    filtered: List[str] = []
    for line in lines:
        if not line:
            continue
        if len(line) <= 3 and line.isdigit():
            continue
        filtered.append(line)
    return "\n".join(filtered)


def load_documents_from_path(path: Path) -> List[Document]:
    """Load PDF/Word/TXT documents from a file or directory."""

    files: Iterable[Path]
    if path.is_dir():
        files = [file for file in path.iterdir() if file.is_file()]
    else:
        files = [path]

    documents: List[Document] = []
    for file in files:
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix.lower() in {".docx", ".doc"}:
            loader = Docx2txtLoader(str(file))
        elif file.suffix.lower() in {".txt", ".md"}:
            loader = TextLoader(str(file), encoding="utf-8")
        else:
            continue

        docs = loader.load()
        for doc in docs:
            content = clean_text(doc.page_content or "")
            if not content.strip():
                continue
            doc.page_content = content
            doc.metadata.setdefault("source", str(file))
            documents.append(doc)

    return documents


def ingest(
    input_path: str,
    persist_dir: str,
    chunk_config: ChunkingConfig | None = None,
    store_config: VectorStoreConfig | None = None,
) -> None:
    """Load, clean, chunk, and persist documents."""

    documents = load_documents_from_path(Path(input_path))
    chunks = split_documents(documents, chunk_config)
    store_config = store_config or VectorStoreConfig(persist_dir=persist_dir)
    build_vectorstore(chunks, store_config)


def main() -> None:
    """CLI entry for ingestion."""

    parser = argparse.ArgumentParser(description="Ingest telecom documents into a vector store")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--persist", required=True, help="Persist directory")
    parser.add_argument("--backend", default="chroma", choices=["chroma", "faiss"])
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=128)
    args = parser.parse_args()

    ingest(
        input_path=args.input,
        persist_dir=args.persist,
        chunk_config=ChunkingConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        ),
        store_config=VectorStoreConfig(backend=args.backend, persist_dir=args.persist),
    )


if __name__ == "__main__":
    main()
