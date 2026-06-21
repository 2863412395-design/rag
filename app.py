"""Streamlit entry for telecom RAG prototype."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

PERSIST_DIR = "data/chroma"


def build_index(raw_dir: Path, backend: str, chunk_size: int, chunk_overlap: int) -> None:
    """Build a vector index from uploaded documents."""

    from rag.chunking import ChunkingConfig
    from rag.ingest import ingest
    from rag.vectorstore import VectorStoreConfig

    ingest(
        input_path=str(raw_dir),
        persist_dir=PERSIST_DIR,
        chunk_config=ChunkingConfig(
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
        ),
        store_config=VectorStoreConfig(backend=backend, persist_dir=PERSIST_DIR),
    )


def run_query(question: str, backend: str, top_k: int):
    """Run retrieval and LLM generation for a question."""

    from rag.qa import LLMConfig, answer_question
    from rag.retrieval import RetrievalConfig, retrieve_chunks
    from rag.vectorstore import VectorStoreConfig, load_vectorstore

    store = load_vectorstore(VectorStoreConfig(backend=backend, persist_dir=PERSIST_DIR))
    docs = retrieve_chunks(store, question, RetrievalConfig(top_k=top_k))
    return answer_question(question, docs, LLMConfig(), None), docs


st.set_page_config(page_title="通信领域 RAG 原型", layout="wide")

st.title("通信领域 RAG 知识问答系统（Prototype）")

with st.sidebar:
    st.header("索引构建")
    backend = st.selectbox("向量库", ["chroma", "faiss"], index=0)
    chunk_size = st.number_input("Chunk Size", min_value=256, max_value=2048, value=1024, step=128)
    chunk_overlap = st.number_input(
        "Chunk Overlap", min_value=0, max_value=512, value=128, step=32
    )
    top_k = st.slider("Top-K", min_value=3, max_value=6, value=4)

    uploaded_files = st.file_uploader("上传 PDF/Word/TXT", accept_multiple_files=True)
    if st.button("构建索引"):
        if not uploaded_files:
            st.warning("请先上传文档。")
        else:
            raw_dir = Path("data/raw")
            raw_dir.mkdir(parents=True, exist_ok=True)
            for uploaded in uploaded_files:
                file_path = raw_dir / uploaded.name
                file_path.write_bytes(uploaded.read())

            try:
                build_index(raw_dir, backend, int(chunk_size), int(chunk_overlap))
                st.success("索引已构建完成。")
            except ModuleNotFoundError as exc:
                st.error(f"依赖未安装：{exc}. 请先安装 requirements.txt。")

st.subheader("提问")
question = st.text_input("请输入问题")

if st.button("生成回答"):
    if not question:
        st.warning("请输入问题。")
    else:
        try:
            (answer, citations), docs = run_query(question, backend, top_k)
        except ModuleNotFoundError as exc:
            st.error(f"依赖未安装：{exc}. 请先安装 requirements.txt。")
            st.stop()

        st.markdown("### 回答")
        st.write(answer)

        st.markdown("### 引用")
        if citations:
            for citation in citations:
                st.write(citation)
        else:
            st.write("未返回引用。")

        st.markdown("### 原文溯源")
        for doc in docs:
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", None)
            page_label = f" 第{page}页" if page is not None else ""
            st.markdown(f"**{source}{page_label}**")
            st.caption(doc.page_content)
