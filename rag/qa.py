"""RAG QA pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from rag.prompt import PromptConfig, build_prompt


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the LLM client."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0


def format_context(docs: Iterable[Document]) -> str:
    """Format retrieved documents with source metadata."""

    blocks: List[str] = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", None)
        page_label = f" 第{page}页" if page is not None else ""
        blocks.append(f"[来源{index}] {source}{page_label}\n{doc.page_content}")
    return "\n\n".join(blocks)


def answer_question(
    question: str,
    docs: Iterable[Document],
    llm_config: LLMConfig | None = None,
    prompt_config: PromptConfig | None = None,
) -> Tuple[str, List[str]]:
    """Generate an answer and citation labels from retrieved docs."""

    llm_config = llm_config or LLMConfig()
    prompt = build_prompt(prompt_config)
    context = format_context(docs)

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("LLM_MODEL", llm_config.model)
    if base_url and "dashscope.aliyuncs.com" in base_url and not os.getenv("LLM_MODEL"):
        model = "qwen-max"
    llm = ChatOpenAI(
        model=model,
        temperature=llm_config.temperature,
        api_key=api_key,
        base_url=base_url,
    )
    messages = prompt.format_messages(question=question, context=context)
    response = llm.invoke(messages)
    answer_text = response.content

    citations = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", None)
        page_label = f" 第{page}页" if page is not None else ""
        citations.append(f"[来源{index}] {source}{page_label}")
    return answer_text, citations
