"""Prompt templates for RAG QA."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate


REFUSAL_TEXT = "未在当前文档中找到相关信息"


@dataclass(frozen=True)
class PromptConfig:
    """Configuration for the QA prompt."""

    refusal_text: str = REFUSAL_TEXT


def build_prompt(config: PromptConfig | None = None) -> ChatPromptTemplate:
    """Build a constrained prompt requiring citations."""

    config = config or PromptConfig()
    template = (
        "你是通信领域知识问答助手。仅根据给定的检索片段作答。\n"
        "如果检索片段中没有回答问题的证据，必须回答：{refusal_text}。\n"
        "回答格式：\n"
        "答案：<一句或多句回答>\n"
        "引用：<列出引用来源，格式如 [来源1], [来源2]>\n\n"
        "检索片段：\n{context}\n\n"
        "问题：{question}"
    )
    return ChatPromptTemplate.from_template(template).partial(refusal_text=config.refusal_text)
