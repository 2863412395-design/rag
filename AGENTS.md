# 通信领域 RAG 知识问答系统（Prototype）

## 项目目标
构建一个基于 RAG（Retrieval-Augmented Generation）的通信领域知识问答原型系统：
- 支持上传/导入通信协议与设备手册（PDF/Word）构建知识库
- 通过语义检索召回 3-5 个最相关片段，再由 LLM 生成答案
- 必须给出引用来源（例如：TS 38.300 第5章）
- 对知识库不存在的问题，必须明确回复“未在当前文档中找到相关信息”
- 交互界面为 Web（Streamlit），目标响应时间 ≤ 5 秒

（以上要求来自开题报告的研究内容与研究目标）

## 技术栈约束
- Python 作为主要语言
- RAG 框架：LangChain
- 向量库：ChromaDB 或 FAISS（本地向量数据库）
- 前端：Streamlit
- LLM：OpenAI 或 DeepSeek API（二选一/可配置）

## 核心模块
1) 数据清洗与预处理（面向通信文档）
- 处理对象：3GPP TS 系列文档、设备手册（PDF/Word）
- 去噪：剔除页眉页脚、目录等干扰
- 重点：设计分块（Chunking）策略，尽量避免语义断裂
- 特别关注缩略语完整性（如 SCS/CORESET/AMF 等）

2) 向量化与检索
- 将清洗后的 chunk 用 Embedding 模型转为向量并入库
- 相似度：可用余弦相似度
- 输入自然语言问题时，快速准确召回 Top 3-5 片段
- 需要验证不同 chunk size（如 512/1024 tokens）与不同 embedding 模型的效果，以召回率为指标择优

3) 检索增强生成与交互
- 用 LangChain 搭建 RAG 流水线
- 设计 Prompt Template：强约束“仅根据召回文档作答”，并强制输出引用
- Web 界面：提问 → 输出答案 + 引用 + 原文溯源（可视化）

## 质量与验收标准
- 功能：上传/导入文档→构建索引→可问答→答案附引用
- 幻觉抑制：无相关证据必须拒答（固定文案）
- 性能：端到端响应时间目标 ≤ 5 秒（原型阶段可记录并优化）
- 对比评测：在通信专业测试集（约 20 个典型问题，如 5G 帧结构、RRC 状态机等）上，
  RAG 系统回答准确率需明显优于“直接询问通用大模型”，且引用需准确

## 建议目录结构
- app.py                         # Streamlit 入口
- rag/
  - ingest.py                    # 文档加载、清洗、分块、入库
  - chunking.py                  # 通信文档 chunking 策略（可独立调参/评测）
  - embeddings.py                # embedding 模型封装与切换
  - vectorstore.py               # Chroma/FAISS 适配
  - retrieval.py                 # 检索（top-k）与可选 rerank
  - prompt.py                    # 提示词模板（强制引用/拒答）
  - qa.py                        # RAG 问答链（retrieve -> generate）
- data/
  - raw/                         # 原始文档
  - chroma/ or faiss/            # 本地持久化索引
  - eval/                        # 测试集与评测结果
- tests/                         # 单元测试/回归测试
- requirements.txt

## 运行命令（约定）
- 安装：pip install -r requirements.txt
- 启动：streamlit run app.py
- （可选）离线构建索引：python -m rag.ingest --input data/raw --persist data/chroma
- 测试：pytest -q

## 开发习惯（给 Codex）
- 每次改动后输出：变更摘要 + 如何运行验证
- 关键函数写类型注解与 docstring
- 所有回答输出必须包含：答案正文 + 引用列表（至少 1 条，否则触发拒答）
