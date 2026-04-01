# RAGStack

企业级检索增强生成（RAG）框架，采用父子分片策略与混合检索技术，为中文文档提供高精度问答能力。

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 核心特性

- **父子分片策略**：父块保证语义完整，子块保证内容精准
- **混合检索**：语义检索 + BM25 关键词检索 + RRF 融合
- **双层重排**：父块重排 + 子块重排，提升召回精度
- **中文优化**：基于 BGE 中文模型，适配中文分词
- **模块化设计**：易于扩展和定制

## 架构概览

```
用户查询
    │
    ▼
┌─────────────────────────────────────┐
│  第一阶段：父块粗召回                  │
│  语义检索 + BM25 → RRF融合           │
└────────────┬────────────────────────┘
             ▼
┌─────────────────────────────────────┐
│  第二阶段：父块精排（Reranker）       │
└────────────┬────────────────────────┘
             ▼
┌─────────────────────────────────────┐
│  第三阶段：子块召回与精排              │
└────────────┬────────────────────────┘
             ▼
        LLM 生成答案
```

## 快速开始

### 安装

```bash
pip install ragstack
```

或从源码安装：

```bash
git clone https://github.com/yourusername/ragstack.git
cd ragstack
pip install -e .
```

### 基础用法

```python
from ragstack import RAGPipeline

# 初始化 RAG 管道
rag = RAGPipeline(
    embedding_model="BAAI/bge-small-zh-v1.5",
    reranker_model="BAAI/bge-reranker-base",
    llm_api_key="your-api-key",
    llm_base_url="https://api.example.com",
    llm_model="your-model-name"
)

# 构建索引
rag.build_index("path/to/your/document.md")

# 查询
answer = rag.query("你的问题？")
print(answer)
```

### 高级用法

```python
from ragstack import RAGPipeline, RetrievalConfig

# 自定义检索配置
config = RetrievalConfig(
    parent_chunk_size=1000,
    parent_chunk_overlap=100,
    child_chunk_size=300,
    child_chunk_overlap=50,
    top_k_parents=5,
    top_m_parents=2,
    top_n_children=3
)

rag = RAGPipeline(config=config, ...)
```

## 项目结构

```
ragstack/
├── ragstack/           # 核心包
│   ├── core/          # 核心流程
│   ├── indexing/      # 索引构建
│   ├── retrieval/     # 检索模块
│   ├── generation/    # 生成模块
│   └── utils/         # 工具函数
├── tests/             # 测试用例
├── examples/          # 使用示例
└── docs/            # 文档
```


## 技术栈

- **Embedding**: BAAI/bge-small-zh-v1.5
- **Reranker**: BAAI/bge-reranker-base
- **向量数据库**: ChromaDB
- **分词**: jieba
- **关键词检索**: BM25Okapi

## 配置说明

通过环境变量配置：

```bash
export RAGSTACK_LLM_API_KEY="your-api-key"
export RAGSTACK_LLM_BASE_URL="https://api.example.com"
export RAGSTACK_LLM_MODEL="your-model-name"
```

## 贡献指南

欢迎提交 Issue 和 PR！

## 许可证

MIT License
