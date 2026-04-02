# RAGStack

![RAGStack Cover](./ragstack.png)

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
┌─────────────────────────────────────────────────────────┐
│                      用户查询                            │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  第一阶段：检索父块                        │
│  ┌──────────────┐    ┌──────────────┐                  │
│  │  语义检索      │    │  BM25关键词检索 │                 │
│  │ bge-small-zh  │    │   + jieba     │                 │
│  └──────┬───────┘    └──────┬───────┘                  │
│         │                    │                          │
│         └────────┬───────────┘                          │
│                  ▼                                       │
│           ┌───────────┐                                 │
│           │ RRF 融合   │   k = 60                        │
│           └─────┬─────┘                                 │
│                 ▼                                       │
│         召回 Top-K 父块                                  │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│                  第二阶段：检索子块                        │
│  从召回的每个父块中提取子块                                │
│  ┌──────────────┐    ┌──────────────┐                  │
│  │  语义重排    │    │  BM25关键词检索 │                 │
│  │ bge-reranker │    │   + jieba     │                 │
│  └──────┬───────┘    └──────┬───────┘                  │
│         │                    │                          │
│         └────────┬───────────┘                          │
│                  ▼                                       │
│           ┌───────────┐                                 │
│           │ RRF 融合   │                                 │
│           └─────┬─────┘                                 │
│                 ▼                                       │
│         召回 Top-K 子块                                  │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│                    第三阶段：重排                          │
│  ┌──────────────────────┐  ┌──────────────────────┐    │
│  │   父块语义重排         │  │   子块语义重排         │    │
│  │ bge-reranker-base   │  │ bge-reranker-base   │    │
│  │   保留 Top-M 父块     │  │   保留 Top-N 子块     │    │
│  └──────────────────────┘  └──────────────────────┘    │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│                  第四阶段：LLM 生成                       │
│         API 调用 (OpenAI/Anthropic/自定义)               │
│         Prompt 注入召回子块 → 生成最终回答                 │
└─────────────────────────────────────────────────────────┘
```

## 快速开始

### 安装

```bash
pip install ragstack
```

或从源码安装：

```bash
git clone https://github.com/ZilongXiao-00/ragstack.git
cd ragstack
pip install -e .
```

### 基础用法

```python
from ragstack import RAGPipeline

# 初始化 RAG 管道
rag = RAGPipeline(
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
from ragstack import RAGPipeline, RAGConfig, RetrievalConfig

# 自定义检索配置
config = RAGConfig(
    retrieval=RetrievalConfig(
        top_k_parents=5,      # 粗召回父块数量
        top_m_parents=2,      # 精排后父块数量
        top_n_children=3,     # 精排后子块数量
        rrf_k=60              # RRF融合参数
    )
)

rag = RAGPipeline(config=config, ...)
```

## 项目结构

```
ragstack/
├── ragstack/           # 核心包
│   ├── core/          # 核心流程 (pipeline, config)
│   ├── indexing/      # 索引构建 (chunking, builder)
│   ├── retrieval/     # 检索模块
│   │   ├── semantic.py    # 语义检索
│   │   ├── bm25.py        # BM25关键词检索
│   │   ├── fusion.py      # RRF融合
│   │   ├── reranker.py    # 语义重排
│   │   └── orchestrator.py # 检索协调
│   ├── generation/    # 生成模块
│   └── utils/         # 工具函数
├── tests/             # 测试用例
├── examples/          # 使用示例
└── docs/            # 文档
```

## 技术栈

| 模块 | 技术 | 说明 |
|-----|------|------|
| **Embedding** | BAAI/bge-small-zh-v1.5 | 中文语义向量编码 |
| **Reranker** | BAAI/bge-reranker-base | 交叉编码器重排 |
| **向量数据库** | ChromaDB | 向量存储与检索 |
| **关键词检索** | BM25Okapi + jieba | 中文分词与关键词匹配 |
| **文本分片** | RecursiveCharacterTextSplitter | 递归语义分片 |

## 配置说明

通过环境变量配置 LLM：

```bash
export RAGSTACK_LLM_API_KEY="your-api-key"
export RAGSTACK_LLM_BASE_URL="https://api.example.com"
export RAGSTACK_LLM_MODEL="your-model-name"
```

支持的 LLM 提供商：
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- DeepSeek
- MiniMax
- 任何兼容 OpenAI API 格式的服务

## 核心概念

### 父子分片策略

```
原始文档
  │
  ▼ 第一刀：大粒度分割（父块）
  ├─ 父块 0 [1000字符，语义完整]
  │    ├─ 子块 0 [300字符]
  │    └─ 子块 1 [300字符]
  ├─ 父块 1 [1000字符，语义完整]
  │    ├─ 子块 0 [300字符]
  │    └─ 子块 1 [300字符]
```

| 层级 | 粒度 | 用途 | 优势 |
|------|------|------|------|
| 父块 | 大（1000字符） | 检索 | 语义完整，召回率高 |
| 子块 | 小（300字符） | 生成 | 内容精准，噪声少 |

### 混合检索原理

**为什么需要混合检索？**

| 检索方式 | 优势 | 劣势 |
|---------|------|------|
| 语义检索 | 理解意图，泛化能力强 | 可能漏掉专有名词 |
| BM25关键词 | 精确匹配专有名词 | 缺乏语义理解 |
| **RRF融合** | **结合两者优势** | - |

**RRF (Reciprocal Rank Fusion) 公式：**

```
RRF_score = 1/(k + rank_semantic) + 1/(k + rank_bm25)
```

其中 k=60 为平滑参数，避免排名靠后的结果得分过低。

### 双层重排机制

1. **父块重排**：从召回的 Top-K 父块中筛选最相关的 Top-M
2. **子块重排**：从选中父块的子块中筛选最相关的 Top-N
3. **优势**：逐步缩小范围，提高最终精度

## 贡献指南

欢迎提交 Issue 和 PR！

```bash
# 开发环境设置
git clone https://github.com/ZilongXiao-00/ragstack.git
cd ragstack
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 相关项目

- [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding) - 北京智源人工智能研究院的Embedding模型
- [ChromaDB](https://github.com/chroma-core/chroma) - 开源向量数据库
- [LangChain](https://github.com/langchain-ai/langchain) - LLM应用开发框架
