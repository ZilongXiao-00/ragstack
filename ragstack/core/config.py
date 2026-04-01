"""核心配置模块"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChunkConfig:
    """分片配置"""
    parent_chunk_size: int = 1000
    parent_chunk_overlap: int = 100
    child_chunk_size: int = 300
    child_chunk_overlap: int = 50
    separators: List[str] = field(default_factory=lambda: [
        "\n\n",           # 段落
        "\n",             # 行
        "。", "！", "？", # 句子
        "，", "；",       # 短语
        " ",              # 空格
        ""                # 字符
    ])


@dataclass
class RetrievalConfig:
    """检索配置"""
    top_k_parents: int = 5          # 粗召回父块数量
    top_m_parents: int = 2          # 精排后父块数量
    top_n_children: int = 3         # 精排后子块数量
    rrf_k: int = 60                 # RRF融合参数
    bm25_top_k_multiplier: int = 2  # BM25检索倍数


@dataclass
class GenerationConfig:
    """生成配置"""
    max_tokens: int = 500
    temperature: float = 0.1
    system_prompt: Optional[str] = None


@dataclass
class RAGConfig:
    """RAG全局配置"""
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # 向量库配置
    collection_name: str = "ragstack_collection"
    persist_directory: str = "./ragstack_db"
