"""检索协调器"""

import json
from typing import List, Dict
import chromadb

from ..core.config import RAGConfig
from .semantic import SemanticRetriever
from .bm25 import BM25Retriever
from .fusion import rrf_fusion
from .reranker import Reranker


class RetrievalOrchestrator:
    """检索协调器：整合多种检索策略"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.semantic_retriever = SemanticRetriever(config.embedding_model)
        self.bm25_retriever = BM25Retriever()
        self.reranker = Reranker(config.reranker_model)

    def retrieve_parents(
        self,
        collection: chromadb.Collection,
        query: str
    ) -> List[Dict]:
        """
        父块检索流程：语义 + BM25 + RRF + 重排

        Args:
            collection: ChromaDB集合
            query: 查询文本

        Returns:
            重排后的父块列表
        """
        cfg = self.config.retrieval

        # 1. 语义检索
        semantic_results = self.semantic_retriever.retrieve(
            collection, query, top_k=cfg.top_k_parents * cfg.bm25_top_k_multiplier
        )

        # 2. BM25检索
        self.bm25_retriever.build_index(collection)
        bm25_results = self.bm25_retriever.retrieve(
            query, top_k=cfg.top_k_parents * cfg.bm25_top_k_multiplier
        )

        # 3. RRF融合
        fused_results = rrf_fusion(semantic_results, bm25_results, k=cfg.rrf_k)

        # 4. 父块重排
        top_parents = self.reranker.rerank_parents(
            query, fused_results[:cfg.top_k_parents], top_m=cfg.top_m_parents
        )

        return top_parents

    def retrieve_children(
        self,
        query: str,
        parents: List[Dict]
    ) -> List[str]:
        """
        子块检索流程：提取子块 + 重排

        Args:
            query: 查询文本
            parents: 父块列表

        Returns:
            重排后的子块内容列表
        """
        # 提取所有子块
        all_child_contents = []
        for parent in parents:
            try:
                child_contents = json.loads(parent["child_contents"])
                all_child_contents.extend(child_contents)
            except (json.JSONDecodeError, KeyError):
                continue

        # 子块重排
        top_children = self.reranker.rerank_children(
            query, all_child_contents, top_n=self.config.retrieval.top_n_children
        )

        return top_children
