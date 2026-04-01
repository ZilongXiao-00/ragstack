"""检索模块 - BM25关键词检索"""

from typing import List, Dict
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
import chromadb


def preprocess_text_for_bm25(text: str) -> List[str]:
    """
    中文文本分词

    Args:
        text: 输入文本

    Returns:
        分词后的token列表
    """
    text = str(text).strip()
    tokens = list(jieba.cut(text))
    return [token.strip() for token in tokens if token.strip()]


class BM25Retriever:
    """BM25关键词检索器"""

    def __init__(self):
        self.bm25 = None
        self.parent_docs = []
        self.parent_metadatas = []

    def build_index(self, collection: chromadb.Collection) -> None:
        """
        从Chroma集合构建BM25索引

        Args:
            collection: ChromaDB集合
        """
        all_data = collection.get(include=["documents", "metadatas"])
        self.parent_docs = all_data["documents"]
        self.parent_metadatas = all_data["metadatas"]

        tokenized_docs = [preprocess_text_for_bm25(doc) for doc in self.parent_docs]
        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        BM25检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        if self.bm25 is None:
            raise ValueError("BM25索引未构建，请先调用build_index()")

        tokenized_query = preprocess_text_for_bm25(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                results.append({
                    "parent_id": self.parent_metadatas[idx]["parent_id"],
                    "parent_content": self.parent_docs[idx],
                    "score": scores[idx],
                    "child_contents": self.parent_metadatas[idx]["child_contents"],
                    "child_count": self.parent_metadatas[idx]["child_count"],
                    "rank": rank
                })

        return results
