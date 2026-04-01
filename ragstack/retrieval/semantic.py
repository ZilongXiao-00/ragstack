"""检索模块 - 语义检索"""

from typing import List, Dict, Tuple
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    """语义检索器"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(model_name)

    def retrieve(
        self,
        collection: chromadb.Collection,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        语义检索父块

        Args:
            collection: ChromaDB集合
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        query_embedding = self.model.encode(query, normalize_embeddings=True)

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )

        retrieved = []
        for i in range(len(results["metadatas"][0])):
            metadata = results["metadatas"][0][i]
            retrieved.append({
                "parent_id": metadata["parent_id"],
                "parent_content": results["documents"][0][i],
                "similarity_distance": results["distances"][0][i],
                "child_contents": metadata["child_contents"],
                "child_count": metadata["child_count"],
                "rank": i
            })

        return retrieved

    def encode(self, texts: List[str]) -> np.ndarray:
        """编码文本"""
        return self.model.encode(texts, normalize_embeddings=True)
