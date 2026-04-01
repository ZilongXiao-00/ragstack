"""重排模块"""

from typing import List, Dict
from FlagEmbedding import FlagReranker


class Reranker:
    """语义重排器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", use_fp16: bool = False):
        self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank_parents(
        self,
        query: str,
        parent_results: List[Dict],
        top_m: int = 2
    ) -> List[Dict]:
        """
        对父块进行重排

        Args:
            query: 查询文本
            parent_results: 召回的父块列表
            top_m: 保留的父块数量

        Returns:
            重排后的父块列表
        """
        if not parent_results:
            return []

        pairs = [[query, parent.get("parent_content", "")] for parent in parent_results]
        scores = self.model.compute_score(pairs)

        parent_with_scores = list(zip(parent_results, scores))
        parent_with_scores.sort(key=lambda x: x[1], reverse=True)

        top_parents = []
        for parent, score in parent_with_scores[:top_m]:
            parent_copy = parent.copy()
            parent_copy["rerank_score"] = score
            top_parents.append(parent_copy)

        return top_parents

    def rerank_children(
        self,
        query: str,
        child_contents: List[str],
        top_n: int = 3
    ) -> List[str]:
        """
        对子块进行重排

        Args:
            query: 查询文本
            child_contents: 子块内容列表
            top_n: 保留的子块数量

        Returns:
            重排后的子块内容列表
        """
        if not child_contents:
            return []

        # 过滤空子块
        child_contents = [c for c in child_contents if c.strip()]

        pairs = [[query, child] for child in child_contents]
        scores = self.model.compute_score(pairs)

        child_with_scores = list(zip(child_contents, scores))
        child_with_scores.sort(key=lambda x: x[1], reverse=True)

        return [child for child, _ in child_with_scores[:top_n]]
