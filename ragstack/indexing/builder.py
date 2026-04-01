"""索引构建模块"""

import json
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions

from ..core.config import RAGConfig
from .chunking import split_into_parent_child_chunks


class IndexBuilder:
    """索引构建器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = chromadb.PersistentClient(path=config.persist_directory)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model
        )

    def build_index(self, content: str) -> chromadb.Collection:
        """
        构建父子分片索引

        Args:
            content: 文档内容

        Returns:
            ChromaDB Collection
        """
        # 获取或创建集合
        collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            embedding_function=self.embedding_func,
            metadata={"description": "RAGStack父子分片索引"}
        )

        # 执行分片
        chunks = split_into_parent_child_chunks(
            content,
            parent_chunk_size=self.config.chunk.parent_chunk_size,
            parent_chunk_overlap=self.config.chunk.parent_chunk_overlap,
            child_chunk_size=self.config.chunk.child_chunk_size,
            child_chunk_overlap=self.config.chunk.child_chunk_overlap,
            separators=self.config.chunk.separators
        )

        # 准备数据
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            parent_id = f"parent_{chunk['parent_id']}"
            child_contents = [child["child_content"] for child in chunk["children"]]

            ids.append(parent_id)
            documents.append(chunk["parent_content"])
            metadatas.append({
                "parent_id": chunk["parent_id"],
                "child_count": len(child_contents),
                "child_contents": json.dumps(child_contents, ensure_ascii=False),
                "parent_length": len(chunk["parent_content"])
            })

        # 批量插入
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        return collection

    def load_collection(self) -> chromadb.Collection:
        """加载已有集合"""
        return self.client.get_collection(
            name=self.config.collection_name,
            embedding_function=self.embedding_func
        )
