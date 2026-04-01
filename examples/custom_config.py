"""自定义配置示例"""

import os
from ragstack import RAGPipeline, RAGConfig, RetrievalConfig, ChunkConfig

# 自定义配置
chunk_config = ChunkConfig(
    parent_chunk_size=1200,
    child_chunk_size=400
)

retrieval_config = RetrievalConfig(
    top_k_parents=10,
    top_m_parents=3,
    top_n_children=5
)

config = RAGConfig(
    chunk=chunk_config,
    retrieval=retrieval_config,
    embedding_model="BAAI/bge-large-zh-v1.5",  # 使用更大的模型
    persist_directory="./my_rag_db"
)

# 初始化
rag = RAGPipeline(
    llm_api_key="your-api-key",
    llm_base_url="https://api.example.com",
    llm_model="your-model-name",
    config=config
)

rag.build_index("document.md")
answer = rag.query("你的问题？")
print(answer)
