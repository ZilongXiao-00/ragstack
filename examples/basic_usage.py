"""基础使用示例"""

import os
from ragstack import RAGPipeline

# 设置环境变量（或在代码中传入参数）
os.environ["RAGSTACK_LLM_API_KEY"] = "your-api-key"
os.environ["RAGSTACK_LLM_BASE_URL"] = "https://api.example.com"
os.environ["RAGSTACK_LLM_MODEL"] = "your-model-name"

# 初始化RAG管道
rag = RAGPipeline()

# 构建索引
rag.build_index("path/to/your/document.md")

# 查询
answer = rag.query("你的问题？")
print(answer)
