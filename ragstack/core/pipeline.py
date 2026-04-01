"""核心流程管道"""

import os
from typing import Optional
import chromadb

from ..core.config import RAGConfig
from ..indexing.builder import IndexBuilder
from ..retrieval.orchestrator import RetrievalOrchestrator
from ..generation.llm import LLMGenerator


class RAGPipeline:
    """RAG主流程管道"""

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        config: Optional[RAGConfig] = None
    ):
        """
        初始化RAG管道

        Args:
            llm_api_key: LLM API密钥，默认从环境变量 RAGSTACK_LLM_API_KEY 读取
            llm_base_url: LLM API基础URL，默认从环境变量 RAGSTACK_LLM_BASE_URL 读取
            llm_model: LLM模型名称，默认从环境变量 RAGSTACK_LLM_MODEL 读取
            config: RAG配置，默认使用默认配置
        """
        self.config = config or RAGConfig()

        # 读取LLM配置
        self.llm_api_key = llm_api_key or os.getenv("RAGSTACK_LLM_API_KEY")
        self.llm_base_url = llm_base_url or os.getenv("RAGSTACK_LLM_BASE_URL")
        self.llm_model = llm_model or os.getenv("RAGSTACK_LLM_MODEL")

        if not all([self.llm_api_key, self.llm_base_url, self.llm_model]):
            raise ValueError(
                "请提供LLM配置参数或设置环境变量: "
                "RAGSTACK_LLM_API_KEY, RAGSTACK_LLM_BASE_URL, RAGSTACK_LLM_MODEL"
            )

        # 初始化组件
        self.index_builder = IndexBuilder(self.config)
        self.retrieval_orchestrator = RetrievalOrchestrator(self.config)
        self.llm_generator = LLMGenerator(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            model=self.llm_model,
            max_tokens=self.config.generation.max_tokens,
            temperature=self.config.generation.temperature
        )

        self.collection: Optional[chromadb.Collection] = None

    def build_index(self, file_path: str) -> None:
        """
        从文件构建索引

        Args:
            file_path: 文档文件路径
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.collection = self.index_builder.build_index(content)

    def load_index(self) -> None:
        """加载已有索引"""
        self.collection = self.index_builder.load_collection()

    def query(self, question: str) -> str:
        """
        执行查询

        Args:
            question: 用户问题

        Returns:
            生成的回答
        """
        if self.collection is None:
            raise ValueError("索引未构建或加载，请先调用build_index()或load_index()")

        # 1. 检索父块
        parents = self.retrieval_orchestrator.retrieve_parents(self.collection, question)

        # 2. 检索子块
        children = self.retrieval_orchestrator.retrieve_children(question, parents)

        # 3. 生成回答
        if not children:
            return "未找到与该问题相关的有效信息"

        answer = self.llm_generator.generate(question, children)

        return answer
