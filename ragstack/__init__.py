"""
RAGStack - 企业级检索增强生成框架
"""

from .core.pipeline import RAGPipeline
from .core.config import RAGConfig, RetrievalConfig, GenerationConfig

__version__ = "0.1.0"
__all__ = ["RAGPipeline", "RAGConfig", "RetrievalConfig", "GenerationConfig"]
