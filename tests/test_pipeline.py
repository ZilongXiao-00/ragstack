import pytest
from ragstack import RAGPipeline, RAGConfig


def test_config_creation():
    """测试配置创建"""
    config = RAGConfig()
    assert config.chunk.parent_chunk_size == 1000
    assert config.retrieval.top_k_parents == 5


def test_pipeline_initialization():
    """测试管道初始化（需要环境变量）"""
    # 这个测试需要设置环境变量才能通过
    import os
    if not os.getenv("RAGSTACK_LLM_API_KEY"):
        pytest.skip("需要设置LLM环境变量")

    rag = RAGPipeline()
    assert rag.collection is None


def test_pipeline_without_config_raises():
    """测试未配置LLM时抛出异常"""
    import os
    # 清除环境变量
    for key in ["RAGSTACK_LLM_API_KEY", "RAGSTACK_LLM_BASE_URL", "RAGSTACK_LLM_MODEL"]:
        os.environ.pop(key, None)

    with pytest.raises(ValueError):
        RAGPipeline()
