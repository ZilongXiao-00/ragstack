import pytest
from ragstack.indexing.chunking import split_into_parent_child_chunks


def test_basic_chunking():
    """测试基础分片功能"""
    content = "这是第一段。\n\n这是第二段，包含多个句子。这是第二个句子。\n\n这是第三段。"

    result = split_into_parent_child_chunks(
        content,
        parent_chunk_size=100,
        child_chunk_size=50
    )

    assert len(result) > 0
    assert "parent_id" in result[0]
    assert "parent_content" in result[0]
    assert "children" in result[0]


def test_empty_content():
    """测试空内容"""
    result = split_into_parent_child_chunks("")
    assert result == []


def test_single_paragraph():
    """测试单段落"""
    content = "这是一个段落。"
    result = split_into_parent_child_chunks(content)

    assert len(result) == 1
    assert result[0]["parent_id"] == 0
    assert len(result[0]["children"]) >= 1
