"""文档分片模块"""

import json
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_into_parent_child_chunks(
    content: str,
    parent_chunk_size: int = 1000,
    parent_chunk_overlap: int = 100,
    child_chunk_size: int = 300,
    child_chunk_overlap: int = 50,
    separators: List[str] = None
) -> List[Dict]:
    """
    父子分片：将文档分割为父块和子块

    Args:
        content: 文档内容
        parent_chunk_size: 父块大小
        parent_chunk_overlap: 父块重叠
        child_chunk_size: 子块大小
        child_chunk_overlap: 子块重叠
        separators: 分隔符列表

    Returns:
        分片结果列表，每个元素包含父块和子块信息
    """
    if separators is None:
        separators = [
            "\n\n", "\n",
            "。", "！", "？",
            "，", "；",
            " ", ""
        ]

    # 父块分割器
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=separators,
        length_function=len,
    )
    parent_chunks = parent_splitter.split_text(content)

    # 子块分割器
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=separators,
        length_function=len,
    )

    result = []
    for parent_idx, parent_content in enumerate(parent_chunks):
        children_texts = child_splitter.split_text(parent_content)
        children = [
            {"child_id": child_idx, "child_content": child_text}
            for child_idx, child_text in enumerate(children_texts)
        ]

        result.append({
            "parent_id": parent_idx,
            "parent_content": parent_content,
            "children": children
        })

    return result
