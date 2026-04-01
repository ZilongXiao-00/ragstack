"""RRF融合模块"""

from typing import List, Dict


def rrf_fusion(
    semantic_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60
) -> List[Dict]:
    """
    RRF（倒数排名融合）算法

    将语义检索和BM25检索的结果按排名融合

    Args:
        semantic_results: 语义检索结果
        bm25_results: BM25检索结果
        k: RRF平滑参数，越大各检索方式权重越均衡

    Returns:
        融合后的结果列表，按RRF得分降序排列
    """
    rrf_scores = {}
    result_map = {}

    # 处理语义检索结果
    for item in semantic_results:
        parent_id = item["parent_id"]
        rank = item.get("rank", 0)
        score = 1.0 / (k + rank + 1)

        if parent_id in rrf_scores:
            rrf_scores[parent_id] += score
        else:
            rrf_scores[parent_id] = score
            result_map[parent_id] = item

    # 处理BM25结果
    for item in bm25_results:
        parent_id = item["parent_id"]
        rank = item.get("rank", 0)
        score = 1.0 / (k + rank + 1)

        if parent_id in rrf_scores:
            rrf_scores[parent_id] += score
        else:
            rrf_scores[parent_id] = score
            result_map[parent_id] = item

    # 按RRF得分排序
    sorted_results = sorted(
        [(pid, score) for pid, score in rrf_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )

    # 组装最终结果
    final_results = []
    for parent_id, rrf_score in sorted_results:
        item = result_map[parent_id].copy()
        item["rrf_score"] = rrf_score
        final_results.append(item)

    return final_results
