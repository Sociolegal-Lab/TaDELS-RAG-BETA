"""
nDCG@k — 檢索排序品質評估

衡量模型在前 k 筆檢索結果中的排序品質。
使用 binary relevance：文件相關 = 1，不相關 = 0。

支援：
  - 單一 doc_id（single-hop）
  - 多個 ref_doc_id（multi-hop，或同一案件有多份相關文件）
"""

import math
from typing import Dict, List, Set, Optional


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int = 5) -> float:
    """
    計算 nDCG@k (binary relevance)。

    Args:
        retrieved: 模型回傳的 doc_id 排序列表（由高到低）
        relevant:  正確答案的 doc_id 集合
        k:         取前 k 筆計算

    Returns:
        nDCG@k 分數，介於 0.0 ~ 1.0
    """
    if not relevant:
        return 0.0

    retrieved_k = retrieved[:k]

    # DCG: sum of 1/log2(rank+1) for relevant docs
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank starts at 1

    # IDCG: best possible DCG with |relevant| hits at top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def build_relevant_set(
    ground_truth: dict,
    case_links: Optional[Dict[str, Set[str]]] = None,
) -> Set[str]:
    """
    從 ground truth 建立 relevant doc_id 集合。

    優先使用 ref_doc_id（多份相關文件），
    否則用 doc_id（單一文件）。
    若提供 case_links，自動擴展同案件的相關文件。

    Args:
        ground_truth: 包含 doc_id 或 ref_doc_id 的 ground truth dict
        case_links:   {doc_id: set(related_doc_ids)} 案件關聯表（可選）

    Returns:
        relevant doc_id 的集合
    """
    if "ref_doc_id" in ground_truth and ground_truth["ref_doc_id"]:
        relevant = set(ground_truth["ref_doc_id"])
    elif "doc_id" in ground_truth and ground_truth["doc_id"]:
        relevant = {ground_truth["doc_id"]}
    else:
        return set()

    # 擴展同案件相關文件
    if case_links:
        expanded = set()
        for doc_id in relevant:
            expanded.update(case_links.get(doc_id, set()))
        relevant |= expanded

    return relevant


def evaluate_retrieval(
    retrieved: Optional[List[str]],
    ground_truth: dict,
    k: int = 5,
    case_links: Optional[Dict[str, Set[str]]] = None,
) -> Optional[float]:
    """
    評估單筆檢索結果的 nDCG@k。

    Args:
        retrieved:    模型回傳的 doc_id 列表，None 表示未做檢索
        ground_truth: ground truth dict
        k:            取前 k 筆
        case_links:   案件關聯表，同案件文件視為 relevant

    Returns:
        nDCG@k 分數，或 None（未提供檢索結果時）
    """
    if retrieved is None:
        return None
    relevant = build_relevant_set(ground_truth, case_links)
    return ndcg_at_k(retrieved, relevant, k=k)
