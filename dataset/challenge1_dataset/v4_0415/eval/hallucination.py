"""
幻覺分數模組

檢查系統輸出中是否有 Ground Truth 沒標注的 extra entity。
extra entity 分三種：
  - extra-supported: 多講但有文件支持 → 不罰 (0)
  - extra-unsupported: 多講且無文件支持 → 輕度懲罰 (α=0.05)
  - extra-contradictory: 多講且與文件衝突 → 重度懲罰 (β=0.15)

P_h(q) = min(γ, Σ p(e))

幻覺分數獨立呈現，不併入 Entity Matching Score。
"""

from typing import Any, Dict, Optional, Set, Tuple

# 懲罰權重
ALPHA = 0.05   # unsupported
BETA = 0.15    # contradictory
GAMMA = 0.50   # 上限


def _to_str_set(val: Any) -> Set[str]:
    """將值正規化為字串集合，統一處理 str / list 型態。"""
    if isinstance(val, list):
        return {str(v).strip() for v in val if str(v).strip()}
    s = str(val).strip()
    return {s} if s else set()


def classify_extra_entity(
    field: str,
    pred_val: Any,
    doc_entities: Optional[Dict[str, Any]] = None,
    doc_text: Optional[str] = None,
) -> str:
    """
    判斷一個 extra entity（GT 沒標但系統多講的）的幻覺類型。

    Args:
        field:        entity 欄位名
        pred_val:     系統輸出的值
        doc_entities: 該文件的完整 entity 資料（用來判斷文件是否支持）
        doc_text:     該文件的原文（entity 資料查不到時的 fallback）

    Returns:
        "supported" / "unsupported" / "contradictory"
    """
    if doc_entities is None and doc_text is None:
        return "unsupported"

    # 如果文件的 entity 資料裡有這個欄位
    if doc_entities and field in doc_entities:
        doc_val = doc_entities[field]

        # 正規化為 set，統一處理 str / list 型態
        pred_set = _to_str_set(pred_val)
        doc_set = _to_str_set(doc_val)

        if not pred_set or not doc_set:
            return "unsupported"

        # 預測值與文件值有交集 → supported
        if pred_set & doc_set:
            return "supported"

        # 也檢查子字串關係（處理值不完全相同但語意包含的情況）
        for p in pred_set:
            for d in doc_set:
                if p in d or d in p:
                    return "supported"

        # 完全無交集 → contradictory
        return "contradictory"

    # entity 資料沒有這個欄位 → 查文件原文
    if doc_text:
        pred_set = _to_str_set(pred_val)
        for p in pred_set:
            if p in doc_text:
                return "supported"

    return "unsupported"


def hallucination_penalty(
    pred_entities: Dict[str, Any],
    gt_entities: Dict[str, Any],
    doc_entities: Optional[Dict[str, Any]] = None,
    doc_text: Optional[str] = None,
) -> Tuple[float, Dict[str, dict]]:
    """
    計算一題的幻覺懲罰分數。

    Args:
        pred_entities: 系統輸出的 entities
        gt_entities:   Ground Truth 的 entities
        doc_entities:  該文件的完整 entity 資料（可選，用來判斷 supported/contradictory）
        doc_text:      該文件的原文（可選，entity 資料查不到時的 fallback）

    Returns:
        (P_h, detail)
        - P_h: 幻覺懲罰分數
        - detail: 每個 extra entity 的分類 {field: {"value": ..., "type": ..., "penalty": ...}}
    """
    gt_fields = set(gt_entities.keys()) if gt_entities else set()
    pred_fields = set(pred_entities.keys()) if pred_entities else set()

    # extra = 系統有但 GT 沒有的
    extra_fields = pred_fields - gt_fields

    detail = {}
    total_penalty = 0.0

    for field in extra_fields:
        pred_val = pred_entities[field]
        classification = classify_extra_entity(field, pred_val, doc_entities, doc_text=doc_text)

        if classification == "supported":
            p = 0.0
        elif classification == "contradictory":
            p = BETA
        else:
            p = ALPHA

        detail[field] = {
            "value": pred_val,
            "type": classification,
            "penalty": p,
        }
        total_penalty += p

    p_h = min(GAMMA, total_penalty)

    # 幻覺率：有問題的 extra entity 佔 pred 總 entity 數的比例
    n_pred = len(pred_fields)
    n_hallucinated = sum(1 for d in detail.values() if d["type"] in ("unsupported", "contradictory"))
    h_rate = n_hallucinated / n_pred if n_pred > 0 else 0.0

    return p_h, detail, h_rate
