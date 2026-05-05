"""
Entity Matching Score 評分模組

依 entity_schema.csv 的評估類型（A/B/C）計算每個 entity 的匹配分數 s_i，
再取平均得到 S_match(q)。
"""

import re
import csv
import requests
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# ---------- 設定 ----------

EMBED_URL = "http://140.116.158.98:22225/api/embed"
EMBED_MODEL = "embeddinggemma:latest"

# ---------- 載入 schema ----------

_EVAL_TYPE_CACHE: Optional[Dict[str, str]] = None


def _load_eval_types(schema_path: Optional[str] = None) -> Dict[str, str]:
    """從 entity_schema.json 或 entity_schema.csv 讀取每個欄位的評估類型（A/B/C）"""
    global _EVAL_TYPE_CACHE
    if _EVAL_TYPE_CACHE is not None:
        return _EVAL_TYPE_CACHE

    if schema_path is None:
        schema_path = str(Path(__file__).parent.parent / "entity_schema.json")

    mapping = {}
    if schema_path.endswith(".json"):
        import json as _json
        with open(schema_path, encoding="utf-8") as f:
            schema = _json.load(f)
        for section in schema.values():
            for field_name, field_info in section.get("fields", {}).items():
                eval_type = (field_info.get("eval") or "").strip()
                if eval_type in ("A", "B", "C"):
                    mapping[field_name] = eval_type
    else:
        with open(schema_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = (row.get("英文標籤") or "").strip()
                eval_type = (row.get("評估類型") or "").strip()
                if label and eval_type in ("A", "B", "C"):
                    mapping[label] = eval_type
    _EVAL_TYPE_CACHE = mapping
    return mapping


# ---------- 標準化工具 ----------

# 國字數字對照
_CN_DIGIT = {
    "零": 0, "〇": 0,
    "壹": 1, "一": 1,
    "貳": 2, "二": 2,
    "參": 3, "三": 3,
    "肆": 4, "四": 4,
    "伍": 5, "五": 5,
    "陸": 6, "六": 6,
    "柒": 7, "七": 7,
    "捌": 8, "八": 8,
    "玖": 9, "九": 9,
}
_CN_UNIT = {
    "拾": 10, "十": 10,
    "佰": 100, "百": 100,
    "仟": 1000, "千": 1000,
    "萬": 10000, "万": 10000,
    "億": 100000000, "亿": 100000000,
}


def _cn_number_to_int(s: str) -> Optional[int]:
    """將國字數字字串轉為整數，如 '參萬伍仟元' -> 35000，也支援混合形式如 '5千' -> 5000

    支援高位單位（萬、億）：參拾萬 → 300000，壹億貳仟萬 → 120000000
    """
    total = 0
    section = 0  # 萬/億 以下的累積值
    current = 0
    has_digit = False
    for ch in s:
        if ch.isdigit():
            current = current * 10 + int(ch)
            has_digit = True
        elif ch in _CN_DIGIT:
            current = _CN_DIGIT[ch]
            has_digit = True
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if unit >= 10000:
                # 高位單位（萬、億）：把 section + current 乘上去
                section += current
                if section == 0:
                    section = 1
                total += section * unit
                section = 0
                current = 0
            else:
                # 低位單位（拾、佰、仟）
                if current == 0 and has_digit:
                    pass  # 零 + 單位
                elif current == 0:
                    current = 1
                section += current * unit
                current = 0
            has_digit = True
    section += current
    total += section
    return total if has_digit else None


_PLATFORM_ALIASES = {
    "臉書": "Facebook", "facebook": "Facebook", "FACEBOOK": "Facebook",
    "FB": "Facebook", "fb": "Facebook",
    "IG": "Instagram", "ig": "Instagram", "instagram": "Instagram",
    "line": "LINE", "Line": "LINE",
    "ptt": "PTT", "Ptt": "PTT", "批踢踢": "PTT",
    "推特": "Twitter", "twitter": "Twitter",
    "微信": "WeChat", "wechat": "WeChat",
    "噗浪": "Plurk", "plurk": "Plurk",
}


def normalize_text(s: str) -> str:
    """
    B 型標準化：
    1. 平台名稱統一
    2. 全形數字轉半形
    3. 國字數字轉阿拉伯（含金額、法條）
    4. 日期只保留民國年月日
    5. 去除多餘空白
    """
    if not s:
        return ""

    # 平台名稱統一
    if s.strip() in _PLATFORM_ALIASES:
        return _PLATFORM_ALIASES[s.strip()]

    # 全形數字轉半形
    result = ""
    for ch in s:
        if "\uff10" <= ch <= "\uff19":
            result += chr(ord(ch) - 0xFEE0)
        else:
            result += ch
    s = result

    # 去除多餘空白
    s = re.sub(r"\s+", "", s)

    # 去除尾部中文標點
    s = re.sub(r"[。，、；！？]+$", "", s)

    # 去除引號 「」『』
    s = re.sub(r"[「」『』]", "", s)

    # 統一用字：新台幣 → 新臺幣
    s = s.replace("新台幣", "新臺幣")

    # 去除數字中的逗號：4,000 → 4000
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = re.sub(r"(\d),(\d)", r"\1\2", s)  # 再跑一次處理 1,000,000

    # 法條國字數字轉阿拉伯：第六十三條 → 第63條
    def _replace_legal_cn(m):
        cn_str = m.group(1)
        num = _cn_number_to_int(cn_str)
        if num is not None:
            return f"第{num}"
        return m.group(0)

    s = re.sub(r"第([零壹貳參肆伍陸柒捌玖拾佰一二三四五六七八九十百]+)", _replace_legal_cn, s)

    # 金額國字轉阿拉伯：新臺幣參萬伍仟元 → 新臺幣35000元
    def _replace_money_cn(m):
        prefix = m.group(1)
        cn_str = m.group(2)
        suffix = m.group(3)
        num = _cn_number_to_int(cn_str)
        if num is not None:
            return f"{prefix}{num}{suffix}"
        return m.group(0)

    s = re.sub(
        r"(新臺幣|新台幣|臺幣|NT\$?)([零壹貳參肆伍陸柒捌玖拾佰仟萬億一二三四五六七八九十百千万亿0-9]+)(元)",
        _replace_money_cn,
        s,
    )

    # 刑期國字轉阿拉伯：拘役伍拾日 → 拘役50日
    def _replace_penalty_cn(m):
        cn_str = m.group(1)
        suffix = m.group(2)
        num = _cn_number_to_int(cn_str)
        if num is not None:
            return f"{num}{suffix}"
        return m.group(0)

    s = re.sub(
        r"([零壹貳參肆伍陸柒捌玖拾佰仟萬億一二三四五六七八九十百千万亿0-9]+)(日|月|年)",
        _replace_penalty_cn,
        s,
    )

    # 民國 前綴去除：民國109年 → 109年
    s = re.sub(r"民國(\d+年)", r"\1", s)

    # 日期：只留民國年月日，去掉時分秒和「許」
    m_date = re.search(r"(\d+年\d+月\d+日)", s)
    if m_date and re.search(r"\d+年\d+月\d+日.*[時分秒許]", s):
        # 如果整串就是一個日期+時間，只保留年月日
        if re.fullmatch(r".*?(\d+年\d+月\d+日).*", s):
            s = re.sub(r"(\d+年\d+月\d+日).*", r"\1", s)

    return s


# ---------- Type A: Exact Match ----------

def _extract_label(s: str) -> str:
    """從 '肯定：理由...' 中取出 '肯定'"""
    s = s.strip()
    for sep in ("：", ":"):
        if sep in s:
            return s.split(sep, 1)[0].strip()
    return s


def score_type_a(pred_val: Any, gt_val: Any) -> float:
    """分類標籤型：取冒號前的標籤，標準化後完全一致 = 1，否則 = 0"""
    pred = _extract_label(normalize_text(str(pred_val)))
    gt = _extract_label(normalize_text(str(gt_val)))
    return 1.0 if pred == gt else 0.0


# ---------- Type B: 標準化 Exact Match / F1 ----------

def _normalize_set_element(s: str) -> str:
    """標準化 list 中的單一元素"""
    return normalize_text(s).strip()


def _fuzzy_match(a: str, b: str) -> bool:
    """檢查兩個標準化後的字串是否匹配（exact 或 substring containment）"""
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _maybe_split(val: Any) -> Any:
    """如果字串含有 ；或 、分隔符，自動拆成 list"""
    if isinstance(val, str) and re.search(r"[；、]", val):
        parts = [p.strip() for p in re.split(r"[；、]", val) if p.strip()]
        if len(parts) > 1:
            return parts
    return val


def score_type_b(pred_val: Any, gt_val: Any) -> float:
    """
    短值型：
    - 自動將含 ；、 分隔的字串拆成 list
    - 任一方是 list → 統一成 list，用 F1（含 substring matching）
    - 都是純量 → 標準化後 exact match，再嘗試 substring containment
    """
    # 自動拆分分隔符字串
    pred_val = _maybe_split(pred_val)
    gt_val = _maybe_split(gt_val)

    # 任一方是 list → 統一成 list，用 F1
    if isinstance(gt_val, list) or isinstance(pred_val, list):
        pred_list = pred_val if isinstance(pred_val, list) else [pred_val]
        gt_list = gt_val if isinstance(gt_val, list) else [gt_val]
        gt_set = {_normalize_set_element(str(v)) for v in gt_list}
        pred_set = {_normalize_set_element(str(v)) for v in pred_list}

        if not gt_set and not pred_set:
            return 1.0
        if not gt_set or not pred_set:
            return 0.0

        matched_gt = {g for g in gt_set if any(_fuzzy_match(g, p) for p in pred_set)}
        matched_pred = {p for p in pred_set if any(_fuzzy_match(g, p) for g in gt_set)}
        precision = len(matched_pred) / len(pred_set)
        recall = len(matched_gt) / len(gt_set)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # 都是純量 → exact match，再嘗試 substring containment
    pred = normalize_text(str(pred_val))
    gt = normalize_text(str(gt_val))
    if pred == gt:
        return 1.0
    if _fuzzy_match(pred, gt):
        return 1.0
    return 0.0


# ---------- Type C: Embedding Cosine Similarity ----------

_embed_cache: Dict[str, np.ndarray] = {}


def _get_embedding(text: str) -> np.ndarray:
    """呼叫 embedding API，有快取"""
    if text in _embed_cache:
        return _embed_cache[text]
    try:
        resp = requests.post(
            EMBED_URL,
            json={"model": EMBED_MODEL, "input": text},
            timeout=300,
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["embeddings"][0], dtype=np.float32)
    except Exception as e:
        print(f"[WARNING] Embedding API 失敗: {e}")
        vec = np.zeros(768, dtype=np.float32)  # fallback
    _embed_cache[text] = vec
    return vec


def score_type_c(pred_val: Any, gt_val: Any) -> float:
    """長文型：embedding cosine similarity"""
    pred_text = str(pred_val).strip()
    gt_text = str(gt_val).strip()

    if not pred_text or not gt_text:
        return 0.0

    pred_emb = _get_embedding(pred_text)
    gt_emb = _get_embedding(gt_text)

    norm_pred = np.linalg.norm(pred_emb)
    norm_gt = np.linalg.norm(gt_emb)
    if norm_pred == 0 or norm_gt == 0:
        return 0.0

    return float(np.dot(pred_emb, gt_emb) / (norm_pred * norm_gt))


# ---------- 主函式 ----------

def score_entity(
    field: str,
    pred_val: Any,
    gt_val: Any,
    eval_types: Optional[Dict[str, str]] = None,
) -> float:
    """計算單一 entity 的匹配分數 s_i"""
    if eval_types is None:
        eval_types = _load_eval_types()

    eval_type = eval_types.get(field, "B")  # 預設 B

    # OTHER 欄位：先用子字串包含比對，沒命中才走 embedding
    if field == "OTHER":
        pred_norm = normalize_text(str(pred_val))
        gt_norm = normalize_text(str(gt_val))
        if pred_norm and gt_norm and (pred_norm in gt_norm or gt_norm in pred_norm):
            return 1.0
        return score_type_c(pred_val, gt_val)

    if eval_type == "A":
        return score_type_a(pred_val, gt_val)
    elif eval_type == "B":
        return score_type_b(pred_val, gt_val)
    elif eval_type == "C":
        return score_type_c(pred_val, gt_val)
    else:
        return score_type_b(pred_val, gt_val)


def entity_matching_score(
    pred_entities: Dict[str, Any],
    gt_entities: Dict[str, Any],
    eval_types: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[float], Dict[str, float]]:
    """
    計算一題的 Entity Matching Score。

    Args:
        pred_entities: 系統輸出的 entities
        gt_entities:   Ground Truth 的 entities

    Returns:
        (S_match, detail_scores)
        - S_match: 平均分數，GT 為空時回傳 None
        - detail_scores: 每個 entity 的分數 {field: s_i}
    """
    if eval_types is None:
        eval_types = _load_eval_types()

    if not gt_entities:
        return None, {}

    detail = {}
    for field, gt_val in gt_entities.items():
        if field in pred_entities:
            s = score_entity(field, pred_entities[field], gt_val, eval_types)
        else:
            s = 0.0  # GT 有但系統沒輸出
        detail[field] = s

    n = len(gt_entities)
    s_match = sum(detail.values()) / n if n > 0 else 0.0
    return s_match, detail
