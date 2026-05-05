"""
主評估腳本

用法：
    python eval/eval.py --results_file result_test.json

讀取系統輸出和 Ground Truth，計算：
  1. Entity Matching Score（主分數）
  2. 幻覺分數（獨立呈現）
  3. nDCG@k（檢索品質，有提供 reference 時才計算）
  4. Unanswerable Exact Match

輸出平均分數 + 每題評分明細 JSON。
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

# 確保從任何目錄都能 import 同層模組
sys.path.insert(0, str(Path(__file__).parent))

from dataset.covid_19_discourse.eval.entities import (
    entity_matching_score, _load_eval_types, normalize_text,
    score_type_a, score_type_b, score_type_c, score_entity,
)
from dataset.covid_19_discourse.eval.hallucination import hallucination_penalty
from dataset.covid_19_discourse.eval.ndcg import evaluate_retrieval


def load_ground_truth(qa_path: str) -> Dict[str, dict]:
    """載入 QA dataset，以 question_id 為 key"""
    with open(qa_path, encoding="utf-8") as f:
        data = json.load(f)
    gt = {}
    for doc in data:
        doc_id = doc["doc_id"]
        for qa in doc["qa_pairs"]:
            qa["doc_id"] = doc_id
            gt[qa["question_id"]] = qa
    return gt


def load_entity_data(entity_path: str) -> Dict[str, Dict[str, Any]]:
    """載入 dataset_entities.json，以 doc_id 為 key，值為 flat entity dict"""
    with open(entity_path, encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for item in data:
        flat = {}
        for section_name, section in item.items():
            if section_name in ("doc_id", "title"):
                continue
            if isinstance(section, dict):
                for k, v in section.items():
                    if v is not None:
                        flat[k] = v
        result[item["doc_id"]] = flat
    return result


def load_doc_titles(entity_path: str) -> Dict[str, str]:
    """從 dataset_entities.json 載入 doc_id → title 對照"""
    with open(entity_path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["doc_id"]: item["title"] for item in data if "title" in item}


def load_doc_texts(full_content_dir: str, doc_titles: Dict[str, str]) -> Dict[str, str]:
    """載入文件原文，以 doc_id 為 key"""
    content_dir = Path(full_content_dir)
    if not content_dir.is_dir():
        raise FileNotFoundError(f"文件原文目錄不存在: {full_content_dir}")

    doc_texts = {}
    missing = []
    for doc_id, title in doc_titles.items():
        fpath = content_dir / f"{title}.txt"
        if fpath.exists():
            doc_texts[doc_id] = fpath.read_text(encoding="utf-8")
        else:
            missing.append(f"{doc_id} ({title}.txt)")

    if missing:
        print(f"[WARNING] {len(missing)} 份文件原文找不到，前 5 筆: {missing[:5]}")

    return doc_texts


def load_case_links(links_path: str) -> Dict[str, set]:
    """載入 case_links.json"""
    with open(links_path, encoding="utf-8") as f:
        data = json.load(f)
    links = {}
    for group in data:
        doc_ids = set(group.get("doc_ids", []))
        for did in doc_ids:
            links[did] = doc_ids - {did}
    return links


def evaluate_one(
    pred: dict,
    gt_item: dict,
    doc_entities: Optional[Dict[str, Any]],
    eval_types: Dict[str, str],
    case_links: Optional[Dict[str, set]] = None,
    doc_text: Optional[str] = None,
) -> dict:
    """評估單一題目"""
    qid = pred["question_id"]
    result = {"question_id": qid, "type": gt_item.get("type", "unknown")}

    # --- Unanswerable: Exact Match ---
    if gt_item.get("type") == "unanswerable":
        gt_answer = gt_item.get("answer", "無法從文件判斷")
        pred_answer = pred.get("answer", "")
        em = 1.0 if pred_answer.strip() == gt_answer.strip() else 0.0
        result["unanswerable_em"] = em
        result["entity_matching_score"] = None
        result["hallucination_penalty"] = None

        # nDCG 還是要算
        retrieved = pred.get("reference")
        ndcg = evaluate_retrieval(retrieved, gt_item, k=5, case_links=case_links)
        result["ndcg@5"] = ndcg
        return result

    # --- Entity Matching Score ---
    gt_entities = gt_item.get("entities", {})
    pred_entities = pred.get("entities", {})

    s_match, entity_detail = entity_matching_score(
        pred_entities, gt_entities, eval_types
    )

    # --- Answer-level fallback ---
    # 用途：當 entity 欄位名對不上（如 LAW vs LAWS）時補救
    pred_answer = str(pred.get("answer", "")).strip()
    gt_answer = str(gt_item.get("answer", "")).strip()
    pred_answer_norm = normalize_text(pred_answer)
    gt_answer_norm = normalize_text(gt_answer)

    # Step 1: 先比對整段答案，normalize 後一致或包含就直接滿分
    answer_exact = 0.0
    if pred_answer_norm and gt_answer_norm:
        if pred_answer_norm == gt_answer_norm:
            answer_exact = 1.0
        elif gt_answer_norm in pred_answer_norm or pred_answer_norm in gt_answer_norm:
            answer_exact = 1.0

    # Step 2: 逐 entity 比對，依 eval type 決定方式
    answer_detail = {}
    if pred_answer and gt_entities:
        for field, gt_val in gt_entities.items():
            et = eval_types.get(field, "B")
            gt_val_norm = normalize_text(str(gt_val))
            # 所有類型都先試 normalize + 文字包含比對
            if gt_val_norm and gt_val_norm in pred_answer_norm:
                s = 1.0
            elif et in ("A", "B"):
                # 結構化值：沒中就是 0
                s = 0.0
            else:  # Type C
                # 自由文本：沒中才走 embedding
                s = score_type_c(str(gt_val), pred_answer)
            answer_detail[field] = s

    answer_entity_score = sum(answer_detail.values()) / len(answer_detail) if answer_detail else 0.0
    answer_score = max(answer_exact, answer_entity_score)

    # 最終分數：fallback 只在實體提取完全失效時才觸發
    if s_match is not None:
        if s_match == 0:
            final_score = answer_score
        else:
            final_score = s_match
    else:
        final_score = answer_score if answer_score > 0 else None

    result["entity_matching_score"] = final_score
    result["entity_score"] = s_match          # 純 entity 欄位比對分數
    result["answer_fallback_score"] = answer_score  # 純 answer fallback 分數
    result["entity_detail"] = entity_detail    # 每個欄位的 entity 比對明細
    result["answer_detail"] = answer_detail    # 每個欄位的 answer fallback 明細

    # --- 幻覺分數 ---
    p_h, hall_detail, h_rate = hallucination_penalty(
        pred_entities, gt_entities, doc_entities, doc_text=doc_text
    )
    result["hallucination_penalty"] = p_h
    result["hallucination_rate"] = h_rate
    result["hallucination_detail"] = hall_detail

    # --- nDCG@k ---
    retrieved = pred.get("reference")
    ndcg = evaluate_retrieval(retrieved, gt_item, k=5, case_links=case_links)
    result["ndcg@5"] = ndcg

    return result


def main():
    parser = argparse.ArgumentParser(description="評估 RAG 系統輸出")
    _base_dir = Path(__file__).parent.parent

    parser.add_argument("--results_file", required=True, help="系統輸出 JSON 檔案路徑")
    parser.add_argument("--gt_file", default=str(_base_dir / "qa_dataset_final_v4.json"),
                        help="Ground Truth QA JSON")
    parser.add_argument("--entity_file", default=str(_base_dir / "dataset_entities_v4.json"),
                        help="dataset_entities_v4.json 路徑")
    parser.add_argument("--schema_file", default=str(_base_dir / "entity_schema.json"),
                        help="entity_schema.json 路徑")
    parser.add_argument("--case_links_file", default=str(_base_dir / "case_links.json"),
                        help="case_links.json 路徑")
    parser.add_argument("--full_content_dir", default=str(_base_dir.parent / "full_content"),
                        help="文件原文目錄路徑（用於幻覺檢測）")
    parser.add_argument("--output", default=None, help="評分明細輸出 JSON 路徑")
    args = parser.parse_args()

    gt_path = args.gt_file
    entity_path = args.entity_file
    schema_path = args.schema_file
    links_path = args.case_links_file

    # 載入資料
    print(f"載入 Ground Truth: {gt_path}")
    gt = load_ground_truth(gt_path)

    print(f"載入 Entity 資料: {entity_path}")
    entity_data = load_entity_data(entity_path)

    print(f"載入 Schema: {schema_path}")
    eval_types = _load_eval_types(schema_path)

    case_links = None
    if Path(links_path).exists():
        print(f"載入 Case Links: {links_path}")
        case_links = load_case_links(links_path)

    # 載入文件原文（用於幻覺檢測）
    full_content_dir = args.full_content_dir
    doc_titles = load_doc_titles(entity_path)
    print(f"載入文件原文: {full_content_dir}")
    doc_texts = load_doc_texts(full_content_dir, doc_titles)
    print(f"[Info] 成功載入 {len(doc_texts)}/{len(doc_titles)} 份文件原文")

    print(f"載入系統輸出: {args.results_file}")
    with open(args.results_file, encoding="utf-8") as f:
        predictions = json.load(f)

    # 逐題評估
    results = []
    for pred in predictions:
        qid = pred["question_id"]
        if qid not in gt:
            print(f"[WARNING] {qid} 不在 Ground Truth 中，跳過")
            continue

        gt_item = gt[qid]
        doc_id = gt_item.get("doc_id")
        doc_ent = entity_data.get(doc_id, {})
        doc_text = doc_texts.get(doc_id)

        r = evaluate_one(pred, gt_item, doc_ent, eval_types, case_links, doc_text=doc_text)
        results.append(r)

    # 統計平均
    em_scores = [r["unanswerable_em"] for r in results if r.get("unanswerable_em") is not None]
    entity_scores = [r["entity_matching_score"] for r in results if r.get("entity_matching_score") is not None]
    ent_only_scores = [r["entity_score"] for r in results if r.get("entity_score") is not None]
    ans_fb_scores = [r["answer_fallback_score"] for r in results if r.get("answer_fallback_score") is not None]
    hall_scores = [r["hallucination_penalty"] for r in results if r.get("hallucination_penalty") is not None]
    hrate_scores = [r["hallucination_rate"] for r in results if r.get("hallucination_rate") is not None]
    ndcg_scores = [r["ndcg@5"] for r in results if r.get("ndcg@5") is not None]

    avg_em = sum(em_scores) / len(em_scores) if em_scores else None
    avg_entity = sum(entity_scores) / len(entity_scores) if entity_scores else None
    avg_ent_only = sum(ent_only_scores) / len(ent_only_scores) if ent_only_scores else None
    avg_ans_fb = sum(ans_fb_scores) / len(ans_fb_scores) if ans_fb_scores else None
    avg_hall = sum(hall_scores) / len(hall_scores) if hall_scores else None
    avg_hrate = sum(hrate_scores) / len(hrate_scores) if hrate_scores else None
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else None

    summary = {
        "total_questions": len(results),
        "unanswerable_count": len(em_scores),
        "answerable_count": len(entity_scores),
        "avg_unanswerable_em": avg_em,
        "avg_entity_matching_score": avg_entity,
        "avg_entity_score": avg_ent_only,
        "avg_answer_fallback_score": avg_ans_fb,
        "avg_hallucination_penalty": avg_hall,
        "avg_hallucination_rate": avg_hrate,
        "avg_ndcg@5": avg_ndcg,
    }

    print("\n========== 評估結果 ==========")
    print(f"總題數:              {summary['total_questions']}")
    print(f"Unanswerable:        {summary['unanswerable_count']} 題, 平均 EM = {avg_em:.4f}" if avg_em is not None else "Unanswerable:        N/A")
    print(f"Entity Matching:     {summary['answerable_count']} 題, 平均 = {avg_entity:.4f}" if avg_entity is not None else "Entity Matching:     N/A")
    print(f"  Entity Score:      平均 = {avg_ent_only:.4f}" if avg_ent_only is not None else "  Entity Score:      N/A")
    print(f"  Answer Fallback:   平均 = {avg_ans_fb:.4f}" if avg_ans_fb is not None else "  Answer Fallback:   N/A")
    print(f"幻覺懲罰:            平均 P_h = {avg_hall:.4f}" if avg_hall is not None else "幻覺懲罰:            N/A")
    print(f"幻覺率:              平均 = {avg_hrate:.4f}" if avg_hrate is not None else "幻覺率:              N/A")
    print(f"nDCG@5:              平均 = {avg_ndcg:.4f}" if avg_ndcg is not None else "nDCG@5:              N/A")
    print("================================")

    # 輸出明細
    output = {"summary": summary, "details": results}

    if args.output:
        out_path = args.output
    else:
        # 預設存到 w1/eval/ 下
        eval_dir = Path(__file__).resolve().parents[4] / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        result_name = Path(args.results_file).stem + "_eval.json"
        out_path = str(eval_dir / result_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n評分明細已存至: {out_path}")


if __name__ == "__main__":
    main()
