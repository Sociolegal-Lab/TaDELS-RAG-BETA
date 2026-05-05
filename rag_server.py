"""
RAG Web Server
- /chat          → 自由對話介面
- /results       → 評估結果瀏覽（支援多組 predictions 切換）
- /api/ask       → POST, RAG 單題查詢
- /api/results   → GET, 載入預測 + ground truth + Entity Matching 分數
- /api/doc/:id   → GET, 取得單篇文件全文
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi import Request
from pydantic import BaseModel

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "dataset" / "challenge1_dataset" / "v4_0415" / "eval"))

from rag_pipeline import (
    DocumentStore,
    _retrieve_and_build_candidates,
    _format_entity_context,
    _build_structured_result,
    query_rewrite,
    generate_hyde,
    rerank_with_llm,
    generate_answer,
    generate_structured_answer,
    extract_entities_from_answer,
    entity_query,
    resolve_extremum,
    resolve_aggregation,
    TOP_K_RERANK,
)
from entities import entity_matching_score, _load_eval_types, normalize_text, score_type_c
from hallucination import hallucination_penalty
from ndcg import ndcg_at_k

# ── Global state ──────────────────────────────────────────────

store: DocumentStore = None
eval_types = None
WEB_DIR = Path(__file__).parent / "web"
BASE_DIR = Path(__file__).parent
V4_DIR = BASE_DIR / "dataset" / "challenge1_dataset" / "v4_0415"
SPLIT_DIR = V4_DIR / "dataset_split_v4"
ENTITIES_PATH = V4_DIR / "dataset_entities_v4.json"
SCHEMA_PATH = V4_DIR / "entity_schema.json"


# ── Lifespan ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    global store, eval_types
    store = DocumentStore()
    store.load()
    store.build_index()
    eval_types = _load_eval_types(str(SCHEMA_PATH))
    yield

app = FastAPI(title="Legal RAG", lifespan=lifespan)


# ── Pages ─────────────────────────────────────────────────────

@app.get("/favicon.png")
def favicon():
    return FileResponse(WEB_DIR / "favicon.png", media_type="image/png")


@app.get("/entities", response_class=HTMLResponse)
def entities_page():
    return (WEB_DIR / "entities.html").read_text(encoding="utf-8")


@app.get("/api/entities")
def api_entities():
    return {"entities": [e for e in store.entities]}


@app.get("/qa", response_class=HTMLResponse)
def qa_page():
    return (WEB_DIR / "qa_viewer.html").read_text(encoding="utf-8")


@app.get("/api/qa")
def api_qa():
    qa_path = V4_DIR / "qa_dataset_final_v4.json"
    with open(qa_path, encoding="utf-8") as f:
        data = json.load(f)
    return {"cases": data}


@app.get("/", response_class=HTMLResponse)
@app.get("/intro", response_class=HTMLResponse)
def intro_page():
    return (WEB_DIR / "intro.html").read_text(encoding="utf-8")


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return (WEB_DIR / "chat.html").read_text(encoding="utf-8")


@app.get("/results", response_class=HTMLResponse)
def results_page():
    return (WEB_DIR / "results.html").read_text(encoding="utf-8")


@app.get("/api/intro")
def api_intro():
    md_path = WEB_DIR / "intro.md"
    if not md_path.exists():
        raise HTTPException(404, "intro.md not found")
    return {"content": md_path.read_text(encoding="utf-8")}


# ── API: Ask ──────────────────────────────────────────────────

class AskReq(BaseModel):
    question: str
    use_rewrite: bool = True
    use_hyde: bool = True
    use_rerank: bool = True
    use_dense: bool = True
    use_bm25: bool = True
    use_metadata: bool = True
    use_entities: bool = True
    filter_titles: list[str] | None = None


@app.post("/api/ask")
def api_ask(req: AskReq):
    """Streaming RAG: sends step-by-step progress via newline-delimited JSON."""
    def _generate():
        def step(name, detail=""):
            return json.dumps({"type": "step", "step": name, "detail": detail}, ensure_ascii=False) + "\n"

        try:
            # Step 1: Query rewrite (includes query_type + sub_queries)
            rw = {"rewritten": req.question, "query_type": "single", "sub_queries": []}
            if req.use_rewrite:
                yield step("Rewriting query...")
                rw = query_rewrite(req.question)

            rewritten = rw["rewritten"]
            query_type = rw["query_type"]
            sub_queries = rw.get("sub_queries", [])

            # Step 2: HyDE
            hyde_passage = None
            if req.use_hyde:
                yield step("Generating HyDE passage...")
                hyde_passage = generate_hyde(req.question)

            # Step 3: Retrieve
            yield step(f"Retrieving documents ({query_type})...")
            if query_type == "multi_hop" and sub_queries:
                all_candidates = {}
                for sq in sub_queries:
                    sq_candidates = _retrieve_and_build_candidates(
                        sq, store, rewritten=None, hyde_passage=None,
                        use_dense=req.use_dense, use_bm25=req.use_bm25,
                        use_metadata=req.use_metadata)
                    for c in sq_candidates:
                        if c["doc_id"] not in all_candidates:
                            all_candidates[c["doc_id"]] = c
                candidates = list(all_candidates.values())
            elif query_type in ("extremum", "aggregation"):
                yield step(f"Querying entity database ({query_type})...")
                eq_result = entity_query(req.question, query_type)
                if query_type == "extremum":
                    target_indices = resolve_extremum(req.question, store, eq_result)
                else:
                    target_indices = resolve_aggregation(req.question, store, eq_result)
                    target_indices = target_indices[:10]
                if target_indices:
                    candidates = []
                    for idx in target_indices:
                        candidates.append({
                            "doc_idx": idx,
                            "doc_id": store.doc_ids[idx],
                            "title": store.titles[idx],
                            "snippet": store.full_texts[idx][:500],
                            "metadata_text": store.metadata_text[idx],
                            "full_text": store.full_texts[idx],
                            "entity_context": _format_entity_context(store.entities[idx]),
                        })
                else:
                    candidates = _retrieve_and_build_candidates(
                        req.question, store, rewritten=rewritten, hyde_passage=hyde_passage,
                        use_dense=req.use_dense, use_bm25=req.use_bm25,
                        use_metadata=req.use_metadata)
            else:
                candidates = _retrieve_and_build_candidates(
                    req.question, store, rewritten=rewritten, hyde_passage=hyde_passage,
                    use_dense=req.use_dense, use_bm25=req.use_bm25,
                    use_metadata=req.use_metadata)

            # Step 4.5: Filter by selected titles
            if req.filter_titles is not None:
                allowed = set(req.filter_titles)
                before = len(candidates)
                candidates = [c for c in candidates if c["title"] in allowed]
                print(f"  [Filter] {len(candidates)}/{before} candidates kept ({len(allowed)} titles allowed)")
                if not candidates:
                    yield json.dumps({"type": "result", "answer": "篩選範圍內無相關文件。請調整資料來源篩選後重試。", "reference": [], "refs": [], "entities": {}, "debug": {"query_type": query_type, "rewritten": rewritten, "sub_queries": sub_queries, "hyde": hyde_passage[:200] if hyde_passage else None}}, ensure_ascii=False) + "\n"
                    return

            # Step 5: Rerank or structured answer
            if query_type in ("extremum", "aggregation") and target_indices:
                structured_result = _build_structured_result(query_type, eq_result, target_indices, store)
                yield step(f"Generating structured answer ({query_type})...")
                answer = generate_structured_answer(req.question, candidates, structured_result)
                reranked = candidates
                entities = {}
            else:
                if req.use_rerank and len(candidates) > TOP_K_RERANK:
                    yield step(f"Re-ranking top {TOP_K_RERANK}...")
                    reranked = rerank_with_llm(req.question, candidates, TOP_K_RERANK)
                else:
                    reranked = candidates[:TOP_K_RERANK]

                # Generate answer
                yield step("Generating answer...")
                answer = generate_answer(req.question, reranked)

                # Extract entities
                entities = {}
                if req.use_entities and answer != "無法從文件判斷":
                    yield step("Extracting entities...")
                    reranked_entities = [store.entities[d["doc_idx"]] for d in reranked]
                    entities = extract_entities_from_answer(req.question, answer, reranked_entities)

            # Final result
            refs = [{
                "doc_id": d["doc_id"],
                "title": d["title"],
                "snippet": d["snippet"],
                "metadata_text": d["metadata_text"],
            } for d in reranked]

            result = {
                "type": "result",
                "answer": answer,
                "reference": [d["doc_id"] for d in reranked],
                "refs": refs,
                "entities": entities,
                "debug": {
                    "query_type": query_type,
                    "rewritten": rewritten,
                    "sub_queries": sub_queries,
                    "hyde": hyde_passage[:200] if hyde_passage else None,
                },
            }
            yield json.dumps(result, ensure_ascii=False) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "detail": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


# ── Helper: Load entity data ─────────────────────────────────

_entity_data_cache = None

def _load_entity_data():
    global _entity_data_cache
    if _entity_data_cache is None:
        with open(ENTITIES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _entity_data_cache = {}
        for item in data:
            flat = {}
            for section_name, section in item.items():
                if section_name in ("doc_id", "title"):
                    continue
                if isinstance(section, dict):
                    for k, v in section.items():
                        if v is not None:
                            flat[k] = v
            _entity_data_cache[item["doc_id"]] = flat
    return _entity_data_cache


# ── API: Results ──────────────────────────────────────────────

@app.get("/api/results")
def api_results(name: str = "val"):
    """載入 predictions + 評估結果。優先讀取 eval 快取檔，沒有才即時計算。"""
    pred_path = BASE_DIR / "results" / f"predictions_{name}.json"
    eval_path = BASE_DIR / "eval" / f"predictions_{name}_eval.json"

    if not pred_path.exists():
        raise HTTPException(404, f"results/predictions_{name}.json not found.")

    with open(pred_path, encoding="utf-8") as f:
        preds = json.load(f)

    # 載入 GT
    qa_path = V4_DIR / "qa_dataset_final_v4.json"
    with open(qa_path, encoding="utf-8") as f:
        qa_data = json.load(f)

    gt_map = {}
    doc_metadata_map = {}  # doc_id → 原始來源 metadata（資料編號/資料發生時間/關鍵字...）
    for doc in qa_data:
        for qa in doc["qa_pairs"]:
            gt_map[qa["question_id"]] = {**qa, "doc_id": doc["doc_id"]}
        if doc.get("metadata"):
            doc_metadata_map[doc["doc_id"]] = doc["metadata"]

    # 讀取 pipeline 執行紀錄（info/{split}.json），供介面顯示 debug info
    info_map: dict = {}
    info_path = BASE_DIR / "results" / "info" / f"{name}.json"
    if info_path.exists():
        try:
            with open(info_path, encoding="utf-8") as f:
                info_map = json.load(f)
        except Exception:
            info_map = {}

    # 讀取預算好的 eval 分數（如果有）
    eval_map = {}
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as f:
            eval_data = json.load(f)
        for d in eval_data.get("details", []):
            eval_map[d["question_id"]] = d

    results = []
    scores_by_type = {"short": [], "long": [], "unanswerable": []}
    hall_scores = []
    ndcg_scores = []

    # doc_id → idx lookup for enriching refs
    doc_id_to_idx = {did: i for i, did in enumerate(store.doc_ids)} if store else {}

    for p in preds:
        qid = p["question_id"]
        gt = gt_map.get(qid)
        if not gt:
            continue

        qtype = gt["type"]
        gt_answer = gt["answer"]
        pred_answer = p.get("answer") or p.get("predicted_answer", "")
        cached = eval_map.get(qid)

        reference_ids = p.get("reference") or p.get("retrieved_doc_ids", [])
        refs_list = []
        for did in reference_ids:
            idx = doc_id_to_idx.get(did)
            if idx is None:
                continue
            refs_list.append({
                "doc_id": did,
                "title": store.titles[idx],
                "snippet": store.full_texts[idx][:500],
                "metadata_text": store.metadata_text[idx],
                "source_metadata": doc_metadata_map.get(did, {}),
            })

        item = {
            "question_id": qid,
            "question": gt["question"],
            "type": qtype,
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
            "gt_entities": gt.get("entities", {}),
            "pred_entities": p.get("entities", {}),
            "reference": reference_ids,
            "refs": refs_list,
            "debug_info": info_map.get(qid),
        }

        if cached:
            # 使用預算好的分數
            if qtype == "unanswerable":
                em = cached.get("unanswerable_em", 1.0 if pred_answer.strip() == gt_answer.strip() else 0.0)
                item["unanswerable_em"] = em
                scores_by_type["unanswerable"].append(em)
            else:
                item["entity_matching_score"] = cached.get("entity_matching_score")
                item["entity_detail"] = cached.get("entity_detail", {})
                if item["entity_matching_score"] is not None:
                    scores_by_type[qtype].append(item["entity_matching_score"])

                item["hallucination_penalty"] = cached.get("hallucination_penalty", 0)
                item["hallucination_detail"] = cached.get("hallucination_detail", {})
                hall_scores.append(item["hallucination_penalty"])

            ndcg = cached.get("ndcg@5") or cached.get("ndcg5")
            if ndcg is not None:
                item["ndcg5"] = round(ndcg, 4)
                ndcg_scores.append(ndcg)
        else:
            # 沒有快取，即時計算
            if qtype == "unanswerable":
                em = 1.0 if pred_answer.strip() == gt_answer.strip() else 0.0
                item["unanswerable_em"] = em
                scores_by_type["unanswerable"].append(em)
            else:
                gt_ent = gt.get("entities", {})
                pred_ent = p.get("entities", {})

                s_match, detail = entity_matching_score(pred_ent, gt_ent, eval_types)
                item["entity_detail"] = detail

                # Answer-level fallback (對齊 eval.py:137-154)
                answer_score = 0.0
                pa, ga = pred_answer.strip(), gt_answer.strip()
                if pa and ga:
                    pa_norm, ga_norm = normalize_text(pa), normalize_text(ga)
                    if pa_norm == ga_norm:
                        answer_score = 1.0
                    elif pa_norm and ga_norm and (pa_norm in ga_norm or ga_norm in pa_norm):
                        answer_score = 1.0
                    else:
                        answer_score = score_type_c(pa, ga)

                if s_match is not None:
                    s_match = max(s_match, answer_score)
                else:
                    s_match = answer_score if answer_score > 0 else None

                item["entity_matching_score"] = s_match
                item["answer_score"] = answer_score
                if s_match is not None:
                    scores_by_type[qtype].append(s_match)

                entity_data = _load_entity_data()
                doc_ent = entity_data.get(gt.get("doc_id", ""), {})
                p_h, hall_detail, h_rate = hallucination_penalty(pred_ent, gt_ent, doc_ent)
                item["hallucination_penalty"] = p_h
                item["hallucination_detail"] = hall_detail
                hall_scores.append(p_h)

            retrieved = p.get("reference") or p.get("retrieved_doc_ids")
            if retrieved:
                relevant = {gt.get("doc_id")} if gt.get("doc_id") else set()
                ndcg = ndcg_at_k(retrieved, relevant, k=5)
                item["ndcg5"] = round(ndcg, 4)
                ndcg_scores.append(ndcg)

        results.append(item)

    # Summary
    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    summary = {
        "name": name,
        "total": len(results),
        "short": {"count": len(scores_by_type["short"]), "avg": avg(scores_by_type["short"])},
        "long": {"count": len(scores_by_type["long"]), "avg": avg(scores_by_type["long"])},
        "unanswerable": {"count": len(scores_by_type["unanswerable"]), "avg": avg(scores_by_type["unanswerable"])},
        "entity_matching_avg": avg(scores_by_type["short"] + scores_by_type["long"]),
        "hallucination_avg": avg(hall_scores),
        "ndcg5_avg": avg(ndcg_scores),
    }

    return {"summary": summary, "results": results}


# ── API: Available Predictions ────────────────────────────────

@app.get("/api/predictions")
def api_predictions():
    """列出所有可用的 predictions 檔案（val 優先）"""
    # 偏好順序：val 在前，然後 test、train，其餘按字母
    priority = {"val": 0, "test": 1, "train": 2}
    def sort_key(f):
        name = f.stem.replace("predictions_", "")
        return (priority.get(name, 99), name)

    available = []
    for f in sorted((BASE_DIR / "results").glob("predictions_*.json"), key=sort_key):
        if "_eval" in f.name:
            continue
        name = f.stem.replace("predictions_", "")
        with open(f, encoding="utf-8") as fp:
            count = len(json.load(fp))
        eval_path = BASE_DIR / "eval" / f"predictions_{name}_eval.json"
        available.append({
            "name": name,
            "count": count,
            "has_eval": eval_path.exists(),
        })
    return {"predictions": available}


# ── API: Document ─────────────────────────────────────────────

@app.get("/api/doc/{doc_id}")
def api_doc(doc_id: str):
    if store is None:
        raise HTTPException(503, "Store not loaded")
    try:
        idx = store.doc_ids.index(doc_id)
    except ValueError:
        raise HTTPException(404, f"Document {doc_id} not found")

    return {
        "doc_id": doc_id,
        "title": store.titles[idx],
        "full_text": store.full_texts[idx],
        "metadata_text": store.metadata_text[idx],
        "entities": store.entities[idx],
    }


# ── Run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8866)
