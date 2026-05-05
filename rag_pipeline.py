"""
TDELS RAG Pipeline

用法：
    # 跑 val set
    python rag_pipeline.py --split val

    # 跑 test set
    python rag_pipeline.py --split test

    # 測試前 5 題
    python rag_pipeline.py --split val --limit 5

    # 用 ollama
    python rag_pipeline.py --split val --llm ollama --llm-model gemma3:27b

    # 關掉 query rewrite 和 HyDE
    python rag_pipeline.py --split val --no-rewrite --no-hyde

    # 純 Dense（關掉 BM25 和 metadata）
    python rag_pipeline.py --split val --no-bm25 --no-metadata

    # 純 BM25（關掉 Dense 和 metadata）
    python rag_pipeline.py --split val --no-dense --no-metadata

    # 關掉 LLM re-rank
    python rag_pipeline.py --split val --no-rerank

    # 最簡版（只用 Dense，不 rewrite、不 HyDE、不 rerank）
    python rag_pipeline.py --split val --no-bm25 --no-metadata --no-rewrite --no-hyde --no-rerank

流程：
    Query → Metadata 抽取 → Query Rewrite → HyDE
    → Dense + BM25 + Metadata Match → RRF Fusion
    → LLM Re-rank → 生成答案（原文 + 結構化 entity 資料）
    → Entity 抽取（同 entity_tagging 做法）→ 輸出

輸出格式（對齊 eval）：
    {
        "question_id": "...",
        "answer": "...",
        "reference": ["doc_id_1", ...],
        "entities": {"VERDICT_CATEGORY": "不罰", ...}
    }

評估：
    python eval/eval.py --results_file results/predictions_val.json
"""

import os
import json
import re
import time
import math
import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────
EMBED_URL = os.getenv("EMBED_URL", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma:latest")

LLM_BACKEND = os.getenv("LLM_BACKEND", "claude")  # "claude" or "ollama"
LLM_OLLAMA_URL = os.getenv("LLM_OLLAMA_URL", "")
LLM_OLLAMA_MODEL = os.getenv("LLM_OLLAMA_MODEL", "gemma4:31b")
LLM_CLAUDE_URL = os.getenv("LLM_CLAUDE_URL", "")

DATA_DIR = Path(__file__).parent / "dataset" / "challenge1_dataset" / "full_content"
ENTITIES_PATH = Path(__file__).parent / "dataset" / "challenge1_dataset" / "v4_0415" / "dataset_entities_v4.json"
SPLIT_DIR = Path(__file__).parent / "dataset" / "challenge1_dataset" / "v4_0415" / "dataset_split_v4"
SCHEMA_PATH = Path(__file__).parent / "dataset" / "challenge1_dataset" / "v4_0415" / "entity_schema.json"
CACHE_DIR = Path(__file__).parent / "cache"
PROMPTS_DIR = Path(__file__).parent / "prompts"

TOP_K_RETRIEVE = 10   # candidates before re-rank
TOP_K_RERANK = 3      # docs sent to LLM for answer generation
RRF_K = 60            # RRF constant

EMBED_BATCH_SIZE = 8
LLM_TIMEOUT = 300


# ── Prompt Loader ─────────────────────────────────────────────

def load_prompt(name: str) -> str:
    """Load prompt template from prompts/ directory."""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


# ── Embedding ─────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """Call Ollama /api/embed endpoint. Returns (N, dim) array."""
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        resp = requests.post(EMBED_URL, json={
            "model": EMBED_MODEL,
            "input": batch,
        }, timeout=120)
        resp.raise_for_status()
        all_embeddings.extend(resp.json()["embeddings"])
    return np.array(all_embeddings, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between query vector a (dim,) and matrix b (N, dim)."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


# ── BM25 ──────────────────────────────────────────────────────

class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)

        # tokenize: split on whitespace + every CJK character as a token
        self.doc_tokens = [self._tokenize(doc) for doc in corpus]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size else 1

        # Build inverted index: term -> list of (doc_idx, term_freq)
        self.df = {}  # term -> doc freq
        self.tf = {}  # term -> {doc_idx: freq}
        for idx, tokens in enumerate(self.doc_tokens):
            seen = set()
            freq = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            for t, f in freq.items():
                if t not in self.tf:
                    self.tf[t] = {}
                self.tf[t][idx] = f
                if t not in seen:
                    self.df[t] = self.df.get(t, 0) + 1
                    seen.add(t)

    def _tokenize(self, text: str) -> list[str]:
        tokens = []
        buf = []
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
                if buf:
                    tokens.append(''.join(buf))
                    buf = []
                tokens.append(ch)
            elif ch.isalnum():
                buf.append(ch.lower())
            else:
                if buf:
                    tokens.append(''.join(buf))
                    buf = []
        if buf:
            tokens.append(''.join(buf))
        return tokens

    def score(self, query: str) -> np.ndarray:
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for t in query_tokens:
            if t not in self.df:
                continue
            idf = math.log((self.corpus_size - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
            for doc_idx, freq in self.tf[t].items():
                dl = self.doc_lens[doc_idx]
                tf_norm = (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                scores[doc_idx] += idf * tf_norm
        return scores


# ── Document Store ────────────────────────────────────────────

class DocumentStore:
    def __init__(self):
        self.doc_ids: list[str] = []
        self.titles: list[str] = []
        self.full_texts: list[str] = []
        self.entities: list[dict] = []
        self.metadata_text: list[str] = []  # flattened metadata for BM25/embed

        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25] = None

        # Metadata lookup tables
        self.defendant_to_docs: dict[str, list[int]] = {}
        self.case_no_to_doc: dict[str, int] = {}

        # Reverse metadata mapping: [(value, doc_idx, field_name), ...]
        self.entity_values: list[tuple[str, int, str]] = []

    def load(self):
        """Load entities + full texts."""
        print("[DocumentStore] Loading entities...")
        with open(ENTITIES_PATH, encoding="utf-8") as f:
            entities_list = json.load(f)

        for i, ent in enumerate(entities_list):
            doc_id = ent["doc_id"]
            title = ent["title"]
            self.doc_ids.append(doc_id)
            self.titles.append(title)
            self.entities.append(ent)

            # Load full text
            txt_path = DATA_DIR / f"{title}.txt"
            if txt_path.exists():
                full_text = txt_path.read_text(encoding="utf-8")
            else:
                full_text = ""
                print(f"  [WARN] Missing text file for {title}")
            self.full_texts.append(full_text)

            # Build metadata text (for embedding/BM25 - flattened key fields)
            info = ent.get("basic_info", {})
            def _str(v):
                return "、".join(v) if isinstance(v, list) else (v or "")
            meta_parts = [
                f"案號：{_str(info.get('CASE_NO', ''))}",
                f"法院：{_str(info.get('COURT', ''))}",
                f"被告/被移送人：{_str(info.get('DEFENDANT', ''))}",
                f"法條：{_str(info.get('LAW', ''))}",
                f"行為時間：{_str(info.get('ACT_TIME', ''))}",
                f"平台：{_str(info.get('PLATFORM', ''))}",
                f"裁判結果：{_str(info.get('VERDICT_CATEGORY', ''))}",
                f"處罰內容：{_str(info.get('SANCTION', ''))}",
                f"謠言內容：{_str(info.get('RUMOR_CONTENT', ''))}",
            ]
            self.metadata_text.append("\n".join(meta_parts))

            # Build lookup tables
            defendant = info.get("DEFENDANT", "")
            if isinstance(defendant, list):
                for d in defendant:
                    if d:
                        self.defendant_to_docs.setdefault(d, []).append(i)
            elif defendant:
                self.defendant_to_docs.setdefault(defendant, []).append(i)
            case_no = info.get("CASE_NO", "")
            if isinstance(case_no, list):
                for cn in case_no:
                    if cn:
                        self.case_no_to_doc[cn] = i
            elif case_no:
                self.case_no_to_doc[case_no] = i
            related = info.get("RELATED_CASE_NO")
            if isinstance(related, list):
                for r in related:
                    if r:
                        self.case_no_to_doc[r] = i
            elif related:
                self.case_no_to_doc[related] = i

            # Build reverse metadata values for query matching
            for section_name, section in ent.items():
                if section_name in ("doc_id", "title"):
                    continue
                if isinstance(section, dict):
                    for field, val in section.items():
                        if val is None:
                            continue
                        vals = val if isinstance(val, list) else [val]
                        for v in vals:
                            v_str = str(v).strip()
                            if len(v_str) >= 2:
                                self.entity_values.append((v_str, i, field))
            # Also add title
            if title and len(title) >= 2:
                self.entity_values.append((title, i, "TITLE"))

        print(f"[DocumentStore] Loaded {len(self.doc_ids)} documents")

    def build_index(self, force_rebuild: bool = False):
        """Build embeddings and BM25 index."""
        CACHE_DIR.mkdir(exist_ok=True)

        # Compute hash of doc_ids to detect data changes
        data_hash = hashlib.md5(json.dumps(self.doc_ids).encode()).hexdigest()[:8]
        embed_cache = CACHE_DIR / f"embeddings_{data_hash}.npy"
        bm25_cache = CACHE_DIR / f"bm25_{data_hash}.pkl"

        # ── Embeddings ──
        if not force_rebuild and embed_cache.exists():
            print("[Index] Loading cached embeddings...")
            self.embeddings = np.load(embed_cache)
        else:
            print("[Index] Computing embeddings (this may take a while)...")
            # Combine metadata + truncated full text for embedding
            texts_to_embed = []
            for i in range(len(self.doc_ids)):
                combined = self.metadata_text[i] + "\n\n" + self.full_texts[i][:2000]
                texts_to_embed.append(combined)
            self.embeddings = embed_texts(texts_to_embed)
            np.save(embed_cache, self.embeddings)
            print(f"[Index] Embeddings saved to {embed_cache}")

        # ── BM25 (always build from corpus — fast enough, avoids pickle issues) ──
        print("[Index] Building BM25 index...")
        corpus = [self.metadata_text[i] + "\n" + self.full_texts[i] for i in range(len(self.doc_ids))]
        self.bm25 = BM25(corpus)
        print("[Index] BM25 ready.")

        print("[Index] Ready.")


# ── Pre-Retrieval ─────────────────────────────────────────────

def metadata_mapping(query: str, store: "DocumentStore") -> np.ndarray:
    """Reverse metadata mapping: check which document metadata values appear in the query."""
    scores = np.zeros(len(store.doc_ids), dtype=np.float32)
    for val, doc_idx, field in store.entity_values:
        if val in query:
            # Case number and person name get higher weight
            if field in ("CASE_NO", "RELATED_CASE_NO"):
                scores[doc_idx] += 100.0
            elif field in ("DEFENDANT", "VICTIM", "APPELLANT", "JUDGE", "PROSECUTOR"):
                scores[doc_idx] += 80.0
            elif field == "TITLE":
                scores[doc_idx] += 60.0
            elif field in ("COURT", "TRANSFER_AGENCY", "PROSECUTION"):
                scores[doc_idx] += 30.0
            else:
                scores[doc_idx] += 10.0
    return scores


def call_llm(message: str, timeout: int = LLM_TIMEOUT) -> str:
    """Call LLM via configured backend (claude or ollama)."""
    if LLM_BACKEND == "ollama":
        resp = requests.post(LLM_OLLAMA_URL, json={
            "model": LLM_OLLAMA_MODEL,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        }, timeout=timeout + 300)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
    else:
        resp = requests.post(LLM_CLAUDE_URL, json={"message": message, "timeout": timeout}, timeout=timeout + 30)
        resp.raise_for_status()
        return resp.json().get("response", "")


def query_rewrite(query: str) -> dict:
    """Use LLM to rewrite query, classify type, and decompose if multi_hop.

    Returns:
        {"rewritten": str, "query_type": str, "sub_queries": list[str]}
    """
    prompt = load_prompt("query_rewrite").format(query=query)
    fallback = {"rewritten": query, "query_type": "single", "sub_queries": []}
    try:
        raw = call_llm(prompt, timeout=120)
        raw = raw.strip()
        # Parse JSON from response
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            result = json.loads(raw[start:end + 1])
            # Ensure required fields
            result.setdefault("rewritten", query)
            result.setdefault("query_type", "single")
            result.setdefault("sub_queries", [])
            return result
        return fallback
    except Exception as e:
        print(f"  [WARN] Query rewrite failed: {e}")
        return fallback


def generate_hyde(query: str) -> str:
    """Generate a hypothetical document passage that would answer the query (HyDE)."""
    prompt = load_prompt("hyde").format(query=query)
    try:
        result = call_llm(prompt, timeout=120)
        return result.strip()
    except Exception as e:
        print(f"  [WARN] HyDE generation failed: {e}")
        return ""


def entity_query(query: str, query_type: str) -> dict:
    """Second-stage LLM call: determine which entity field to query for extremum/aggregation."""
    schema_path = ENTITIES_PATH.parent / "entity_schema.json"
    schema_text = schema_path.read_text(encoding="utf-8")
    prompt = load_prompt("entity_query").format(
        schema=schema_text, query_type=query_type, query=query
    )
    fallback = {}
    try:
        raw = call_llm(prompt, timeout=120)
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            raw = "\n".join(lines).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start:end + 1])
        return fallback
    except Exception as e:
        print(f"  [WARN] Entity query failed: {e}")
        return fallback


def _parse_money(s: str) -> Optional[float]:
    """Parse a money string to a numeric value. e.g. '新臺幣貳拾萬元' → 200000

    Only parses monetary amounts (containing 元 or 新臺幣). Ignores 刑期 like 拘役/有期徒刑.
    """
    if not s or s == "null":
        return None
    # Only parse monetary values
    if "元" not in s and "新臺幣" not in s and "NT" not in s:
        return None
    # Try to find Arabic numbers first
    nums = re.findall(r'[\d,]+', s)
    if nums:
        try:
            return float(nums[0].replace(',', ''))
        except ValueError:
            pass
    # Try Chinese number conversion
    import sys
    eval_path = str(ENTITIES_PATH.parent / "eval")
    if eval_path not in sys.path:
        sys.path.insert(0, eval_path)
    from entities import _cn_number_to_int
    m = re.search(r'([零壹貳參肆伍陸柒捌玖拾佰仟萬億一二三四五六七八九十百千万亿]+)', s)
    if m:
        val = _cn_number_to_int(m.group(1))
        if val is not None:
            return float(val)
    return None


def resolve_extremum(query: str, store: "DocumentStore", eq_result: dict) -> list[int]:
    """Find document indices matching an extremum query by scanning entity data."""
    field = eq_result.get("target_field", "")
    order = eq_result.get("order", "max")
    if not field:
        return []

    scored = []
    for idx, ent in enumerate(store.entities):
        # Search all sections for the field
        for section_name, section in ent.items():
            if section_name in ("doc_id", "title"):
                continue
            if isinstance(section, dict) and field in section:
                val = section[field]
                if val is None:
                    continue
                # Handle list values (take first or join)
                if isinstance(val, list):
                    val = val[0] if val else None
                if val is None:
                    continue
                num = _parse_money(str(val))
                if num is not None:
                    scored.append((idx, num))
                break

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=(order == "max"))
    # Return top results
    return [idx for idx, _ in scored[:5]]


def resolve_aggregation(query: str, store: "DocumentStore", eq_result: dict) -> list[int]:
    """Find document indices matching aggregation filter conditions."""
    filters = eq_result.get("filters", [])
    if not filters:
        return []

    matched_indices = set(range(len(store.doc_ids)))  # start with all

    for f in filters:
        field = f.get("field", "")
        op = f.get("op", "eq")
        value = f.get("value", "")
        if not field or not value:
            continue

        current_match = set()
        for idx, ent in enumerate(store.entities):
            for section_name, section in ent.items():
                if section_name in ("doc_id", "title"):
                    continue
                if isinstance(section, dict) and field in section:
                    val = section[field]
                    if val is None:
                        continue
                    val_str = "、".join(val) if isinstance(val, list) else str(val)
                    if op == "eq" and val_str == value:
                        current_match.add(idx)
                    elif op == "contains" and value in val_str:
                        current_match.add(idx)
                    break

        matched_indices &= current_match

    return list(matched_indices)


# ── Retrieval ─────────────────────────────────────────────────



def reciprocal_rank_fusion(*rank_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.
    Each rank_list is a list of doc indices sorted by relevance (best first).
    Returns fused ranking.
    """
    scores = {}
    for rank_list in rank_lists:
        for rank, doc_idx in enumerate(rank_list):
            scores[doc_idx] = scores.get(doc_idx, 0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def retrieve(
    query: str,
    store: DocumentStore,
    rewritten_query: Optional[str] = None,
    hyde_passage: Optional[str] = None,
    top_k: int = TOP_K_RETRIEVE,
    use_dense: bool = True,
    use_bm25: bool = True,
    use_metadata: bool = True,
) -> list[int]:
    """
    Hybrid retrieval: Dense + BM25 + Metadata Mapping → RRF fusion.
    可透過參數關閉個別檢索方式。
    """
    rankings = []

    # Dense retrieval
    if use_dense:
        q_emb = embed_texts([query])[0]
        dense_scores = cosine_similarity(q_emb, store.embeddings)
        rankings.append(np.argsort(-dense_scores).tolist())

        if rewritten_query and rewritten_query != query:
            rw_emb = embed_texts([rewritten_query])[0]
            rw_scores = cosine_similarity(rw_emb, store.embeddings)
            rankings.append(np.argsort(-rw_scores).tolist())

        if hyde_passage:
            hyde_emb = embed_texts([hyde_passage])[0]
            hyde_scores = cosine_similarity(hyde_emb, store.embeddings)
            rankings.append(np.argsort(-hyde_scores).tolist())

    # BM25
    if use_bm25:
        bm25_scores = store.bm25.score(query)
        rankings.append(np.argsort(-bm25_scores).tolist())

        if rewritten_query and rewritten_query != query:
            rw_bm25_scores = store.bm25.score(rewritten_query)
            rankings.append(np.argsort(-rw_bm25_scores).tolist())

    # Metadata mapping (reverse lookup)
    if use_metadata:
        meta_scores = metadata_mapping(query, store)
        if meta_scores.max() > 0:
            meta_ranking = np.argsort(-meta_scores).tolist()
            rankings.append(meta_ranking)
            rankings.append(meta_ranking)  # double weight

    if not rankings:
        return list(range(min(top_k, len(store.doc_ids))))

    fused = reciprocal_rank_fusion(*rankings)
    return fused[:top_k]


# ── Post-Retrieval: Re-ranking ────────────────────────────────

def rerank_with_llm(query: str, candidates: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    """
    Use LLM to re-rank candidate documents by relevance.
    candidates: list of {"doc_idx": int, "doc_id": str, "title": str, "snippet": str}
    """
    if len(candidates) <= top_k:
        return candidates

    # Build candidate list for LLM
    cand_text = ""
    for i, c in enumerate(candidates):
        cand_text += f"\n[文件{i+1}] {c['title']}\n{c['snippet'][:500]}\n"

    prompt = load_prompt("rerank").format(query=query, cand_text=cand_text, top_k=top_k)

    try:
        result = call_llm(prompt, timeout=270)
        # Parse numbers from response
        numbers = re.findall(r'\d+', result)
        reranked = []
        seen = set()
        for n in numbers:
            idx = int(n) - 1
            if 0 <= idx < len(candidates) and idx not in seen:
                reranked.append(candidates[idx])
                seen.add(idx)
            if len(reranked) >= top_k:
                break
        # Fill remaining if LLM didn't return enough
        for c in candidates:
            if len(reranked) >= top_k:
                break
            if c not in reranked:
                reranked.append(c)
        return reranked
    except Exception as e:
        print(f"  [WARN] Re-ranking failed: {e}")
        return candidates[:top_k]


# ── Answer Generation ─────────────────────────────────────────

def _format_entity_context(ent: dict) -> str:
    """把 dataset_entities.json 的一筆 entity 資料格式化成 LLM 可讀的文字。"""
    lines = []
    for section_name, section in ent.items():
        if section_name in ("doc_id", "title"):
            continue
        if isinstance(section, dict):
            for k, v in section.items():
                if v is not None:
                    if isinstance(v, list):
                        val = "、".join(str(x) if not isinstance(x, dict) else json.dumps(x, ensure_ascii=False) for x in v)
                    else:
                        val = str(v)
                    if len(val) > 200:
                        val = val[:200] + "..."
                    lines.append(f"  {k}：{val}")
    return "\n".join(lines)


def generate_answer(query: str, docs: list[dict], question_type_hint: str = "") -> str:
    """Generate answer using LLM with type-specific prompts."""

    # Build context from retrieved docs (原文 + 結構化 entity 資料)
    context = ""
    for i, doc in enumerate(docs):
        context += f"\n\n===== 文件{i+1}：{doc['title']} =====\n"
        context += f"【結構化資訊】\n{doc['metadata_text']}\n"
        if doc.get("entity_context"):
            context += f"【案件實體資料】\n{doc['entity_context']}\n"
        context += f"【全文】\n{doc['full_text'][:3000]}\n"

    prompt = load_prompt("answer").format(context=context, query=query)

    try:
        answer = call_llm(prompt)
        # Clean up common prefixes
        answer = answer.strip()
        for prefix in ["根據文件，", "根據提供的文件，", "根據判決書，", "答：", "回答："]:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        # Normalize unanswerable: if the answer contains the key phrase, standardize it
        if "無法從文件判斷" in answer:
            answer = "無法從文件判斷"
        return answer
    except Exception as e:
        print(f"  [ERROR] Answer generation failed: {e}")
        return "無法從文件判斷"


def _build_structured_result(query_type: str, eq_result: dict, indices: list[int], store: "DocumentStore") -> str:
    """Build a human-readable structured result string for the LLM."""
    field = eq_result.get("target_field", "")
    lines = []

    if query_type == "extremum":
        order = eq_result.get("order", "max")
        order_label = "最高" if order == "max" else "最低"
        lines.append(f"查詢類型：極值查詢（{order_label}）")
        lines.append(f"查詢欄位：{field}")
        lines.append(f"結果（依{order_label}排序）：")
        for rank, idx in enumerate(indices):
            ent = store.entities[idx]
            doc_id = store.doc_ids[idx]
            title = store.titles[idx]
            # Find field value
            val = None
            for section_name, section in ent.items():
                if section_name in ("doc_id", "title"):
                    continue
                if isinstance(section, dict) and field in section:
                    val = section[field]
                    break
            lines.append(f"  第{rank+1}名：{title}（{doc_id}）→ {field} = {val}")

    elif query_type == "aggregation":
        filters = eq_result.get("filters", [])
        filter_desc = ", ".join(f"{f['field']} {f['op']} {f['value']}" for f in filters)
        lines.append(f"查詢類型：篩選查詢")
        lines.append(f"篩選條件：{filter_desc}")
        lines.append(f"符合條件的案件數：{len(indices)} 件")
        lines.append(f"案件列表：")
        for idx in indices:
            doc_id = store.doc_ids[idx]
            title = store.titles[idx]
            info = store.entities[idx].get("basic_info", {})
            defendant = info.get("DEFENDANT", "")
            if isinstance(defendant, list):
                defendant = "、".join(defendant)
            verdict = info.get("VERDICT_CATEGORY", "")
            sanction = info.get("SANCTION", "")
            lines.append(f"  - {title}（{doc_id}）被告：{defendant}，結果：{verdict}，處罰：{sanction or '無'}")

    return "\n".join(lines)


def generate_structured_answer(query: str, docs: list[dict], structured_result: str) -> str:
    """Generate answer for extremum/aggregation using structured query result."""
    context = ""
    for i, doc in enumerate(docs):
        context += f"\n\n===== 文件{i+1}：{doc['title']} =====\n"
        context += f"【結構化資訊】\n{doc['metadata_text']}\n"
        if doc.get("entity_context"):
            context += f"【案件實體資料】\n{doc['entity_context']}\n"
        context += f"【全文】\n{doc['full_text'][:3000]}\n"

    prompt = load_prompt("answer_structured").format(
        structured_result=structured_result, context=context, query=query
    )
    try:
        answer = call_llm(prompt)
        answer = answer.strip()
        for prefix in ["根據文件，", "根據提供的文件，", "根據判決書，", "答：", "回答："]:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        return answer
    except Exception as e:
        print(f"  [ERROR] Structured answer generation failed: {e}")
        return "無法從文件判斷"


_SCHEMA_CACHE = None

def _load_schema():
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = SCHEMA_PATH.read_text(encoding="utf-8")
    return _SCHEMA_CACHE


_ENTITY_SECTION_KEYS = {
    "basic_info", "elements", "procedure", "cited_authority",
    "concurrence", "prior_record", "civil_tort", "law_role",
}


def _flatten_entities(ents: dict) -> dict:
    """LLM 偶爾會把欄位巢狀在 section 下（如 basic_info/elements），攤平成扁平 dict。"""
    if not isinstance(ents, dict):
        return ents
    if not any(k in _ENTITY_SECTION_KEYS for k in ents.keys()):
        return ents
    flat = {}
    for k, v in ents.items():
        if k in _ENTITY_SECTION_KEYS and isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    return flat


def extract_entities_from_answer(question: str, answer: str, doc_entities: list[dict] = None) -> dict:
    """用 LLM 判斷答案涵蓋哪些 entities。做法同 entity_tagging。"""

    # 從 reranked docs 收集 entity 資料
    available = []
    if doc_entities:
        seen = set()
        for ent in doc_entities:
            for section_name, section in ent.items():
                if section_name in ("doc_id", "title"):
                    continue
                if isinstance(section, dict):
                    for k, v in section.items():
                        if v is not None and k not in seen:
                            val_str = str(v) if not isinstance(v, list) else "、".join(str(x) for x in v)
                            if len(val_str) > 100:
                                val_str = val_str[:100] + "..."
                            available.append(f"  {k} = {val_str}")
                            seen.add(k)

    schema_csv = _load_schema()
    entity_data = chr(10).join(available) if available else '（無）'

    prompt = load_prompt("entity_extraction").format(
        question=question,
        answer=answer,
        entity_data=entity_data,
        schema_csv=schema_csv,
    )

    try:
        raw = call_llm(prompt, timeout=270)
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            raw = "\n".join(lines).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            return _flatten_entities(json.loads(raw[start:end + 1]))
        return {}
    except Exception as e:
        print(f"  [ERROR] Entity extraction failed: {e}")
        return {}


# ── Single Query (for API) ────────────────────────────────────

def _retrieve_and_build_candidates(
    query: str,
    store: DocumentStore,
    rewritten: Optional[str] = None,
    hyde_passage: Optional[str] = None,
    top_k: int = TOP_K_RETRIEVE,
    use_dense: bool = True,
    use_bm25: bool = True,
    use_metadata: bool = True,
) -> list[dict]:
    """Retrieve documents and build candidate list."""
    top_indices = retrieve(query, store, rewritten, hyde_passage,
                           top_k=top_k, use_dense=use_dense, use_bm25=use_bm25,
                           use_metadata=use_metadata)
    candidates = []
    for idx in top_indices:
        candidates.append({
            "doc_idx": idx,
            "doc_id": store.doc_ids[idx],
            "title": store.titles[idx],
            "snippet": store.full_texts[idx][:500],
            "metadata_text": store.metadata_text[idx],
            "full_text": store.full_texts[idx],
            "entity_context": _format_entity_context(store.entities[idx]),
        })
    return candidates


def run_single_query(
    query: str,
    store: DocumentStore,
    use_rewrite: bool = True,
    use_hyde: bool = True,
    use_rerank: bool = True,
    use_dense: bool = True,
    use_bm25: bool = True,
    use_metadata: bool = True,
) -> dict:
    """Run RAG pipeline on a single free-form query. Returns answer + refs."""

    # Start time
    start_time = time.time()
    
    # ── Rewrite (includes query_type + sub_queries) ──
    rw = query_rewrite(query) if use_rewrite else {"rewritten": query, "query_type": "single", "sub_queries": []}
    rewritten = rw["rewritten"]
    query_type = rw["query_type"]
    sub_queries = rw.get("sub_queries", [])
    
    # Rewrite time
    rewrite_time = time.time() - start_time
    print(f"  [{query_type}] Rewritten: {rewritten[:80]}... (rewrite time: {rewrite_time:.2f}s)")

    hyde_passage = generate_hyde(query) if use_hyde else None
    
    # HyDE time
    if use_hyde:
        hyde_time = time.time() - start_time - rewrite_time
        print(f"  HyDE passage generated (HyDE time: {hyde_time:.2f}s)")

    # ── Retrieval (varies by query_type) ──
    if query_type == "multi_hop" and sub_queries:
        all_candidates = {}
        for sq in sub_queries:
            sq_candidates = _retrieve_and_build_candidates(
                sq, store, rewritten=None, hyde_passage=None,
                use_dense=use_dense, use_bm25=use_bm25, use_metadata=use_metadata)
            for c in sq_candidates:
                if c["doc_id"] not in all_candidates:
                    all_candidates[c["doc_id"]] = c
        candidates = list(all_candidates.values())
    elif query_type in ("extremum", "aggregation"):
        eq_result = entity_query(query, query_type)
        if query_type == "extremum":
            target_indices = resolve_extremum(query, store, eq_result)
        else:
            target_indices = resolve_aggregation(query, store, eq_result)
            target_indices = target_indices[:10]  # aggregation 限制 top 10
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
            # Build structured result string
            structured_result = _build_structured_result(query_type, eq_result, target_indices, store)
            # Skip rerank, use structured answer
            answer = generate_structured_answer(query, candidates, structured_result)
            return {
                "answer": answer,
                "reference": [c["doc_id"] for c in candidates],
                "entities": {},
                "query_type": query_type,
            }
        else:
            candidates = _retrieve_and_build_candidates(
                query, store, rewritten=rewritten, hyde_passage=hyde_passage,
                use_dense=use_dense, use_bm25=use_bm25, use_metadata=use_metadata)
    else:
        candidates = _retrieve_and_build_candidates(
            query, store, rewritten=rewritten, hyde_passage=hyde_passage,
            use_dense=use_dense, use_bm25=use_bm25, use_metadata=use_metadata)
    # Retrieval time
    retrieval_time = time.time() - start_time - rewrite_time - (hyde_time if use_hyde else 0)
    print(f"  Retrieved {len(candidates)} candidates (retrieval time: {retrieval_time:.2f}s)")
    
    if use_rerank and len(candidates) > TOP_K_RERANK:
        reranked = rerank_with_llm(query, candidates, TOP_K_RERANK)
        rerank_time = time.time() - start_time - rewrite_time - retrieval_time - (hyde_time if use_hyde else 0)
        print(f"  Re-ranked candidates (rerank time: {rerank_time:.2f}s)")
    else:
        reranked = candidates[:TOP_K_RERANK]

    answer = generate_answer(query, reranked)
    # Answer generation time
    answer_time = time.time() - start_time - rewrite_time - retrieval_time - (rerank_time if use_rerank and len(candidates) > TOP_K_RERANK else 0) - (hyde_time if use_hyde else 0)
    print(f"  Generated answer (answer generation time: {answer_time:.2f}s)")
    
    if answer == "無法從文件判斷":
        entities = {}
    else:
        reranked_entities = [store.entities[d["doc_idx"]] for d in reranked]
        entities = extract_entities_from_answer(query, answer, reranked_entities)
        # Entity extraction time
        entity_time = time.time() - start_time - rewrite_time - retrieval_time - (rerank_time if use_rerank and len(candidates) > TOP_K_RERANK else 0) - answer_time - (hyde_time if use_hyde else 0)
        print(f"  Extracted entities (entity extraction time: {entity_time:.2f}s)")
    
    return {
        "answer": answer,
        "reference": [d["doc_id"] for d in reranked],
        "entities": entities,
        "query_type": query_type,
    }


# ── Main Pipeline ─────────────────────────────────────────────

def run_pipeline(
    questions: list[dict],
    store: DocumentStore,
    use_rewrite: bool = True,
    use_hyde: bool = True,
    use_rerank: bool = True,
    use_dense: bool = True,
    use_bm25: bool = True,
    use_metadata: bool = True,
    sleep_sec: float = 1.0,
    output_path: str = None,
) -> list[dict]:
    """
    Run full RAG pipeline on a list of questions.
    Saves incrementally after each question if output_path is provided.
    Resumes from existing predictions if output_path exists.
    """
    # Load existing predictions for resume support
    predictions = []
    done_qids = set()
    if output_path and Path(output_path).exists():
        with open(output_path, encoding="utf-8") as f:
            predictions = json.load(f)
        done_qids = {p["question_id"] for p in predictions}
        if done_qids:
            print(f"[Pipeline] Resuming: {len(done_qids)} already done, skipping them.")

    total = len(questions)

    # Info log directory
    info_dir = Path(__file__).parent / "results" / "info"
    info_dir.mkdir(parents=True, exist_ok=True)

    # Time cost tracking
    time_cost_path = Path(__file__).parent / "results" / "time_cost.json"
    time_costs = {}
    if time_cost_path.exists():
        with open(time_cost_path, encoding="utf-8") as f:
            time_costs = json.load(f)
    # Derive run name from output_path
    run_name = Path(output_path).stem if output_path else "default"
    if run_name not in time_costs:
        time_costs[run_name] = {}

    # Per-run info file：results/info/{split}.json（例如 info/val.json、info/test.json）
    split_name = run_name.replace("predictions_", "") or "default"
    info_file = info_dir / f"{split_name}.json"
    info_map: dict = {}
    if info_file.exists():
        try:
            with open(info_file, encoding="utf-8") as f:
                info_map = json.load(f)
        except Exception:
            info_map = {}

    for qi, q in enumerate(questions):
        qid = q["question_id"]
        query = q["question"]

        if qid in done_qids:
            continue

        info = {"question_id": qid, "question": query}
        tc = {}  # time cost for this question
        t_question_start = time.time()
        print(f"\n[{LLM_BACKEND}][{qi+1}/{total}] {qid}: {query[:60]}...")

        # ── Pre-Retrieval: Rewrite ──
        rw = {"rewritten": query, "query_type": "single", "sub_queries": []}
        if use_rewrite:
            t0 = time.time()
            rw = query_rewrite(query)
            tc["rewrite"] = round(time.time() - t0, 2)
            print(f"  [{rw['query_type']}] Rewritten: {rw['rewritten'][:80]}... ({tc['rewrite']}s)")
            if rw["sub_queries"]:
                print(f"  Sub-queries: {rw['sub_queries']}")
            time.sleep(sleep_sec)

        rewritten = rw["rewritten"]
        query_type = rw["query_type"]
        sub_queries = rw.get("sub_queries", [])
        info["query_type"] = query_type
        info["rewritten"] = rewritten
        info["sub_queries"] = sub_queries

        hyde_passage = None
        if use_hyde:
            t0 = time.time()
            hyde_passage = generate_hyde(query)
            tc["hyde"] = round(time.time() - t0, 2)
            if hyde_passage:
                print(f"  HyDE: {hyde_passage[:80]}... ({tc['hyde']}s)")
            info["hyde"] = hyde_passage
            time.sleep(sleep_sec)

        # ── Retrieval ──
        t0 = time.time()
        if query_type == "multi_hop" and sub_queries:
            all_candidates = {}
            for sq in sub_queries:
                sq_candidates = _retrieve_and_build_candidates(
                    sq, store, rewritten=None, hyde_passage=None,
                    use_dense=use_dense, use_bm25=use_bm25, use_metadata=use_metadata)
                for c in sq_candidates:
                    if c["doc_id"] not in all_candidates:
                        all_candidates[c["doc_id"]] = c
            candidates = list(all_candidates.values())
            tc["retrieve"] = round(time.time() - t0, 2)
            info["retrieved"] = [c["doc_id"] for c in candidates[:10]]
            print(f"  Retrieved (multi_hop merged): {info['retrieved']} ({tc['retrieve']}s)")
        elif query_type in ("extremum", "aggregation"):
            t0_eq = time.time()
            eq_result = entity_query(query, query_type)
            tc["entity_query"] = round(time.time() - t0_eq, 2)
            print(f"  Entity query: {eq_result} ({tc['entity_query']}s)")
            time.sleep(sleep_sec)
            if query_type == "extremum":
                target_indices = resolve_extremum(query, store, eq_result)
            else:
                target_indices = resolve_aggregation(query, store, eq_result)
                target_indices = target_indices[:10]
            tc["retrieve"] = round(time.time() - t0, 2)
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
                info["retrieved"] = [c["doc_id"] for c in candidates[:10]]
                info["entity_query"] = eq_result
                print(f"  Retrieved ({query_type}): {info['retrieved']}")
                # Structured answer — skip rerank
                structured_result = _build_structured_result(query_type, eq_result, target_indices, store)
                print(f"  Structured result:\n{structured_result}")
                t0_ans = time.time()
                answer = generate_structured_answer(query, candidates, structured_result)
                tc["answer"] = round(time.time() - t0_ans, 2)
                tc["total"] = round(time.time() - t_question_start, 2)
                info["structured_result"] = structured_result
                info["answer"] = answer
                print(f"  Answer: {answer[:100]}... ({tc['answer']}s)")
                predictions.append({
                    "question_id": qid,
                    "answer": answer,
                    "reference": [c["doc_id"] for c in candidates],
                    "entities": {},
                })
                if output_path:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(predictions, f, ensure_ascii=False, indent=2)
                time_costs[run_name][qid] = tc
                with open(time_cost_path, "w", encoding="utf-8") as f:
                    json.dump(time_costs, f, ensure_ascii=False, indent=2)
                info_map[qid] = info
                with open(info_file, "w", encoding="utf-8") as f:
                    json.dump(info_map, f, ensure_ascii=False, indent=2)
                continue
            else:
                print(f"  [WARN] {query_type} query returned no results, falling back to normal retrieval")
                candidates = _retrieve_and_build_candidates(
                    query, store, rewritten=rewritten, hyde_passage=hyde_passage,
                    use_dense=use_dense, use_bm25=use_bm25, use_metadata=use_metadata)
                tc["retrieve"] = round(time.time() - t0, 2)
                info["retrieved"] = [c["doc_id"] for c in candidates[:5]]
                print(f"  Retrieved (fallback): {info['retrieved']} ({tc['retrieve']}s)")
        else:
            candidates = _retrieve_and_build_candidates(
                query, store, rewritten=rewritten, hyde_passage=hyde_passage,
                use_dense=use_dense, use_bm25=use_bm25, use_metadata=use_metadata)
            tc["retrieve"] = round(time.time() - t0, 2)
            info["retrieved"] = [c["doc_id"] for c in candidates[:5]]
            print(f"  Retrieved: {info['retrieved']} ({tc['retrieve']}s)")

        # ── Post-Retrieval: Re-rank ──
        t0 = time.time()
        if use_rerank and len(candidates) > TOP_K_RERANK:
            reranked = rerank_with_llm(query, candidates, TOP_K_RERANK)
            tc["rerank"] = round(time.time() - t0, 2)
            info["reranked"] = [d["doc_id"] for d in reranked]
            print(f"  Reranked: {info['reranked']} ({tc['rerank']}s)")
            time.sleep(sleep_sec)
        else:
            reranked = candidates[:TOP_K_RERANK]

        # ── Answer Generation ──
        t0 = time.time()
        answer = generate_answer(query, reranked)
        tc["answer"] = round(time.time() - t0, 2)
        info["answer"] = answer
        print(f"  Answer: {answer[:100]}... ({tc['answer']}s)")

        # ── Entity Extraction ──
        t0 = time.time()
        if answer == "無法從文件判斷":
            entities = {}
        else:
            reranked_entities = [store.entities[d["doc_idx"]] for d in reranked]
            entities = extract_entities_from_answer(query, answer, reranked_entities)
            print(f"  Entities: {list(entities.keys())}")
            time.sleep(sleep_sec)
        tc["entity_extract"] = round(time.time() - t0, 2)
        tc["total"] = round(time.time() - t_question_start, 2)

        info["entities"] = entities
        info["time_cost"] = tc

        predictions.append({
            "question_id": qid,
            "answer": answer,
            "reference": [d["doc_id"] for d in reranked],
            "entities": entities,
        })

        # Incremental save
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
            print(f"  [Saved] {len(predictions)} predictions → {output_path}")

        # Save info log（per-split 合併在一個檔，key = qid）
        info_map[qid] = info
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(info_map, f, ensure_ascii=False, indent=2)

        # Save time cost
        time_costs[run_name][qid] = tc
        with open(time_cost_path, "w", encoding="utf-8") as f:
            json.dump(time_costs, f, ensure_ascii=False, indent=2)
        print(f"  [Time] total={tc['total']}s")

        time.sleep(sleep_sec)

    return predictions


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="COVID-19 Legal QA RAG Pipeline")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"],
                        help="Which split to run on (default: val)")
    parser.add_argument("--output", default=None,
                        help="Output predictions JSON path (default: results/predictions_{split}.json)")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Force rebuild embeddings and BM25 index")
    parser.add_argument("--no-rewrite", action="store_true",
                        help="Disable query rewriting")
    parser.add_argument("--no-hyde", action="store_true",
                        help="Disable HyDE")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Disable LLM re-ranking")
    parser.add_argument("--no-dense", action="store_true",
                        help="Disable dense (embedding) retrieval")
    parser.add_argument("--no-bm25", action="store_true",
                        help="Disable BM25 retrieval")
    parser.add_argument("--no-metadata", action="store_true",
                        help="Disable metadata match retrieval")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N questions (for testing)")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Sleep between LLM calls (default: 1.0s)")
    parser.add_argument("--llm", default=None, choices=["claude", "ollama"],
                        help="LLM backend (default: use config in script)")
    parser.add_argument("--llm-model", default=None,
                        help="Ollama model name (e.g. gemma3:27b)")
    args = parser.parse_args()
    

    # Apply LLM backend override
    global LLM_BACKEND, LLM_OLLAMA_MODEL
    if args.llm:
        LLM_BACKEND = args.llm
    if args.llm_model:
        LLM_OLLAMA_MODEL = args.llm_model
    print(f"[Config] LLM backend: {LLM_BACKEND}" +
        (f" ({LLM_OLLAMA_MODEL})" if LLM_BACKEND == "ollama" else ""))
    
    # Load data
    store = DocumentStore()
    store.load()
    store.build_index(force_rebuild=args.rebuild_index)

    # Load questions
    split_path = SPLIT_DIR / f"qa_{args.split}.json"
    print(f"\n[Pipeline] Loading questions from {split_path}")
    with open(split_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"[Pipeline] {len(questions)} questions loaded")

    if args.limit:
        questions = questions[:args.limit]
        print(f"[Pipeline] Limited to first {args.limit} questions")

    # Output path
    output_path = args.output or str(Path(__file__).parent / "results" / f"predictions_{args.split}.json")

    # Run pipeline (saves incrementally)
    predictions = run_pipeline(
        questions, store,
        use_rewrite=not args.no_rewrite,
        use_hyde=not args.no_hyde,
        use_rerank=not args.no_rerank,
        use_dense=not args.no_dense,
        use_bm25=not args.no_bm25,
        use_metadata=not args.no_metadata,
        sleep_sec=args.sleep,
        output_path=output_path,
    )

    print(f"\n[Pipeline] Done! {len(predictions)} predictions in {output_path}")
    print(f"[Pipeline] Run evaluation with:")
    print(f"  python dataset/challenge1_dataset/qa_eval.py --mode batch \\")
    print(f"    --pred {output_path} \\")
    print(f"    --gt dataset/challenge1_dataset/dataset_split/qa_{args.split}.json")


if __name__ == "__main__":
    main()
