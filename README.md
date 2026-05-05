# TaDELS-RAG-BETA

Legal-domain RAG system for Taiwan Sociolegal datasets (challenge1 / human-rights subset).

## Features

- `/chat` — Streaming RAG conversation (query rewrite → HyDE → hybrid retrieval → rerank → answer + entity extraction)
- `/results` — Per-question evaluation viewer (entity matching, hallucination penalty, NDCG@5)
- `/qa` — QA dataset browser
- `/entities` — Document entity index
- `/intro` — Project overview

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env       # then edit .env with your endpoints
```

The system depends on two external HTTP services configured via `.env`:

- **Embedding** (Ollama-compatible `/api/embed`)
- **LLM** — either an Ollama `/api/chat` endpoint or a Claude-proxy `/ask` endpoint

Switch backends with `LLM_BACKEND=claude` or `LLM_BACKEND=ollama`.

## Run

```bash
python rag_server.py
```

Server listens on `0.0.0.0:8866`. Open <http://localhost:8866/>.

On first launch, embeddings and the BM25 index are built from `dataset/challenge1_dataset/full_content/` (211 documents) and cached under `cache/`.

## Layout

```
rag_server.py                  FastAPI server
rag_pipeline.py                Retrieval + RAG pipeline (DocumentStore, hybrid retrieval, rerank, answer)
prompts/                       LLM prompt templates
web/                           Front-end pages
results/predictions_*.json     Pre-computed predictions (val / test / train)
eval/predictions_*_eval.json   Cached evaluation scores
dataset/challenge1_dataset/
  full_content/                211 legal documents (.txt corpus)
  v4_0415/
    qa_dataset_final_v4.json   QA pairs (ground truth)
    dataset_entities_v4.json   Per-document entities
    entity_schema.json/.csv    Entity type schema
    dataset_split_v4/          train / val / test splits
    eval/                      Scoring modules: entities, hallucination, ndcg
```

## Evaluation

```bash
python rag_pipeline.py --split val           # writes results/predictions_val.json
python dataset/challenge1_dataset/v4_0415/eval/eval.py \
    --pred results/predictions_val.json \
    --gt   dataset/challenge1_dataset/v4_0415/dataset_split_v4/qa_val.json
```

## Notes

- This is a **beta extract**: experiment variants, ablation predictions, and the ColPali/Gemma visual-retrieval viewer (`/retrieval`) have been stripped.
- `cache/`, `results/info/`, and `results/time_cost.json` are runtime artifacts and gitignored.
