# 挑戰一：TDELS RAG 系統

## 系統概述

本系統針對 211 篇 COVID-19 相關台灣法院裁判文書，建構 Retrieval-Augmented Generation (RAG) 問答系統。涵蓋社會秩序維護法裁定、傳染病防治法判決、特別條例判決等多種文書類型。

## 資料集

| 項目          | 數量                                             |
| ------------- | ------------------------------------------------ |
| 法律文件      | 211 篇                                           |
| Single-hop QA | 650 題 (short 212 / long 225 / unanswerable 213) |
| Multi-hop QA  | 18 題                                            |
| 結構化欄位    | 39 種 (5 大模組)                                 |

### 資料切分

| 集合      | 題數 | short | long | unanswerable |
| --------- | ---- | ----- | ---- | ------------ |
| Train     | 454  | 148   | 157  | 149          |
| Val       | 98   | 32    | 34   | 32           |
| Test      | 98   | 32    | 34   | 32           |
| Multi-hop | 18   | —    | —   | —           |

## RAG Pipeline 架構

```
Query
  │
  ▼
┌─────────────────────────────────────┐
│         Pre-Retrieval               │
│  ┌───────────┐  ┌────────────────┐  │
│  │ Metadata  │  │ Query Rewrite  │  │
│  │ Extract   │  │   (LLM)       │  │
│  │ 案號/姓名  │  └────────────────┘  │
│  └───────────┘  ┌────────────────┐  │
│                 │   HyDE         │  │
│                 │ 假設性文件生成    │  │
│                 └────────────────┘  │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│         Retrieval (RRF Fusion)      │
│                                     │
│  Dense (embeddinggemma) ──┐         │
│  Dense (rewritten query) ─┤         │
│  Dense (HyDE passage) ────┤ → RRF   │
│  BM25 (original query) ───┤   Fusion │
│  BM25 (rewritten query) ──┤         │
│  Metadata Match (2x) ─────┘         │
│  (案號: 100分 / 被告名: 80分)        │
│                                     │
└─────────────────────────────────────┘
  │ Top-10
  ▼
┌─────────────────────────────────────┐
│         Post-Retrieval              │
│  ┌────────────────────────────────┐ │
│  │  LLM Re-ranking               │ │
│  │  Top-10 → Top-3               │ │
│  └────────────────────────────────┘ │
│  ┌────────────────────────────────┐ │
│  │  Answer Generation             │ │
│  │  Type-specific prompts         │ │
│  │  + Few-shot examples           │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
  │
  ▼
Answer + References
```

## Pre-Retrieval 策略

### 1. Metadata Extraction

從 query 中用 regex 抽取案號和被告姓名：

- **案號**：`109年度新秩字第19號` → 精確匹配文件
- **被告姓名**：`被告李家豪`、`蔡嘉祥散布...` → 匹配 DEFENDANT 欄位

### 2. Query Rewriting

使用 LLM 將口語化問題改寫為法律檢索語句，補充相關法律術語。

例：`有沒有被罰` → `裁判結果 罰鍰 社會秩序維護法 罰金`

### 3. HyDE (Hypothetical Document Embedding)

讓 LLM 生成一段假設性的判決書段落，用這段文字的 embedding 進行檢索。解決 query 與 document 用語差異大的問題。

## Retrieval 策略

### Hybrid Search + Reciprocal Rank Fusion (RRF)

| 檢索信號          | 方式                      | 權重 |
| ----------------- | ------------------------- | ---- |
| Dense (original)  | embeddinggemma cosine sim | 1x   |
| Dense (rewritten) | embeddinggemma cosine sim | 1x   |
| Dense (HyDE)      | embeddinggemma cosine sim | 1x   |
| BM25 (original)   | 字元級 tokenize + BM25    | 1x   |
| BM25 (rewritten)  | 字元級 tokenize + BM25    | 1x   |
| Metadata Match    | 案號/被告名精確匹配       | 2x   |

使用 **RRF** （Cormack et al., 2009）合併所有排序：

$$
RRF(d) = \sum_{r \in rankings} \frac{1}{k + rank_r(d)}
$$

其中 $k = 60$（原始論文實驗測出來的最佳常數）。

### 為什麼用 Hybrid？

- **Dense**：語意相似度，能處理同義詞和改述
- **BM25**：在全文裡搜關鍵字，例如搜尋法律專有名詞（案號、法條、人名）
- **Metadata**：在結構化欄位裡精確比對（因 query 中會提及案號或被告名）

## Post-Retrieval 策略

### 1. LLM Re-ranking

從 Top-10 候選文件中，讓 LLM 根據 query 重新排序，選出最相關的 Top-3。

### 2. Type-Specific Answer Generation

使用 Few-shot prompting，從 Train split 隨機條選一題作為例子，針對三種題型有不同的回答策略：

| 題型                   | 策略                                                 | 評估指標 |
| ---------------------- | ---------------------------------------------------- | -------- |
| **short**        | 從原文擷取實體，使用原始文字，不改寫。法條需完整名稱 | ANLS     |
| **long**         | 一段連貫摘要 (100-200字)，不用標題/條列式            | ROUGE-L  |
| **unanswerable** | 只回答「無法從文件判斷」七個字                       | ANLS     |

### Few-shot Examples (from train set)

```
Q: 謝芯甯散播不實疫情訊息所違反的法條為何？
A: 嚴重特殊傳染性肺炎防治及紓困振興特別條例第14條

Q: 109年度秩字第25號裁定依據社會秩序維護法第幾條第幾項？
A: 社會秩序維護法第四十五條第二項

Q: 被移送人郭華岡在張貼PTT貼文後，是否曾主動向警方說明？
A: 無法從文件判斷
```

## 技術選擇

| 組件            | 選擇                      | 說明                      |
| --------------- | ------------------------- | ------------------------- |
| Embedding Model | embeddinggemma (768d)     | Ollama API, gemma3 family |
| LLM             | Claude (via FastAPI)      | localhost:8899/ask        |
| BM25            | 自製實作                  | CJK 字元級 tokenize       |
| Vector Store    | NumPy in-memory           | 211 篇可全部載入記憶體    |
| Web Framework   | FastAPI + vanilla HTML/JS | 輕量、無前端框架          |

## 評估指標

| 指標              | 適用題型             | 說明                              |
| ----------------- | -------------------- | --------------------------------- |
| **ANLS**    | short + unanswerable | 基於 Levenshtein 距離的字串相似度 |
| **ROUGE-L** | long                 | 基於 LCS 的 F1 score              |
| **nDCG@5**  | 所有題型             | 檢索排序品質 (Top-5)              |

---

## 附錄：BM25 與 RRF 公式說明

### BM25 (Best Match 25)

Robertson 等人 (1994) 在 Okapi 資訊檢索系統中嘗試多種排序公式，第 25 版效果最好，沿用至今。

對 query 中的每個詞 $t$，計算它在文件 $d$ 中的得分後加總：

$$
BM25(q, d) = \sum_{t \in q} IDF(t) \times TF\_norm(t, d)
$$

**IDF — 詞的稀有度**

$$
IDF(t) = \log\frac{N - df(t) + 0.5}{df(t) + 0.5} + 1
$$

- $N$ = 總文件數 (211)
- $df(t)$ = 包含詞 $t$ 的文件數

舉例（$N = 211$）：

| 詞     | $df$ | IDF   | 意義               |
| ------ | ------ | ----- | ------------------ |
| `豪` | 8      | 3.21  | 稀有，鑑別力高     |
| `罰` | 180    | 0.17  | 常見，鑑別力低     |
| `的` | 211    | 0.002 | 到處都有，幾乎無用 |

**TF_norm — 詞頻（有飽和 + 長度懲罰）**

$$
TF\_norm(t, d) = \frac{tf(t,d) \times (k_1 + 1)}{tf(t,d) + k_1 \times (1 - b + b \times \frac{|d|}{avgdl})}
$$

- $tf(t,d)$ = 詞 $t$ 在文件 $d$ 中出現次數
- $|d|$ = 文件 $d$ 的長度，$avgdl$ = 平均文件長度
- $k_1 = 1.5$，$b = 0.75$（標準常數）

詞頻飽和效果（假設文件長度 = 平均）：

| 出現次數 | TF_norm | 說明                   |
| -------- | ------- | ---------------------- |
| 0        | 0       | 沒出現，不得分         |
| 1        | 1.00    | 出現1次 vs 0次差距最大 |
| 2        | 1.40    | 邊際遞減               |
| 5        | 1.83    | 持續遞減               |
| 100      | 2.44    | 接近上限，不會無限增長 |

$b = 0.75$ 的作用：同樣出現 3 次，短文件的得分比長文件高，因為長文件本來就容易包含更多詞。

### RRF (Reciprocal Rank Fusion) — 倒數排名融合

Cormack et al. (2009) 提出，用於合併多個排序結果：

$$
RRF(d) = \sum_{r \in rankings} \frac{1}{k + rank_r(d)}
$$

- $rank_r(d)$ = 文件 $d$ 在排序 $r$ 中的名次（從 1 開始）
- $k = 60$（論文實驗得出的最佳常數）

$k = 60$ 的作用：壓平排名差距。

|        | 無 k：$1/rank$ | 有$k$=60：$1/(60+rank)$ |
| ------ | ---------------- | --------------------------- |
| 第1名  | 1.000            | 0.0164                      |
| 第2名  | 0.500 (差2倍)    | 0.0161 (差1.8%)             |
| 第10名 | 0.100 (差10倍)   | 0.0143 (差13%)              |
