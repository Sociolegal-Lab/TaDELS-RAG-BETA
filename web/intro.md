# TaDELS RAG：與法律文件對話

本系統旨在建構一套結合 Retrieval-Augmented Generation（RAG）技術的**問答系統**，讓法律研究者、法律專業人員及對法律文件有興趣的一般民眾，能透過 **AI 快速檢索**臺灣**法律文件資料庫**中的裁判書、法規與行政處分，定位所需資訊，並取得彙整後的回應。

TaDELS RAG 處於**開發測試階段**，已收錄 **211 篇 COVID-19 相關台灣法院裁判文書**作為第一批測試資料；未來會在此基礎上擴展至更廣的法律文書範圍。

涵蓋的測試文書類型包含：社會秩序維護法裁定、傳染病防治法判決、特別條例判決等。

> 向下捲動可查看評估資料、系統實際表現，以及完整的 RAG 設計細節。

<!--END-HERO-->

<details open>
<summary><strong>評估資料：測試 RAG 系統表現所用的資料集</strong></summary>

本節呈現開發階段所使用的測試資料集。為了驗證 RAG pipeline 在不同題型 (short / long / unanswerable) 下的表現，我們以 211 篇 COVID-19 相關裁判文書為基礎，整理出 668 題 QA 與 54 種結構化欄位，建立可重複測量的評估環境。

| 項目          | 數量                                             |
| ------------- | ------------------------------------------------ |
| 法律文件      | 211 篇                                           |
| Single-hop QA | 650 題 (short 212 / long 225 / unanswerable 213) |
| Multi-hop QA  | 18 題                                            |
| 結構化欄位    | 54 欄位 (7 大類)                                 |

#### 結構化欄位類別

| 類別           | 欄位數 | 內容                                   |
| -------------- | -----: | -------------------------------------- |
| 案件基本資訊   |     29 | 文書類型、案號、被告、法官、行為時地等 |
| 構成要件分析   |      8 | 訊息不實性、公共危害、故意認定等       |
| 競合罪名分析   |      5 | 競合類型、誹謗、恐嚇、個資法認定等     |
| 民事侵權分析   |      6 | 侵權類型、名譽損害、損害賠償等         |
| 程序歷程       |      3 | 程序階段、前/後階段文書                |
| 程序與前案資訊 |      2 | 前科/累犯、累犯加重認定                |
| 援引法源       |      1 | 援引法源彙整                           |

### 資料切分

| 集合      | 題數 | short | long | unanswerable |
| --------- | ---- | ----- | ---- | ------------ |
| Train     | 454  | 148   | 157  | 149          |
| Val       | 98   | 32    | 34   | 32           |
| Test      | 98   | 32    | 34   | 32           |
| Multi-hop | 18   | —    | —   | —           |

### 探索系統表現

<div class="nav-cards">
<a class="nav-card" href="/results">
<div class="nav-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><line x1="3" y1="20" x2="21" y2="20"/><rect x="5" y="12" width="3" height="8" rx="0.5"/><rect x="10.5" y="6" width="3" height="14" rx="0.5"/><rect x="16" y="9" width="3" height="11" rx="0.5"/></svg></div>
<div class="nav-title">Results</div>
<div class="nav-desc">每題評分與整體表現分數彙整</div>
<div class="arrow-cue">→</div>
</a>
<a class="nav-card" href="/qa">
<div class="nav-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="3"/><path d="M9 10a3 3 0 0 1 5.5-1.5c.8 1.5-.5 2.7-2 3.3V14"/><circle cx="12.5" cy="17" r="0.9" fill="currentColor" stroke="none"/></svg></div>
<div class="nav-title">QA Viewer</div>
<div class="nav-desc">瀏覽問題、檢索文件與系統回答</div>
<div class="arrow-cue">→</div>
</a>
<a class="nav-card" href="/entities">
<div class="nav-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><path d="M3 12V5a2 2 0 0 1 2-2h7l9 9-9 9-9-9z"/><circle cx="8" cy="8" r="1.6" fill="currentColor" stroke="none"/></svg></div>
<div class="nav-title">Entities</div>
<div class="nav-desc">結構化欄位 Type A / B / C 比對</div>
<div class="arrow-cue">→</div>
</a>
<a class="nav-card" href="/chat">
<div class="nav-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round"><path d="M3 5h18v12H8l-5 4z"/><circle cx="9" cy="11" r="0.9" fill="currentColor" stroke="none"/><circle cx="13" cy="11" r="0.9" fill="currentColor" stroke="none"/><circle cx="17" cy="11" r="0.9" fill="currentColor" stroke="none"/></svg></div>
<div class="nav-title">Chat</div>
<div class="nav-desc">即時提問互動式問答</div>
<div class="arrow-cue">→</div>
</a>
</div>

</details>

<details>
<summary><strong>RAG 設計細節</strong></summary>

### Pipeline 架構

<div class="pipeline-placeholder"></div>

### Pre-Retrieval 策略

#### 1. Metadata Extraction

從 query 中用 regex 抽取案號和被告姓名：

- **案號**：`109年度新秩字第19號` → 精確匹配文件
- **被告姓名**：`被告李家豪`、`蔡嘉祥散布...` → 匹配 DEFENDANT 欄位

#### 2. Query Rewriting

使用 LLM 將口語化問題改寫為法律檢索語句，補充相關法律術語。

例：`有沒有被罰` → `裁判結果 罰鍰 社會秩序維護法 罰金`

#### 3. HyDE (Hypothetical Document Embedding)

讓 LLM 生成一段假設性的判決書段落，用這段文字的 embedding 進行檢索。解決 query 與 document 用語差異大的問題。

### Retrieval 策略

#### Hybrid Search + Reciprocal Rank Fusion (RRF)

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

#### 為什麼用 Hybrid？

- **Dense**：語意相似度，能處理同義詞和改述
- **BM25**：在全文裡搜關鍵字，例如搜尋法律專有名詞（案號、法條、人名）
- **Metadata**：在結構化欄位裡精確比對（因 query 中會提及案號或被告名）

### Post-Retrieval 策略

#### LLM Re-ranking

從 Top-10 候選文件中，讓 LLM 根據 query 重新排序，選出最相關的 Top-3。

### Generation 策略

#### Type-Specific Answer Generation

使用 Few-shot prompting，從 Train split 隨機條選一題作為例子，針對三種題型有不同的回答策略：

| 題型                   | 策略                                                 |
| ---------------------- | ---------------------------------------------------- |
| **short**        | 從原文擷取實體，使用原始文字，不改寫。法條需完整名稱 |
| **long**         | 一段連貫摘要 (100-200字)，不用標題/條列式            |
| **unanswerable** | 只回答「無法從文件判斷」七個字                       |

#### Few-shot Examples (from train set)

```
Q: 謝芯甯散播不實疫情訊息所違反的法條為何？
A: 嚴重特殊傳染性肺炎防治及紓困振興特別條例第14條

Q: 109年度秩字第25號裁定依據社會秩序維護法第幾條第幾項？
A: 社會秩序維護法第四十五條第二項

Q: 被移送人郭華岡在張貼PTT貼文後，是否曾主動向警方說明？
A: 無法從文件判斷
```

### 技術選擇

| 組件            | 選擇                                          |
| --------------- | --------------------------------------------- |
| Embedding Model | embeddinggemma (768d)                         |
| LLM             | Claude Sonnet / Gemma4:31b                    |
| BM25            | 字元級 tokenize（待優化）                     |
| Vector Store    | NumPy in-memory（211 篇可全部載入記憶體）     |
| Web Framework   | FastAPI + vanilla HTML/JS（輕量、無前端框架） |

### 評估指標

| 指標                         | 適用題型     | 說明                                             |
| ---------------------------- | ------------ | ------------------------------------------------ |
| **Entity Matching**    | short + long | 結構化欄位逐欄比對，整合 Type A/B/C 三種比對方式 |
| **Exact Match**        | unanswerable | 是否準確回答「無法從文件判斷」                   |
| **Hallucination Rate** | short + long | 答案中無法被檢索文件支持的虛構內容比例           |
| **nDCG@5**             | 所有題型     | 檢索排序品質 (Top-5)                             |

#### Entity Matching 子分數

| 類型             | 適用欄位                    | 比對方式                             |
| ---------------- | --------------------------- | ------------------------------------ |
| **Type A** | 數值/年份/案號等精確欄位    | 數值正規化後完全比對                 |
| **Type B** | 集合/多值欄位（如多名被告） | 集合 fuzzy match，計算交集比例       |
| **Type C** | 文字敘述欄位（如裁判結果）  | embedding cosine similarity 語意比對 |

### 附錄：BM25 與 RRF 公式說明

#### BM25 (Best Match 25)

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

#### RRF (Reciprocal Rank Fusion)

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

</details>
