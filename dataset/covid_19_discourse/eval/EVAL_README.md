# 評估系統說明文件

## 用法

```bash
python eval/eval.py --results_file results/predictions_val.json
```

可選參數（皆有預設值）：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--results_file` | （必填） | 系統輸出 JSON |
| `--gt_file` | `../qa_dataset_final_v4.json` | Ground Truth |
| `--entity_file` | `../dataset_entities_v4.json` | 文件 entity 資料 |
| `--schema_file` | `../entity_schema.json` | 欄位評估類型定義 |
| `--case_links_file` | `../case_links.json` | 案件關聯表 |
| `--full_content_dir` | `../../full_content` | 文件原文目錄 |
| `--output` | `{results_file}_eval.json` | 評分明細輸出路徑 |

---

## 評估流程總覽

每一題依題型走不同流程：

```
題目
 ├─ unanswerable → Exact Match + nDCG@5
 └─ short / long → Entity Matching Score + Answer Fallback + 幻覺懲罰 + nDCG@5
```

### 判斷邏輯

1. **Entity Matching Score**：逐欄位比對 Pred entities vs GT entities
2. **Answer Fallback**：僅在 entity_score = 0（實體提取完全失效）時觸發，比對整段 answer 文字
3. **幻覺懲罰**：獨立計算，檢查 Pred 多出的 entity 是否有文件支持
4. **nDCG@5**：所有題型都計算，評估檢索品質

---

## 評估指標

系統輸出 5 個指標，各自獨立計算：

### 指標一：Entity Matching Score（主分數）

衡量系統輸出的結構化 entity 是否與 Ground Truth 一致。

#### 計算流程

```
若 entity_score > 0：final_score = entity_score
若 entity_score = 0：final_score = answer_score（fallback 觸發）
若 entity_score = None：final_score = answer_score（GT 無 entity）
```

其中：

**entity_score** = 各欄位分數的平均：

```
entity_score = (1/N) × Σ s_i ,  N = GT entity 欄位數
```

每個欄位的分數 s_i 依 entity_schema.json 定義的評估類型決定：

**Type A — 分類標籤型（Exact Match）**

適用欄位：VERDICT_CATEGORY, SPREAD_TYPE, OBJ_FALSITY 等固定分類標籤。

```
s_i = 1  if extract_label(normalize(pred)) == extract_label(normalize(gt))
s_i = 0  otherwise
```

- `extract_label`：取冒號前的標籤（「肯定：理由...」→「肯定」）

**Type B — 短值型（標準化 Exact Match / F1）**

適用欄位：SANCTION, COURT, DEFENDANT, LAW, ACT_TIME 等具體名稱、數字、日期、法條。

純量比對：
```
s_i = 1  if normalize(pred) == normalize(gt)         # exact match
s_i = 1  if normalize(pred) ⊂ normalize(gt) 或反向    # substring containment
s_i = 0  otherwise
```

List 比對（任一方為 list，或字串含「；」「、」自動拆分）：
```
s_i = F1 = 2PR / (P + R)

P = |matched_pred| / |pred_set|
R = |matched_gt|   / |gt_set|
```

匹配判定使用 fuzzy matching（exact 或 substring containment）。

**Type C — 長文型（Embedding Cosine Similarity）**

適用欄位：ACT_TYPE, RUMOR_CONTENT, RATIONALE 等描述性長文。

```
s_i = cosine_similarity(embed(pred), embed(gt))
```

使用 embeddinggemma 模型計算向量。

**OTHER 欄位（特殊處理）**

OTHER 欄位先做 normalize 後子字串包含比對，命中直接 1.0；沒命中才 fallback 到 embedding cosine similarity。

```
s_i = 1.0  if normalize(pred) ⊂ normalize(gt) 或反向
s_i = cosine_similarity(embed(pred), embed(gt))  otherwise
```

**answer_score（fallback，僅 entity_score = 0 時觸發）**：

Step 1 — 整段 answer 比對：
```
answer_exact = 1.0  if normalize(pred_answer) == normalize(gt_answer)
answer_exact = 1.0  if substring containment（任一方包含另一方）
answer_exact = 0.0  otherwise
```

Step 2 — 逐 entity 欄位在 answer 中查找：
```
對每個 GT entity 欄位：
  若 normalize(gt_val) 出現在 normalize(pred_answer) 中 → 1.0
  若 Type A/B → 0.0
  若 Type C → cosine_similarity(embed(gt_val), embed(pred_answer))

answer_entity_score = 各欄位平均
```

最終：
```
answer_score = max(answer_exact, answer_entity_score)
```

#### normalize_text 標準化規則

比對前統一執行以下標準化（依序）：

1. 平台名稱統一（臉書 → Facebook, line → LINE 等）
2. 全形數字轉半形（０１２ → 012）
3. 去除空白
4. 去除尾部中文標點（。，、；！？）
5. 去除引號（「」『』）
6. 統一用字（新台幣 → 新臺幣）
7. 去除數字逗號（4,000 → 4000）
8. 法條國字轉阿拉伯（第六十三條 → 第63條）
9. 金額國字轉阿拉伯（新臺幣伍仟元 → 新臺幣5000元），支援混合形式（5千 → 5000）
10. 刑期國字轉阿拉伯（拘役參拾伍日 → 拘役35日）
11. 去除「民國」前綴（民國109年 → 109年）
12. 日期去除時分秒（109年2月20日凌晨1時許 → 109年2月20日）

---

### 指標二：Unanswerable Exact Match

針對 type=unanswerable 的題目，檢查系統是否正確回答「無法從文件判斷」。

```
EM = 1  if pred_answer.strip() == gt_answer.strip()
EM = 0  otherwise
```

---

### 指標三：Hallucination Penalty（P_h）

衡量系統是否輸出了 GT 沒有標注的 extra entity，獨立呈現，**不併入主分數**。

#### 計算流程

```
extra_fields = pred 的 entity 欄位 - GT 的 entity 欄位
```

每個 extra field 分類：

| 分類 | 判定邏輯 | 懲罰 |
|------|---------|------|
| supported | entity 資料有該欄位且值一致，或文件原文包含該值 | 0 |
| unsupported | entity 資料無該欄位，且文件原文也找不到該值 | α = 0.05 |
| contradictory | entity 資料有該欄位但值完全不一致 | β = 0.15 |

判定優先順序：
1. 先查 dataset_entities_v4.json 的結構化 entity 資料
2. 若該欄位不存在，fallback 查文件原文（full_content/*.txt）
3. 文件原文也找不到 → unsupported

```
P_h(q) = min(γ, Σ penalty(e))  ,  γ = 0.50（上限）
```

---

### 指標四：Hallucination Rate（幻覺率）

更直覺的幻覺程度指標，獨立呈現，**不併入主分數**。

```
hallucination_rate = (unsupported 數 + contradictory 數) / pred 總 entity 數
```

- 0 = 沒有幻覺
- 0.2 = 20% 的輸出 entity 有幻覺疑慮
- 1.0 = 全部都是幻覺

---

### 指標五：nDCG@5（檢索品質）

衡量檢索模組回傳的文件排序品質，使用 binary relevance。

```
nDCG@k = DCG@k / IDCG@k

DCG@k  = Σ rel(i) / log₂(i + 1)  ,  i = 1..k
IDCG@k = 最佳排序的 DCG@k
```

- `rel(i) = 1` 如果第 i 筆檢索結果是相關文件，否則 0
- 相關文件來源：GT 的 doc_id 或 ref_doc_id，加上 case_links 擴展的同案件文件

---

## 輸出格式

### 終端輸出

```
========== 評估結果 ==========
總題數:              101
Unanswerable:        33 題, 平均 EM = 1.0000
Entity Matching:     68 題, 平均 = 0.9110
  Entity Score:      平均 = 0.9110
  Answer Fallback:   平均 = 0.7500
幻覺懲罰:            平均 P_h = 0.0167
幻覺率:              平均 = 0.0200
nDCG@5:              平均 = 0.9478
================================
```

### JSON 輸出（_eval.json）

```json
{
  "summary": {
    "total_questions": 101,
    "unanswerable_count": 33,
    "answerable_count": 68,
    "avg_unanswerable_em": 1.0,
    "avg_entity_matching_score": 0.9110,
    "avg_entity_score": 0.9110,
    "avg_answer_fallback_score": 0.7500,
    "avg_hallucination_penalty": 0.0167,
    "avg_hallucination_rate": 0.02,
    "avg_ndcg@5": 0.9478
  },
  "details": [
    {
      "question_id": "...",
      "type": "short",
      "entity_matching_score": 1.0,
      "entity_score": 1.0,
      "answer_fallback_score": 1.0,
      "entity_detail": {"SANCTION": 1.0, "VERDICT_CATEGORY": 1.0},
      "answer_detail": {"SANCTION": 0.0, "VERDICT_CATEGORY": 0.0},
      "hallucination_penalty": 0.0,
      "hallucination_rate": 0.0,
      "hallucination_detail": {},
      "ndcg@5": 1.0
    }
  ]
}
```

---

## 載入資料

| 資料 | 來源 | 用途 |
|------|------|------|
| Ground Truth | qa_dataset_final_v4.json | 每題的正確答案、entities、doc_id |
| Entity 資料 | dataset_entities_v4.json | 每份文件的完整結構化 entity（幻覺檢測用） |
| Entity Schema | entity_schema.json | 定義每個欄位的評估類型（A/B/C） |
| Case Links | case_links.json | 同案件文件關聯（nDCG 擴展 relevant set） |
| 文件原文 | full_content/*.txt | 幻覺檢測 fallback（entity 資料查不到時查原文） |
| Doc Titles | dataset_entities_v4.json 的 title 欄位 | doc_id → 檔名對照，用於載入文件原文 |

---

## 更動紀錄

### v4 更新

1. **Answer Fallback 邏輯調整**：改為僅在 entity_score = 0 時觸發（原本為 max(entity_score, answer_score)）
2. **OTHER 欄位特殊處理**：先做 normalize 後子字串包含比對，沒命中才走 embedding
3. **normalize_text 標準化增強**：去除尾部中文標點、引號、「民國」前綴，金額/刑期國字轉阿拉伯數字
4. **_cn_number_to_int 混合數字支援**：支援阿拉伯數字與國字混合（5千 → 5000）
5. **score_type_b 比對邏輯增強**：自動拆分「；」「、」分隔字串、substring containment、fuzzy F1
6. **幻覺檢測加入文件原文 fallback**：entity 資料查不到時查 full_content/*.txt
7. **新增幻覺率指標 hallucination_rate**
8. **資料檔名更新**：v3 → v4，schema 從 csv 改為 json
