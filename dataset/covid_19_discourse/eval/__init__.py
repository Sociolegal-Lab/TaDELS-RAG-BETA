"""
評估模組

子模組：
  - entities: Entity Matching Score（A/B/C 三類）
  - hallucination: 幻覺懲罰（獨立呈現）
  - ndcg: 檢索品質 nDCG@k
"""

from .entities import entity_matching_score, score_entity
from .hallucination import hallucination_penalty
from .ndcg import ndcg_at_k, evaluate_retrieval
