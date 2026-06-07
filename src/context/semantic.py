"""轻量级语义相似度 — 无 embedding 模型依赖的字符 n-gram 向量。"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", "", text)
    return text


def text_to_ngram_vector(text: str, n: int = 2) -> Dict[str, float]:
    text = normalize_text(text)
    if not text:
        return {}
    grams: Dict[str, float] = {}
    for i in range(len(text) - n + 1):
        gram = text[i : i + n]
        grams[gram] = grams.get(gram, 0) + 1.0
    norm = math.sqrt(sum(v * v for v in grams.values())) or 1.0
    return {k: v / norm for k, v in grams.items()}


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    return max(0.0, min(dot, 1.0))


def semantic_similarity(text_a: str, text_b: str, n: int = 2) -> float:
    return cosine_similarity(text_to_ngram_vector(text_a, n), text_to_ngram_vector(text_b, n))


def max_similarity_to_any(text: str, candidates: Iterable[str]) -> float:
    return max((semantic_similarity(text, c) for c in candidates if c), default=0.0)
