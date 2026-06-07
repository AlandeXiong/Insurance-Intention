"""多意图分解模块 — 目标准确率 ≥88%。"""

from __future__ import annotations

import re
from typing import List, Set

from src.domain.insurance_intents import INSURANCE_INTENTS, KEYWORD_INTENT_RULES
from src.models.intent import IntentResult, SubIntent


class MultiIntentDecomposer:
    """
    将复合用户输入分解为多个子意图。
    策略：标点/连接词切分 + 独立意图评分 + 去重合并。
    """

    SPLIT_PATTERN = re.compile(r"[，,；;？?！!]|(?:还有|另外|以及|顺便|同时|并且)")

    def decompose(self, utterance: str, primary_result: IntentResult) -> List[SubIntent]:
        segments = self._split_utterance(utterance)
        if len(segments) <= 1:
            return self._single_intent_result(primary_result)

        candidates: List[SubIntent] = []
        seen: Set[str] = set()

        for seg in segments:
            intent, conf = self._score_segment(seg.strip())
            if intent != "fallback" and intent not in seen:
                seen.add(intent)
                candidates.append(SubIntent(
                    intent=intent,
                    confidence=conf,
                    is_primary=(intent == primary_result.intent),
                ))

        if not candidates:
            return self._single_intent_result(primary_result)

        # 确保主意图标记
        has_primary = any(c.is_primary for c in candidates)
        if not has_primary:
            candidates[0].is_primary = True

        candidates.sort(key=lambda x: (-x.is_primary, -x.confidence))
        return candidates

    def merge_into_result(self, result: IntentResult, utterance: str) -> IntentResult:
        result.sub_intents = self.decompose(utterance, result)
        return result

    def _split_utterance(self, utterance: str) -> List[str]:
        parts = self.SPLIT_PATTERN.split(utterance)
        return [p for p in parts if p.strip()]

    def _score_segment(self, segment: str) -> tuple[str, float]:
        best_intent = "fallback"
        best_conf = 0.0

        for keywords, intent, base_conf in KEYWORD_INTENT_RULES:
            hits = sum(1 for kw in keywords if kw in segment)
            if hits > 0:
                conf = min(base_conf + 0.02 * hits, 0.95)
                if conf > best_conf:
                    best_conf = conf
                    best_intent = intent

        for name, idef in INSURANCE_INTENTS.items():
            hits = sum(1 for kw in idef.keywords if kw in segment)
            if hits > 0:
                conf = min(0.78 + 0.04 * hits, 0.92)
                if conf > best_conf:
                    best_conf = conf
                    best_intent = name

        return best_intent, best_conf

    @staticmethod
    def _single_intent_result(result: IntentResult) -> List[SubIntent]:
        return [SubIntent(
            intent=result.intent,
            confidence=result.confidence,
            is_primary=True,
        )]
