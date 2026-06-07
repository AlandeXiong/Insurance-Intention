"""Multi-intent decomposition module — target accuracy ≥88%."""

from __future__ import annotations

import re
from typing import List, Set

from src.domain.insurance_intents import INSURANCE_INTENTS, KEYWORD_INTENT_RULES
from src.models.intent import IntentResult, SubIntent


class MultiIntentDecomposer:
    """
    Decompose compound user input into multiple sub-intents.
    Strategy: punctuation/conjunction split + per-segment intent scoring + dedupe merge.
    """

    SPLIT_PATTERN = re.compile(
        r"[,;?!]|(?:\band\b|\balso\b|\bplus\b|\bfurthermore\b|\bbesides\b|\bmeanwhile\b|\bin addition\b|\bas well\b)",
        re.IGNORECASE,
    )

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

        # Ensure primary intent is marked
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
        segment_lower = segment.lower()

        for keywords, intent, base_conf in KEYWORD_INTENT_RULES:
            hits = sum(1 for kw in keywords if kw.lower() in segment_lower)
            if hits > 0:
                conf = min(base_conf + 0.02 * hits, 0.95)
                if conf > best_conf:
                    best_conf = conf
                    best_intent = intent

        for name, idef in INSURANCE_INTENTS.items():
            hits = sum(1 for kw in idef.keywords if kw.lower() in segment_lower)
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
