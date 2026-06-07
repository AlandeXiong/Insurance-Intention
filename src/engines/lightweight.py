"""Frontend lightweight intent engine — target latency ≤150ms."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Dict, List, Optional, Tuple

from config.settings import settings
from src.domain.insurance_intents import (
    IMPLICIT_INTENT_RULES,
    INSURANCE_INTENTS,
    KEYWORD_INTENT_RULES,
)
from src.models.intent import IntentResult, IntentSource, SessionContext


class LightweightIntentEngine:
    """
    Dual-path lightweight engine:
    1. Keyword rule matching (deterministic, high confidence)
    2. Character n-gram TF-IDF similarity (fallback)
    """

    def __init__(self, confidence_threshold: float | None = None) -> None:
        self.confidence_threshold = confidence_threshold or settings.lightweight_confidence_threshold
        self._intent_vectors = self._build_intent_vectors()

    def predict(
        self,
        utterance: str,
        ctx: Optional[SessionContext] = None,
    ) -> IntentResult:
        start = time.perf_counter()

        rule_intent, rule_conf, rule_scores = self._rule_match(utterance)
        vector_intent, vector_conf, vector_scores = self._vector_match(utterance)

        if rule_conf >= vector_conf:
            intent, confidence, scores = rule_intent, rule_conf, rule_scores
        else:
            intent, confidence, scores = vector_intent, vector_conf, vector_scores

        # Context boost: continue prior-turn related intent
        if ctx and ctx.active_intent and confidence < self.confidence_threshold:
            boosted = self._context_boost(utterance, ctx, scores)
            if boosted:
                intent, confidence = boosted

        implicit = self._detect_implicit(utterance)
        latency = (time.perf_counter() - start) * 1000

        return IntentResult(
            intent=intent,
            confidence=confidence,
            source=IntentSource.LIGHTWEIGHT,
            implicit_intents=implicit,
            latency_ms=latency,
            raw_scores=scores,
        )

    def is_confident(self, result: IntentResult) -> bool:
        return result.confidence >= self.confidence_threshold

    def _rule_match(self, utterance: str) -> Tuple[str, float, Dict[str, float]]:
        scores: Dict[str, float] = {}
        best_intent = "fallback"
        best_conf = 0.0

        for keywords, intent, base_conf in KEYWORD_INTENT_RULES:
            hits = sum(1 for kw in keywords if kw in utterance)
            if hits > 0:
                conf = min(base_conf + 0.03 * (hits - 1), 0.98)
                scores[intent] = conf
                if conf > best_conf:
                    best_conf = conf
                    best_intent = intent

        for intent_def in INSURANCE_INTENTS.values():
            kw_hits = sum(1 for kw in intent_def.keywords if kw in utterance)
            if kw_hits > 0:
                conf = min(0.75 + 0.05 * kw_hits, 0.93)
                scores[intent_def.name] = max(scores.get(intent_def.name, 0), conf)
                if scores[intent_def.name] > best_conf:
                    best_conf = scores[intent_def.name]
                    best_intent = intent_def.name

        return best_intent, best_conf, scores

    def _vector_match(self, utterance: str) -> Tuple[str, float, Dict[str, float]]:
        query_vec = self._text_to_vector(utterance)
        scores: Dict[str, float] = {}
        best_intent = "fallback"
        best_conf = 0.0

        for intent, vec in self._intent_vectors.items():
            sim = self._cosine_similarity(query_vec, vec)
            scores[intent] = sim
            if sim > best_conf:
                best_conf = sim
                best_intent = intent

        return best_intent, best_conf, scores

    def _context_boost(
        self,
        utterance: str,
        ctx: SessionContext,
        scores: Dict[str, float],
    ) -> Optional[Tuple[str, float]]:
        active = INSURANCE_INTENTS.get(ctx.active_intent or "")
        if not active:
            return None

        for related in active.related_intents:
            if related in scores:
                boosted_conf = min(scores[related] + 0.15, 0.94)
                if any(kw in utterance for kw in INSURANCE_INTENTS[related].keywords):
                    return related, boosted_conf

        if len(utterance) < 20 and ctx.active_intent:
            return ctx.active_intent, 0.80

        return None

    def _detect_implicit(self, utterance: str) -> List[str]:
        found = []
        for trigger, intent in IMPLICIT_INTENT_RULES.items():
            if trigger in utterance:
                found.append(intent)
        return list(dict.fromkeys(found))

    def _build_intent_vectors(self) -> Dict[str, Dict[str, float]]:
        vectors = {}
        for name, idef in INSURANCE_INTENTS.items():
            corpus = " ".join(idef.keywords + idef.implicit_triggers + [idef.description])
            vectors[name] = self._text_to_vector(corpus)
        return vectors

    @staticmethod
    def _text_to_vector(text: str, n: int = 2) -> Dict[str, float]:
        text = text.lower().strip()
        grams: Dict[str, float] = {}
        for i in range(len(text) - n + 1):
            gram = text[i : i + n]
            grams[gram] = grams.get(gram, 0) + 1.0
        norm = math.sqrt(sum(v * v for v in grams.values())) or 1.0
        return {k: v / norm for k, v in grams.items()}

    @staticmethod
    def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in set(a) | set(b))
        return max(0.0, min(dot, 1.0))
