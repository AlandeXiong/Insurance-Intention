"""Intent drift detection — multi-signal fusion (production SOTA practice)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from config.settings import settings
from src.context.semantic import max_similarity_to_any, semantic_similarity
from src.domain.insurance_domain import get_category_name
from src.drift.category_graph import graph_distance, is_related_switch
from src.models.intent import DriftType, IntentResult, SessionContext

# Explicit topic-shift markers
TOPIC_SHIFT_MARKERS = [
    "by the way", "change topic", "also want to ask", "another question", "incidentally",
    "actually no", "not that", "never mind", "forget it", "different question",
    "on another note", "switching topics", "also tell me", "also want to know",
    "another thing", "one more thing", "separately",
]

# Clarification / follow-up markers — not drift
CLARIFICATION_MARKERS = [
    "what do you mean", "don't understand", "say again", "explain", "specifically",
    "why", "how to understand", "more detail", "don't get it",
]

# Continuation / reference markers — strong context dependency, not drift
CONTINUATION_MARKERS = [
    "it", "this", "that", "this plan", "and also", "right",
    "just now", "above", "before", "continue", "what about",
]


@dataclass
class DriftSignals:
    """Per-signal drift components — interpretable and debuggable."""
    category_distance: float = 0.0
    utterance_semantic_shift: float = 0.0
    intent_label_shift: float = 0.0
    topic_stack_divergence: float = 0.0
    product_focus_change: float = 0.0
    explicit_marker_boost: float = 0.0
    continuation_penalty: float = 0.0
    llm_drift_signal: float = 0.0
    fused_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "category_distance": self.category_distance,
            "utterance_semantic_shift": self.utterance_semantic_shift,
            "intent_label_shift": self.intent_label_shift,
            "topic_stack_divergence": self.topic_stack_divergence,
            "product_focus_change": self.product_focus_change,
            "explicit_marker_boost": self.explicit_marker_boost,
            "continuation_penalty": self.continuation_penalty,
            "llm_drift_signal": self.llm_drift_signal,
            "fused_score": self.fused_score,
        }


class IntentDriftDetector:
    """
    Five-layer fused drift detection (aligned with SITS / Rasa CALM / Forth AI practice):
    1. Category graph distance
    2. Utterance semantic shift (n-gram similarity)
    3. Intent label semantic shift
    4. Topic stack / product focus consistency
    5. Explicit markers + LLM signal fusion
    """

    DEFAULT_WEIGHTS = {
        "category_distance": 0.25,
        "utterance_semantic_shift": 0.25,
        "intent_label_shift": 0.15,
        "topic_stack_divergence": 0.10,
        "product_focus_change": 0.10,
        "explicit_marker_boost": 0.10,
        "llm_drift_signal": 0.15,
    }

    def __init__(
        self,
        threshold: float | None = None,
        weights: Dict[str, float] | None = None,
    ) -> None:
        self.threshold = threshold or settings.drift_similarity_threshold
        self.weights = weights or self.DEFAULT_WEIGHTS

    def detect(
        self,
        ctx: SessionContext,
        result: IntentResult,
        utterance: str,
    ) -> Tuple[bool, DriftType, float, DriftSignals]:
        new_category = result.category
        signals = self._compute_signals(ctx, result, utterance)

        if not ctx.active_category or ctx.active_category == new_category:
            return False, DriftType.NONE, signals.fused_score, signals

        if ctx.active_category in ("greeting_chitchat", "other"):
            return False, DriftType.NONE, signals.fused_score, signals

        if ctx.pending_clarification.active:
            return False, DriftType.CLARIFICATION, signals.fused_score, signals

        if is_related_switch(ctx.active_category, new_category):
            drift_type = DriftType.CLARIFICATION if self._is_clarification(utterance) else DriftType.SUB_INTENT_SWITCH
            return False, drift_type, signals.fused_score, signals

        if any(m in utterance.lower() for m in CONTINUATION_MARKERS) and signals.fused_score < self.threshold + 0.15:
            return False, DriftType.SUB_INTENT_SWITCH, signals.fused_score, signals

        drifted = signals.fused_score >= self.threshold
        if not drifted:
            return False, DriftType.NONE, signals.fused_score, signals

        if self._is_clarification(utterance):
            drift_type = DriftType.CLARIFICATION
        elif self._is_return_to_prior_topic(ctx, new_category):
            drift_type = DriftType.SUB_INTENT_SWITCH
            drifted = False
        else:
            drift_type = DriftType.TOPIC_SHIFT

        return drifted, drift_type, signals.fused_score, signals

    def annotate(self, result: IntentResult, ctx: SessionContext, utterance: str) -> IntentResult:
        drifted, drift_type, score, signals = self.detect(ctx, result, utterance)

        if result.drift_detected and result.drift_reason:
            drifted = drifted or result.drift_detected
            if result.drift_type != DriftType.NONE:
                drift_type = result.drift_type

        result.drift_detected = drifted
        result.drift_type = drift_type if drifted or drift_type == DriftType.SUB_INTENT_SWITCH else DriftType.NONE
        if drifted and not result.drift_reason:
            result.drift_reason = self._build_reason(ctx, result, signals)
        result.raw_scores["drift_score"] = score
        result.raw_scores.update({f"drift_{k}": v for k, v in signals.to_dict().items()})
        return result

    def _compute_signals(
        self, ctx: SessionContext, result: IntentResult, utterance: str
    ) -> DriftSignals:
        sig = DriftSignals()

        if ctx.active_category:
            sig.category_distance = graph_distance(ctx.active_category, result.category)

        prev_user = ctx.get_recent_user_utterances(1)
        if prev_user:
            sig.utterance_semantic_shift = 1.0 - semantic_similarity(utterance, prev_user[0])

        if ctx.active_intent_label:
            sig.intent_label_shift = 1.0 - semantic_similarity(
                result.intent_label, ctx.active_intent_label
            )

        if ctx.category_history:
            if result.category in ctx.category_history[-4:-1]:
                sig.topic_stack_divergence = 0.15
            elif ctx.topic_stack:
                sig.topic_stack_divergence = 1.0 - max_similarity_to_any(
                    result.intent_label, ctx.topic_stack[-3:]
                )
            else:
                sig.topic_stack_divergence = 0.5

        new_product = result.slots.get("product_name") or result.slots.get("product")
        new_product_val = new_product.value if new_product else None
        if ctx.active_product and new_product_val and str(new_product_val) != ctx.active_product:
            sig.product_focus_change = 0.8

        utterance_lower = utterance.lower()
        if any(m in utterance_lower for m in TOPIC_SHIFT_MARKERS):
            sig.explicit_marker_boost = 0.35

        if any(m in utterance_lower for m in CONTINUATION_MARKERS):
            sig.continuation_penalty = 0.25

        if result.drift_detected:
            sig.llm_drift_signal = 0.85

        sig.fused_score = self._fuse(sig)
        return sig

    def _fuse(self, sig: DriftSignals) -> float:
        w = self.weights
        score = (
            w["category_distance"] * sig.category_distance
            + w["utterance_semantic_shift"] * sig.utterance_semantic_shift
            + w["intent_label_shift"] * sig.intent_label_shift
            + w["topic_stack_divergence"] * sig.topic_stack_divergence
            + w["product_focus_change"] * sig.product_focus_change
            + w["explicit_marker_boost"] * sig.explicit_marker_boost
            + w["llm_drift_signal"] * sig.llm_drift_signal
            - sig.continuation_penalty
        )
        # Explicit topic-shift marker + category jump → high-confidence drift (production rule boost)
        if sig.explicit_marker_boost > 0 and sig.category_distance >= 0.3:
            score = max(score, 0.58)
        if sig.llm_drift_signal > 0 and sig.category_distance >= 0.2:
            score = max(score, 0.62)
        return max(0.0, min(score, 1.0))

    @staticmethod
    def _is_clarification(utterance: str) -> bool:
        utterance_lower = utterance.lower()
        return any(m in utterance_lower for m in CLARIFICATION_MARKERS)

    @staticmethod
    def _is_return_to_prior_topic(ctx: SessionContext, new_category: str) -> bool:
        if new_category in ctx.category_history[-4:-1]:
            return True
        return False

    @staticmethod
    def _build_reason(ctx: SessionContext, result: IntentResult, signals: DriftSignals) -> str:
        from_name = get_category_name(ctx.active_category or "")
        to_name = get_category_name(result.category)
        parts = [f"Switched from \"{ctx.active_intent_label}\" ({from_name}) to \"{result.intent_label}\" ({to_name})"]
        if signals.explicit_marker_boost > 0:
            parts.append("contains explicit topic-shift marker")
        if signals.product_focus_change > 0:
            parts.append("product focus changed")
        if signals.llm_drift_signal > 0:
            parts.append("LLM detected topic shift")
        parts.append(f"fused score={signals.fused_score:.2f}")
        return "; ".join(parts)
