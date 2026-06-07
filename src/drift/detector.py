"""意图漂移检测 — 多信号融合（SOTA 工业实践）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from config.settings import settings
from src.context.semantic import max_similarity_to_any, semantic_similarity
from src.domain.insurance_domain import get_category_name
from src.drift.category_graph import graph_distance, is_related_switch
from src.models.intent import DriftType, IntentResult, SessionContext

# 显式话题切换标记
TOPIC_SHIFT_MARKERS = [
    "另外", "换个话题", "还想问", "再问", "顺便", "对了",
    "不对", "不是", "算了", "不问了", "换个问题",
]

# 澄清/追问标记 — 非漂移
CLARIFICATION_MARKERS = [
    "什么意思", "不太懂", "再说一遍", "解释", "具体是",
    "为什么", "怎么理解", "能详细", "听不懂",
]

# 延续/指代标记 — 强上下文依赖，非漂移
CONTINUATION_MARKERS = [
    "它", "这个", "那个", "这款", "还有", "呢", "吗",
    "刚才", "上面", "之前", "继续",
]


@dataclass
class DriftSignals:
    """各检测信号分量 — 可解释、可调试。"""
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
    五层融合漂移检测（对齐 SITS/Rasa CALM/Forth AI 工业实践）：
    1. 分类图距离
    2. 话语语义偏移（n-gram 相似度）
    3. 意图描述语义偏移
    4. 主题栈/产品焦点一致性
    5. 显式标记 + LLM 信号融合
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

        if any(m in utterance for m in CONTINUATION_MARKERS) and signals.fused_score < self.threshold + 0.15:
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

        if any(m in utterance for m in TOPIC_SHIFT_MARKERS):
            sig.explicit_marker_boost = 0.35

        if any(m in utterance for m in CONTINUATION_MARKERS):
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
        # 显式换题标记 + 分类跳变 → 高置信漂移（工业规则增强）
        if sig.explicit_marker_boost > 0 and sig.category_distance >= 0.3:
            score = max(score, 0.58)
        if sig.llm_drift_signal > 0 and sig.category_distance >= 0.2:
            score = max(score, 0.62)
        return max(0.0, min(score, 1.0))

    @staticmethod
    def _is_clarification(utterance: str) -> bool:
        return any(m in utterance for m in CLARIFICATION_MARKERS)

    @staticmethod
    def _is_return_to_prior_topic(ctx: SessionContext, new_category: str) -> bool:
        if new_category in ctx.category_history[-4:-1]:
            return True
        return False

    @staticmethod
    def _build_reason(ctx: SessionContext, result: IntentResult, signals: DriftSignals) -> str:
        from_name = get_category_name(ctx.active_category or "")
        to_name = get_category_name(result.category)
        parts = [f"从「{ctx.active_intent_label}」({from_name}) 切换到「{result.intent_label}」({to_name})"]
        if signals.explicit_marker_boost > 0:
            parts.append("含显式换题标记")
        if signals.product_focus_change > 0:
            parts.append("产品焦点变更")
        if signals.llm_drift_signal > 0:
            parts.append("LLM 判定话题切换")
        parts.append(f"融合分={signals.fused_score:.2f}")
        return "；".join(parts)
