"""Intent decision and fusion module."""

from __future__ import annotations

from typing import Optional

from config.settings import settings
from src.models.intent import IntentResult, IntentSource, Slot, SubIntent


class IntentFusionEngine:
    """
    Dual-engine fusion strategy (2026 production practice):
    - High-confidence lightweight result → adopt directly, LLM validates
    - Low confidence → LLM leads, lightweight weighted in
    - Conflict → take higher confidence, flag for human review
    """

    def __init__(self, fusion_min: float | None = None) -> None:
        self.fusion_min = fusion_min or settings.quality.fusion_confidence_min

    def fuse(
        self,
        lightweight: IntentResult,
        llm: Optional[IntentResult],
    ) -> IntentResult:
        if llm is None:
            lightweight.source = IntentSource.FUSED
            return lightweight

        lw_conf = lightweight.confidence
        llm_conf = llm.confidence

        # Lightweight high-confidence fast path
        if lw_conf >= settings.lightweight_confidence_threshold and lw_conf >= llm_conf:
            fused = self._build_fused(lightweight, llm, lw_weight=0.7, llm_weight=0.3)
        elif llm_conf >= self.fusion_min:
            fused = self._build_fused(llm, lightweight, lw_weight=0.3, llm_weight=0.7)
        else:
            fused = self._build_fused(
                llm if llm_conf >= lw_conf else lightweight,
                lightweight if llm_conf >= lw_conf else llm,
                lw_weight=0.5,
                llm_weight=0.5,
            )

        fused.source = IntentSource.FUSED
        fused.latency_ms = lightweight.latency_ms + (llm.latency_ms if llm else 0)
        fused.implicit_intents = list(dict.fromkeys(
            lightweight.implicit_intents + (llm.implicit_intents if llm else [])
        ))
        fused.sub_intents = self._merge_sub_intents(lightweight, llm)
        fused.slots = self._merge_slots(lightweight, llm)
        fused.drift_detected = llm.drift_detected or lightweight.drift_detected
        fused.drift_type = llm.drift_type if llm.drift_detected else lightweight.drift_type
        fused.resolved_references = {**lightweight.resolved_references, **llm.resolved_references}

        return fused

    def _build_fused(
        self,
        primary: IntentResult,
        secondary: IntentResult,
        lw_weight: float,
        llm_weight: float,
    ) -> IntentResult:
        if primary.source == IntentSource.LIGHTWEIGHT:
            lw, llm_r = primary, secondary
        else:
            lw, llm_r = secondary, primary

        merged_scores = {}
        all_keys = set(lw.raw_scores) | set(llm_r.raw_scores)
        for k in all_keys:
            merged_scores[k] = lw_weight * lw.raw_scores.get(k, 0) + llm_weight * llm_r.raw_scores.get(k, 0)

        best_intent = max(merged_scores, key=merged_scores.get, default=primary.intent)
        fused_conf = merged_scores.get(best_intent, primary.confidence)

        return IntentResult(
            intent=best_intent if merged_scores else primary.intent,
            confidence=min(fused_conf, 0.99),
            source=IntentSource.FUSED,
            raw_scores=merged_scores,
        )

    def _merge_sub_intents(
        self,
        lw: IntentResult,
        llm: Optional[IntentResult],
    ) -> list[SubIntent]:
        if not llm or not llm.sub_intents:
            return lw.sub_intents
        if not lw.sub_intents:
            return llm.sub_intents

        merged: dict[str, SubIntent] = {}
        for si in llm.sub_intents + lw.sub_intents:
            if si.intent not in merged or si.confidence > merged[si.intent].confidence:
                merged[si.intent] = si
        return sorted(merged.values(), key=lambda x: (-x.is_primary, -x.confidence))

    def _merge_slots(
        self,
        lw: IntentResult,
        llm: Optional[IntentResult],
    ) -> dict[str, Slot]:
        slots = dict(lw.slots)
        if llm:
            for name, slot in llm.slots.items():
                if name not in slots or slot.confidence > slots[name].confidence:
                    slots[name] = slot
        return slots
