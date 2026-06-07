"""
多轮对话意图捕捉主管道
架构：LLM 动态意图捕获 + 实体抽取辅助

用户输入 → 上下文管理/指代消解 → LLM 动态意图分析
         → 实体槽位合并 → 漂移补充 → 业务输出
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import settings
from src.clarification.guidance import ClarificationEngine
from src.clarification.resolver import ClarificationResolver
from src.context.manager import ContextManager
from src.domain.insurance_domain import get_category_name
from src.drift.detector import IntentDriftDetector
from src.engines.entity_extractor import EntityExtractor
from src.engines.llm_engine import LLMIntentEngine
from src.models.clarification import ClarificationGuide, PendingClarification
from src.models.intent import IntentResult, SessionContext, Slot


@dataclass
class PipelineResponse:
    session_id: str
    utterance: str
    resolved_utterance: str
    intent: IntentResult
    missing_slots: List[str] = field(default_factory=list)
    should_clarify: bool = False
    clarification: ClarificationGuide = field(default_factory=ClarificationGuide)
    total_latency_ms: float = 0.0
    engine_path: str = "llm_dynamic"
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntentPipeline:
    """LLM 驱动的动态意图识别管道。"""

    def __init__(self) -> None:
        self.context_mgr = ContextManager(max_history_turns=settings.max_history_turns)
        self.llm_engine = LLMIntentEngine()
        self.entity_extractor = EntityExtractor()
        self.drift_detector = IntentDriftDetector()
        self.clarification_engine = ClarificationEngine()
        self.clarification_resolver = ClarificationResolver()
        self._sessions: Dict[str, SessionContext] = {}

    def _engine_path_prefix(self, llm_available: bool) -> str:
        if not llm_available:
            return "rule"
        return settings.llm.provider

    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionContext:
        sid = session_id or str(uuid.uuid4())
        if sid not in self._sessions:
            self._sessions[sid] = self.context_mgr.create_session(sid)
        return self._sessions[sid]

    def _merge_entity_slots(
        self, result: IntentResult, utterance: str, ctx: SessionContext
    ) -> IntentResult:
        """LLM 槽位 + 规则实体抽取合并，LLM 优先。"""
        entities = self.entity_extractor.extract_entities(utterance, ctx)
        if result.category == "product_compare":
            entities.update(self.entity_extractor.extract_compare_products(utterance))

        merged = dict(entities)
        for name, slot in result.slots.items():
            if slot.value is not None:
                merged[name] = slot
            elif name in entities:
                merged[name] = entities[name]

        result.slots = merged
        return result

    def _finalize(
        self,
        ctx: SessionContext,
        utterance: str,
        resolved: str,
        references: dict,
        result: IntentResult,
        start: float,
        engine_path: str,
    ) -> PipelineResponse:
        result.resolved_references = references
        resolve_meta = getattr(ctx, "_clarification_resolve_meta", {}) or {}
        was_clarification_followup = bool(resolve_meta.get("is_clarification_followup"))
        pending_snapshot = ctx.pending_clarification.model_copy(deep=True) if ctx.pending_clarification.active else None

        result = self.drift_detector.annotate(result, ctx, resolved)

        result = self._merge_entity_slots(result, resolved, ctx)

        slot_values = {k: v.value for k, v in result.slots.items() if v.value is not None}
        merged = self.context_mgr.merge_slots(ctx, slot_values, len(ctx.turns))
        missing = result.missing_info or []

        clarification = self.clarification_engine.evaluate(
            result, utterance, ctx, llm_clarification=result.llm_clarification
        )

        # 澄清回复后 refinement：合并原始意图 + 用户补充
        if was_clarification_followup and pending_snapshot:
            result, refinement = self.clarification_resolver.refine_intent_after_clarification(
                result, pending_snapshot, utterance, resolve_meta
            )
            clarification = refinement
            engine_path = f"{self._engine_path_prefix(True)}_clarification_refined"
            ctx.pending_clarification = PendingClarification()
        elif clarification.needs_clarification:
            ctx.pending_clarification = self.clarification_resolver.create_pending_state(
                clarification, result, utterance, len(ctx.turns)
            )
        else:
            ctx.pending_clarification = PendingClarification()

        should_clarify = clarification.needs_clarification

        self.context_mgr.add_turn(
            ctx, "user", utterance,
            intent_label=result.intent_label,
            category=result.category,
            slots=merged,
            resolved_content=resolved,
        )
        self.context_mgr.update_active_state(ctx, result.intent_label, result.category, merged)

        if should_clarify and clarification.guide_response:
            self.context_mgr.add_turn(ctx, "assistant", clarification.guide_response)

        total_ms = (time.perf_counter() - start) * 1000

        return PipelineResponse(
            session_id=ctx.session_id,
            utterance=utterance,
            resolved_utterance=resolved,
            intent=result,
            missing_slots=missing,
            should_clarify=should_clarify,
            clarification=clarification,
            total_latency_ms=total_ms,
            engine_path=engine_path,
            metadata={
                "active_product": ctx.active_product,
                "topic_stack": ctx.topic_stack,
                "category_history": ctx.category_history,
                "dialogue_phase": ctx.dialogue_phase,
                "dst_snapshot": self.context_mgr.build_state_snapshot(ctx),
                "profile_hints": ctx.user_profile_hints,
                "category_name": get_category_name(result.category),
                "within_latency_budget": total_ms <= settings.latency.total_ms,
                "llm_provider": settings.llm.provider,
                "llm_model": settings.llm.model,
                "llm_enabled": self.llm_engine.is_available(),
                "clarification_resolved": clarification.resolved_from_clarification,
                "was_clarification_followup": was_clarification_followup,
                "context_dependent": self.context_mgr.should_inherit_context(ctx, utterance),
                "drift_signals": {
                    k: v for k, v in result.raw_scores.items() if k.startswith("drift_")
                },
            },
        )

    def process(self, utterance: str, session_id: Optional[str] = None) -> PipelineResponse:
        start = time.perf_counter()
        ctx = self.get_or_create_session(session_id)

        resolved, references = self.context_mgr.resolve_references(ctx, utterance)
        hints = self.context_mgr.extract_profile_hints(resolved)
        ctx.user_profile_hints.extend(h for h in hints if h not in ctx.user_profile_hints)

        context_window = self.context_mgr.build_context_window(ctx)
        llm_available = self.llm_engine.is_available()

        clarify_block = self.clarification_resolver.build_clarification_context_block(ctx)
        if clarify_block:
            context_window = f"{context_window}\n\n{clarify_block}"

        llm_input = resolved
        if self.clarification_resolver.is_followup_to_clarification(ctx):
            llm_input, resolve_meta = self.clarification_resolver.enrich_utterance(ctx, resolved)
            ctx._clarification_resolve_meta = resolve_meta

        result = self.llm_engine.predict(llm_input, ctx, context_window)

        prefix = self._engine_path_prefix(llm_available)
        path = f"{prefix}_dynamic" if llm_available else "rule_fallback"
        if getattr(ctx, "_clarification_resolve_meta", {}).get("is_clarification_followup"):
            path = f"{prefix}_clarification_refined" if llm_available else "rule_clarification_refined"
        return self._finalize(ctx, utterance, resolved, references, result, start, path)

    async def process_async(self, utterance: str, session_id: Optional[str] = None) -> PipelineResponse:
        start = time.perf_counter()
        ctx = self.get_or_create_session(session_id)

        resolved, references = self.context_mgr.resolve_references(ctx, utterance)
        hints = self.context_mgr.extract_profile_hints(resolved)
        ctx.user_profile_hints.extend(h for h in hints if h not in ctx.user_profile_hints)

        context_window = self.context_mgr.build_context_window(ctx)
        llm_available = self.llm_engine.is_available()

        clarify_block = self.clarification_resolver.build_clarification_context_block(ctx)
        if clarify_block:
            context_window = f"{context_window}\n\n{clarify_block}"

        llm_input = resolved
        if self.clarification_resolver.is_followup_to_clarification(ctx):
            llm_input, resolve_meta = self.clarification_resolver.enrich_utterance(ctx, resolved)
            ctx._clarification_resolve_meta = resolve_meta

        result = await self.llm_engine.predict_async(llm_input, ctx, context_window)

        prefix = self._engine_path_prefix(llm_available)
        path = f"{prefix}_async" if llm_available else "async_rule_fallback"
        if getattr(ctx, "_clarification_resolve_meta", {}).get("is_clarification_followup"):
            path = f"{prefix}_async_clarification_refined" if llm_available else "async_rule_clarification_refined"
        return self._finalize(ctx, utterance, resolved, references, result, start, path)

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def format_response(resp: PipelineResponse) -> str:
    intent = resp.intent
    cat_name = resp.metadata.get("category_name", intent.category)
    lines = [
        f"会话: {resp.session_id}",
        f"用户输入: {resp.utterance}",
        f"消解后: {resp.resolved_utterance}",
        f"意图描述: {intent.intent_label}",
        f"参考分类: {cat_name} ({intent.category})",
        f"置信度: {intent.confidence:.2f} | 来源: {intent.source.value} | 路径: {resp.engine_path}",
        f"延迟: {resp.total_latency_ms:.1f}ms",
    ]
    if intent.reasoning:
        lines.append(f"推理: {intent.reasoning}")
    if intent.sub_intents and len(intent.sub_intents) > 1:
        subs = ", ".join(f"{s.intent_label}({s.confidence:.2f})" for s in intent.sub_intents)
        lines.append(f"子意图: {subs}")
    if intent.implicit_intents:
        imps = ", ".join(f"{i.intent_label}" for i in intent.implicit_intents)
        lines.append(f"隐式意图: {imps}")
    if intent.drift_detected:
        lines.append(f"意图漂移: {intent.drift_type.value} — {intent.drift_reason}")
    if intent.resolved_references:
        lines.append(f"指代消解: {intent.resolved_references}")
    filled = {k: v.value for k, v in intent.slots.items() if v.value}
    if filled:
        lines.append(f"槽位: {filled}")
    if resp.missing_slots:
        lines.append(f"待澄清: {resp.missing_slots}")
    if resp.clarification.needs_clarification:
        lines.append(f"澄清原因: {resp.clarification.reason.value}")
        lines.append(f"引导回复: {resp.clarification.guide_response}")
        if resp.clarification.suggested_options:
            lines.append("建议选项: " + " | ".join(resp.clarification.suggested_options))
    return "\n".join(lines)
