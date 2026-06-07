"""
Multi-turn context management — production-grade DST (Dialogue State Tracking)

SOTA-aligned capabilities:
- Structured dialogue state snapshot (TopicFrame + slot_memory + entity_salience)
- Multi-layer reference resolution (pronouns / demonstratives / ellipsis)
- Cross-turn slot inheritance and conflict resolution
- Layered context window (state summary + recent dialogue + clarification state)
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple

from src.context.semantic import semantic_similarity
from src.domain.insurance_domain import PRODUCT_ENTITIES
from src.models.intent import DialogueTurn, SessionContext, Slot, SlotStatus, TopicFrame

# Pronoun / demonstrative lexicon (extended)
PRONOUNS = {
    "its", "it", "this", "that", "this one", "that one",
    "this product", "that product", "this plan", "that plan",
    "the product", "this policy", "that policy",
}
INDIRECT_REFS = {
    "as mentioned above", "mentioned above", "previously mentioned",
    "mentioned earlier", "said earlier", "the one above",
    "this one", "that one",
}
ELLIPSIS_PATTERNS = [
    re.compile(r"^(waiting period|premium|claims|coverage|renewal|surrender|cancellation)(\?)?$", re.IGNORECASE),
    re.compile(r"^(how much|what about it|how to buy|how do i buy)(\?)?$", re.IGNORECASE),
]

PHASE_BY_CATEGORY = {
    "greeting_chitchat": "greeting",
    "product_inquiry": "inquiry",
    "premium_inquiry": "inquiry",
    "coverage_terms": "inquiry",
    "claims_service": "service",
    "purchase": "transaction",
    "product_compare": "inquiry",
    "policy_service": "service",
    "product_recommend": "inquiry",
    "complaint_feedback": "service",
}


class ContextManager:
    """Production-grade multi-turn dialogue context manager."""

    def __init__(self, max_history_turns: int = 10) -> None:
        self.max_history_turns = max_history_turns

    def create_session(self, session_id: str) -> SessionContext:
        return SessionContext(session_id=session_id)

    def add_turn(
        self,
        ctx: SessionContext,
        role: str,
        content: str,
        intent_label: Optional[str] = None,
        category: Optional[str] = None,
        slots: Optional[dict] = None,
        resolved_content: Optional[str] = None,
    ) -> DialogueTurn:
        turn = DialogueTurn(
            role=role,
            content=content,
            turn_id=len(ctx.turns),
            intent_label=intent_label,
            category=category,
            intent=category,
            slots=slots or {},
            resolved_content=resolved_content,
            timestamp_ms=int(time.time() * 1000),
        )
        ctx.turns.append(turn)
        if len(ctx.turns) > self.max_history_turns:
            ctx.turns = ctx.turns[-self.max_history_turns :]
        return turn

    def build_context_window(self, ctx: SessionContext, n: int = 6) -> str:
        """Layered context window — for LLM Chain-of-Intent."""
        sections: List[str] = []

        snapshot = self.build_state_snapshot(ctx)
        if snapshot:
            sections.append("=== Dialogue State Snapshot (DST) ===")
            sections.append(snapshot)

        sections.append("=== Recent Dialogue ===")
        recent = ctx.turns[-n:]
        for t in recent:
            prefix = "User" if t.role == "user" else "Agent"
            display = t.resolved_content or t.content
            if t.intent_label and t.role == "user":
                cat = f"[{t.category}]" if t.category else ""
                sections.append(f"{prefix} {cat}[{t.intent_label}]: {display}")
            else:
                sections.append(f"{prefix}: {display}")

        if ctx.pending_clarification.active:
            sections.append("=== Clarification State ===")
            sections.append(
                "Awaiting user's reply to clarification follow-ups; "
                "this turn may be a response to those questions"
            )

        return "\n".join(sections)

    def build_state_snapshot(self, ctx: SessionContext) -> str:
        """Structured DST state summary."""
        lines = []
        if ctx.dialogue_phase != "init":
            lines.append(f"Dialogue phase: {ctx.dialogue_phase}")
        if ctx.active_intent_label:
            lines.append(f"Active intent: {ctx.active_intent_label} ({ctx.active_category})")
        if ctx.active_product:
            lines.append(f"Focus product: {ctx.active_product}")
        if ctx.slot_memory:
            filled = {k: s.value for k, s in ctx.slot_memory.items() if s.value}
            if filled:
                lines.append(f"Confirmed slots: {filled}")
        if ctx.user_profile_hints:
            lines.append(f"User profile: {', '.join(ctx.user_profile_hints)}")
        if ctx.topic_frames:
            active = [f.intent_label for f in ctx.topic_frames if f.is_active][-2:]
            if active:
                lines.append(f"Active topic frames: {' → '.join(active)}")
        if ctx.category_history:
            lines.append(f"Category trajectory: {' → '.join(ctx.category_history[-4:])}")
        return "\n".join(lines)

    def resolve_references(self, ctx: SessionContext, utterance: str) -> Tuple[str, dict]:
        """Multi-layer reference resolution: pronouns + demonstratives + ellipsis completion."""
        resolved: dict = {}
        expanded = utterance

        target_product = self._resolve_focus_product(ctx)

        for pronoun in sorted(PRONOUNS, key=len, reverse=True):
            if not target_product:
                break
            pattern = re.compile(rf"\b{re.escape(pronoun)}\b", re.IGNORECASE)
            if pattern.search(expanded):
                expanded = pattern.sub(target_product, expanded)
                resolved[pronoun.lower()] = target_product

        for ref in sorted(INDIRECT_REFS, key=len, reverse=True):
            if ref in expanded.lower() and target_product:
                pattern = re.compile(re.escape(ref), re.IGNORECASE)
                expanded = pattern.sub(target_product, expanded)
                resolved[ref] = target_product

        expanded, ellipsis_resolved = self._resolve_ellipsis(ctx, expanded)
        resolved.update(ellipsis_resolved)

        return expanded, resolved

    def _resolve_focus_product(self, ctx: SessionContext) -> Optional[str]:
        if ctx.active_product:
            return ctx.active_product
        frame = ctx.get_active_topic_frame()
        if frame and frame.product:
            return frame.product
        for turn in reversed(ctx.turns):
            for product, aliases in PRODUCT_ENTITIES.items():
                if product in turn.content:
                    return product
                if any(a in turn.content for a in aliases if len(a) > 1):
                    return product
        salient = sorted(ctx.entity_salience.items(), key=lambda x: -x[1])
        for name, _ in salient:
            if name in PRODUCT_ENTITIES:
                return name
        return None

    def _resolve_ellipsis(self, ctx: SessionContext, utterance: str) -> Tuple[str, dict]:
        """Ellipsis completion, e.g. 'waiting period?' -> '{product} waiting period?'."""
        resolved = {}
        text = utterance.strip()
        if not ctx.active_intent_label or not ctx.active_product:
            return text, resolved

        for pattern in ELLIPSIS_PATTERNS:
            if pattern.match(text):
                product = ctx.active_product
                if product and product not in text:
                    expanded = f"{product} {text}"
                    resolved["ellipsis"] = product
                    return expanded, resolved
        return text, resolved

    def update_active_state(
        self,
        ctx: SessionContext,
        intent_label: str,
        category: str,
        slots: dict,
    ) -> None:
        ctx.active_intent_label = intent_label
        ctx.active_category = category
        ctx.active_intent = category
        ctx.dialogue_phase = PHASE_BY_CATEGORY.get(category, ctx.dialogue_phase)

        product = slots.get("product_name") or slots.get("product_a") or slots.get("product")
        if isinstance(product, dict):
            product = product.get("value")
        if product:
            product = str(product)
            ctx.active_product = product
            self._boost_entity_salience(ctx, product, 1.0)

        self._update_topic_frames(ctx, intent_label, category, product)
        self._update_category_history(ctx, category)

        topic_key = intent_label[:40]
        if not ctx.topic_stack or ctx.topic_stack[-1] != topic_key:
            ctx.topic_stack.append(topic_key)
            if len(ctx.topic_stack) > 10:
                ctx.topic_stack = ctx.topic_stack[-10:]

    def _update_topic_frames(
        self, ctx: SessionContext, intent_label: str, category: str, product: Optional[str]
    ) -> None:
        active = ctx.get_active_topic_frame()
        if active and active.category == category and semantic_similarity(active.intent_label, intent_label) > 0.6:
            active.intent_label = intent_label
            if product:
                active.product = product
            return

        for frame in ctx.topic_frames:
            frame.is_active = False

        ctx.topic_frames.append(TopicFrame(
            intent_label=intent_label,
            category=category,
            product=product,
            turn_id=len(ctx.turns),
            is_active=True,
        ))
        if len(ctx.topic_frames) > 8:
            ctx.topic_frames = ctx.topic_frames[-8:]

    def _update_category_history(self, ctx: SessionContext, category: str) -> None:
        if not ctx.category_history or ctx.category_history[-1] != category:
            ctx.category_history.append(category)
            if len(ctx.category_history) > 12:
                ctx.category_history = ctx.category_history[-12:]

    def _boost_entity_salience(self, ctx: SessionContext, entity: str, delta: float) -> None:
        ctx.entity_salience[entity] = min(ctx.entity_salience.get(entity, 0) + delta, 3.0)

    def merge_slots(self, ctx: SessionContext, new_slots: dict, turn_id: int) -> dict:
        """Cross-turn slot merge + conflict resolution (new values win; high-confidence inheritance)."""
        merged: dict = {}
        for name, slot in ctx.slot_memory.items():
            merged[name] = slot.value

        for key, value in new_slots.items():
            if value is None or value == "":
                continue
            existing = ctx.slot_memory.get(key)
            if existing and existing.value and existing.value != value:
                if existing.source_turn >= turn_id - 2:
                    continue
            ctx.slot_memory[key] = Slot(
                name=key, value=value, status=SlotStatus.FILLED,
                confidence=0.9, source_turn=turn_id,
            )
            merged[key] = value

        for key, slot in ctx.slot_memory.items():
            if key not in merged and slot.value:
                slot.status = SlotStatus.INHERITED
                merged[key] = slot.value

        return merged

    def extract_profile_hints(self, utterance: str) -> List[str]:
        hints = []
        patterns = [
            (r"travel frequently|frequent business travel|travel often", "travels frequently"),
            (r"travel often|frequent travel|travel a lot", "travels frequently"),
            (r"high[- ]risk (job|occupation)|dangerous work", "high-risk occupation"),
            (r"(\d+)\s*(years old|yo|y/o)?", None),
            (r"(\d+)[- ]year[- ]old (child|kid)|child (is )?\d+", "has children"),
        ]
        for pattern, label in patterns:
            m = re.search(pattern, utterance, re.IGNORECASE)
            if m:
                hints.append(label if label else m.group(0))
        return hints

    def should_inherit_context(self, ctx: SessionContext, utterance: str) -> bool:
        """Whether utterance strongly depends on prior context (drift detection assist)."""
        if len(utterance) < 12:
            return True
        utterance_lower = utterance.lower()
        if any(p in utterance_lower for p in PRONOUNS | INDIRECT_REFS):
            return True
        return any(p.match(utterance.strip()) for p in ELLIPSIS_PATTERNS)
