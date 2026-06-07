"""Clarification reply resolution — refine intent using user clarification answers."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.domain.insurance_domain import CATEGORY_BY_CODE, get_category_name
from src.models.clarification import (
    ClarificationGuide,
    ClarificationQuestion,
    ClarificationReason,
    PendingClarification,
)
from src.models.intent import IntentResult, SessionContext


# User option index / keywords → reference category (runtime matching)
OPTION_CATEGORY_HINTS: list[tuple[list[str], str, str]] = [
    (["critical illness", "critical illness insurance"], "product_inquiry", "Learn about critical illness product coverage"),
    (["medical", "health insurance", "medical insurance"], "product_inquiry", "Learn about medical insurance product coverage"),
    (["accident", "accident insurance"], "product_inquiry", "Learn about accident insurance product coverage"),
    (["premium", "how much", "price", "cost", "rate"], "premium_inquiry", "Check insurance product premium"),
    (["claim", "reimbursement", "payout"], "claims_service", "Inquire about insurance claims process"),
    (["compare", "comparison"], "product_compare", "Compare different insurance products"),
    (["recommend", "what should i buy", "suggest"], "product_recommend", "Seek insurance product recommendations"),
    (["renewal", "surrender", "cancel", "policy"], "policy_service", "Handle policy-related services"),
    (["waiting period", "exclusions", "terms", "clauses"], "coverage_terms", "Check coverage terms"),
    (["purchase", "buy", "apply"], "purchase", "Express intent to purchase insurance"),
]


class ClarificationResolver:
    """Handle clarification prompts and user replies to improve intent recognition."""

    def is_followup_to_clarification(self, ctx: SessionContext) -> bool:
        return ctx.pending_clarification.active

    def enrich_utterance(
        self, ctx: SessionContext, utterance: str
    ) -> Tuple[str, dict]:
        """
        Merge clarification reply with original vague input for LLM refinement.
        Returns (enriched_utterance, meta).
        """
        pending = ctx.pending_clarification
        if not pending.active:
            return utterance, {}

        resolved_answer = self._resolve_user_answer(utterance, pending)
        enriched = (
            f"[Clarification context] User originally said: \"{pending.original_utterance}\"; "
            f"after follow-up, user added: \"{utterance}\""
        )
        if resolved_answer.get("matched_option"):
            enriched += f"; user selected: {resolved_answer['matched_option']}"
        if resolved_answer.get("inferred_category"):
            cat_name = get_category_name(resolved_answer["inferred_category"])
            enriched += f"; inferred intent direction: {cat_name}"

        return enriched, {
            "is_clarification_followup": True,
            "original_utterance": pending.original_utterance,
            "tentative_intent": pending.tentative_intent_label,
            "tentative_category": pending.tentative_category,
            **resolved_answer,
        }

    def build_clarification_context_block(self, ctx: SessionContext) -> str:
        pending = ctx.pending_clarification
        if not pending.active:
            return ""

        lines = [
            "[Clarification in progress — user is answering the following follow-ups]",
            f"Original input: {pending.original_utterance}",
            f"Initial understanding: {pending.tentative_intent_label} ({get_category_name(pending.tentative_category)})",
        ]
        for i, q in enumerate(pending.questions, 1):
            lines.append(f"  Follow-up {i}: {q.question}")
        if pending.suggested_options:
            lines.append("Suggested options: " + " | ".join(pending.suggested_options))
        lines.append("Combine the user's latest reply to produce a refined precise intent; needs_clarification should be false")
        return "\n".join(lines)

    def refine_intent_after_clarification(
        self,
        result: IntentResult,
        pending: PendingClarification,
        utterance: str,
        resolve_meta: dict,
    ) -> Tuple[IntentResult, ClarificationGuide]:
        """Boost intent confidence after clarification and mark refinement."""
        matched_cat = resolve_meta.get("inferred_category") or pending.tentative_category
        matched_label = resolve_meta.get("inferred_label") or pending.tentative_intent_label

        # User picked an option index or keyword, and LLM confidence is still low → rule boost
        if resolve_meta.get("matched_option") and result.confidence < 0.8:
            if matched_cat and matched_cat in CATEGORY_BY_CODE:
                result.category = matched_cat
            if matched_label and len(result.intent_label) < 8:
                result.intent_label = f"{matched_label} (user added: {utterance})"
            result.confidence = max(result.confidence, 0.85)

        # Merge original intent with supplemental information
        if pending.tentative_intent_label and pending.original_utterance:
            if utterance not in result.intent_label:
                result.intent_label = (
                    f"{result.intent_label}"
                    if result.confidence >= 0.85
                    else f"{pending.tentative_intent_label}, additional detail: {utterance}"
                )
            result.confidence = max(result.confidence, 0.82)
            result.reasoning = (
                f"Clarification refinement: original input \"{pending.original_utterance}\" → "
                f"user added \"{utterance}\" → {result.reasoning or result.intent_label}"
            )

        note = (
            f"Intent refined via clarification: {pending.original_utterance} + {utterance} "
            f"→ {result.intent_label} (confidence {result.confidence:.0%})"
        )
        guide = ClarificationGuide(
            needs_clarification=False,
            resolved_from_clarification=True,
            refinement_note=note,
        )
        return result, guide

    def create_pending_state(
        self,
        guide: ClarificationGuide,
        result: IntentResult,
        utterance: str,
        turn_id: int,
    ) -> PendingClarification:
        return PendingClarification(
            active=True,
            reason=guide.reason,
            original_utterance=utterance,
            tentative_intent_label=result.intent_label,
            tentative_category=result.category,
            questions=guide.clarification_questions,
            suggested_options=guide.suggested_options,
            asked_at_turn=turn_id,
        )

    def _resolve_user_answer(self, utterance: str, pending: PendingClarification) -> dict:
        text = utterance.strip()
        meta: dict = {}

        # Option index selection: 1 / first / pick 1 / option 2
        num_match = re.match(r"^(?:pick\s+|option\s+|choose\s+|select\s+)?(\d)[\s\.]?$", text, re.IGNORECASE)
        if num_match and pending.suggested_options:
            idx = int(num_match.group(1)) - 1
            if 0 <= idx < len(pending.suggested_options):
                meta["matched_option"] = pending.suggested_options[idx]
                meta.update(self._infer_from_text(pending.suggested_options[idx]))

        # Direct match against an option string
        if not meta.get("matched_option"):
            for opt in pending.suggested_options:
                if opt in text or text in opt:
                    meta["matched_option"] = opt
                    meta.update(self._infer_from_text(opt))
                    break

        # Keyword inference
        if not meta.get("inferred_category"):
            meta.update(self._infer_from_text(text))

        return meta

    def _infer_from_text(self, text: str) -> dict:
        text_lower = text.lower()
        for keywords, category, label in OPTION_CATEGORY_HINTS:
            if any(kw in text_lower for kw in keywords):
                return {"inferred_category": category, "inferred_label": label}
        return {}
