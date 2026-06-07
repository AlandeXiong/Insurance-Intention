"""Entity extractor — lightweight assist; does not perform fixed intent classification."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from src.domain.insurance_domain import CATEGORY_BY_CODE, PRODUCT_ENTITIES
from src.models.intent import SessionContext, Slot, SlotStatus

# Keyword → reference category hint (fallback path only, not enforced classification)
CATEGORY_HINTS: list[tuple[list[str], str, str]] = [
    (["waiting period", "observation period", "exclusion", "coverage term"], "coverage_terms", "Inquire about coverage terms"),
    (["premium", "how much", "price", "rate", "annual payment"], "premium_inquiry", "Inquire about premium"),
    (["claim", "reimburse", "payout"], "claims_service", "Inquire about claims"),
    (["compare", "comparison", "which is better", "difference"], "product_compare", "Compare insurance products"),
    (["buy", "purchase", "enroll", "how to buy"], "purchase", "Express purchase intent"),
    (["renew", "surrender", "policy", "change"], "policy_service", "Policy service request"),
    (["recommend", "what to buy", "suitable for"], "product_recommend", "Seek product recommendation"),
    (["business travel", "travel", "accident"], "product_recommend", "Scenario-based product recommendation"),
    (["hello", "hi", "hey", "are you there"], "greeting_chitchat", "Greeting"),
    (["critical illness", "medical", "coverage scope", "what does it cover"], "product_inquiry", "Inquire about product coverage"),
]


class EntityExtractor:
    """Extract entity slots + intent hints on fallback; does not replace LLM dynamic recognition."""

    SLOT_PATTERNS = {
        "age": re.compile(r"(\d{1,3})\s*(?:years?\s*old|yrs?|yo)\b|age\s*(\d{1,3})\b", re.IGNORECASE),
        "coverage_amount": re.compile(r"(\d+)\s*(?:wan|10k|million|M)\b", re.IGNORECASE),
        "payment_period": re.compile(r"(?<![0-9])(\d{1,2})\s*(?:year|yr)s?\s*(?:payment|term|premium)?"),
        "budget": re.compile(r"budget\s*\$?(\d+)", re.IGNORECASE),
    }

    def infer_intent_hint(
        self, utterance: str, ctx: Optional[SessionContext] = None
    ) -> Tuple[str, str, float]:
        """Fallback path: returns (intent_label, category_code, confidence)."""
        best_label = "Could not recognize user intent"
        best_cat = "other"
        best_conf = 0.3

        for keywords, cat, label in CATEGORY_HINTS:
            hits = sum(1 for kw in keywords if kw.lower() in utterance.lower())
            if hits > 0:
                conf = min(0.65 + 0.05 * hits, 0.85)
                if conf > best_conf:
                    best_conf = conf
                    best_cat = cat
                    best_label = label

        if ctx and ctx.active_intent_label and len(utterance) < 15:
            return ctx.active_intent_label, ctx.active_category or "other", 0.6

        return best_label, best_cat, best_conf

    def extract_entities(
        self, utterance: str, ctx: Optional[SessionContext] = None
    ) -> Dict[str, Slot]:
        slots: Dict[str, Slot] = {}

        product = self._match_product(utterance, ctx)
        if product:
            slots["product_name"] = Slot(
                name="product_name", value=product, status=SlotStatus.FILLED, confidence=0.85
            )

        for name, pattern in self.SLOT_PATTERNS.items():
            m = pattern.search(utterance)
            if m:
                value = next((g for g in m.groups() if g is not None), m.group(0))
                slots[name] = Slot(
                    name=name,
                    value=value,
                    status=SlotStatus.FILLED,
                    confidence=0.85,
                )

        if any(k in utterance.lower() for k in ("frequent business travel", "frequent travel")):
            slots["scenario"] = Slot(name="scenario", value="high-frequency travel", status=SlotStatus.FILLED, confidence=0.8)

        if ctx:
            for name, mem in ctx.slot_memory.items():
                if name not in slots and mem.value:
                    slots[name] = Slot(
                        name=name,
                        value=mem.value,
                        status=SlotStatus.INHERITED,
                        confidence=mem.confidence * 0.9,
                        source_turn=mem.source_turn,
                    )

        return slots

    def extract_compare_products(self, utterance: str) -> Dict[str, Slot]:
        found: list[str] = []
        for product, aliases in PRODUCT_ENTITIES.items():
            if product in utterance or any(a in utterance for a in aliases if len(a) > 1):
                if product not in found:
                    found.append(product)
        slots: Dict[str, Slot] = {}
        if len(found) >= 1:
            slots["product_a"] = Slot(name="product_a", value=found[0], status=SlotStatus.FILLED, confidence=0.88)
        if len(found) >= 2:
            slots["product_b"] = Slot(name="product_b", value=found[1], status=SlotStatus.FILLED, confidence=0.88)
        return slots

    def _match_product(self, utterance: str, ctx: Optional[SessionContext]) -> Optional[str]:
        utterance_lower = utterance.lower()
        for product, aliases in PRODUCT_ENTITIES.items():
            if product.lower() in utterance_lower:
                return product
            for alias in aliases:
                if len(alias) > 1 and alias.lower() in utterance_lower:
                    return product
        if ctx and ctx.active_product:
            return ctx.active_product
        return None
