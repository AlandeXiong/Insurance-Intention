"""Slot filling and tracking module."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.domain.insurance_intents import INSURANCE_INTENTS, PRODUCT_ENTITIES
from src.models.intent import SessionContext, Slot, SlotStatus


class SlotTracker:
    """Rule- and context-based slot extraction and cross-turn tracking."""

    SLOT_PATTERNS = {
        "age": re.compile(r"(\d{1,3})\s*(?:years?\s*old|yrs?|yo)\b|(?:age\s*)?(\d{1,3})\b"),
        "coverage_amount": re.compile(r"(\d+)\s*(?:wan|10k|million|M)\b", re.IGNORECASE),
        "payment_period": re.compile(r"(?<![0-9])(\d{1,2})\s*(?:year|yr)s?\s*(?:payment|term|premium)?"),
        "budget": re.compile(r"budget\s*\$?(\d+)", re.IGNORECASE),
        "compare_dimension": re.compile(r"compar(?:e|ing).{0,10}(premium|coverage|claims|waiting\s*period)", re.IGNORECASE),
    }

    def extract_slots(
        self,
        utterance: str,
        intent: str,
        ctx: Optional[SessionContext] = None,
    ) -> Dict[str, Slot]:
        definition = INSURANCE_INTENTS.get(intent)
        if not definition:
            return {}

        slots: Dict[str, Slot] = {}
        all_slot_names = definition.required_slots + definition.optional_slots

        for slot_name in all_slot_names:
            value = self._extract_single(utterance, slot_name, ctx)
            if value is not None:
                slots[slot_name] = Slot(
                    name=slot_name,
                    value=value,
                    status=SlotStatus.FILLED,
                    confidence=0.85,
                )
            elif ctx and slot_name in ctx.slot_memory:
                inherited = ctx.slot_memory[slot_name]
                slots[slot_name] = Slot(
                    name=slot_name,
                    value=inherited.value,
                    status=SlotStatus.INHERITED,
                    confidence=inherited.confidence * 0.9,
                    source_turn=inherited.source_turn,
                )
            else:
                slots[slot_name] = Slot(name=slot_name, status=SlotStatus.EMPTY)

        return slots

    def _extract_single(
        self,
        utterance: str,
        slot_name: str,
        ctx: Optional[SessionContext],
    ) -> Optional[str]:
        if slot_name in ("product_name", "product_a", "product_b"):
            return self._match_product(utterance, ctx)

        pattern = self.SLOT_PATTERNS.get(slot_name)
        if pattern:
            m = pattern.search(utterance)
            if m:
                return next((g for g in m.groups() if g is not None), m.group(0))

        if slot_name == "travel_frequency" and any(
            k in utterance.lower() for k in ("frequent business travel", "frequent travel")
        ):
            return "high frequency"

        return None

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

    def get_missing_required(self, intent: str, slots: Dict[str, Slot]) -> List[str]:
        definition = INSURANCE_INTENTS.get(intent)
        if not definition:
            return []
        return [
            s for s in definition.required_slots
            if s not in slots or slots[s].status in (SlotStatus.EMPTY, SlotStatus.AMBIGUOUS)
        ]

    def fill_compare_products(self, utterance: str) -> Dict[str, Slot]:
        """Compare intent only: extract two products."""
        found: List[str] = []
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
