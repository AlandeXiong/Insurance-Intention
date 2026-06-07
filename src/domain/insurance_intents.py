"""Insurance industry intent definitions and lightweight rule library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class IntentDefinition:
    name: str
    description: str
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    implicit_triggers: List[str] = field(default_factory=list)
    related_intents: List[str] = field(default_factory=list)


INSURANCE_INTENTS: Dict[str, IntentDefinition] = {
    "query_critical_illness_premium": IntentDefinition(
        name="query_critical_illness_premium",
        description="Query critical illness insurance premium",
        required_slots=["product_name"],
        optional_slots=["age", "coverage_amount", "payment_period"],
        keywords=["critical illness", "CI", "premium", "how much", "price", "rate"],
        related_intents=["query_waiting_period", "compare_products"],
    ),
    "query_waiting_period": IntentDefinition(
        name="query_waiting_period",
        description="Query waiting period",
        required_slots=["product_name"],
        optional_slots=[],
        keywords=["waiting period", "when effective", "observation period"],
        related_intents=["query_critical_illness_premium", "query_medical_claim"],
    ),
    "query_medical_claim": IntentDefinition(
        name="query_medical_claim",
        description="Medical insurance claims process",
        required_slots=["product_name"],
        optional_slots=["claim_type", "hospital"],
        keywords=["claim", "reimburse", "medical insurance", "hospitalization", "outpatient"],
        related_intents=["query_waiting_period"],
    ),
    "recommend_accident_insurance": IntentDefinition(
        name="recommend_accident_insurance",
        description="Recommend accident insurance (including implicit triggers)",
        required_slots=[],
        optional_slots=["travel_frequency", "occupation"],
        keywords=["accident insurance", "accident coverage", "business travel", "travel"],
        implicit_triggers=["frequent business travel", "frequent travel", "high-risk occupation", "outdoor"],
        related_intents=["query_critical_illness_premium"],
    ),
    "compare_products": IntentDefinition(
        name="compare_products",
        description="Product comparison",
        required_slots=["product_a", "product_b"],
        optional_slots=["compare_dimension"],
        keywords=["compare", "comparison", "which is better", "difference"],
        related_intents=["query_critical_illness_premium"],
    ),
    "purchase_intent": IntentDefinition(
        name="purchase_intent",
        description="Purchase / enrollment intent",
        required_slots=["product_name"],
        optional_slots=["budget"],
        keywords=["buy", "purchase", "enroll", "get a policy", "how to buy", "order"],
        related_intents=["query_critical_illness_premium"],
    ),
    "greeting": IntentDefinition(
        name="greeting",
        description="Greeting",
        required_slots=[],
        keywords=["hello", "hi", "hey", "good morning", "are you there"],
    ),
    "fallback": IntentDefinition(
        name="fallback",
        description="Unrecognized intent",
        required_slots=[],
        keywords=[],
    ),
}

# Product entity library — reference resolution
PRODUCT_ENTITIES: Dict[str, List[str]] = {
    "Anxin Critical Illness 2026": ["Anxin", "Anxin CI", "this plan", "it", "this"],
    "Kangle Medical Plus": ["Kangle Medical", "Kangle Plus", "medical plan", "this medical plan"],
    "Changxing Accident Insurance": ["Changxing Accident", "accident insurance", "travel insurance"],
}

# Lightweight engine: keyword → intent fast mapping
KEYWORD_INTENT_RULES: List[Tuple[List[str], str, float]] = [
    (["waiting period", "observation period", "when effective"], "query_waiting_period", 0.92),
    (["claim", "reimburse", "how to claim"], "query_medical_claim", 0.90),
    (["critical illness", "CI", "premium", "how much"], "query_critical_illness_premium", 0.91),
    (["compare", "comparison", "which is better"], "compare_products", 0.88),
    (["buy", "purchase", "how to buy"], "purchase_intent", 0.89),
    (["hello", "hi", "hey"], "greeting", 0.95),
]

# Implicit intent mapping
IMPLICIT_INTENT_RULES: Dict[str, str] = {
    "frequent business travel": "recommend_accident_insurance",
    "frequent travel": "recommend_accident_insurance",
    "high-risk occupation": "recommend_accident_insurance",
    "outdoor": "recommend_accident_insurance",
    "hospitalization": "query_medical_claim",
    "surgery": "query_medical_claim",
}

INTENT_TAXONOMY = list(INSURANCE_INTENTS.keys())
