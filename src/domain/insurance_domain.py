"""Insurance domain configuration — common customer intent reference framework (not fixed taxonomy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class IntentCategory:
    """Common insurance customer intent categories — LLM reference only, not enforced enum."""
    code: str
    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    typical_slots: List[str] = field(default_factory=list)


# Insurance industry customer intent taxonomy (2026 intelligent customer service practice)
INSURANCE_INTENT_CATEGORIES: List[IntentCategory] = [
    IntentCategory(
        code="product_inquiry",
        name="Product Inquiry",
        description="Learn about coverage scope, policy terms, key features, and target demographics for a product",
        examples=["What diseases does this CI plan cover?", "Does medical insurance cover outpatient visits?", "What age group is it suited for?"],
        typical_slots=["product_name", "insurance_type"],
    ),
    IntentCategory(
        code="premium_inquiry",
        name="Premium Inquiry",
        description="Ask about premium amount, rates, payment term, and payment method",
        examples=["How much per year?", "How much for a 30-year-old male with 500k coverage?", "Annual or monthly payment?"],
        typical_slots=["product_name", "age", "gender", "coverage_amount", "payment_period"],
    ),
    IntentCategory(
        code="coverage_terms",
        name="Coverage Terms",
        description="Waiting period, observation period, exclusions, coverage duration, renewal rules",
        examples=["How long is the waiting period?", "When is a claim not paid?", "Is renewal guaranteed?"],
        typical_slots=["product_name", "clause_type"],
    ),
    IntentCategory(
        code="claims_service",
        name="Claims Service",
        description="Claims process, required documents, claim status, reimbursement ratio",
        examples=["How do I file a claim?", "What documents are needed for hospitalization claims?", "How long until payout?"],
        typical_slots=["product_name", "claim_type", "hospital"],
    ),
    IntentCategory(
        code="purchase",
        name="Purchase",
        description="Express purchase intent, ask about enrollment process or how to order",
        examples=["I want to buy one", "How do I enroll?", "Can I buy online?"],
        typical_slots=["product_name", "budget"],
    ),
    IntentCategory(
        code="product_compare",
        name="Product Comparison",
        description="Compare differences between two or more products",
        examples=["Which is better, A or B?", "Help me compare them", "What's the difference?"],
        typical_slots=["product_a", "product_b", "compare_dimension"],
    ),
    IntentCategory(
        code="policy_service",
        name="Policy Service",
        description="Renewal, surrender, policy changes, policy lookup",
        examples=["How do I renew?", "I want to surrender", "Change beneficiary", "Look up my policy"],
        typical_slots=["policy_no", "service_type"],
    ),
    IntentCategory(
        code="value_added",
        name="Value-Added Benefits",
        description="Value-added services, medical concierge, health management, checkup benefits",
        examples=["Is there a green-channel service?", "How do I use the free checkup?"],
        typical_slots=["product_name", "service_type"],
    ),
    IntentCategory(
        code="product_recommend",
        name="Product Recommendation",
        description="Recommend suitable products based on user profile or implicit needs",
        examples=["What insurance for frequent business travel?", "What do you recommend for kids?", "Anything suitable for seniors?"],
        typical_slots=["age", "occupation", "scenario", "budget"],
    ),
    IntentCategory(
        code="complaint_feedback",
        name="Complaint & Feedback",
        description="Complaints, dissatisfaction, suggestions",
        examples=["I want to file a complaint", "The service is terrible", "I have a suggestion"],
        typical_slots=["issue_type"],
    ),
    IntentCategory(
        code="greeting_chitchat",
        name="Greeting & Chitchat",
        description="Greetings, thanks, unrelated small talk",
        examples=["Hello", "Thank you", "Are you there?"],
        typical_slots=[],
    ),
    IntentCategory(
        code="other",
        name="Other",
        description="Requests that do not fit the categories above",
        examples=[],
        typical_slots=[],
    ),
]

CATEGORY_BY_CODE: Dict[str, IntentCategory] = {c.code: c for c in INSURANCE_INTENT_CATEGORIES}

# Product entity library — reference resolution and slot extraction
PRODUCT_ENTITIES: Dict[str, List[str]] = {
    "Anxin Critical Illness 2026": ["Anxin", "Anxin CI", "this plan", "it", "this"],
    "Kangle Medical Plus": ["Kangle Medical", "Kangle Plus", "medical plan", "this medical plan"],
    "Changxing Accident Insurance": ["Changxing Accident", "accident insurance", "travel insurance"],
}

# Common slot definitions (shared across intents; filled dynamically by LLM)
COMMON_SLOT_HINTS = [
    "product_name", "product_a", "product_b",
    "age", "gender", "coverage_amount", "payment_period", "budget",
    "insurance_type", "claim_type", "policy_no", "occupation", "scenario",
]


def build_category_prompt() -> str:
    """Build intent category reference text for the LLM."""
    lines = ["Below are common insurance customer intent categories (reference only — refine as needed):\n"]
    for cat in INSURANCE_INTENT_CATEGORIES:
        lines.append(f"- [{cat.name}] ({cat.code}): {cat.description}")
        if cat.examples:
            lines.append(f"  Examples: {'; '.join(cat.examples[:3])}")
        if cat.typical_slots:
            lines.append(f"  Typical slots: {', '.join(cat.typical_slots)}")
    return "\n".join(lines)


def get_category_name(code: str) -> str:
    cat = CATEGORY_BY_CODE.get(code)
    return cat.name if cat else code
