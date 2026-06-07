"""Intent category relationship graph — semantic distance for drift detection."""

from __future__ import annotations

from src.domain.insurance_domain import CATEGORY_BY_CODE

# Directed business adjacency: A → B means A naturally continues to B in dialogue (sub-intent switch, not drift)
CATEGORY_ADJACENCY: dict[str, set[str]] = {
    "greeting_chitchat": set(CATEGORY_BY_CODE.keys()),
    "product_inquiry": {"premium_inquiry", "coverage_terms", "purchase", "product_compare", "product_recommend"},
    "premium_inquiry": {"product_inquiry", "coverage_terms", "purchase", "product_compare"},
    "coverage_terms": {"product_inquiry", "premium_inquiry", "claims_service", "purchase"},
    "claims_service": {"coverage_terms", "policy_service", "product_inquiry"},
    "purchase": {"product_inquiry", "premium_inquiry", "coverage_terms", "policy_service"},
    "product_compare": {"product_inquiry", "premium_inquiry", "purchase"},
    "product_recommend": {"product_inquiry", "premium_inquiry", "purchase"},
    "policy_service": {"claims_service", "product_inquiry"},
    "value_added": {"product_inquiry", "policy_service"},
    "complaint_feedback": set(CATEGORY_BY_CODE.keys()),
    "other": set(CATEGORY_BY_CODE.keys()),
}

# Cross-domain jumps: almost always treated as topic shift
DISTANT_PAIRS: set[tuple[str, str]] = {
    ("premium_inquiry", "complaint_feedback"),
    ("claims_service", "product_recommend"),
    ("greeting_chitchat", "complaint_feedback"),
    ("policy_service", "product_recommend"),
}


def graph_distance(from_cat: str, to_cat: str) -> float:
    """
    Category graph distance in [0, 1]; 0 = same/strongly related, 1 = unrelated.
    Industry practice: fuse graph distance with semantic similarity.
    """
    if from_cat == to_cat:
        return 0.0
    if (from_cat, to_cat) in DISTANT_PAIRS or (to_cat, from_cat) in DISTANT_PAIRS:
        return 1.0

    if to_cat in CATEGORY_ADJACENCY.get(from_cat, set()):
        return 0.15
    if from_cat in CATEGORY_ADJACENCY.get(to_cat, set()):
        return 0.2

    # Two-hop reachable
    neighbors = CATEGORY_ADJACENCY.get(from_cat, set())
    for mid in neighbors:
        if to_cat in CATEGORY_ADJACENCY.get(mid, set()):
            return 0.35

    return 0.75


def is_related_switch(from_cat: str, to_cat: str) -> bool:
    return graph_distance(from_cat, to_cat) <= 0.2
