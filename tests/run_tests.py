#!/usr/bin/env python3
"""Dynamic intent recognition — automated tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.context.manager import ContextManager
from src.domain.insurance_domain import CATEGORY_BY_CODE
from src.engines.entity_extractor import EntityExtractor
from src.pipeline import IntentPipeline

passed = 0
failed = 0
errors: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        msg = f"  ✗ {name}" + (f" — {detail}" if detail else "")
        print(msg)
        errors.append(msg)


def check_category(resp, expected_categories: set[str], name: str) -> None:
    actual = resp.intent.category
    check(name, actual in expected_categories, f"got {actual} ({resp.intent.intent_label})")


def run_tests() -> None:
    print("=" * 60)
    print("Dynamic Intent Recognition — Automated Tests")
    print("=" * 60)

    pipeline = IntentPipeline()
    extractor = EntityExtractor()

    print("\n[1] Entity extraction")
    slots = extractor.extract_entities("How much is the premium for Anxin Critical Illness 2026?")
    check(
        "Product name extraction",
        slots.get("product_name") and slots["product_name"].value == "Anxin Critical Illness 2026",
    )

    print("\n[2] Coreference resolution")
    sid = "ref-test"
    pipeline.reset_session(sid)
    pipeline.process("How much is the premium for Anxin Critical Illness 2026?", sid)
    resp = pipeline.process("How long is its waiting period?", sid)
    check(
        "Pronoun resolution",
        "Anxin" in resp.resolved_utterance
        and resp.resolved_utterance != "How long is its waiting period?",
    )
    check("Has intent label", len(resp.intent.intent_label) > 2)
    check_category(resp, {"coverage_terms", "product_inquiry"}, "Waiting period → coverage/product")

    print("\n[3] Dynamic intent — reference categories")
    cases = [
        ("Hello", {"greeting_chitchat", "other"}),
        ("How much is Anxin critical illness premium", {"premium_inquiry", "product_inquiry"}),
        ("How do I file a medical insurance claim", {"claims_service"}),
        ("Compare two products for me", {"product_compare"}),
    ]
    for utterance, cats in cases:
        r = pipeline.process(utterance)
        check_category(r, cats, f"'{utterance[:24]}...'")

    print("\n[4] Implicit needs / product recommendation")
    r = pipeline.process("I travel frequently for work — any recommendations?")
    check(
        "Business travel scenario",
        r.intent.category == "product_recommend"
        or any(i.category == "product_recommend" for i in r.intent.implicit_intents)
        or "travel" in r.intent.intent_label.lower()
        or "recommend" in r.intent.intent_label.lower(),
        r.intent.intent_label,
    )

    print("\n[5] Clarification guidance")
    r_vague = pipeline.process("Tell me more")
    check("Vague input triggers clarification", r_vague.clarification.needs_clarification)
    check("Structured clarification questions", len(r_vague.clarification.clarification_questions) >= 1)
    check("Guide response present", len(r_vague.clarification.guide_response) > 10)

    print("\n[5b] Clarification reply refinement")
    sid = "clarify-flow"
    pipeline.reset_session(sid)
    r1 = pipeline.process("Tell me more", sid)
    check("First turn needs clarification", r1.clarification.needs_clarification)
    r2 = pipeline.process("Critical illness insurance", sid)
    check(
        "Intent refined after clarification",
        r2.clarification.resolved_from_clarification
        or r2.intent.confidence >= 0.75
        or "critical illness" in r2.intent.intent_label.lower(),
        r2.intent.intent_label,
    )
    check("Clarification state cleared", not pipeline.get_or_create_session(sid).pending_clarification.active)

    print("\n[6] Multi-turn dialogue flow")
    sid = "flow-test"
    pipeline.reset_session(sid)
    turns = [
        "Hello, I'd like to learn about critical illness insurance",
        "How much is the premium for Anxin Critical Illness 2026?",
        "How long is its waiting period?",
        "Tell me about the medical insurance claims process",
    ]
    for u in turns:
        r = pipeline.process(u, sid)
        check(f"Turn '{u[:28]}...' has intent", bool(r.intent.intent_label))
    ctx = pipeline.get_or_create_session(sid)
    check("Topic stack accumulates", len(ctx.topic_stack) >= 2)

    print("\n[7] Context management")
    mgr = ContextManager()
    ctx2 = mgr.create_session("ctx-test")
    mgr.add_turn(ctx2, "user", "Hello", intent_label="Greeting", category="greeting_chitchat")
    window = mgr.build_context_window(ctx2)
    check("Context includes intent label", "Greeting" in window)

    print("\n[8] Reference category taxonomy")
    check("Category count ≥ 10", len(CATEGORY_BY_CODE) >= 10)

    print("\n[9] API")
    try:
        from fastapi.testclient import TestClient
        from api.server import app
        client = TestClient(app)
        check("GET /health", client.get("/health").status_code == 200)
        check("GET /categories", client.get("/v1/intent/categories").status_code == 200)
        p = client.post(
            "/v1/intent/predict/sync",
            json={"utterance": "How much is critical illness premium?"},
        )
        check("POST predict", p.status_code == 200)
        data = p.json()
        check("Returns intent_label", bool(data.get("intent_label")))
        check("Returns category", bool(data.get("category")))
    except ImportError as e:
        print(f"  ⚠ Skipping API tests: {e}")

    total = passed + failed
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} failed")
        for e in errors:
            print(e)
    else:
        print(" — all passed ✓")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    run_tests()
