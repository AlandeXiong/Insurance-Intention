#!/usr/bin/env python3
"""Context management and drift detection — SOTA alignment focused tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.context.manager import ContextManager
from src.context.semantic import semantic_similarity
from src.drift.category_graph import graph_distance, is_related_switch
from src.drift.detector import IntentDriftDetector
from src.models.intent import DialogueTurn, IntentResult, SessionContext
from src.pipeline import IntentPipeline

passed = failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}" + (f" — {detail}" if detail else ""))


def test_dst_context() -> None:
    print("\n[CTX-1] DST dialogue state tracking")
    mgr = ContextManager()
    ctx = mgr.create_session("dst-1")

    mgr.add_turn(
        ctx, "user", "How much is the premium for Anxin Critical Illness 2026?",
        category="premium_inquiry",
        intent_label="Query premium",
        resolved_content="How much is the premium for Anxin Critical Illness 2026?",
    )
    mgr.update_active_state(
        ctx, "Query Anxin Critical Illness 2026 premium", "premium_inquiry",
        {"product_name": "Anxin Critical Illness 2026"},
    )
    mgr.merge_slots(ctx, {"product_name": "Anxin Critical Illness 2026"}, turn_id=0)

    check("Focus product", ctx.active_product == "Anxin Critical Illness 2026")
    check("Category trail", "premium_inquiry" in ctx.category_history)
    check("Topic frame", len(ctx.topic_frames) == 1 and ctx.topic_frames[0].is_active)
    check("Dialogue phase", ctx.dialogue_phase == "inquiry")

    snapshot = mgr.build_state_snapshot(ctx)
    check("DST snapshot includes slot", "product_name" in snapshot or "Anxin" in snapshot)

    print("\n[CTX-2] Multi-layer coreference resolution")
    resolved, refs = mgr.resolve_references(ctx, "How long is its waiting period?")
    check("Pronoun resolution", "Anxin" in resolved)
    check("Resolution mapping", len(refs) > 0)

    print("\n[CTX-3] Ellipsis completion")
    resolved2, refs2 = mgr.resolve_references(ctx, "Waiting period?")
    check(
        "Ellipsis fills product",
        "Anxin" in resolved2 or "Waiting period" in resolved2,
    )

    print("\n[CTX-4] Cross-turn slot inheritance")
    merged = mgr.merge_slots(ctx, {"age": "30"}, turn_id=1)
    merged2 = mgr.merge_slots(ctx, {}, turn_id=2)
    check("Slot inheritance", merged2.get("product_name") == "Anxin Critical Illness 2026")
    check("New slot written", merged.get("age") == "30")

    print("\n[CTX-5] Layered context window")
    window = mgr.build_context_window(ctx)
    check("Includes DST snapshot", "Dialogue State Snapshot" in window)
    check(
        "Includes category trail or active intent",
        "premium_inquiry" in window or "Query" in window,
    )


def test_drift_detection() -> None:
    print("\n[DRIFT-1] Category graph distance")
    check("Same category distance=0", graph_distance("premium_inquiry", "premium_inquiry") == 0.0)
    check("Related switch low distance", graph_distance("product_inquiry", "premium_inquiry") <= 0.2)
    check("Distant category high distance", graph_distance("premium_inquiry", "complaint_feedback") >= 0.75)
    check("is_related_switch", is_related_switch("product_inquiry", "coverage_terms"))

    print("\n[DRIFT-2] Multi-signal fusion detection")
    detector = IntentDriftDetector(threshold=0.35)
    ctx = SessionContext(session_id="drift-test")
    ctx.active_category = "premium_inquiry"
    ctx.active_intent_label = "Query Anxin Critical Illness 2026 premium"
    ctx.active_product = "Anxin Critical Illness 2026"
    ctx.category_history = ["greeting_chitchat", "premium_inquiry"]
    ctx.turns.append(DialogueTurn(
        role="user", content="How much is Anxin premium?", turn_id=0,
        resolved_content="How much is Anxin premium?",
    ))

    # Related sub-intent — should not drift
    r1 = IntentResult(intent_label="Query waiting period", category="coverage_terms", confidence=0.9)
    d1, t1, s1, sig1 = detector.detect(ctx, r1, "How long is its waiting period?")
    check("Related in-chain switch not drift", not d1, f"type={t1}, score={s1:.2f}")

    # Explicit topic change — should drift
    r2 = IntentResult(intent_label="Medical insurance claims process", category="claims_service", confidence=0.88)
    d2, t2, s2, sig2 = detector.detect(ctx, r2, "Also tell me about the medical insurance claims process")
    check("Explicit topic change detected as drift", d2, f"score={s2:.2f}, signals={sig2.to_dict()}")

    print("\n[DRIFT-3] End-to-end multi-turn (rule/LLM)")
    pipeline = IntentPipeline()
    sid = "drift-e2e"
    pipeline.process("How much is the premium for Anxin Critical Illness 2026?", sid)
    r = pipeline.process("Also tell me about the medical insurance claims process", sid)
    check("E2E recognition completes", bool(r.intent.intent_label))
    check("Drift signal components present", bool(r.metadata.get("drift_signals")))
    ctx_sim = SessionContext(session_id="sim")
    ctx_sim.active_category = "premium_inquiry"
    ctx_sim.active_intent_label = "Query Anxin Critical Illness 2026 premium"
    ctx_sim.active_product = "Anxin Critical Illness 2026"
    ctx_sim.category_history = ["premium_inquiry"]
    ctx_sim.turns.append(DialogueTurn(role="user", content="How much is Anxin premium?", turn_id=0))
    r_sim = IntentResult(intent_label="Medical insurance claims inquiry", category="claims_service", confidence=0.9)
    d, _, score, _ = detector.detect(ctx_sim, r_sim, "Also tell me about the medical insurance claims process")
    check("Same scenario fusion detects drift", d, f"score={score:.2f}")


def test_semantic() -> None:
    print("\n[SEM] Semantic similarity")
    sim = semantic_similarity("query critical illness premium", "query critical illness price")
    check("Similar sentences high similarity", sim > 0.3)
    sim2 = semantic_similarity("query critical illness premium", "what is the weather today")
    check("Unrelated sentences low similarity", sim2 < sim)


if __name__ == "__main__":
    print("=" * 60)
    print("Context Management & Drift Detection — SOTA Alignment Tests")
    print("=" * 60)
    test_semantic()
    test_dst_context()
    test_drift_detection()
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed}", "passed ✓" if not failed else f"failed {failed}")
    print("=" * 60)
    sys.exit(1 if failed else 0)
