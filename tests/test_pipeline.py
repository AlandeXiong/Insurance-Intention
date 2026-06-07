"""Multi-turn intent recognition — automated tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.context.manager import ContextManager
from src.drift.detector import IntentDriftDetector
from src.engines.lightweight import LightweightIntentEngine
from src.pipeline import IntentPipeline
from src.slots.tracker import SlotTracker


@pytest.fixture
def pipeline() -> IntentPipeline:
    return IntentPipeline()


@pytest.fixture
def session_id(pipeline: IntentPipeline) -> str:
    ctx = pipeline.get_or_create_session("test-session-001")
    return ctx.session_id


class TestReferenceResolution:
    """Reference resolution tests."""

    def test_pronoun_resolves_to_active_product(self, pipeline: IntentPipeline, session_id: str):
        pipeline.process("How much is the premium for Anxin Critical Illness 2026?", session_id)
        resp = pipeline.process("How long is its waiting period?", session_id)

        assert "Anxin Critical Illness 2026" in resp.resolved_utterance
        assert resp.intent.resolved_references.get("its") == "Anxin Critical Illness 2026"
        assert resp.intent.category in {"coverage_terms", "product_inquiry"}


class TestIntentRecognition:
    """Intent recognition accuracy tests."""

    @pytest.mark.parametrize("utterance,expected_category", [
        ("Hello", "greeting_chitchat"),
        ("How much is Anxin critical illness premium", "premium_inquiry"),
        ("How long is the waiting period", "coverage_terms"),
        ("How do I file a medical insurance claim", "claims_service"),
        ("Compare two products for me", "product_compare"),
        ("I want to buy this insurance", "purchase"),
    ])
    def test_single_turn_intent(
        self, pipeline: IntentPipeline, utterance: str, expected_category: str
    ):
        resp = pipeline.process(utterance)
        assert resp.intent.category == expected_category, (
            f"Input: {utterance}, expected: {expected_category}, actual: {resp.intent.category}"
        )
        assert resp.intent.confidence >= 0.75


class TestImplicitIntent:
    """Implicit intent tests."""

    def test_business_travel_triggers_accident_insurance(self, pipeline: IntentPipeline):
        resp = pipeline.process("I travel frequently for work")
        assert (
            resp.intent.category == "product_recommend"
            or any(i.category == "product_recommend" for i in resp.intent.implicit_intents)
            or "travel" in resp.intent.intent_label.lower()
        )


class TestSlotTracking:
    """Slot filling and cross-turn inheritance."""

    def test_product_slot_filled(self, pipeline: IntentPipeline, session_id: str):
        resp = pipeline.process("How much is the premium for Anxin Critical Illness 2026?", session_id)
        assert resp.intent.slots["product_name"].value == "Anxin Critical Illness 2026"

    def test_slot_inherited_across_turns(self, pipeline: IntentPipeline, session_id: str):
        pipeline.process("How much is the premium for Anxin Critical Illness 2026?", session_id)
        resp = pipeline.process("What about the waiting period?", session_id)
        assert resp.intent.slots["product_name"].value == "Anxin Critical Illness 2026"

    def test_compare_products_dual_slots(self, pipeline: IntentPipeline):
        resp = pipeline.process("Compare Anxin Critical Illness 2026 and Kangle Medical Plus")
        assert resp.intent.slots["product_a"].value == "Anxin Critical Illness 2026"
        assert resp.intent.slots["product_b"].value == "Kangle Medical Plus"


class TestDriftDetection:
    """Intent drift detection."""

    def test_topic_shift_detected(self, pipeline: IntentPipeline, session_id: str):
        pipeline.process("How much is the premium for Anxin Critical Illness 2026?", session_id)
        resp = pipeline.process("Also tell me about the medical insurance claims process", session_id)
        assert resp.intent.drift_detected is True
        assert resp.intent.category == "claims_service"

    def test_related_intent_not_drift(self, pipeline: IntentPipeline, session_id: str):
        pipeline.process("How much is the premium for Anxin Critical Illness 2026?", session_id)
        resp = pipeline.process("How long is its waiting period?", session_id)
        assert resp.intent.drift_detected is False


class TestMultiIntent:
    """Multi-intent decomposition."""

    def test_compound_utterance_decomposed(self, pipeline: IntentPipeline, session_id: str):
        pipeline.process("Anxin Critical Illness 2026", session_id)
        resp = pipeline.process("How much is the premium, and how long is the waiting period?", session_id)
        assert len(resp.intent.sub_intents) >= 1


class TestLatency:
    """Latency budget."""

    def test_within_latency_budget(self, pipeline: IntentPipeline):
        resp = pipeline.process("Hello, I'd like to learn about critical illness insurance")
        assert resp.total_latency_ms <= settings.latency.total_ms
        assert resp.metadata["within_latency_budget"] is True


class TestMultiTurnFlow:
    """Full multi-turn dialogue flow."""

    def test_full_insurance_dialogue(self, pipeline: IntentPipeline):
        sid = "full-flow-test"
        turns = [
            ("Hello, I'd like to learn about critical illness insurance", "greeting_chitchat"),
            ("How much is the premium for Anxin Critical Illness 2026?", "premium_inquiry"),
            ("How long is its waiting period?", "coverage_terms"),
            ("Tell me about the medical insurance claims process", "claims_service"),
            ("I travel frequently for work", "product_recommend"),
            ("Compare Anxin Critical Illness 2026 and Kangle Medical Plus", "product_compare"),
        ]
        for utterance, expected in turns:
            resp = pipeline.process(utterance, sid)
            assert resp.intent.category == expected, (
                f"Turn '{utterance}': expected {expected}, got {resp.intent.category}"
            )

        ctx = pipeline.get_or_create_session(sid)
        assert len(ctx.topic_stack) >= 4


class TestLightweightEngine:
    """Lightweight engine unit tests."""

    def test_high_confidence_greeting(self):
        engine = LightweightIntentEngine()
        result = engine.predict("Hello")
        assert result.intent == "greeting"
        assert engine.is_confident(result)

    def test_latency_under_budget(self):
        engine = LightweightIntentEngine()
        result = engine.predict("How much is critical illness premium?")
        assert result.latency_ms < settings.latency.frontend_engine_ms


class TestContextManager:
    """Context manager unit tests."""

    def test_build_context_window(self):
        mgr = ContextManager()
        ctx = mgr.create_session("ctx-test")
        mgr.add_turn(ctx, "user", "Hello")
        mgr.add_turn(ctx, "assistant", "Hello! How can I help you today?")
        window = mgr.build_context_window(ctx)
        assert "User" in window
        assert "Agent" in window
