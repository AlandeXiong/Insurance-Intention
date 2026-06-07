"""多轮意图识别 — 自动化测试。"""

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
    """指代消解测试。"""

    def test_pronoun_resolves_to_active_product(self, pipeline: IntentPipeline, session_id: str):
        pipeline.process("安心保重疾险2026年保费多少？", session_id)
        resp = pipeline.process("那它的等待期是多久？", session_id)

        assert resp.resolved_utterance == "那安心保重疾险2026的等待期是多久？"
        assert resp.intent.resolved_references.get("它") == "安心保重疾险2026"
        assert resp.intent.intent == "query_waiting_period"


class TestIntentRecognition:
    """意图识别准确率测试。"""

    @pytest.mark.parametrize("utterance,expected_intent", [
        ("你好", "greeting"),
        ("安心保重疾险保费多少", "query_critical_illness_premium"),
        ("等待期是多久", "query_waiting_period"),
        ("医疗险怎么理赔", "query_medical_claim"),
        ("帮我对比一下两款产品", "compare_products"),
        ("我想购买这份保险", "purchase_intent"),
    ])
    def test_single_turn_intent(
        self, pipeline: IntentPipeline, utterance: str, expected_intent: str
    ):
        resp = pipeline.process(utterance)
        assert resp.intent.intent == expected_intent, (
            f"输入: {utterance}, 期望: {expected_intent}, 实际: {resp.intent.intent}"
        )
        assert resp.intent.confidence >= 0.75


class TestImplicitIntent:
    """隐式意图测试。"""

    def test_business_travel_triggers_accident_insurance(self, pipeline: IntentPipeline):
        resp = pipeline.process("我最近经常出差")
        assert "recommend_accident_insurance" in resp.intent.implicit_intents or \
               resp.intent.intent == "recommend_accident_insurance"


class TestSlotTracking:
    """槽位填充与跨轮继承。"""

    def test_product_slot_filled(self, pipeline: IntentPipeline, session_id: str):
        resp = pipeline.process("安心保重疾险2026年保费多少？", session_id)
        assert resp.intent.slots["product_name"].value == "安心保重疾险2026"

    def test_slot_inherited_across_turns(self, pipeline: IntentPipeline, session_id: str):
        pipeline.process("安心保重疾险2026年保费多少？", session_id)
        resp = pipeline.process("等待期呢？", session_id)
        assert resp.intent.slots["product_name"].value == "安心保重疾险2026"

    def test_compare_products_dual_slots(self, pipeline: IntentPipeline):
        resp = pipeline.process("对比一下安心保和康乐医疗险Plus")
        assert resp.intent.slots["product_a"].value == "安心保重疾险2026"
        assert resp.intent.slots["product_b"].value == "康乐医疗险Plus"


class TestDriftDetection:
    """意图漂移检测。"""

    def test_topic_shift_detected(self, pipeline: IntentPipeline, session_id: str):
        pipeline.process("安心保重疾险2026年保费多少？", session_id)
        resp = pipeline.process("另外我想了解医疗险理赔流程", session_id)
        assert resp.intent.drift_detected is True
        assert resp.intent.intent == "query_medical_claim"

    def test_related_intent_not_drift(self, pipeline: IntentPipeline, session_id: str):
        pipeline.process("安心保重疾险2026年保费多少？", session_id)
        resp = pipeline.process("它的等待期是多久？", session_id)
        assert resp.intent.drift_detected is False


class TestMultiIntent:
    """多意图分解。"""

    def test_compound_utterance_decomposed(self, pipeline: IntentPipeline, session_id: str):
        pipeline.process("安心保重疾险2026", session_id)
        resp = pipeline.process("保费多少，还有等待期是多久？", session_id)
        sub_names = {s.intent for s in resp.intent.sub_intents}
        assert len(resp.intent.sub_intents) >= 1


class TestLatency:
    """延迟预算。"""

    def test_within_latency_budget(self, pipeline: IntentPipeline):
        resp = pipeline.process("你好，我想了解重疾险")
        assert resp.total_latency_ms <= settings.latency.total_ms
        assert resp.metadata["within_latency_budget"] is True


class TestMultiTurnFlow:
    """完整多轮对话流程。"""

    def test_full_insurance_dialogue(self, pipeline: IntentPipeline):
        sid = "full-flow-test"
        turns = [
            ("你好，我想了解一下重疾险", "greeting"),
            ("安心保重疾险2026年保费多少？", "query_critical_illness_premium"),
            ("那它的等待期是多久？", "query_waiting_period"),
            ("另外医疗险理赔流程是怎样的？", "query_medical_claim"),
            ("我最近经常出差", "recommend_accident_insurance"),
            ("帮我对比一下安心保和康乐医疗险Plus", "compare_products"),
        ]
        for utterance, expected in turns:
            resp = pipeline.process(utterance, sid)
            assert resp.intent.intent == expected, (
                f"Turn '{utterance}': expected {expected}, got {resp.intent.intent}"
            )

        ctx = pipeline.get_or_create_session(sid)
        assert len(ctx.topic_stack) >= 4


class TestLightweightEngine:
    """轻量引擎单元测试。"""

    def test_high_confidence_greeting(self):
        engine = LightweightIntentEngine()
        result = engine.predict("你好")
        assert result.intent == "greeting"
        assert engine.is_confident(result)

    def test_latency_under_budget(self):
        engine = LightweightIntentEngine()
        result = engine.predict("重疾险保费多少钱")
        assert result.latency_ms < settings.latency.frontend_engine_ms


class TestContextManager:
    """上下文管理单元测试。"""

    def test_build_context_window(self):
        mgr = ContextManager()
        ctx = mgr.create_session("ctx-test")
        mgr.add_turn(ctx, "user", "你好")
        mgr.add_turn(ctx, "assistant", "您好，有什么可以帮您？")
        window = mgr.build_context_window(ctx)
        assert "用户" in window
        assert "客服" in window
