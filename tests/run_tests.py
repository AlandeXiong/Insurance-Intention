#!/usr/bin/env python3
"""动态意图识别 — 自动化测试。"""

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
    print("动态意图识别 — 自动化测试")
    print("=" * 60)

    pipeline = IntentPipeline()
    extractor = EntityExtractor()

    print("\n[1] 实体抽取")
    slots = extractor.extract_entities("安心保重疾险2026年保费多少？")
    check("产品名抽取", slots.get("product_name") and slots["product_name"].value == "安心保重疾险2026")

    print("\n[2] 指代消解")
    sid = "ref-test"
    pipeline.reset_session(sid)
    pipeline.process("安心保重疾险2026年保费多少？", sid)
    resp = pipeline.process("那它的等待期是多久？", sid)
    check("代词消解", "安心保" in resp.resolved_utterance and resp.resolved_utterance != "那它的等待期是多久？")
    check("有意图描述", len(resp.intent.intent_label) > 2)
    check_category(resp, {"coverage_terms", "product_inquiry"}, "等待期 → 保障/产品类")

    print("\n[3] 动态意图 — 参考分类")
    cases = [
        ("你好", {"greeting_chitchat", "other"}),
        ("安心保重疾险保费多少", {"premium_inquiry", "product_inquiry"}),
        ("医疗险怎么理赔", {"claims_service"}),
        ("帮我对比一下两款产品", {"product_compare"}),
    ]
    for utterance, cats in cases:
        r = pipeline.process(utterance)
        check_category(r, cats, f"'{utterance[:12]}...'")

    print("\n[4] 隐式需求 / 产品推荐")
    r = pipeline.process("我最近经常出差，有什么推荐？")
    check(
        "出差场景",
        r.intent.category == "product_recommend"
        or any(i.category == "product_recommend" for i in r.intent.implicit_intents)
        or "出差" in r.intent.intent_label
        or "推荐" in r.intent.intent_label,
        r.intent.intent_label,
    )

    print("\n[5] 意图澄清引导")
    r_vague = pipeline.process("了解一下")
    check("模糊输入触发澄清", r_vague.clarification.needs_clarification)
    check("有结构化澄清问题", len(r_vague.clarification.clarification_questions) >= 1)
    check("有引导话术", len(r_vague.clarification.guide_response) > 10)

    print("\n[5b] 澄清回复 refinement")
    sid = "clarify-flow"
    pipeline.reset_session(sid)
    r1 = pipeline.process("了解一下", sid)
    check("首轮待澄清", r1.clarification.needs_clarification)
    r2 = pipeline.process("重疾险", sid)
    check(
        "澄清后意图完善",
        r2.clarification.resolved_from_clarification or r2.intent.confidence >= 0.75 or "重疾" in r2.intent.intent_label,
        r2.intent.intent_label,
    )
    check("澄清状态已清除", not pipeline.get_or_create_session(sid).pending_clarification.active)

    print("\n[6] 多轮对话流程")
    sid = "flow-test"
    pipeline.reset_session(sid)
    turns = [
        "你好，我想了解重疾险",
        "安心保重疾险2026年保费多少？",
        "那它的等待期是多久？",
        "另外医疗险理赔流程是怎样的？",
    ]
    for u in turns:
        r = pipeline.process(u, sid)
        check(f"Turn '{u[:10]}...' 有意图", bool(r.intent.intent_label))
    ctx = pipeline.get_or_create_session(sid)
    check("主题栈累积", len(ctx.topic_stack) >= 2)

    print("\n[7] 上下文管理")
    mgr = ContextManager()
    ctx2 = mgr.create_session("ctx-test")
    mgr.add_turn(ctx2, "user", "你好", intent_label="寒暄问候", category="greeting_chitchat")
    window = mgr.build_context_window(ctx2)
    check("上下文含意图描述", "寒暄" in window)

    print("\n[8] 参考分类体系")
    check("分类数量 ≥ 10", len(CATEGORY_BY_CODE) >= 10)

    print("\n[9] API")
    try:
        from fastapi.testclient import TestClient
        from api.server import app
        client = TestClient(app)
        check("GET /health", client.get("/health").status_code == 200)
        check("GET /categories", client.get("/v1/intent/categories").status_code == 200)
        p = client.post("/v1/intent/predict/sync", json={"utterance": "重疾险保费多少"})
        check("POST predict", p.status_code == 200)
        data = p.json()
        check("返回 intent_label", bool(data.get("intent_label")))
        check("返回 category", bool(data.get("category")))
    except ImportError as e:
        print(f"  ⚠ 跳过 API 测试: {e}")

    total = passed + failed
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过", end="")
    if failed:
        print(f", {failed} 失败")
        for e in errors:
            print(e)
    else:
        print(" — 全部通过 ✓")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    run_tests()
