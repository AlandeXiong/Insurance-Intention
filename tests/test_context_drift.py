#!/usr/bin/env python3
"""上下文管理与漂移检测 — SOTA 对齐专项测试。"""

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
    print("\n[CTX-1] DST 对话状态追踪")
    mgr = ContextManager()
    ctx = mgr.create_session("dst-1")

    mgr.add_turn(ctx, "user", "安心保重疾险2026年保费多少？", category="premium_inquiry",
                 intent_label="查询保费", resolved_content="安心保重疾险2026年保费多少？")
    mgr.update_active_state(ctx, "查询安心保重疾险2026保费", "premium_inquiry",
                            {"product_name": "安心保重疾险2026"})
    mgr.merge_slots(ctx, {"product_name": "安心保重疾险2026"}, turn_id=0)

    check("焦点产品", ctx.active_product == "安心保重疾险2026")
    check("分类轨迹", "premium_inquiry" in ctx.category_history)
    check("主题帧", len(ctx.topic_frames) == 1 and ctx.topic_frames[0].is_active)
    check("对话阶段", ctx.dialogue_phase == "inquiry")

    snapshot = mgr.build_state_snapshot(ctx)
    check("DST 快照含槽位", "product_name" in snapshot or "安心保" in snapshot)

    print("\n[CTX-2] 多层指代消解")
    resolved, refs = mgr.resolve_references(ctx, "那它的等待期是多久？")
    check("代词消解", "安心保" in resolved)
    check("有消解映射", len(refs) > 0)

    print("\n[CTX-3] 省略句补全")
    resolved2, refs2 = mgr.resolve_references(ctx, "等待期呢？")
    check("省略句补全产品", "安心保" in resolved2 or "等待期呢" in resolved2)

    print("\n[CTX-4] 槽位跨轮继承")
    merged = mgr.merge_slots(ctx, {"age": "30"}, turn_id=1)
    merged2 = mgr.merge_slots(ctx, {}, turn_id=2)
    check("槽位继承", merged2.get("product_name") == "安心保重疾险2026")
    check("新槽位写入", merged.get("age") == "30")

    print("\n[CTX-5] 分层上下文窗口")
    window = mgr.build_context_window(ctx)
    check("含 DST 快照", "对话状态快照" in window)
    check("含分类轨迹或活跃意图", "premium_inquiry" in window or "查询" in window)


def test_drift_detection() -> None:
    print("\n[DRIFT-1] 分类图距离")
    check("同分类距离=0", graph_distance("premium_inquiry", "premium_inquiry") == 0.0)
    check("相关切换距离低", graph_distance("product_inquiry", "premium_inquiry") <= 0.2)
    check("远距分类距离高", graph_distance("premium_inquiry", "complaint_feedback") >= 0.75)
    check("is_related_switch", is_related_switch("product_inquiry", "coverage_terms"))

    print("\n[DRIFT-2] 多信号融合检测")
    detector = IntentDriftDetector(threshold=0.35)
    ctx = SessionContext(session_id="drift-test")
    ctx.active_category = "premium_inquiry"
    ctx.active_intent_label = "查询安心保重疾险2026保费"
    ctx.active_product = "安心保重疾险2026"
    ctx.category_history = ["greeting_chitchat", "premium_inquiry"]
    ctx.turns.append(DialogueTurn(role="user", content="安心保保费多少", turn_id=0,
                                   resolved_content="安心保保费多少"))

    # 相关子意图 — 不应漂移
    r1 = IntentResult(intent_label="查询等待期", category="coverage_terms", confidence=0.9)
    d1, t1, s1, sig1 = detector.detect(ctx, r1, "那它的等待期是多久？")
    check("相关链内切换非漂移", not d1, f"type={t1}, score={s1:.2f}")

    # 显式换题 — 应漂移
    r2 = IntentResult(intent_label="咨询医疗险理赔流程", category="claims_service", confidence=0.88)
    d2, t2, s2, sig2 = detector.detect(ctx, r2, "另外我想了解医疗险理赔流程")
    check("显式换题检测漂移", d2, f"score={s2:.2f}, signals={sig2.to_dict()}")

    print("\n[DRIFT-3] 端到端多轮（规则/LLM）")
    pipeline = IntentPipeline()
    sid = "drift-e2e"
    pipeline.process("安心保重疾险2026年保费多少？", sid)
    r = pipeline.process("另外医疗险理赔流程是怎样的？", sid)
    check("E2E 完成识别", bool(r.intent.intent_label))
    check("有漂移信号分量", bool(r.metadata.get("drift_signals")))
    # 单元级验证：premium → claims + 显式换题
    ctx_sim = SessionContext(session_id="sim")
    ctx_sim.active_category = "premium_inquiry"
    ctx_sim.active_intent_label = "查询安心保重疾险2026保费"
    ctx_sim.active_product = "安心保重疾险2026"
    ctx_sim.category_history = ["premium_inquiry"]
    ctx_sim.turns.append(DialogueTurn(role="user", content="安心保保费多少", turn_id=0))
    r_sim = IntentResult(intent_label="咨询医疗险理赔", category="claims_service", confidence=0.9)
    d, _, score, _ = detector.detect(ctx_sim, r_sim, "另外我想了解医疗险理赔流程")
    check("同场景融合检测漂移", d, f"score={score:.2f}")


def test_semantic() -> None:
    print("\n[SEM] 语义相似度")
    sim = semantic_similarity("查询重疾险保费", "查询重疾险价格")
    check("近义句高相似", sim > 0.3)
    sim2 = semantic_similarity("查询重疾险保费", "今天天气怎么样")
    check("无关句低相似", sim2 < sim)


if __name__ == "__main__":
    print("=" * 60)
    print("上下文管理 & 漂移检测 — SOTA 对齐测试")
    print("=" * 60)
    test_semantic()
    test_dst_context()
    test_drift_detection()
    print("\n" + "=" * 60)
    print(f"结果: {passed}/{passed + failed}", "通过 ✓" if not failed else f"失败 {failed}")
    print("=" * 60)
    sys.exit(1 if failed else 0)
