"""交互式多轮对话 — LLM 动态意图分析。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.domain.insurance_domain import get_category_name
from src.pipeline import IntentPipeline, PipelineResponse

EXIT_COMMANDS = {"quit", "exit", "q", "退出", "再见"}
RESET_COMMANDS = {"reset", "新对话", "重新开始"}
HELP_COMMANDS = {"help", "帮助", "?"}
HISTORY_COMMANDS = {"history", "历史", "上下文"}
CATEGORIES_COMMANDS = {"categories", "分类", "意图分类"}


def print_banner() -> None:
    llm = settings.llm
    if llm.is_configured:
        mode = f"{llm.display_name} 动态意图 ({llm.model})"
    else:
        mode = f"规则降级（请配置 {llm.api_key_env} 并设置 LLM_PROVIDER={llm.provider}）"
    print()
    print("=" * 62)
    print("  保险智能客服 — 多轮对话动态意图分析")
    print("  基于保险行业常见意图框架，由大模型动态理解用户诉求")
    print(f"  引擎: {mode}")
    print("=" * 62)
    print()


def print_help() -> None:
    print("""
可用命令:
  help / 帮助       显示此帮助
  categories / 分类  查看保险行业意图参考分类
  history / 历史    查看当前会话上下文
  reset / 新对话    清空会话，重新开始
  quit / 退出       结束对话

示例对话:
  > 你好，我想了解重疾险
  > 安心保重疾险2026年保费多少？
  > 那它的等待期是多久？
  > 另外医疗险理赔流程是怎样的？
  > 我最近经常出差，有什么推荐的？
""")


def print_categories() -> None:
    from src.domain.insurance_domain import INSURANCE_INTENT_CATEGORIES
    print("\n--- 保险客户常见意图参考分类 ---")
    for cat in INSURANCE_INTENT_CATEGORIES:
        print(f"  · {cat.name} ({cat.code}): {cat.description}")
    print("  （以上供 LLM 参考，实际意图由模型动态捕获）\n")


def print_session_history(pipeline: IntentPipeline, session_id: str) -> None:
    ctx = pipeline.get_or_create_session(session_id)
    print("\n--- 会话上下文 ---")
    print(f"会话 ID     : {ctx.session_id[:8]}...")
    print(f"活跃产品    : {ctx.active_product or '（无）'}")
    print(f"当前意图    : {ctx.active_intent_label or '（无）'}")
    if ctx.active_category:
        print(f"参考分类    : {get_category_name(ctx.active_category)}")
    print(f"主题栈      : {' → '.join(ctx.topic_stack[-4:]) if ctx.topic_stack else '（空）'}")
    if ctx.user_profile_hints:
        print(f"用户画像    : {', '.join(ctx.user_profile_hints)}")
    print(f"\n对话历史 ({len(ctx.turns)} 轮):")
    for t in ctx.turns:
        role = "用户" if t.role == "user" else "客服"
        tag = f" [{t.intent_label}]" if t.intent_label else ""
        print(f"  [{t.turn_id + 1}] {role}{tag}: {t.content}")
    print()


def print_analysis(resp: PipelineResponse, turn: int) -> None:
    intent = resp.intent
    cat_name = resp.metadata.get("category_name", get_category_name(intent.category))

    print()
    print("─" * 62)
    print(f"  第 {turn} 轮 — 动态意图分析")
    print("─" * 62)

    print(f"\n【用户输入】\n  {resp.utterance}")

    if resp.resolved_utterance != resp.utterance:
        print(f"\n【指代消解】\n  {resp.resolved_utterance}")
        for pronoun, entity in intent.resolved_references.items():
            print(f"  '{pronoun}' → {entity}")

    print(f"\n【动态意图】")
    print(f"  {intent.intent_label}")
    print(f"  参考分类: {cat_name}")
    print(f"  置信度: {intent.confidence:.0%}  |  路径: {resp.engine_path}")

    if intent.reasoning:
        print(f"\n【推理过程】\n  {intent.reasoning}")

    extras = [s for s in intent.sub_intents if not s.is_primary]
    if extras:
        print(f"\n【附加意图】")
        for si in extras:
            print(f"  · {si.intent_label}  [{get_category_name(si.category)}]  {si.confidence:.0%}")

    if intent.implicit_intents:
        print(f"\n【隐式需求】")
        for imp in intent.implicit_intents:
            trigger = f"（触发: {imp.trigger}）" if imp.trigger else ""
            print(f"  · {imp.intent_label}{trigger}")

    if intent.drift_detected:
        print(f"\n【话题切换】\n  {intent.drift_reason or intent.drift_type.value}")
        drift_sigs = resp.metadata.get("drift_signals", {})
        if drift_sigs.get("drift_fused_score") is not None:
            print(f"  漂移融合分: {drift_sigs.get('drift_fused_score', 0):.2f}")
    elif resp.metadata.get("context_dependent"):
        print(f"\n【上下文延续】强依赖上文，判定为同主题追问")

    filled = {k: v.value for k, v in intent.slots.items() if v.value is not None}
    if filled:
        print(f"\n【关键信息】")
        for name, value in filled.items():
            print(f"  · {name}: {value}")

    if resp.missing_slots:
        print(f"\n【待澄清信息】")
        for item in resp.missing_slots:
            print(f"  · {item}")

    clar = resp.clarification
    if clar.resolved_from_clarification:
        print(f"\n{'─' * 62}")
        print(f"  【意图已完善】— 澄清回复 refinement")
        print(f"{'─' * 62}")
        print(f"\n  {clar.refinement_note}")
        print(f"\n  完善后意图: {intent.intent_label}")
        print(f"  参考分类: {cat_name}  |  置信度: {intent.confidence:.0%}")
        if intent.reasoning:
            print(f"  推理: {intent.reasoning}")
    elif clar.needs_clarification:
        print(f"\n{'=' * 62}")
        print(f"  【客服引导】— 意图待澄清 ({clar.reason.value})")
        print(f"{'=' * 62}")
        print(f"\n  {clar.guide_response}")
        if clar.clarification_questions:
            print(f"\n  📋 澄清问题：")
            for i, q in enumerate(clar.clarification_questions, 1):
                purpose = f"（{q.purpose}）" if q.purpose else ""
                print(f"    {i}. {q.question}{purpose}")
        if clar.suggested_options:
            print(f"\n  或直接选择：")
            for i, opt in enumerate(clar.suggested_options, 1):
                print(f"    {i}. {opt}")

    print(f"\n【性能】\n  延迟: {resp.total_latency_ms:.0f}ms")
    print()


def run_chat() -> None:
    pipeline = IntentPipeline()
    session_id: str | None = None
    turn = 0

    print_banner()
    print("开始对话（输入 help 查看帮助）\n")

    while True:
        try:
            user_input = input("用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n对话结束，再见！")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in EXIT_COMMANDS:
            print("\n对话结束，再见！")
            break
        if cmd in HELP_COMMANDS:
            print_help()
            continue
        if cmd in CATEGORIES_COMMANDS:
            print_categories()
            continue
        if cmd in RESET_COMMANDS:
            if session_id:
                pipeline.reset_session(session_id)
            session_id = None
            turn = 0
            print("\n✓ 已开启新对话\n")
            continue
        if cmd in HISTORY_COMMANDS:
            if session_id:
                print_session_history(pipeline, session_id)
            else:
                print("\n（当前无对话历史）\n")
            continue

        turn += 1
        resp = pipeline.process(user_input, session_id=session_id)
        session_id = resp.session_id
        print_analysis(resp, turn)


if __name__ == "__main__":
    run_chat()
