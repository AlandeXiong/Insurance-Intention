"""演示多轮对话意图捕捉 — 保险客服典型场景。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 path 中
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.pipeline import IntentPipeline, format_response


def run_demo() -> None:
    pipeline = IntentPipeline()
    session_id = None

    scenarios = [
        ("你好，我想了解一下重疾险", "首轮：问候 + 产品咨询"),
        ("安心保重疾险2026年保费多少？", "槽位填充：产品 + 保费查询"),
        ("那它的等待期是多久？", "指代消解：'它' → 安心保"),
        ("另外医疗险理赔流程是怎样的？", "意图漂移 + 多意图"),
        ("我最近经常出差", "隐式意图：意外险推荐"),
        ("帮我对比一下安心保和康乐医疗险Plus", "多槽位：产品对比"),
    ]

    print("=" * 60)
    print("多轮对话意图捕捉演示 — 保险智能客服")
    print("模式: LLM 动态意图捕获 + 保险行业参考分类")
    llm = settings.llm
    if llm.is_configured:
        print(f"LLM: {llm.display_name} ({llm.model})")
    else:
        print("LLM: 未配置，使用规则降级")
    print("=" * 60)

    for i, (utterance, desc) in enumerate(scenarios, 1):
        print(f"\n--- Turn {i}: {desc} ---")
        resp = pipeline.process(utterance, session_id=session_id)
        session_id = resp.session_id
        print(format_response(resp))

    print("\n" + "=" * 60)
    print("演示完成")
    print(f"最终会话状态 — 活跃产品: {pipeline.get_or_create_session(session_id).active_product}")
    print(f"主题栈: {pipeline.get_or_create_session(session_id).topic_stack}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多轮对话意图捕捉系统")
    parser.add_argument(
        "--mode",
        choices=["demo", "chat"],
        default="demo",
        help="demo=预设场景演示, chat=交互式多轮对话",
    )
    args = parser.parse_args()

    if args.mode == "chat":
        from chat import run_chat
        run_chat()
    else:
        run_demo()
