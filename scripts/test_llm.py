#!/usr/bin/env python3
"""LLM API 连通性测试（按 LLM_PROVIDER 自动选择 DeepSeek / 千问）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.engines.llm_engine import LLMIntentEngine
from src.models.intent import SessionContext


def main() -> None:
    llm = settings.llm
    print(f"{llm.display_name} API 连通性测试")
    print(f"  Provider : {llm.provider}")
    print(f"  API Base : {llm.api_base}")
    print(f"  Model    : {llm.model}")
    print(f"  URL      : {llm.chat_completions_url}")
    print(f"  Key      : {'已配置 (' + llm.api_key[:8] + '...)' if llm.is_configured else '未配置'}")

    if not llm.is_configured:
        print(f"\n请在 .env 中设置 {llm.api_key_env}，并确认 LLM_PROVIDER={llm.provider}")
        print("参考 .env.example")
        sys.exit(1)

    engine = LLMIntentEngine()
    ctx = SessionContext(session_id=f"{llm.provider}-test")
    context = "[当前焦点产品: 安心保重疾险2026]"

    test_cases = [
        "那它的等待期是多久？",
        "我最近经常出差，有什么保险推荐？",
        "保费多少，还有等待期是多久？",
    ]

    for utterance in test_cases:
        print(f"\n--- 输入: {utterance} ---")
        try:
            result = engine.predict(utterance, ctx, context)
            print(f"  意图    : {result.intent_label} ({result.confidence:.2f})")
            print(f"  延迟    : {result.latency_ms:.0f}ms")
            if result.sub_intents:
                print(f"  子意图  : {[s.intent_label for s in result.sub_intents]}")
            if result.implicit_intents:
                print(f"  隐式意图: {result.implicit_intents}")
            slots = {k: v.value for k, v in result.slots.items() if v.value}
            if slots:
                print(f"  槽位    : {slots}")
        except Exception as exc:
            print(f"  失败    : {exc}")
            sys.exit(1)

    print(f"\n✓ {llm.display_name} API 测试通过")


if __name__ == "__main__":
    main()
