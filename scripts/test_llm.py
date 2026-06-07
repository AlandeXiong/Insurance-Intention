#!/usr/bin/env python3
"""LLM API connectivity test (auto-selects DeepSeek / Qwen via LLM_PROVIDER)."""

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
    print(f"{llm.display_name} API connectivity test")
    print(f"  Provider : {llm.provider}")
    print(f"  API Base : {llm.api_base}")
    print(f"  Model    : {llm.model}")
    print(f"  URL      : {llm.chat_completions_url}")
    print(f"  Key      : {'configured (' + llm.api_key[:8] + '...)' if llm.is_configured else 'not configured'}")

    if not llm.is_configured:
        print(f"\nSet {llm.api_key_env} in .env and confirm LLM_PROVIDER={llm.provider}")
        print("See .env.example")
        sys.exit(1)

    engine = LLMIntentEngine()
    ctx = SessionContext(session_id=f"{llm.provider}-test")
    context = "[Current focus product: Anxin Critical Illness 2026]"

    test_cases = [
        "How long is its waiting period?",
        "I travel frequently for work — any insurance recommendations?",
        "How much is the premium, and how long is the waiting period?",
    ]

    for utterance in test_cases:
        print(f"\n--- Input: {utterance} ---")
        try:
            result = engine.predict(utterance, ctx, context)
            print(f"  Intent     : {result.intent_label} ({result.confidence:.2f})")
            print(f"  Latency    : {result.latency_ms:.0f}ms")
            if result.sub_intents:
                print(f"  Sub-intents: {[s.intent_label for s in result.sub_intents]}")
            if result.implicit_intents:
                print(f"  Implicit   : {result.implicit_intents}")
            slots = {k: v.value for k, v in result.slots.items() if v.value}
            if slots:
                print(f"  Slots      : {slots}")
        except Exception as exc:
            print(f"  Failed     : {exc}")
            sys.exit(1)

    print(f"\n✓ {llm.display_name} API test passed")


if __name__ == "__main__":
    main()
