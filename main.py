"""Demo multi-turn dialogue intent capture — typical insurance customer service scenarios."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.pipeline import IntentPipeline, format_response


def run_demo() -> None:
    pipeline = IntentPipeline()
    session_id = None

    scenarios = [
        ("Hello, I'd like to learn about critical illness insurance", "Turn 1: Greeting + product inquiry"),
        ("How much is the premium for Anxin Critical Illness 2026?", "Slot filling: product + premium query"),
        ("How long is its waiting period?", "Coreference: 'its' → Anxin Critical Illness 2026"),
        ("Tell me about the medical insurance claims process", "Intent drift + multi-intent"),
        ("I travel frequently for work", "Implicit intent: accident insurance recommendation"),
        ("Compare Anxin Critical Illness 2026 and Kangle Medical Plus for me", "Multi-slot: product comparison"),
    ]

    print("=" * 60)
    print("Multi-Turn Intent Capture Demo — Insurance Smart Assistant")
    print("Mode: LLM dynamic intent capture + insurance reference taxonomy")
    llm = settings.llm
    if llm.is_configured:
        print(f"LLM: {llm.display_name} ({llm.model})")
    else:
        print("LLM: Not configured — using rule fallback")
    print("=" * 60)

    for i, (utterance, desc) in enumerate(scenarios, 1):
        print(f"\n--- Turn {i}: {desc} ---")
        resp = pipeline.process(utterance, session_id=session_id)
        session_id = resp.session_id
        print(format_response(resp))

    print("\n" + "=" * 60)
    print("Demo complete")
    print(f"Final session state — active product: {pipeline.get_or_create_session(session_id).active_product}")
    print(f"Topic stack: {pipeline.get_or_create_session(session_id).topic_stack}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-turn dialogue intent recognition system")
    parser.add_argument(
        "--mode",
        choices=["demo", "chat"],
        default="demo",
        help="demo=preset scenario demo, chat=interactive multi-turn dialogue",
    )
    args = parser.parse_args()

    if args.mode == "chat":
        from chat import run_chat
        run_chat()
    else:
        run_demo()
