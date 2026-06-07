"""Interactive multi-turn dialogue — LLM dynamic intent analysis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.domain.insurance_domain import get_category_name
from src.pipeline import IntentPipeline, PipelineResponse

EXIT_COMMANDS = {"quit", "exit", "q"}
RESET_COMMANDS = {"reset"}
HELP_COMMANDS = {"help", "?"}
HISTORY_COMMANDS = {"history", "context"}
CATEGORIES_COMMANDS = {"categories"}


def print_banner() -> None:
    llm = settings.llm
    if llm.is_configured:
        mode = f"{llm.display_name} dynamic intent ({llm.model})"
    else:
        mode = f"Rule fallback (configure {llm.api_key_env} and set LLM_PROVIDER={llm.provider})"
    print()
    print("=" * 62)
    print("  Insurance Smart Assistant — Multi-Turn Dynamic Intent Analysis")
    print("  Insurance industry intent framework powered by LLM understanding")
    print(f"  Engine: {mode}")
    print("=" * 62)
    print()


def print_help() -> None:
    print("""
Available commands:
  help              Show this help
  categories        View insurance intent reference categories
  history / context View current session context
  reset             Clear session and start over
  quit / exit       End the conversation

Example dialogue:
  > Hello, I'd like to learn about critical illness insurance
  > How much is the premium for Anxin Critical Illness 2026?
  > How long is its waiting period?
  > Tell me about the medical insurance claims process
  > I travel frequently for work — any recommendations?
""")


def print_categories() -> None:
    from src.domain.insurance_domain import INSURANCE_INTENT_CATEGORIES
    print("\n--- Insurance Customer Intent Reference Categories ---")
    for cat in INSURANCE_INTENT_CATEGORIES:
        print(f"  · {cat.name} ({cat.code}): {cat.description}")
    print("  (Reference for the LLM; actual intents are captured dynamically)\n")


def print_session_history(pipeline: IntentPipeline, session_id: str) -> None:
    ctx = pipeline.get_or_create_session(session_id)
    print("\n--- Session Context ---")
    print(f"Session ID     : {ctx.session_id[:8]}...")
    print(f"Active product : {ctx.active_product or '(none)'}")
    print(f"Current intent : {ctx.active_intent_label or '(none)'}")
    if ctx.active_category:
        print(f"Reference cat. : {get_category_name(ctx.active_category)}")
    print(f"Topic stack    : {' → '.join(ctx.topic_stack[-4:]) if ctx.topic_stack else '(empty)'}")
    if ctx.user_profile_hints:
        print(f"User profile   : {', '.join(ctx.user_profile_hints)}")
    print(f"\nDialogue history ({len(ctx.turns)} turns):")
    for t in ctx.turns:
        role = "User" if t.role == "user" else "Agent"
        tag = f" [{t.intent_label}]" if t.intent_label else ""
        print(f"  [{t.turn_id + 1}] {role}{tag}: {t.content}")
    print()


def print_analysis(resp: PipelineResponse, turn: int) -> None:
    intent = resp.intent
    cat_name = resp.metadata.get("category_name", get_category_name(intent.category))

    print()
    print("─" * 62)
    print(f"  Turn {turn} — Dynamic Intent Analysis")
    print("─" * 62)

    print(f"\n[User Input]\n  {resp.utterance}")

    if resp.resolved_utterance != resp.utterance:
        print(f"\n[Coreference Resolution]\n  {resp.resolved_utterance}")
        for pronoun, entity in intent.resolved_references.items():
            print(f"  '{pronoun}' → {entity}")

    print(f"\n[Dynamic Intent]")
    print(f"  {intent.intent_label}")
    print(f"  Reference category: {cat_name}")
    print(f"  Confidence: {intent.confidence:.0%}  |  Path: {resp.engine_path}")

    if intent.reasoning:
        print(f"\n[Reasoning]\n  {intent.reasoning}")

    extras = [s for s in intent.sub_intents if not s.is_primary]
    if extras:
        print(f"\n[Additional Intents]")
        for si in extras:
            print(f"  · {si.intent_label}  [{get_category_name(si.category)}]  {si.confidence:.0%}")

    if intent.implicit_intents:
        print(f"\n[Implicit Needs]")
        for imp in intent.implicit_intents:
            trigger = f" (trigger: {imp.trigger})" if imp.trigger else ""
            print(f"  · {imp.intent_label}{trigger}")

    if intent.drift_detected:
        print(f"\n[Topic Shift]\n  {intent.drift_reason or intent.drift_type.value}")
        drift_sigs = resp.metadata.get("drift_signals", {})
        if drift_sigs.get("drift_fused_score") is not None:
            print(f"  Drift fusion score: {drift_sigs.get('drift_fused_score', 0):.2f}")
    elif resp.metadata.get("context_dependent"):
        print(f"\n[Context Continuation] Strongly depends on prior context — same-topic follow-up")

    filled = {k: v.value for k, v in intent.slots.items() if v.value is not None}
    if filled:
        print(f"\n[Key Information]")
        for name, value in filled.items():
            print(f"  · {name}: {value}")

    if resp.missing_slots:
        print(f"\n[Information Needed]")
        for item in resp.missing_slots:
            print(f"  · {item}")

    clar = resp.clarification
    if clar.resolved_from_clarification:
        print(f"\n{'─' * 62}")
        print(f"  [Intent Refined] — Clarification reply refinement")
        print(f"{'─' * 62}")
        print(f"\n  {clar.refinement_note}")
        print(f"\n  Refined intent: {intent.intent_label}")
        print(f"  Reference category: {cat_name}  |  Confidence: {intent.confidence:.0%}")
        if intent.reasoning:
            print(f"  Reasoning: {intent.reasoning}")
    elif clar.needs_clarification:
        print(f"\n{'=' * 62}")
        print(f"  [Agent Guidance] — Intent needs clarification ({clar.reason.value})")
        print(f"{'=' * 62}")
        print(f"\n  {clar.guide_response}")
        if clar.clarification_questions:
            print(f"\n  Clarification questions:")
            for i, q in enumerate(clar.clarification_questions, 1):
                purpose = f" ({q.purpose})" if q.purpose else ""
                print(f"    {i}. {q.question}{purpose}")
        if clar.suggested_options:
            print(f"\n  Or choose directly:")
            for i, opt in enumerate(clar.suggested_options, 1):
                print(f"    {i}. {opt}")

    print(f"\n[Performance]\n  Latency: {resp.total_latency_ms:.0f}ms")
    print()


def run_chat() -> None:
    pipeline = IntentPipeline()
    session_id: str | None = None
    turn = 0

    print_banner()
    print("Start chatting (type help for commands)\n")

    while True:
        try:
            user_input = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nConversation ended. Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in EXIT_COMMANDS:
            print("\nConversation ended. Goodbye!")
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
            print("\n✓ New conversation started\n")
            continue
        if cmd in HISTORY_COMMANDS:
            if session_id:
                print_session_history(pipeline, session_id)
            else:
                print("\n(No conversation history yet)\n")
            continue

        turn += 1
        resp = pipeline.process(user_input, session_id=session_id)
        session_id = resp.session_id
        print_analysis(resp, turn)


if __name__ == "__main__":
    run_chat()
