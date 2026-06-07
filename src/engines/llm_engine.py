"""Backend LLM intent engine — DeepSeek / Alibaba Qwen dynamic intent capture."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List

import httpx

from config.settings import settings
from src.domain.insurance_domain import CATEGORY_BY_CODE, build_category_prompt
from src.models.intent import (
    ImplicitIntent,
    IntentResult,
    IntentSource,
    SessionContext,
    Slot,
    SlotStatus,
    SubIntent,
)

logger = logging.getLogger(__name__)

# Replace placeholder separately to avoid JSON braces conflicting with format
_CATEGORY_PLACEHOLDER = "{{CATEGORY_FRAMEWORK}}"

SYSTEM_PROMPT = f"""You are a senior insurance intelligent customer service intent analysis expert. Based on conversation context, dynamically understand the user's true intent in their latest input.

{_CATEGORY_PLACEHOLDER}

## Core Principles
1. **Dynamic capture**: Do not mechanically apply fixed labels; use natural language to precisely describe the user's current intent (intent_label) in English
2. **Reference categories**: Select the closest category from the reference list above (code); refinement and combination are allowed
3. **Multi-turn understanding**: Use conversation history to resolve references ("it/this/that" — which product?) and inherit context
4. **Implicit mining**: Identify unstated but inferable needs (e.g., "frequent business travel" → may need accident insurance)
5. **Multiple intents**: When one utterance contains multiple questions, split into sub_intents
6. **Intent clarification**: When the user's statement is vague, information is insufficient, confidence is below 0.72, or category is other, you must set needs_clarification=true and:
   - Generate guide_response (customer service guidance in English)
   - Provide clarification_questions (1-3 specific follow-ups, each with question_id, question, purpose, fills_slot)
   - Provide options (2-5 possible directions)
7. **Clarification refinement**: If context shows clarification is in progress and the current input is the user's answer to a follow-up, you must:
   - Combine original_utterance + user supplement to produce a refined precise intent
   - Set needs_clarification to false; confidence should be ≥ 0.85
   - Explain the before/after intent change in reasoning

## Output Requirements
Output only a JSON object — no markdown code blocks:
{{
  "primary_intent": {{
    "intent_label": "One precise English sentence describing user intent, e.g.: Query the waiting period for Anxin Critical Illness 2026",
    "category": "reference category code, e.g. coverage_terms",
    "confidence": 0.0-1.0
  }},
  "sub_intents": [
    {{"intent_label": "...", "category": "...", "confidence": 0.0-1.0, "is_primary": false}}
  ],
  "implicit_intents": [
    {{"intent_label": "...", "category": "...", "trigger": "source text fragment that triggered implicit inference", "confidence": 0.0-1.0}}
  ],
  "slots": {{"slot_name": "value or null"}},
  "drift_detected": false,
  "drift_reason": "If topic shift detected, explain why; otherwise empty string",
  "missing_info": ["information still needed to fulfill intent, e.g. product_name, age"],
  "reasoning": "Brief reasoning chain (reference resolution, intent judgment basis)",
  "clarification": {{
    "needs_clarification": true,
    "reason": "vague_utterance/missing_info/low_confidence/ambiguous/unrecognized",
    "guide_response": "Friendly, natural customer service guidance in English — speak directly to the user to help clarify intent",
    "clarification_questions": [
      {{"question_id": "q1", "question": "specific follow-up question", "purpose": "purpose of the question", "fills_slot": "product_name"}}
    ],
    "options": ["Option 1: ...", "Option 2: ..."],
    "follow_up_questions": ["follow-up 1", "follow-up 2"]
  }}
}}
"""


class LLMIntentEngine:
    """Chain-of-Intent — dynamic intent capture, not fixed taxonomy (OpenAI-compatible API)."""

    def __init__(self) -> None:
        self.config = settings.llm

    def is_available(self) -> bool:
        return self.config.is_configured

    async def predict_async(
        self,
        utterance: str,
        ctx: SessionContext,
        context_window: str,
    ) -> IntentResult:
        start = time.perf_counter()
        result = await self._predict(utterance, ctx, context_window)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def predict(
        self,
        utterance: str,
        ctx: SessionContext,
        context_window: str,
    ) -> IntentResult:
        start = time.perf_counter()
        if self.is_available():
            try:
                result = self._call_llm_sync(utterance, context_window, ctx)
            except Exception as exc:
                logger.warning("%s sync call failed, falling back: %s", self.config.display_name, exc)
                result = self._fallback_predict(utterance, ctx)
        else:
            result = self._fallback_predict(utterance, ctx)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    async def _predict(
        self,
        utterance: str,
        ctx: SessionContext,
        context_window: str,
    ) -> IntentResult:
        if self.is_available():
            try:
                return await self._call_llm(utterance, context_window, ctx)
            except Exception as exc:
                logger.warning("%s call failed, falling back: %s", self.config.display_name, exc)
        return self._fallback_predict(utterance, ctx)

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT.replace(_CATEGORY_PLACEHOLDER, build_category_prompt())

    def _build_messages(self, utterance: str, context_window: str, ctx: SessionContext) -> list:
        user_msg = (
            f"Conversation context:\n{context_window}\n\n"
            f"User profile hints: {ctx.user_profile_hints or 'none'}\n\n"
            f"User's latest input: {utterance}"
        )
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_msg},
        ]

    def _build_request_body(self, utterance: str, context_window: str, ctx: SessionContext) -> dict:
        return {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": self._build_messages(utterance, context_window, ctx),
            "response_format": {"type": "json_object"},
        }

    def _call_llm_sync(
        self, utterance: str, context_window: str, ctx: SessionContext
    ) -> IntentResult:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.config.timeout_s) as client:
            resp = client.post(
                self.config.chat_completions_url,
                headers=headers,
                json=self._build_request_body(utterance, context_window, ctx),
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return self._parse_llm_response(content)

    async def _call_llm(
        self, utterance: str, context_window: str, ctx: SessionContext
    ) -> IntentResult:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
            resp = await client.post(
                self.config.chat_completions_url,
                headers=headers,
                json=self._build_request_body(utterance, context_window, ctx),
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return self._parse_llm_response(content)

    def _fallback_predict(self, utterance: str, ctx: SessionContext) -> IntentResult:
        """Minimal fallback when LLM is unavailable — keyword coarse match + reference category only."""
        from src.engines.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        label, category, conf = extractor.infer_intent_hint(utterance, ctx)
        slots = extractor.extract_entities(utterance, ctx)

        return IntentResult(
            intent_label=label,
            category=category,
            confidence=conf,
            source=IntentSource.LLM,
            sub_intents=[SubIntent(intent_label=label, category=category, confidence=conf, is_primary=True)],
            slots=slots,
            reasoning=f"Rule-based fallback inference ({self.config.display_name} unavailable)",
        )

    def _parse_llm_response(self, content: str) -> IntentResult:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError(f"Failed to parse LLM JSON: {content[:200]}")
            data = json.loads(match.group())

        primary = data.get("primary_intent") or {}
        intent_label = primary.get("intent_label") or "Could not recognize user intent"
        category = self._normalize_category(primary.get("category", "other"))
        confidence = float(primary.get("confidence", 0.5))

        sub_intents = self._parse_sub_intents(data.get("sub_intents", []), intent_label, category, confidence)
        implicit = self._parse_implicit(data.get("implicit_intents", []))
        slots = self._parse_slots(data.get("slots") or {})

        drift_detected = bool(data.get("drift_detected", False))
        from src.models.intent import DriftType
        drift_type = DriftType.TOPIC_SHIFT if drift_detected else DriftType.NONE

        return IntentResult(
            intent_label=intent_label,
            category=category,
            confidence=confidence,
            source=IntentSource.LLM,
            sub_intents=sub_intents,
            implicit_intents=implicit,
            slots=slots,
            drift_detected=drift_detected,
            drift_type=drift_type,
            drift_reason=data.get("drift_reason", ""),
            reasoning=data.get("reasoning", ""),
            missing_info=data.get("missing_info") or [],
            raw_scores={category: confidence},
            llm_clarification=data.get("clarification") or {},
        )

    def _normalize_category(self, code: str) -> str:
        code = (code or "other").strip()
        return code if code in CATEGORY_BY_CODE else "other"

    def _parse_sub_intents(
        self, items: List[Any], primary_label: str, primary_cat: str, primary_conf: float
    ) -> List[SubIntent]:
        result = []
        for s in items:
            label = s.get("intent_label", "")
            if not label:
                continue
            result.append(SubIntent(
                intent_label=label,
                category=self._normalize_category(s.get("category", "other")),
                confidence=float(s.get("confidence", 0.5)),
                is_primary=bool(s.get("is_primary", False)),
            ))
        if not result:
            result.append(SubIntent(
                intent_label=primary_label,
                category=primary_cat,
                confidence=primary_conf,
                is_primary=True,
            ))
        elif not any(s.is_primary for s in result):
            result[0].is_primary = True
        return result

    def _parse_implicit(self, items: List[Any]) -> List[ImplicitIntent]:
        result = []
        for item in items:
            label = item.get("intent_label", "")
            if not label:
                continue
            result.append(ImplicitIntent(
                intent_label=label,
                category=self._normalize_category(item.get("category", "product_recommend")),
                trigger=item.get("trigger", ""),
                confidence=float(item.get("confidence", 0.5)),
            ))
        return result

    def _parse_slots(self, raw: Dict[str, Any]) -> Dict[str, Slot]:
        slots = {}
        for name, value in raw.items():
            if value is not None and value != "":
                slots[name] = Slot(name=name, value=value, status=SlotStatus.FILLED, confidence=0.9)
        return slots
