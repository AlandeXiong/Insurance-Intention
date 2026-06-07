"""后端大模型意图引擎 — DeepSeek / 阿里云千问 动态意图捕获。"""

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

# 占位符单独替换，避免 JSON 花括号与 format 冲突
_CATEGORY_PLACEHOLDER = "{{CATEGORY_FRAMEWORK}}"

SYSTEM_PROMPT = f"""你是资深保险智能客服意图分析专家。请基于对话上下文，动态理解用户最新输入的真实意图。

{_CATEGORY_PLACEHOLDER}

## 核心原则
1. **动态捕获**：不要机械套用固定标签，用自然语言精确描述用户当前意图（intent_label）
2. **参考分类**：category 字段从上述参考分类中选择最接近的一项（code），允许细化和组合
3. **多轮理解**：结合历史对话消解指代（"它/这个/那个"指什么产品）、继承上下文
4. **隐式挖掘**：识别用户未明说但可推断的需求（如"经常出差"→可能需要交通意外险）
5. **多意图并存**：一句含多个问题时，拆分为 sub_intents
6. **意图澄清**：当用户表述模糊、信息不足、置信度低于 0.72 或 category 为 other 时，必须设置 needs_clarification=true，并：
   - 生成 guide_response（客服引导话术）
   - 给出 clarification_questions（1-3 个具体追问，每项含 question_id、question、purpose、fills_slot）
   - 给出 options（2-5 个可选方向）
7. **澄清回复 refinement**：若上下文显示「澄清进行中」，用户当前输入是对追问的回答，必须：
   - 结合 original_utterance + 用户补充，给出 refined 精确意图
   - needs_clarification 设为 false，confidence 应 ≥ 0.85
   - reasoning 中说明澄清前后意图变化

## 输出要求
必须只输出 JSON 对象，不要 markdown 代码块：
{{
  "primary_intent": {{
    "intent_label": "用一句中文精确描述用户意图，如：查询安心保重疾险的等待期",
    "category": "参考分类 code，如 coverage_terms",
    "confidence": 0.0-1.0
  }},
  "sub_intents": [
    {{"intent_label": "...", "category": "...", "confidence": 0.0-1.0, "is_primary": false}}
  ],
  "implicit_intents": [
    {{"intent_label": "...", "category": "...", "trigger": "触发隐式推断的原文片段", "confidence": 0.0-1.0}}
  ],
  "slots": {{"槽位名": "值或null"}},
  "drift_detected": false,
  "drift_reason": "若检测到话题切换则说明原因，否则空字符串",
  "missing_info": ["完成该意图还需追问的信息，如 product_name, age"],
  "reasoning": "简短推理链（指代消解、意图判断依据）",
  "clarification": {{
    "needs_clarification": true,
    "reason": "vague_utterance/missing_info/low_confidence/ambiguous/unrecognized",
    "guide_response": "亲切自然的客服引导话术，直接对用户说，帮助其明确意图",
    "clarification_questions": [
      {{"question_id": "q1", "question": "具体追问问题", "purpose": "追问目的", "fills_slot": "product_name"}}
    ],
    "options": ["选项1：...", "选项2：..."],
    "follow_up_questions": ["追问1", "追问2"]
  }}
}}
"""


class LLMIntentEngine:
    """Chain-of-Intent — 动态意图捕获，非固定分类（OpenAI 兼容 API）。"""

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
                logger.warning("%s 同步调用失败，降级: %s", self.config.display_name, exc)
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
                logger.warning("%s 调用失败，降级: %s", self.config.display_name, exc)
        return self._fallback_predict(utterance, ctx)

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT.replace(_CATEGORY_PLACEHOLDER, build_category_prompt())

    def _build_messages(self, utterance: str, context_window: str, ctx: SessionContext) -> list:
        user_msg = (
            f"对话上下文：\n{context_window}\n\n"
            f"用户画像线索：{ctx.user_profile_hints or '无'}\n\n"
            f"用户最新输入：{utterance}"
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
        """无 LLM 时的最小降级 — 仅做关键词粗匹配 + 参考分类。"""
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
            reasoning=f"规则降级推断（{self.config.display_name} 不可用）",
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
                raise ValueError(f"无法解析 LLM JSON: {content[:200]}")
            data = json.loads(match.group())

        primary = data.get("primary_intent") or {}
        intent_label = primary.get("intent_label") or "未能识别用户意图"
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
