"""澄清回复解析 — 结合用户澄清回答 refinement 意图。"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.domain.insurance_domain import CATEGORY_BY_CODE, get_category_name
from src.models.clarification import (
    ClarificationGuide,
    ClarificationQuestion,
    ClarificationReason,
    PendingClarification,
)
from src.models.intent import IntentResult, SessionContext


# 用户选序号 / 关键词 → 参考分类
OPTION_CATEGORY_HINTS: list[tuple[list[str], str, str]] = [
    (["重疾", "重疾险"], "product_inquiry", "了解重疾险产品保障"),
    (["医疗", "医疗险"], "product_inquiry", "了解医疗险产品保障"),
    (["意外", "意外险"], "product_inquiry", "了解意外险产品保障"),
    (["保费", "多少钱", "价格"], "premium_inquiry", "查询保险产品保费"),
    (["理赔", "报销", "赔付"], "claims_service", "咨询保险理赔流程"),
    (["对比", "比较"], "product_compare", "对比不同保险产品"),
    (["推荐", "买什么"], "product_recommend", "寻求保险产品推荐"),
    (["续保", "退保", "保单"], "policy_service", "办理保单相关服务"),
    (["等待期", "免责", "条款"], "coverage_terms", "查询保障条款"),
    (["购买", "投保"], "purchase", "表达投保购买意向"),
]


class ClarificationResolver:
    """处理澄清追问与用户回复，完善意图识别。"""

    def is_followup_to_clarification(self, ctx: SessionContext) -> bool:
        return ctx.pending_clarification.active

    def enrich_utterance(
        self, ctx: SessionContext, utterance: str
    ) -> Tuple[str, dict]:
        """
        将澄清回复与原始模糊输入合并，供 LLM  refinement。
        返回 (enriched_utterance, meta)
        """
        pending = ctx.pending_clarification
        if not pending.active:
            return utterance, {}

        resolved_answer = self._resolve_user_answer(utterance, pending)
        enriched = (
            f"[澄清上下文] 用户最初说：「{pending.original_utterance}」；"
            f"客服追问后，用户补充：「{utterance}」"
        )
        if resolved_answer.get("matched_option"):
            enriched += f"；用户选择了：{resolved_answer['matched_option']}"
        if resolved_answer.get("inferred_category"):
            cat_name = get_category_name(resolved_answer["inferred_category"])
            enriched += f"；推断意图方向：{cat_name}"

        return enriched, {
            "is_clarification_followup": True,
            "original_utterance": pending.original_utterance,
            "tentative_intent": pending.tentative_intent_label,
            "tentative_category": pending.tentative_category,
            **resolved_answer,
        }

    def build_clarification_context_block(self, ctx: SessionContext) -> str:
        pending = ctx.pending_clarification
        if not pending.active:
            return ""

        lines = [
            "[澄清进行中 — 用户正在回答以下追问]",
            f"原始输入: {pending.original_utterance}",
            f"初步理解: {pending.tentative_intent_label} ({get_category_name(pending.tentative_category)})",
        ]
        for i, q in enumerate(pending.questions, 1):
            lines.append(f"  追问{i}: {q.question}")
        if pending.suggested_options:
            lines.append("可选方向: " + " | ".join(pending.suggested_options))
        lines.append("请结合用户最新回复，给出 refined 后的精确意图，needs_clarification 应为 false")
        return "\n".join(lines)

    def refine_intent_after_clarification(
        self,
        result: IntentResult,
        pending: PendingClarification,
        utterance: str,
        resolve_meta: dict,
    ) -> Tuple[IntentResult, ClarificationGuide]:
        """澄清回复后提升意图置信度并标记 refinement。"""
        matched_cat = resolve_meta.get("inferred_category") or pending.tentative_category
        matched_label = resolve_meta.get("inferred_label") or pending.tentative_intent_label

        # 用户选了序号或关键词，且 LLM 置信度仍低 → 规则补强
        if resolve_meta.get("matched_option") and result.confidence < 0.8:
            if matched_cat and matched_cat in CATEGORY_BY_CODE:
                result.category = matched_cat
            if matched_label and len(result.intent_label) < 8:
                result.intent_label = f"{matched_label}（用户补充：{utterance}）"
            result.confidence = max(result.confidence, 0.85)

        # 合并原始意图与补充信息
        if pending.tentative_intent_label and pending.original_utterance:
            if utterance not in result.intent_label:
                result.intent_label = (
                    f"{result.intent_label}"
                    if result.confidence >= 0.85
                    else f"{pending.tentative_intent_label}，补充说明：{utterance}"
                )
            result.confidence = max(result.confidence, 0.82)
            result.reasoning = (
                f"澄清 refinement：原输入「{pending.original_utterance}」→ "
                f"用户补充「{utterance}」→ {result.reasoning or result.intent_label}"
            )

        note = (
            f"已通过澄清完善意图：{pending.original_utterance} + {utterance} "
            f"→ {result.intent_label}（置信度 {result.confidence:.0%}）"
        )
        guide = ClarificationGuide(
            needs_clarification=False,
            resolved_from_clarification=True,
            refinement_note=note,
        )
        return result, guide

    def create_pending_state(
        self,
        guide: ClarificationGuide,
        result: IntentResult,
        utterance: str,
        turn_id: int,
    ) -> PendingClarification:
        return PendingClarification(
            active=True,
            reason=guide.reason,
            original_utterance=utterance,
            tentative_intent_label=result.intent_label,
            tentative_category=result.category,
            questions=guide.clarification_questions,
            suggested_options=guide.suggested_options,
            asked_at_turn=turn_id,
        )

    def _resolve_user_answer(self, utterance: str, pending: PendingClarification) -> dict:
        text = utterance.strip()
        meta: dict = {}

        # 序号选择：1 / 第一个 / 选1
        num_match = re.match(r"^[选]?(\d)[\s\.、]?$", text)
        if num_match and pending.suggested_options:
            idx = int(num_match.group(1)) - 1
            if 0 <= idx < len(pending.suggested_options):
                meta["matched_option"] = pending.suggested_options[idx]
                meta.update(self._infer_from_text(pending.suggested_options[idx]))

        # 直接匹配某个选项文本
        if not meta.get("matched_option"):
            for opt in pending.suggested_options:
                if opt in text or text in opt:
                    meta["matched_option"] = opt
                    meta.update(self._infer_from_text(opt))
                    break

        # 关键词推断
        if not meta.get("inferred_category"):
            meta.update(self._infer_from_text(text))

        return meta

    def _infer_from_text(self, text: str) -> dict:
        for keywords, category, label in OPTION_CATEGORY_HINTS:
            if any(kw in text for kw in keywords):
                return {"inferred_category": category, "inferred_label": label}
        return {}
