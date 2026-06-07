"""意图澄清引导 — 检测模糊意图并生成客服引导话术。"""

from __future__ import annotations

import re
from typing import List, Optional

from config.settings import settings
from src.domain.insurance_domain import CATEGORY_BY_CODE, get_category_name
from src.models.clarification import ClarificationGuide, ClarificationQuestion, ClarificationReason, PendingClarification
from src.models.intent import IntentResult, SessionContext

# 模糊表述特征
VAGUE_PATTERNS = [
    r"^(了解|咨询|问问|想知道|帮我|看看).{0,6}$",
    r"^(那个|这个|它).{0,8}$",
    r"^怎么(办|样|弄)$",
    r"^有关?保险",
    r"^关于",
]

# 槽位名 → 中文追问
SLOT_LABELS = {
    "product_name": "想了解哪款保险产品",
    "product_a": "第一款对比产品",
    "product_b": "第二款对比产品",
    "age": "您的年龄",
    "gender": "您的性别",
    "coverage_amount": "期望的保额（如50万）",
    "payment_period": "计划缴费年限",
    "budget": "预算范围",
    "insurance_type": "险种类型（重疾/医疗/意外/寿险）",
    "claim_type": "理赔类型（住院/门诊/手术等）",
    "policy_no": "保单号",
    "scenario": "使用场景（如出差、育儿、养老）",
}

# 参考分类 → 缺信息时的引导模板
CATEGORY_GUIDE_TEMPLATES = {
    "product_inquiry": (
        "您好！我可以帮您介绍保险产品的保障内容。"
        "请问您想了解哪个险种？例如：重疾险、医疗险、意外险或寿险？"
    ),
    "premium_inquiry": (
        "您好！保费会根据产品、年龄、保额和缴费方式有所不同。"
        "请问您想了解哪款产品的保费？或者告诉我您的年龄和期望保额，我来帮您估算。"
    ),
    "coverage_terms": (
        "您好！关于保障条款，不同产品规则不同。"
        "请问您想查询哪款产品？是想了解等待期、免责条款还是续保规则？"
    ),
    "claims_service": (
        "您好！我可以为您介绍理赔流程。"
        "请问您是哪个险种的理赔（医疗/重疾/意外）？是否已有保单？"
    ),
    "purchase": (
        "您好！很高兴您有投保意向。"
        "请问您想购买哪类产品？是否已有心仪的产品名称？"
    ),
    "product_compare": (
        "您好！我可以帮您对比保险产品。"
        "请告诉我您想对比哪两款产品，或说明您关注的对比维度（保费/保障/理赔）。"
    ),
    "policy_service": (
        "您好！请问您需要办理哪项保单服务？"
        "例如：续保、退保、变更受益人，或查询保单信息。"
    ),
    "product_recommend": (
        "您好！我可以根据您的情况推荐合适的保险。"
        "请问您的年龄是多少？主要关注哪方面保障（健康/意外/养老/子女教育）？"
    ),
    "other": (
        "您好！为了更好地帮助您，我需要再确认一下您的需求。"
        "请问您想了解以下哪方面？"
    ),
}

# 参考分类 → 结构化澄清问题模板
CATEGORY_CLARIFICATION_QUESTIONS: dict[str, list[tuple[str, str, str | None]]] = {
    "product_inquiry": [
        ("q_insurance_type", "您想了解哪类保险？例如重疾险、医疗险、意外险还是寿险？", "insurance_type"),
        ("q_product_name", "您是否有具体的产品名称？", "product_name"),
    ],
    "premium_inquiry": [
        ("q_product", "请问您想查询哪款产品的保费？", "product_name"),
        ("q_age_coverage", "请告知您的年龄和期望保额（如30岁、50万），方便估算。", "age"),
    ],
    "coverage_terms": [
        ("q_product", "请问您想查询哪款产品？", "product_name"),
        ("q_clause", "您想了解等待期、免责条款还是续保规则？", "clause_type"),
    ],
    "claims_service": [
        ("q_insurance_type", "请问是哪种险种的理赔？医疗险、重疾险还是意外险？", "claim_type"),
        ("q_claim_detail", "请问是住院、门诊还是其他类型的理赔？", "claim_type"),
    ],
    "product_compare": [
        ("q_products", "请告诉我您想对比哪两款产品？", "product_a"),
        ("q_dimension", "您主要关注保费、保障范围还是理赔条件？", "compare_dimension"),
    ],
    "product_recommend": [
        ("q_scenario", "请问您的主要场景是什么？如育儿、养老、出差出行？", "scenario"),
        ("q_age", "请问您的年龄是多少？", "age"),
    ],
    "purchase": [
        ("q_product", "请问您想购买哪款产品？", "product_name"),
        ("q_insurance_type", "或者您想购买哪类保险？", "insurance_type"),
    ],
    "policy_service": [
        ("q_service_type", "请问您需要办理续保、退保、变更还是查询保单？", "service_type"),
        ("q_policy_no", "如有保单号请提供，方便快速查询。", "policy_no"),
    ],
}

# 意图不明时的通用选项
DEFAULT_OPTIONS = [
    "了解某款保险产品的保障内容",
    "查询保费或费率",
    "咨询理赔流程或申请理赔",
    "对比不同保险产品",
    "根据我的情况推荐合适的保险",
    "办理续保、退保等保单服务",
]

# 通用澄清问题（意图完全不明时）
DEFAULT_CLARIFICATION_QUESTIONS = [
    ("q_intent", "您主要想了解保险哪方面？产品咨询、保费、理赔还是保单服务？", None),
    ("q_insurance_type", "您关注哪类保险？重疾/医疗/意外/寿险？", "insurance_type"),
    ("q_product", "您是否有具体的产品名称或保单？", "product_name"),
]


class ClarificationEngine:
    """判断是否需要澄清，并生成引导话术。"""

    def __init__(
        self,
        confidence_threshold: float | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold or settings.quality.clarification_confidence_threshold

    def evaluate(
        self,
        result: IntentResult,
        utterance: str,
        ctx: SessionContext,
        llm_clarification: Optional[dict] = None,
    ) -> ClarificationGuide:
        llm_clarification = llm_clarification or {}

        # LLM 已明确给出澄清引导时优先采用
        if llm_clarification.get("needs_clarification") and llm_clarification.get("guide_response"):
            questions = self._parse_llm_questions(llm_clarification)
            follow_ups = llm_clarification.get("follow_up_questions") or [q.question for q in questions]
            return ClarificationGuide(
                needs_clarification=True,
                reason=self._parse_reason(llm_clarification.get("reason", "")),
                guide_response=llm_clarification["guide_response"],
                clarification_questions=questions,
                suggested_options=llm_clarification.get("options") or [],
                follow_up_questions=follow_ups,
            )

        reason = self._detect_reason(result, utterance, ctx)
        if reason == ClarificationReason.NONE:
            return ClarificationGuide()

        guide = self._build_guide(result, utterance, ctx, reason)
        return guide

    def _detect_reason(
        self, result: IntentResult, utterance: str, ctx: SessionContext
    ) -> ClarificationReason:
        if result.category == "other" or result.confidence < self.confidence_threshold:
            if self._is_vague(utterance):
                return ClarificationReason.VAGUE_UTTERANCE
            if result.category == "other":
                return ClarificationReason.UNRECOGNIZED
            return ClarificationReason.LOW_CONFIDENCE

        if result.missing_info:
            return ClarificationReason.MISSING_INFO

        if self._is_ambiguous(result):
            return ClarificationReason.AMBIGUOUS

        if self._is_vague(utterance) and result.confidence < 0.85:
            return ClarificationReason.VAGUE_UTTERANCE

        # 分类明确但关键槽位缺失
        if self._has_critical_missing_slots(result):
            return ClarificationReason.MISSING_INFO

        return ClarificationReason.NONE

    def _is_vague(self, utterance: str) -> bool:
        text = utterance.strip()
        if len(text) <= 4:
            return True
        return any(re.search(p, text) for p in VAGUE_PATTERNS)

    def _is_ambiguous(self, result: IntentResult) -> bool:
        if len(result.sub_intents) < 2:
            return False
        confs = sorted([s.confidence for s in result.sub_intents], reverse=True)
        if len(confs) >= 2 and confs[0] - confs[1] < 0.15:
            return True
        categories = {s.category for s in result.sub_intents}
        return len(categories) >= 2 and result.confidence < 0.88

    def _has_critical_missing_slots(self, result: IntentResult) -> bool:
        critical_by_category = {
            "premium_inquiry": ["product_name"],
            "coverage_terms": ["product_name"],
            "claims_service": ["product_name"],
            "purchase": ["product_name"],
            "product_compare": ["product_a", "product_b"],
        }
        required = critical_by_category.get(result.category, [])
        if not required:
            return False
        filled = {k for k, v in result.slots.items() if v.value is not None}
        return any(s not in filled for s in required)

    def _build_guide(
        self,
        result: IntentResult,
        utterance: str,
        ctx: SessionContext,
        reason: ClarificationReason,
    ) -> ClarificationGuide:
        options = self._build_options(result, reason)
        follow_ups = self._build_follow_up_questions(result, reason)

        if reason == ClarificationReason.VAGUE_UTTERANCE:
            guide = (
                "您好！我注意到您的描述比较简略，为了更准确地帮到您，"
                "请问您具体想了解以下哪方面呢？"
            )
        elif reason == ClarificationReason.UNRECOGNIZED:
            guide = (
                "抱歉，我还不太确定您的具体需求。"
                "作为保险智能客服，我可以帮您处理以下常见问题，请选择或补充说明："
            )
        elif reason == ClarificationReason.LOW_CONFIDENCE:
            guide = (
                f"您好！我理解您可能是想「{result.intent_label}」，"
                f"但还不太确定。能否请您再具体说明一下？"
            )
        elif reason == ClarificationReason.AMBIGUOUS:
            labels = [s.intent_label for s in result.sub_intents[:3]]
            guide = (
                "您好！我注意到您的问题可能包含多个方面："
                + "；".join(labels)
                + "。请问您想先了解哪一个？"
            )
        elif reason == ClarificationReason.MISSING_INFO:
            guide = self._build_missing_info_guide(result, ctx)
        else:
            guide = CATEGORY_GUIDE_TEMPLATES.get(result.category, CATEGORY_GUIDE_TEMPLATES["other"])

        if not options:
            options = DEFAULT_OPTIONS[:4]

        questions = self._build_clarification_questions(result, reason, follow_ups)
        guide = self._compose_guide_with_questions(guide, questions)

        return ClarificationGuide(
            needs_clarification=True,
            reason=reason,
            guide_response=guide,
            clarification_questions=questions,
            suggested_options=options,
            follow_up_questions=follow_ups or [q.question for q in questions],
        )

    def _build_missing_info_guide(self, result: IntentResult, ctx: SessionContext) -> str:
        base = CATEGORY_GUIDE_TEMPLATES.get(result.category, "")
        missing = result.missing_info or self._infer_missing_slots(result)
        if not missing:
            return base or "您好！还需要您补充一些信息，以便我更好地为您服务。"

        parts = []
        for item in missing[:3]:
            label = SLOT_LABELS.get(item, item)
            parts.append(label)

        missing_text = "、".join(parts)
        if base:
            return f"{base}\n\n另外，还需要您提供：{missing_text}。"
        return f"您好！为了准确处理您的「{result.intent_label}」，请问您能告诉我{missing_text}吗？"

    def _infer_missing_slots(self, result: IntentResult) -> List[str]:
        critical = {
            "premium_inquiry": ["product_name"],
            "coverage_terms": ["product_name"],
            "product_compare": ["product_a", "product_b"],
        }.get(result.category, [])
        filled = {k for k, v in result.slots.items() if v.value is not None}
        return [s for s in critical if s not in filled]

    def _build_options(self, result: IntentResult, reason: ClarificationReason) -> List[str]:
        if reason == ClarificationReason.AMBIGUOUS:
            return [s.intent_label for s in result.sub_intents[:4]]

        if reason in (ClarificationReason.VAGUE_UTTERANCE, ClarificationReason.UNRECOGNIZED):
            return DEFAULT_OPTIONS[:5]

        if reason == ClarificationReason.MISSING_INFO:
            cat = CATEGORY_BY_CODE.get(result.category)
            if cat and cat.examples:
                return [f"例如：{cat.examples[0]}"] + cat.examples[1:3]
            return DEFAULT_OPTIONS[:3]

        return DEFAULT_OPTIONS[:4]

    def _build_follow_up_questions(
        self, result: IntentResult, reason: ClarificationReason
    ) -> List[str]:
        questions = []
        missing = result.missing_info or self._infer_missing_slots(result)
        for slot in missing[:3]:
            label = SLOT_LABELS.get(slot, slot)
            questions.append(f"请问您的{label}是？")
        if not questions and reason == ClarificationReason.VAGUE_UTTERANCE:
            questions = [
                "您想了解哪类保险（重疾/医疗/意外/寿险）？",
                "您是否有具体的产品名称？",
            ]
        return questions

    @staticmethod
    def _parse_reason(raw: str) -> ClarificationReason:
        mapping = {
            "low_confidence": ClarificationReason.LOW_CONFIDENCE,
            "missing_info": ClarificationReason.MISSING_INFO,
            "ambiguous": ClarificationReason.AMBIGUOUS,
            "vague": ClarificationReason.VAGUE_UTTERANCE,
            "vague_utterance": ClarificationReason.VAGUE_UTTERANCE,
            "unrecognized": ClarificationReason.UNRECOGNIZED,
            "模糊": ClarificationReason.VAGUE_UTTERANCE,
            "信息不足": ClarificationReason.MISSING_INFO,
            "意图不明": ClarificationReason.UNRECOGNIZED,
        }
        return mapping.get(raw.lower(), ClarificationReason.LOW_CONFIDENCE)

    def _build_clarification_questions(
        self,
        result: IntentResult,
        reason: ClarificationReason,
        follow_ups: List[str],
    ) -> List[ClarificationQuestion]:
        questions: List[ClarificationQuestion] = []

        if reason == ClarificationReason.AMBIGUOUS and result.sub_intents:
            for i, si in enumerate(result.sub_intents[:3], 1):
                questions.append(ClarificationQuestion(
                    question_id=f"q_ambig_{i}",
                    question=f"您是想「{si.intent_label}」吗？",
                    purpose=f"确认是否为{get_category_name(si.category)}",
                    fills_slot=None,
                    priority=i,
                ))

        templates = CATEGORY_CLARIFICATION_QUESTIONS.get(result.category, [])
        if reason in (ClarificationReason.MISSING_INFO, ClarificationReason.LOW_CONFIDENCE) and templates:
            for i, (qid, qtext, slot) in enumerate(templates, len(questions) + 1):
                questions.append(ClarificationQuestion(
                    question_id=qid, question=qtext, purpose="补充关键信息",
                    fills_slot=slot, priority=i,
                ))

        if reason in (ClarificationReason.VAGUE_UTTERANCE, ClarificationReason.UNRECOGNIZED):
            for i, (qid, qtext, slot) in enumerate(DEFAULT_CLARIFICATION_QUESTIONS, len(questions) + 1):
                questions.append(ClarificationQuestion(
                    question_id=qid, question=qtext, purpose="明确用户诉求",
                    fills_slot=slot, priority=i,
                ))

        for i, fq in enumerate(follow_ups, len(questions) + 1):
            if not any(q.question == fq for q in questions):
                questions.append(ClarificationQuestion(
                    question_id=f"q_follow_{i}", question=fq, purpose="追问细节", priority=i,
                ))

        return questions[:3]

    @staticmethod
    def _compose_guide_with_questions(guide: str, questions: List[ClarificationQuestion]) -> str:
        if not questions:
            return guide
        q_lines = "\n".join(f"  {i}. {q.question}" for i, q in enumerate(questions, 1))
        return f"{guide}\n\n为了更准确帮到您，请回答以下问题（可直接回复序号或描述）：\n{q_lines}"

    @staticmethod
    def _parse_llm_questions(llm_clarification: dict) -> List[ClarificationQuestion]:
        raw = llm_clarification.get("clarification_questions") or []
        questions = []
        for i, item in enumerate(raw):
            if isinstance(item, str):
                questions.append(ClarificationQuestion(
                    question_id=f"q_llm_{i}", question=item, purpose="LLM 生成追问", priority=i + 1,
                ))
            elif isinstance(item, dict) and item.get("question"):
                questions.append(ClarificationQuestion(
                    question_id=item.get("question_id", f"q_llm_{i}"),
                    question=item["question"],
                    purpose=item.get("purpose", ""),
                    fills_slot=item.get("fills_slot"),
                    priority=item.get("priority", i + 1),
                ))
        if not questions:
            for i, fq in enumerate(llm_clarification.get("follow_up_questions") or []):
                questions.append(ClarificationQuestion(
                    question_id=f"q_llm_{i}", question=fq, purpose="LLM 追问", priority=i + 1,
                ))
        return questions[:3]
