"""Intent clarification guidance — detect vague intents and generate agent guidance copy."""

from __future__ import annotations

import re
from typing import List, Optional

from config.settings import settings
from src.domain.insurance_domain import CATEGORY_BY_CODE, get_category_name
from src.models.clarification import ClarificationGuide, ClarificationQuestion, ClarificationReason, PendingClarification
from src.models.intent import IntentResult, SessionContext

# Vague utterance patterns
VAGUE_PATTERNS = [
    r"^(learn|inquire|ask|want to know|help me|look into).{0,20}$",
    r"^(that|this|it)( one| product| plan)?\.?$",
    r"^how (do i|can i|does it work)\??$",
    r"^about (the )?insurance",
    r"^regarding",
]

# Slot name → English follow-up labels (runtime user-facing)
SLOT_LABELS = {
    "product_name": "which insurance product you'd like to learn about",
    "product_a": "first product to compare",
    "product_b": "second product to compare",
    "age": "your age",
    "gender": "your gender",
    "coverage_amount": "desired coverage amount (e.g., $500K)",
    "payment_period": "planned payment term",
    "budget": "budget range",
    "insurance_type": "insurance type (critical illness / medical / accident / life)",
    "claim_type": "claim type (inpatient / outpatient / surgery, etc.)",
    "policy_no": "policy number",
    "scenario": "use case (e.g., business travel, parenting, retirement)",
}

# Reference category → guidance templates when info is missing (runtime user-facing)
CATEGORY_GUIDE_TEMPLATES = {
    "product_inquiry": (
        "Hello! I can help you learn about insurance product coverage. "
        "Which type of insurance are you interested in? For example: critical illness, "
        "medical, accident, or life insurance?"
    ),
    "premium_inquiry": (
        "Hello! Premiums vary by product, age, coverage amount, and payment method. "
        "Which product's premium would you like to check? Or share your age and desired "
        "coverage amount and I can estimate it for you."
    ),
    "coverage_terms": (
        "Hello! Coverage terms differ by product. "
        "Which product would you like to ask about? Are you looking for the waiting period, "
        "exclusions, or renewal rules?"
    ),
    "claims_service": (
        "Hello! I can walk you through the claims process. "
        "Which type of insurance claim is this (medical / critical illness / accident)? "
        "Do you already have a policy?"
    ),
    "purchase": (
        "Hello! Glad to hear you're interested in purchasing coverage. "
        "Which type of product are you looking for? Do you already have a product in mind?"
    ),
    "product_compare": (
        "Hello! I can help you compare insurance products. "
        "Tell me which two products you'd like to compare, or what matters most "
        "(premium / coverage / claims)."
    ),
    "policy_service": (
        "Hello! Which policy service do you need help with? "
        "For example: renewal, surrender, beneficiary change, or policy lookup."
    ),
    "product_recommend": (
        "Hello! I can recommend suitable insurance based on your situation. "
        "How old are you? What coverage matters most (health / accident / retirement / children's education)?"
    ),
    "other": (
        "Hello! To help you better, I need to confirm your request. "
        "Which of the following are you interested in?"
    ),
}

# Reference category → structured clarification question templates (runtime user-facing)
CATEGORY_CLARIFICATION_QUESTIONS: dict[str, list[tuple[str, str, str | None]]] = {
    "product_inquiry": [
        ("q_insurance_type", "Which type of insurance are you interested in? Critical illness, medical, accident, or life?", "insurance_type"),
        ("q_product_name", "Do you have a specific product name in mind?", "product_name"),
    ],
    "premium_inquiry": [
        ("q_product", "Which product's premium would you like to check?", "product_name"),
        ("q_age_coverage", "Please share your age and desired coverage amount (e.g., age 30, $500K) so I can estimate.", "age"),
    ],
    "coverage_terms": [
        ("q_product", "Which product would you like to ask about?", "product_name"),
        ("q_clause", "Are you asking about the waiting period, exclusions, or renewal rules?", "clause_type"),
    ],
    "claims_service": [
        ("q_insurance_type", "Which type of insurance claim is this? Medical, critical illness, or accident?", "claim_type"),
        ("q_claim_detail", "Is this for inpatient, outpatient, or another type of claim?", "claim_type"),
    ],
    "product_compare": [
        ("q_products", "Which two products would you like to compare?", "product_a"),
        ("q_dimension", "Do you care most about premium, coverage scope, or claims conditions?", "compare_dimension"),
    ],
    "product_recommend": [
        ("q_scenario", "What is your main scenario? Parenting, retirement, business travel?", "scenario"),
        ("q_age", "How old are you?", "age"),
    ],
    "purchase": [
        ("q_product", "Which product would you like to purchase?", "product_name"),
        ("q_insurance_type", "Or which type of insurance are you looking for?", "insurance_type"),
    ],
    "policy_service": [
        ("q_service_type", "Do you need renewal, surrender, a change, or a policy lookup?", "service_type"),
        ("q_policy_no", "If you have a policy number, please share it for a faster lookup.", "policy_no"),
    ],
}

# Default options when intent is unclear (runtime user-facing)
DEFAULT_OPTIONS = [
    "Learn about coverage for a specific insurance product",
    "Check premium or rates",
    "Ask about the claims process or file a claim",
    "Compare different insurance products",
    "Get a recommendation based on my situation",
    "Handle renewal, surrender, or other policy services",
]

# Default clarification questions when intent is fully unclear (runtime user-facing)
DEFAULT_CLARIFICATION_QUESTIONS = [
    ("q_intent", "What would you like help with? Product info, premium, claims, or policy services?", None),
    ("q_insurance_type", "Which type of insurance interests you? Critical illness / medical / accident / life?", "insurance_type"),
    ("q_product", "Do you have a specific product name or policy?", "product_name"),
]


class ClarificationEngine:
    """Decide whether clarification is needed and generate guidance copy."""

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

        # Prefer LLM-provided clarification guidance when explicitly present
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

        # Category is clear but critical slots are missing
        if self._has_critical_missing_slots(result):
            return ClarificationReason.MISSING_INFO

        return ClarificationReason.NONE

    def _is_vague(self, utterance: str) -> bool:
        text = utterance.strip()
        if len(text) <= 4:
            return True
        return any(re.search(p, text, re.IGNORECASE) for p in VAGUE_PATTERNS)

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
                "Hello! Your message is a bit brief, so I'd like to understand you better. "
                "Which of the following are you interested in?"
            )
        elif reason == ClarificationReason.UNRECOGNIZED:
            guide = (
                "Sorry, I'm not quite sure what you need yet. "
                "As your insurance assistant, I can help with the following — please choose or add details:"
            )
        elif reason == ClarificationReason.LOW_CONFIDENCE:
            guide = (
                f"Hello! I think you may want to \"{result.intent_label}\", "
                f"but I'm not fully sure yet. Could you tell me a bit more?"
            )
        elif reason == ClarificationReason.AMBIGUOUS:
            labels = [s.intent_label for s in result.sub_intents[:3]]
            guide = (
                "Hello! Your question may cover several topics: "
                + "; ".join(labels)
                + ". Which one would you like to start with?"
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
            return base or "Hello! I still need a bit more information to help you better."

        parts = []
        for item in missing[:3]:
            label = SLOT_LABELS.get(item, item)
            parts.append(label)

        missing_text = ", ".join(parts)
        if base:
            return f"{base}\n\nI also need: {missing_text}."
        return f"Hello! To handle your \"{result.intent_label}\" request accurately, could you share {missing_text}?"

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
                return [f"e.g.: {cat.examples[0]}"] + cat.examples[1:3]
            return DEFAULT_OPTIONS[:3]

        return DEFAULT_OPTIONS[:4]

    def _build_follow_up_questions(
        self, result: IntentResult, reason: ClarificationReason
    ) -> List[str]:
        questions = []
        missing = result.missing_info or self._infer_missing_slots(result)
        for slot in missing[:3]:
            label = SLOT_LABELS.get(slot, slot)
            questions.append(f"Could you share your {label}?")
        if not questions and reason == ClarificationReason.VAGUE_UTTERANCE:
            questions = [
                "Which type of insurance interests you (critical illness / medical / accident / life)?",
                "Do you have a specific product name in mind?",
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
                    question=f"Did you mean \"{si.intent_label}\"?",
                    purpose=f"Confirm whether this is {get_category_name(si.category)}",
                    fills_slot=None,
                    priority=i,
                ))

        templates = CATEGORY_CLARIFICATION_QUESTIONS.get(result.category, [])
        if reason in (ClarificationReason.MISSING_INFO, ClarificationReason.LOW_CONFIDENCE) and templates:
            for i, (qid, qtext, slot) in enumerate(templates, len(questions) + 1):
                questions.append(ClarificationQuestion(
                    question_id=qid, question=qtext, purpose="Fill in key details",
                    fills_slot=slot, priority=i,
                ))

        if reason in (ClarificationReason.VAGUE_UTTERANCE, ClarificationReason.UNRECOGNIZED):
            for i, (qid, qtext, slot) in enumerate(DEFAULT_CLARIFICATION_QUESTIONS, len(questions) + 1):
                questions.append(ClarificationQuestion(
                    question_id=qid, question=qtext, purpose="Clarify user intent",
                    fills_slot=slot, priority=i,
                ))

        for i, fq in enumerate(follow_ups, len(questions) + 1):
            if not any(q.question == fq for q in questions):
                questions.append(ClarificationQuestion(
                    question_id=f"q_follow_{i}", question=fq, purpose="Follow up on details", priority=i,
                ))

        return questions[:3]

    @staticmethod
    def _compose_guide_with_questions(guide: str, questions: List[ClarificationQuestion]) -> str:
        if not questions:
            return guide
        q_lines = "\n".join(f"  {i}. {q.question}" for i, q in enumerate(questions, 1))
        return f"{guide}\n\nTo help you more accurately, please answer the following (reply with a number or describe in your own words):\n{q_lines}"

    @staticmethod
    def _parse_llm_questions(llm_clarification: dict) -> List[ClarificationQuestion]:
        raw = llm_clarification.get("clarification_questions") or []
        questions = []
        for i, item in enumerate(raw):
            if isinstance(item, str):
                questions.append(ClarificationQuestion(
                    question_id=f"q_llm_{i}", question=item, purpose="LLM-generated follow-up", priority=i + 1,
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
                    question_id=f"q_llm_{i}", question=fq, purpose="LLM follow-up", priority=i + 1,
                ))
        return questions[:3]
