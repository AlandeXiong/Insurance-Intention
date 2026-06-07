"""保险行业意图定义与轻量级规则库。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class IntentDefinition:
    name: str
    description: str
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    implicit_triggers: List[str] = field(default_factory=list)
    related_intents: List[str] = field(default_factory=list)


INSURANCE_INTENTS: Dict[str, IntentDefinition] = {
    "query_critical_illness_premium": IntentDefinition(
        name="query_critical_illness_premium",
        description="查询重疾险保费",
        required_slots=["product_name"],
        optional_slots=["age", "coverage_amount", "payment_period"],
        keywords=["重疾险", "保费", "多少钱", "价格", "费率"],
        related_intents=["query_waiting_period", "compare_products"],
    ),
    "query_waiting_period": IntentDefinition(
        name="query_waiting_period",
        description="查询等待期",
        required_slots=["product_name"],
        optional_slots=[],
        keywords=["等待期", "多久生效", "观察期"],
        related_intents=["query_critical_illness_premium", "query_medical_claim"],
    ),
    "query_medical_claim": IntentDefinition(
        name="query_medical_claim",
        description="医疗险理赔流程",
        required_slots=["product_name"],
        optional_slots=["claim_type", "hospital"],
        keywords=["理赔", "报销", "医疗险", "住院", "门诊"],
        related_intents=["query_waiting_period"],
    ),
    "recommend_accident_insurance": IntentDefinition(
        name="recommend_accident_insurance",
        description="推荐意外险（含隐式触发）",
        required_slots=[],
        optional_slots=["travel_frequency", "occupation"],
        keywords=["意外险", "意外保障", "出差", "旅行"],
        implicit_triggers=["经常出差", "经常旅行", "高危职业", "户外"],
        related_intents=["query_critical_illness_premium"],
    ),
    "compare_products": IntentDefinition(
        name="compare_products",
        description="产品对比",
        required_slots=["product_a", "product_b"],
        optional_slots=["compare_dimension"],
        keywords=["对比", "比较", "哪个好", "区别"],
        related_intents=["query_critical_illness_premium"],
    ),
    "purchase_intent": IntentDefinition(
        name="purchase_intent",
        description="购买/投保意向",
        required_slots=["product_name"],
        optional_slots=["budget"],
        keywords=["购买", "投保", "买一份", "怎么买", "下单"],
        related_intents=["query_critical_illness_premium"],
    ),
    "greeting": IntentDefinition(
        name="greeting",
        description="问候",
        required_slots=[],
        keywords=["你好", "您好", "在吗", "hello"],
    ),
    "fallback": IntentDefinition(
        name="fallback",
        description="无法识别",
        required_slots=[],
        keywords=[],
    ),
}

# 产品实体库 — 用于指代消解
PRODUCT_ENTITIES: Dict[str, List[str]] = {
    "安心保重疾险2026": ["安心保", "安心保重疾", "这款重疾", "它", "这个"],
    "康乐医疗险Plus": ["康乐医疗", "康乐Plus", "医疗险", "这款医疗"],
    "畅行意外险": ["畅行意外", "意外险", "出差险"],
}

# 轻量级引擎：关键词 → 意图 快速映射
KEYWORD_INTENT_RULES: List[Tuple[List[str], str, float]] = [
    (["等待期", "观察期", "多久生效"], "query_waiting_period", 0.92),
    (["理赔", "报销", "怎么赔"], "query_medical_claim", 0.90),
    (["重疾险", "保费", "多少钱"], "query_critical_illness_premium", 0.91),
    (["对比", "比较", "哪个好"], "compare_products", 0.88),
    (["购买", "投保", "怎么买"], "purchase_intent", 0.89),
    (["你好", "您好"], "greeting", 0.95),
]

# 隐式意图映射
IMPLICIT_INTENT_RULES: Dict[str, str] = {
    "经常出差": "recommend_accident_insurance",
    "经常旅行": "recommend_accident_insurance",
    "高危职业": "recommend_accident_insurance",
    "户外": "recommend_accident_insurance",
    "住院": "query_medical_claim",
    "手术": "query_medical_claim",
}

INTENT_TAXONOMY = list(INSURANCE_INTENTS.keys())
